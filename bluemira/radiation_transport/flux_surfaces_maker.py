# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A simplified 2-D solver for calculating charged particle heat loads.
"""

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import plot_coordinates
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.flux_surfaces import OpenFluxSurface
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from bluemira.radiation_transport.error import AdvectionTransportError

__all__ = ["FluxSurfaceMaker"]


class FluxSurfaceMaker:
    """
    A class to extract flux surfaces from a given equilibrium.

    Parameters
    ----------
    equilibrium: Equilibrium
        The equilibrium defining flux surfaces.
    dx_mp: float (optional)
        The midplane spatial resolution between flux surfaces [m]
        (default: 0.001).
    """

    def __init__(self, equilibrium, dx_mp: float = 0.001):
        self.eq = equilibrium
        self.dx_mp = dx_mp

        # Constructors
        self.first_wall = None
        self.flux_surfaces_ob_lfs = None
        self.flux_surfaces_ob_hfs = None
        self.flux_surfaces_ib_lfs = None
        self.flux_surfaces_ib_hfs = None
        self.x_sep_omp = None
        self.x_sep_imp = None
        self.result = None

        # Pre-processing
        o_points, _ = self.eq.get_OX_points()
        self._o_point = o_points[0]
        z = self._o_point.z
        self._yz_plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [1, 1, z])

    @property
    def flux_surfaces(self):
        """
        All flux surfaces in the ChargedParticleSolver.

        Returns
        -------
        flux_surfaces: List[PartialOpenFluxSurface]
        """
        flux_surfaces = []
        for group in [
            self.flux_surfaces_ob_lfs,
            self.flux_surfaces_ob_hfs,
            self.flux_surfaces_ib_lfs,
            self.flux_surfaces_ib_hfs,
        ]:
            if group:
                flux_surfaces.extend(group)
        return flux_surfaces

    @staticmethod
    def _process_first_wall(first_wall):
        """
        Force working first wall geometry to be closed and counter-clockwise.
        """
        first_wall = deepcopy(first_wall)

        if not first_wall.check_ccw(axis=[0, 1, 0]):
            bluemira_warn(
                "First wall should be oriented counter-clockwise. Reversing it."
            )
            first_wall.reverse()

        if not first_wall.closed:
            bluemira_warn("First wall should be a closed geometry. Closing it.")
            first_wall.close()
        return first_wall

    @staticmethod
    def _get_array_x_mp(flux_surfaces):
        """
        Get x_mp array of flux surface values.
        """
        return np.array([fs.x_start for fs in flux_surfaces])

    def _get_array_z_mp(flux_surfaces):
        """
        Get z_mp array of flux surface values.
        """
        return np.array([fs.z_start for fs in flux_surfaces])
    
    def _get_array_x_fw(flux_surfaces):
        """
        Get x_fw array of flux surface values.
        """
        return np.array([fs.x_end for fs in flux_surfaces])

    def _get_array_z_fw(flux_surfaces):
        """
        Get z_fw array of flux surface values.
        """
        return np.array([fs.z_end for fs in flux_surfaces])
    
    def _get_array_alpha(flux_surfaces):
        """
        Get alpha angle array of flux surface values.
        """
        return np.array([fs.alpha for fs in flux_surfaces])
    
    def _get_sep_out_intersection(self, outboard=True):
        """
        Find the middle and maximum outboard mid-plane psi norm values
        """
        yz_plane = self._yz_plane
        o_point = self._o_point
        separatrix = self.eq.get_separatrix()

        if not isinstance(separatrix, Coordinates):
            sep1_intersections = coords_plane_intersect(separatrix[0], yz_plane)
            sep2_intersections = coords_plane_intersect(separatrix[1], yz_plane)
            sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - o_point.x))
            sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - o_point.x))
            x_sep1_mp = sep1_intersections.T[0][sep1_arg]
            x_sep2_mp = sep2_intersections.T[0][sep2_arg]
            if outboard:
                x_sep_mp = x_sep1_mp if x_sep1_mp > x_sep2_mp else x_sep2_mp
            else:
                x_sep_mp = x_sep1_mp if x_sep1_mp < x_sep2_mp else x_sep2_mp
        else:
            sep_intersections = coords_plane_intersect(separatrix, yz_plane)
            sep_arg = np.argmin(np.abs(sep_intersections.T[0] - o_point.x))
            x_sep_mp = sep_intersections.T[0][sep_arg]

        out_intersections = coords_plane_intersect(self.first_wall, yz_plane)
        if outboard:
            x_out_mp = np.max(out_intersections.T[0])
        else:
            x_out_mp = np.min(out_intersections.T[0])

        return x_sep_mp, x_out_mp

    def _make_flux_surfaces(self, x, z):
        """
        Make individual PartialOpenFluxSurfaces through a point.
        """
        coords = find_flux_surface_through_point(
            self.eq.x, self.eq.z, self.eq.psi(), x, z, self.eq.psi(x, z)
        )
        coords = Coordinates({"x": coords[0], "z": coords[1]})
        f_s = OpenFluxSurface(coords)
        lfs, hfs = f_s.split(self._o_point, plane=self._yz_plane)
        return lfs, hfs

    def _make_flux_surfaces_ob(self):
        """
        Make the flux surfaces on the outboard.
        """
        self.x_sep_omp, x_out_omp = self._get_sep_out_intersection(outboard=True)

        self.flux_surfaces_ob_lfs = []
        self.flux_surfaces_ob_hfs = []

        for x in np.arange(self.x_sep_omp + self.dx_mp, x_out_omp - EPS, self.dx_mp):
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ob_lfs.append(lfs)
            self.flux_surfaces_ob_hfs.append(hfs)

    def _make_flux_surfaces_ib(self):
        """
        Make the flux surfaces on the inboard.
        """
        self.x_sep_imp, x_out_imp = self._get_sep_out_intersection(outboard=False)

        self.flux_surfaces_ib_lfs = []
        self.flux_surfaces_ib_hfs = []

        for x in np.arange(self.x_sep_imp - self.dx_mp, x_out_imp + EPS, -self.dx_mp):
            lfs, hfs = self._make_flux_surfaces(x, self._o_point.z)
            self.flux_surfaces_ib_lfs.append(lfs)
            self.flux_surfaces_ib_hfs.append(hfs)    

    def _clip_flux_surfaces(self, first_wall):
        """
        Clip the flux surfaces to a first wall. Catch the cases where no intersections
        are found.
        """
        for group in [
            self.flux_surfaces_ob_lfs,
            self.flux_surfaces_ob_hfs,
            self.flux_surfaces_ib_lfs,
            self.flux_surfaces_ib_hfs,
        ]:
            if group:
                for i, flux_surface in enumerate(group):
                    flux_surface.clip(first_wall)
                    if flux_surface.alpha is None:
                        # No intersection detected between flux surface and first wall
                        # Drop the flux surface from the group
                        group.pop(i)

    def analyse(self, first_wall):
        """
        Perform the calculation to obtain charged particle heat fluxes on the
        the specified first_wall

        Parameters
        ----------
        first_wall: Coordinates
            The closed first wall geometry on which to calculate the heat flux

        Returns
        -------
        x: np.array
            The x coordinates of the flux surface intersections
        z: np.array
            The z coordinates of the flux surface intersections
        heat_flux: np.array
            The perpendicular heat fluxes at the intersection points [MW/m^2]
        """
        self.first_wall = self._process_first_wall(first_wall)

        if self.eq.is_double_null:
            self.dx_omp = self._analyse_DN(first_wall)
            return self.dx_omp
        else:
            self.dx_omp, self.dx_imp = self._analyse_SN(first_wall)
            return self.dx_omp, self.dx_imp

    def _analyse_SN(self, first_wall):
        """
        Calculation for the case of single nulls.
        """
        self._make_flux_surfaces_ob()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(first_wall)

        x_omp = self._get_array_x_mp(
            self.flux_surfaces_ob_lfs
        )

        # Calculate values at OMP
        return x_omp - self.x_sep_omp

    def _analyse_DN(self, first_wall):
        """
        Calculation for the case of double nulls.
        """
        self._make_flux_surfaces_ob()
        self._make_flux_surfaces_ib()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(first_wall)

        x_omp = self._get_array_x_mp(self.flux_surfaces_ob_lfs)

        x_imp = self._get_array_x_mp(self.flux_surfaces_ib_lfs)

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        # Calculate values at IMP
        dx_imp = abs(x_imp - self.x_sep_imp)
        
        return dx_omp, dx_imp