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

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.flux_surfaces import OpenFluxSurface
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.wire import BluemiraWire

__all__ = ["analyse_first_wall_flux_surfaces"]


def analyse_first_wall_flux_surfaces(
    equilibrium: Equilibrium, first_wall: BluemiraWire, dx_mp: float
):
    """
    A simplified charged particle transport model along open field lines.

    Parameters
    ----------
    equilibrium:
        The equilibrium defining flux surfaces.
    first_wall:
    dx_mp:
        The midplane spatial resolution between flux surfaces [m]
        (default: 0.001).
    """
    o_point = equilibrium.get_OX_points()[0][0]  # 1st o_point
    z = o_point.z
    yz_plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [1, 1, z])

    first_wall = _process_first_wall(first_wall)

    if equilibrium.is_double_null:
        dx_omp, dx_imp, flux_surfaces, x_sep_omp, x_sep_imp = _analyse_DN(
            first_wall, dx_mp, equilibrium, o_point, yz_plane
        )
    else:
        dx_omp, flux_surfaces, x_sep_omp = _analyse_SN(
            first_wall, dx_mp, equilibrium, o_point, yz_plane
        )
        dx_imp = None
        x_sep_imp = None
    return dx_omp, dx_imp, flux_surfaces, x_sep_omp, x_sep_imp


def _process_first_wall(first_wall: Coordinates) -> Coordinates:
    """
    Force working first wall geometry to be closed and counter-clockwise.
    """
    first_wall = deepcopy(first_wall)

    if not first_wall.check_ccw(axis=[0, 1, 0]):
        bluemira_warn("First wall should be oriented counter-clockwise. Reversing it.")
        first_wall.reverse()

    if not first_wall.closed:
        bluemira_warn("First wall should be a closed geometry. Closing it.")
        first_wall.close()
    return first_wall


def _analyse_SN(first_wall, dx_mp, equilibrium, o_point, yz_plane):
    """
    Calculation for the case of single nulls.
    """
    x_sep_omp, x_out_omp = _get_sep_out_intersection(
        equilibrium.get_separatrix(), first_wall, yz_plane, o_point, outboard=True
    )

    flux_surfaces_ob = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_omp, x_out_omp, outboard=True
    )

    return (
        get_array_x_mp(flux_surfaces_ob[0]) - x_sep_omp,  # Calculate values at OMP
        _clip_flux_surfaces(  # Find flux surface intersections with the first wall
            first_wall, flux_surfaces_ob
        ),
        x_sep_omp,
    )


def _analyse_DN(first_wall, dx_mp, equilibrium, o_point, yz_plane):
    """
    Calculation for the case of double nulls.
    """
    separatrix = equilibrium.get_separatrix()
    x_sep_omp, x_out_omp = _get_sep_out_intersection(
        separatrix, first_wall, yz_plane, o_point, outboard=True
    )
    x_sep_imp, x_out_imp = _get_sep_out_intersection(
        separatrix, first_wall, yz_plane, o_point, outboard=False
    )

    flux_surfaces_ob = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_omp, x_out_omp, outboard=True
    )
    flux_surfaces_ib = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_imp, x_out_imp, outboard=False
    )
    return (
        get_array_x_mp(flux_surfaces_ob[0]) - x_sep_omp,  # Calculate values at OMP
        abs(get_array_x_mp(flux_surfaces_ib[0]) - x_sep_imp),  # Calculate values at IMP
        _clip_flux_surfaces(  # Find flux surface intersections with the first wall
            first_wall,
            [*flux_surfaces_ob, *flux_surfaces_ib],
        ),
        x_sep_omp,
        x_sep_imp,
    )


def _clip_flux_surfaces(first_wall, flux_surfaces):
    """
    Clip the flux surfaces to a first wall. Catch the cases where no intersections
    are found.
    """
    for group in flux_surfaces:
        for flux_surface in group:
            flux_surface.clip(first_wall)
            if flux_surface.alpha is not None:
                # No intersection detected between flux surface and first wall
                # Drop the flux surface from the group
                break
    return flux_surfaces


def get_array_x_mp(flux_surfaces):
    """
    Get x_mp array of flux surface values.
    """
    return np.array([fs.x_start for fs in flux_surfaces])


def get_array_z_mp(flux_surfaces):
    """
    Get z_mp array of flux surface values.
    """
    return np.array([fs.z_start for fs in flux_surfaces])


def get_array_x_fw(flux_surfaces):
    """
    Get x_fw array of flux surface values.
    """
    return np.array([fs.x_end for fs in flux_surfaces])


def get_array_z_fw(flux_surfaces):
    """
    Get z_fw array of flux surface values.
    """
    return np.array([fs.z_end for fs in flux_surfaces])


def get_array_alpha(flux_surfaces):
    """
    Get alpha angle array of flux surface values.
    """
    return np.array([fs.alpha for fs in flux_surfaces])


def _get_sep_out_intersection(
    separatrix, first_wall, yz_plane, o_point, *, outboard=True
):
    """
    Find the middle and maximum outboard mid-plane psi norm values
    """
    if not isinstance(separatrix, Coordinates):
        sep1_intersections = coords_plane_intersect(separatrix[0], yz_plane)
        sep2_intersections = coords_plane_intersect(separatrix[1], yz_plane)
        sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - o_point.x))
        sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - o_point.x))
        x_sep1_mp = sep1_intersections.T[0][sep1_arg]
        x_sep2_mp = sep2_intersections.T[0][sep2_arg]
        x_sep_mp = max(x_sep2_mp, x_sep1_mp) if outboard else min(x_sep2_mp, x_sep1_mp)
    else:
        sep_intersections = coords_plane_intersect(separatrix, yz_plane)
        sep_arg = np.argmin(np.abs(sep_intersections.T[0] - o_point.x))
        x_sep_mp = sep_intersections.T[0][sep_arg]

    out_intersections = coords_plane_intersect(first_wall, yz_plane)
    x_out_mp = (
        np.max(out_intersections.T[0]) if outboard else np.min(out_intersections.T[0])
    )
    return x_sep_mp, x_out_mp


def _make_flux_surfaces(x, z, equilibrium, o_point, yz_plane):
    """
    Make individual PartialOpenFluxSurfaces through a point.
    """
    coords = find_flux_surface_through_point(
        equilibrium.x, equilibrium.z, equilibrium.psi(), x, z, equilibrium.psi(x, z)
    )
    return OpenFluxSurface(Coordinates({"x": coords[0], "z": coords[1]})).split(
        o_point, plane=yz_plane
    )


def _make_flux_surfaces_ibob(
    dx_mp, equilibrium, o_point, yz_plane, x_sep_mp, x_out_mp, *, outboard: bool
):
    """
    Make the flux surfaces on the outboard.
    """
    sign = 1 if outboard else -1

    flux_surfaces_lfs = []
    flux_surfaces_hfs = []

    for x in np.arange(
        x_sep_mp + (sign * dx_mp), x_out_mp - (sign * EPS), (sign * dx_mp)
    ):
        lfs, hfs = _make_flux_surfaces(x, o_point.z, equilibrium, o_point, yz_plane)
        flux_surfaces_lfs.append(lfs)
        flux_surfaces_hfs.append(hfs)
    return flux_surfaces_lfs, flux_surfaces_hfs
