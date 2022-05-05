# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Poloidal field system
"""
from typing import Type

from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.coilCAD import PFSystemCAD
from BLUEPRINT.geometry.geomtools import get_boundary
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class PoloidalFieldCoils(Meshable, ReactorSystem):
    """
    Reactor poloidal field (PF) coil system
    """

    config: Type[ParameterFrame]
    # fmt: off
    default_params = [
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["n_PF", "Number of PF coils", 6, "dimensionless", None, "Input"],
        ["n_CS", "Number of CS coil divisions", 5, "dimensionless", None, "Input"],
    ]
    # fmt: on
    CADConstructor = PFSystemCAD

    def __init__(self, config):
        self.config = config

        self._init_params(self.config)
        self._plotter = PoloidalFieldCoilsPlotter()

    def update_coilset(self, coilset):
        """
        Passes a CoilSet object from equilibria into the ReactorSystem
        """
        self.coils = coilset.coils
        self._coilset = coilset

    def get_solenoid(self):
        """
        Get the central solenoid in the PFsystem.

        Returns
        -------
        solenoid: Solenoid
            The central solenoid object.
        """
        return self._coilset.get_solenoid()

    def generate_cross_sections(self, mesh_sizes, geometry_names=None, verbose=True):
        """
        Generate cross-section objects for this system.

        This cleans the points forming the geometries based on the `min_length`
        and `min_angle` using the :func:`~BLUEPRINT.geometry.geomtools.clean_loop_points`
        algorithm.

        For each requested geometry, clean points are then fed into a sectionproperties
        `CustomSection` object, with corresponding facets and control point. The
        geometry is cleaned, using the sectionproperties `clean_geometry` method, before
        creating a mesh and loading the mesh and geometry into a sectionproperties
        `CrossSection`.

        Also provides the points, facets, control points, and holes are used to generate
        the `CrossSection` objects. Points and facets correspond to the boundary of the
        geometries that have been used to generate the `CrossSection` objects.

        Parameters
        ----------
        mesh_sizes : List[float]
            A list of maximum element areas corresponding to each region
            within the cross-section geometry.
        geometry_names : List[str], optional
            A list of names in the system's geometry dictionary to generate
            cross-sections for.
            If None then all geometries for analysis in the X-Z plane will be
            used, by default None.
        verbose : bool, optional
            Determines if verbose mesh cleaning output should be provided,
            by default True.

        Returns
        -------
        List[`sectionproperties.analysis.cross_section.CrossSection`]
            The `CrossSection` objects representing the system.
        points : List[float, float]
            The points bounding the geometries used to generate the `CrossSection`
            objects.
        facets : List[int, int]
            The facets bounding the geometries used to generate the `CrossSection`
            objects.
        control_points : List[float, float]
            The control points used to generate the `CrossSection` objects.
        holes : List[float, float]
            The holes used to generate the `CrossSection` objects.
        """
        points = []
        facets = []
        control_points = []
        holes = []
        polygons = []
        cross_sections, coil_loops = self._coilset.generate_cross_sections(
            mesh_sizes, geometry_names, verbose
        )
        for loop in coil_loops:
            points += loop.get_points()
            facets += loop.get_closed_facets(start=len(facets))
            control_points += [loop.get_control_point()]
            holes += [loop.get_hole()]

            polygons += [loop.as_shpoly()]
        points, facets = get_boundary(polygons)
        return cross_sections, points, facets, control_points, holes

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        raise NotImplementedError

    def plot_xz(self, ax=None, **kwargs):
        """
        Plot the PoloidalFieldCoilSystem in x-z.
        """
        self._plotter.plot_xz(self._coilset, ax=ax, **kwargs)

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        raise NotImplementedError

    def plot_xy(self, ax=None, **kwargs):
        """
        Plot the PoloidalFieldCoilSystem in x-y.
        """
        raise NotImplementedError

    def design_winding_packs(self):
        """
        Design the winding packs for the PoloidalFieldCoils.
        """
        raise NotImplementedError


class PoloidalFieldCoilsPlotter(ReactorSystemPlotter):
    """
    The plotter for Poloidal Field Coils.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "PF"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the PoloidalFieldCoils in x-z.
        """
        kwargs["facecolor"] = kwargs.get("facecolor", [BLUE["PF"][0], BLUE["CS"][0]])
        kwargs["linewidth"] = kwargs.get("linewidth", 2)
        plot_objects.plot(
            ax=ax,
            label=False,
            **kwargs,
        )

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the PoloidalFieldCoils in x-y.
        """
        raise NotImplementedError
