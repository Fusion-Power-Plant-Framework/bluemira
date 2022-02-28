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
Define shapes & builders for the wall components of EUDEMO's first wall
"""
import copy
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.spatial import ConvexHull

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.shapes import OptimisedShapeBuilder
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PolySpline,
    PrincetonD,
)
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.optimiser import Optimiser


class WallPolySpline(PolySpline):
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on a PolySpline
    """

    _defaults = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "z2": {"value": 0},
        "height": {"value": 9.3},
        "top": {"value": 0.4},
        "upper": {"value": 0.3},
        "dz": {"value": -0.5},
        "tilt": {"value": 0},
        "lower": {"value": 0.5},
        "bottom": {"value": 0.2},
    }

    def __init__(self, var_dict=None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        z2 = self.variables["z2"].value
        height = self.variables["height"].value
        top = self.variables["top"].value
        upper = self.variables["upper"].value
        dz = self.variables["dz"].value
        tilt = self.variables["tilt"].value
        lower = self.variables["lower"].value
        bottom = self.variables["bottom"].value

        self.adjust_variable(
            "x1",
            ib_radius,
            lower_bound=ib_radius - 2,
            upper_bound=ib_radius * 1.1,
        )
        self.adjust_variable(
            "x2",
            value=ob_radius,
            lower_bound=ob_radius * 0.9,
            upper_bound=ob_radius + 2,
        )
        self.adjust_variable("z2", z2, lower_bound=-0.9, upper_bound=0.9)
        self.adjust_variable(
            "height", height, lower_bound=height - 0.001, upper_bound=50
        )
        self.adjust_variable("top", top, lower_bound=0.05, upper_bound=0.75)
        self.adjust_variable("upper", upper, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("dz", dz, lower_bound=-5, upper_bound=5)
        self.adjust_variable("tilt", tilt, lower_bound=-25, upper_bound=25)
        self.adjust_variable("lower", lower, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("bottom", bottom, lower_bound=0.05, upper_bound=0.75)

        # Fix 'flat' to avoid drawing the PolySpline's outer straight.
        # The straight is often optimised to near-zero length, which
        # causes an error when CAD tries to draw it
        self.fix_variable("flat", 0)


class WallPrincetonD(PrincetonD):

    _defaults = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "dz": {"value": -0.5},
    }

    def __init__(self, var_dict=None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        self.adjust_variable(
            "x1", ib_radius, lower_bound=ib_radius - 2, upper_bound=ib_radius * 1.02
        )
        self.adjust_variable(
            "x2", ob_radius, lower_bound=ob_radius * 0.98, upper_bound=ob_radius + 2
        )
        self.adjust_variable(
            "dz", self.variables["dz"].value, lower_bound=-3, upper_bound=3
        )


class MinimiseLength(GeometryOptimisationProblem):
    """
    Optimiser to minimize the length of a geometry parameterisation.
    """

    def __init__(
        self,
        parameterisation: GeometryParameterisation,
        optimiser: Optimiser,
        keep_out_zones: Iterable[BluemiraWire] = None,
        n_koz_points: int = 100,
        koz_con_tol: float = 1e-3,
    ):
        super().__init__(parameterisation, optimiser)

        self.n_koz_points = n_koz_points
        self.keep_out_zones = keep_out_zones
        if self.keep_out_zones is not None:
            self.koz_points = self._make_koz_points(
                self.keep_out_zones, self.n_koz_points
            )
            self.optimiser.add_ineq_constraints(
                self.f_constrain_koz,
                koz_con_tol * np.ones(self.n_koz_points),
            )

    def _make_koz_points(
        self, keep_out_zones: Iterable[BluemiraWire], n_points: int
    ) -> np.ndarray:
        """
        Generate a set of points that combine the given keep-out-zones,
        to use as constraints in the optimisation problem.

        Returns
        -------
        coords: np.ndarray[2, n_points]
            Coordinate of the keep-out-zone points in the xz plane.
        """
        shape_discretizations = []
        for zone in keep_out_zones:
            d = zone.discretize(byedges=True, dl=zone.length / 200).xz
            shape_discretizations.append(d)
        coords = np.hstack(shape_discretizations)
        hull = ConvexHull(coords.T)

        # The hull vertices do not give a closed shape, so add the first
        # point to the end of the array to close it.
        # If we do not close the shape, we may not get the results we
        # expect from the signed distance function
        hull_coords = np.empty((coords.shape[0], len(hull.vertices) + 1))
        hull_coords[:, 0:-1] = coords[:, hull.vertices]
        hull_coords[:, -1] = coords[:, hull.vertices[0]]
        return hull_coords

    def f_constrain_koz(self, constraint, x, grad):
        """
        Geometry constraint function to the keep-out-zone.
        """
        constraint[:] = self.calculate_signed_distance(x)
        if grad.size > 0:
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_signed_distance, x, constraint
            )
        return constraint

    def calculate_signed_distance(self, x):
        """
        Calculate the signed distances from the parameterised shape to
        the keep-out zone.
        """
        self.update_parameterisation(x)

        shape = self.parameterisation.create_shape()
        s = shape.discretize(ndiscr=self.n_koz_points).xz
        return signed_distance_2D_polygon(s.T, self.koz_points.T).T

    def calculate_length(self, x):
        """Calculate the length of the shape being optimised."""
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        """The objective function for the optimiser."""
        length = self.calculate_length(x)
        if grad.size > 0:
            self.optimiser.approx_derivative(self.calculate_length, x, f0=length)
        return length


class WallBuilder(OptimisedShapeBuilder):
    """
    Builder class for the wall of the reactor, this does not include the
    divertor.
    """

    COMPONENT_WALL_BOUNDARY = "wall_boundary"

    _required_params: List[str] = [
        "plasma_type",
        "R_0",  # major radius
        "kappa_95",  # 95th percentile plasma elongation
        "r_fw_ib_in",  # inboard first wall inner radius
        "r_fw_ob_in",  # inboard first wall outer radius
        "A",  # aspect ratio
    ]
    _required_config: List[str] = []
    _params: Configuration
    _default_runmode: str = "mock"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        keep_out_zones: Iterable[BluemiraWire] = None,
    ):
        # boundary should be set by run/mock/read, it is used by the build methods
        self.boundary: BluemiraWire
        # _keep_out_zones should be set by reinitialize
        self._keep_out_zones: Iterable[BluemiraWire]

        super().__init__(params, build_config, keep_out_zones=keep_out_zones)

    def reinitialise(
        self, params: Dict[str, Any], keep_out_zones: Iterable[BluemiraWire] = None
    ) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        super().reinitialise(params)
        self._keep_out_zones = keep_out_zones

    def read(self):
        """Read the result of the design problem from file."""
        return super().read()

    def run(self):
        """
        Run the design problem for wall builder.
        """
        super().run(keep_out_zones=self._keep_out_zones)
        self.boundary = self._shape.create_shape()

    def mock(self):
        """
        Create a basic shape for the wall's boundary.
        """
        self.boundary = self._shape.create_shape()

    def build(self, **kwargs) -> Component:
        """
        Build the components.
        """
        component = super().build(**kwargs)
        component.add_child(self.build_xz())
        return component

    def build_xz(self) -> Component:
        """
        Build the components in the xz plane.
        """
        component = Component("xz")
        component.add_child(
            PhysicalComponent(
                self.COMPONENT_WALL_BOUNDARY,
                BluemiraWire(self.boundary, label=self.COMPONENT_WALL_BOUNDARY),
            )
        )
        component.plot_options.wire_options["color"] = BLUE_PALETTE["DIV"]
        bm_plot_tools.set_component_plane(component, "xz")
        return component

    # TODO(hsaunders1904): 'height' is not a parameter for all shapes.
    # Does it make sense to derive it within this class?
    def _derive_shape_params(self):
        """
        Calculate derived parameters for the GeometryParameterisation.
        """
        params = super()._derive_shape_params()
        if issubclass(self._param_class, PolySpline):
            params["height"] = {"value": self._derive_polyspline_height()}
        return params

    def _derive_polyspline_height(self) -> float:
        """Derive the PolySpline height from relevant parameters."""
        r_minor = self._params.R_0 / self._params.A
        return (self._params.kappa_95 * r_minor) * 2
