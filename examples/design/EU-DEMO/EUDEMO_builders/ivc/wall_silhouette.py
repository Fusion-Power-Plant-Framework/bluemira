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

"""
import copy
import enum
from dataclasses import dataclass
from typing import Dict, Optional, Type

from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.base.solver import RunMode, SolverABC, Task
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points, get_legs
from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PolySpline,
    PrincetonD,
)
from bluemira.geometry.tools import convex_hull_wires_2d, make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_class_from_module, offset_wire


@dataclass
class WallSilhouetteSolverParams(ParameterFrame):
    """Parameters for running the `WallSilhouetteSolver`."""

    plasma_type: Parameter[str]
    R_0: Parameter[float]  # major radius
    kappa_95: Parameter[float]  # 95th percentile plasma elongation
    r_fw_ib_in: Parameter[float]  # inboard first wall inner radius
    r_fw_ob_in: Parameter[float]  # inboard first wall outer radius
    A: Parameter[float]  # aspect ratio


class WallSilhouetteRunMode(RunMode):
    """Run modes for `WallSilhouetteSolver`."""

    RUN = enum.auto()
    MOCK = enum.auto()
    READ = enum.auto()


class _Setup(Task):
    def __init__(
        self,
        params: ParameterFrame,
        param_cls: str,
        variables_map: Dict[str, str],
        file_path: Optional[str] = None,
    ) -> None:
        super().__init__(params)

        self.param_cls: Type[GeometryParameterisation] = get_class_from_module(
            param_cls,
            default_module="bluemira.geometry.parameterisations",
        )
        self.variables_map = variables_map

        self.file_path = file_path

    def run(self) -> GeometryParameterisation:
        return self.param_cls(self._derive_shape_params(self.variables_map))

    def mock(self) -> GeometryParameterisation:
        return self.run()

    def read(self) -> GeometryParameterisation:
        if not self.file_path:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in READ mode: no file path specified."
            )
        return self.param_cls.from_json(file=self.file_path)

    def _derive_shape_params(self, variables_map: Dict[str, str]) -> Dict:
        shape_params = {}
        for key, val in self.variables_map.items():
            if isinstance(val, str):
                val = self.params.get(val)

            if isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = self.params.get(val["value"])
            else:
                val = {"value": val}

            shape_params[key] = val

        if issubclass(self.param_cls, PolySpline):
            shape_params["height"] = {"value": self._derive_polyspline_height()}
        return shape_params

    def _derive_polyspline_height(self) -> float:
        """Derive the PolySpline height from relevant parameters."""
        r_minor = self._params.R_0 / self._params.A
        return (self._params.kappa_95 * r_minor) * 2


class _Run(Task):
    def __init__(
        self,
        params: ParameterFrame,
        equilibrium: Equilibrium,
        problem_class: Optional[str] = None,
        optimisation_config: Optional[str] = None,
        file_path: Optional[str] = None,
        *problem_class_args,
        **problem_class_kwargs,
    ) -> None:

        super().__init__(params)

        if problem_class is not None:
            self.problem_class = get_class_from_module(problem_class)
            self.optimisation_config = optimisation_config

            self.problem_settings = optimisation_config.get("problem_settings", {})
            self.algorithm_name = optimisation_config.get("algorithm_name", "SLSQP")
            self.opt_conditions = optimisation_config.get(
                "opt_conditions", {"max_eval": 100}
            )
            self.opt_parameters = optimisation_config.get("opt_parameters", {})
            self.problem_class_args = problem_class_args
            self.problem_class_kwargs = problem_class_kwargs

        self.equilibrium = equilibrium

    def run(self, parameterisation) -> GeometryParameterisation:
        """
        Optimise the shape using the provided parameterisation and optimiser.
        """
        if not hasattr(self, "problem_class"):
            raise ValueError(
                f"Cannot execute {type(self).__name__} in RUN mode: no problem_class specified."
            )

        bluemira_debug(
            f"Setting up design problem with:\n"
            f"algorithm_name: {self.algorithm_name}\n"
            f"n_variables: {parameterisation.variables.n_free_variables}\n"
            f"opt_conditions: {self.opt_conditions}\n"
            f"opt_parameters: {self.opt_parameters}"
        )

        optimiser = Optimiser(
            self.algorithm_name,
            parameterisation.variables.n_free_variables,
            self.opt_conditions,
            self.opt_parameters,
        )

        if self._problem_settings != {}:
            bluemira_debug(
                f"Applying non-default settings to problem: {self.problem_settings}"
            )
        design_problem = self.problem_class(
            parameterisation,
            optimiser,
            self._make_wall_keep_out_zone(),
            **self.problem_class_kwargs,
            **self._problem_settings,
        )

        bluemira_print(f"Solving design problem: {design_problem.__class__.__name__}")
        if parameterisation.n_ineq_constraints > 0:
            bluemira_debug("Applying shape constraints")
            design_problem.apply_shape_constraints()

        bluemira_debug("Solving...")
        return design_problem.optimise()

    def _make_wall_keep_out_zone(self) -> BluemiraWire:
        """
        Create a "keep-out zone" to be used as a constraint in the
        wall shape optimiser.
        """
        geom_offset = self.params.tk_sol_ib.value
        psi_n = self.params.fw_psi_n.value
        geom_offset = 0.2  # TODO: Unpin
        psi_n = 1.05  # TODO: Unpin
        geom_offset_zone = self._make_geometric_keep_out_zone(geom_offset)
        flux_surface_zone = self._make_flux_surface_keep_out_zone(psi_n)
        leg_zone = self._make_divertor_leg_keep_out_zone(
            self.params.div_L2D_ib.value, self.params.div_L2D_ob.value
        )
        return convex_hull_wires_2d(
            [geom_offset_zone, flux_surface_zone, leg_zone], ndiscr=200, plane="xz"
        )

    def _make_geometric_keep_out_zone(self, offset: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from a geometric offset of the LCFS.
        """
        lcfs = make_polygon(self.equilibrium.get_LCFS().xyz, closed=True)
        return offset_wire(lcfs, offset, join="arc")

    def _make_flux_surface_keep_out_zone(self, psi_n: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's flux surface.
        """
        # TODO: This is currently called twice once here and once for the
        # setup of the remaining ivc
        o_points, _ = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )
        flux_surface_zone = self.equilibrium.get_flux_surface(psi_n)
        # Chop the flux surface to only take the upper half
        indices = flux_surface_zone.z >= o_points[0][1]
        flux_surface_zone = make_polygon(flux_surface_zone.xyz[:, indices], closed=True)
        return flux_surface_zone

    def _make_divertor_leg_keep_out_zone(
        self, leg_length_ib_2D, leg_length_ob_2D
    ) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's divertor legs
        """
        legs = get_legs(self.equilibrium, n_layers=1, dx_off=0.0)
        ib_leg = make_polygon(legs["lower_inner"][0].xyz)
        ob_leg = make_polygon(legs["lower_outer"][0].xyz)

        return make_polygon(
            [
                ib_leg.value_at(distance=leg_length_ib_2D),
                ob_leg.value_at(distance=leg_length_ob_2D),
            ],
            closed=False,
        )


class WallSilhouetteSolver(SolverABC):

    setup_cls = _Setup
    run_cls = _Run
    teardown_cls = None
    run_mode_cls = WallSilhouetteRunMode

    def __init__(
        self, params: ParameterFrame, equilibrium: Equilibrium, build_config: Dict = None
    ):
        self.params = make_parameter_frame(params, WallSilhouetteSolverParams)
        self.build_config = {} if build_config is None else build_config

        read_file_path = self.build_config.get("file_path", None)
        problem_class = self.build_config.get("problem_class", None)
        optimisation_config = self.build_config.get("optimisation_config", {})
        param_cls = self.build_config.get("param_class", None)
        variables_map = self.build_config.get("variables_map", {})

        self._setup = self.setup_cls(
            self.params, param_cls, variables_map, read_file_path
        )
        self._run = self.run_cls(
            self.params, equilibrium, problem_class, optimisation_config
        )


class WallPolySpline(PolySpline):
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on the PolySpline parameterisation.
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

    def __init__(self, var_dict: Dict = None):
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

        if not self.variables["x1"].fixed:
            self.adjust_variable(
                "x1",
                ib_radius,
                lower_bound=ib_radius - 2,
                upper_bound=ib_radius * 1.1,
            )
        if not self.variables["x2"].fixed:
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
    """
    Defines the geometry for reactor first wall, without a divertor,
    based on the PrincetonD parameterisation.
    """

    _defaults = {
        "x1": {"value": 5.8},
        "x2": {"value": 12.1},
        "dz": {"value": -0.5},
    }

    def __init__(self, var_dict: Dict = None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        if not self.variables["x1"].fixed:
            self.adjust_variable(
                "x1", ib_radius, lower_bound=ib_radius - 2, upper_bound=ib_radius * 1.02
            )

        if not self.variables["x2"].fixed:
            self.adjust_variable(
                "x2", ob_radius, lower_bound=ob_radius * 0.98, upper_bound=ob_radius + 2
            )
        self.adjust_variable(
            "dz", self.variables["dz"].value, lower_bound=-3, upper_bound=3
        )
