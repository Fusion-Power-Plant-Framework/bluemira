# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
A simple user-facing reactor example, showing some of the building blocks, and how to
combine them.
"""

# %%
from dataclasses import dataclass

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem, minimise_length
from bluemira.geometry.parameterisations import GeometryParameterisation, PrincetonD
from bluemira.geometry.tools import (
    distance_to,
    make_polygon,
    offset_wire,
    revolve_shape,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_problems import OptimisationConstraint, OptimisationObjective
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from bluemira.utilities.tools import get_class_from_module

# %%[markdown]

# Let's set up some parameters that we're going to use in our reactor design.
# and some `ComponentManagers` to manage acces to our `Components`

# %%


@dataclass
class PlasmaDesignerParams(ParameterFrame):
    R_0: Parameter[float]
    A: Parameter[float]
    z_0: Parameter[float]
    kappa_u: Parameter[float]
    kappa_l: Parameter[float]
    delta_u: Parameter[float]
    delta_l: Parameter[float]
    phi_neg_u: Parameter[float]
    phi_pos_u: Parameter[float]
    phi_pos_l: Parameter[float]
    phi_neg_l: Parameter[float]


@dataclass
class TFCoilBuilderParams(ParameterFrame):
    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]


class Plasma(ComponentManager):
    def lcfs(self):
        return (
            self.component().get_component("xz").get_component("LCFS").shape.boundary[0]
        )


class TFCoil(ComponentManager):
    pass


# %%[markdown]

# Now we want to define a way to optimise the TF coil shape.

# %%


class MyTFCoilOptProblem(GeometryOptimisationProblem):
    """
    A simple geometry optimisation problem for the TF coil current centreline

    Here we:

    minimise: length
        subject to:
            min_distance_to_LCFS >= min_distance
    """

    def __init__(self, geometry_parameterisation, lcfs, optimiser, min_distance):
        objective = OptimisationObjective(
            minimise_length,
            f_objective_args={"parameterisation": geometry_parameterisation},
        )
        constraints = [
            OptimisationConstraint(
                self.f_constraint,
                f_constraint_args={
                    "parameterisation": geometry_parameterisation,
                    "lcfs": lcfs,
                    "min_distance": min_distance,
                    "ad_args": {},
                },
                tolerance=1e-6,
                constraint_type="inequality",
            )
        ]
        super().__init__(
            geometry_parameterisation, optimiser, objective, constraints=constraints
        )

    @staticmethod
    def constraint_value(vector, parameterisation, lcfs, min_distance):
        """
        The constraint evaluation function
        """
        parameterisation.variables.set_values_from_norm(vector)
        shape = parameterisation.create_shape()
        return min_distance - distance_to(shape, lcfs)[0]

    @staticmethod
    def f_constraint(
        constraint, vector, grad, parameterisation, lcfs, min_distance, ad_args=None
    ):
        """
        Constraint function
        """
        function = MyTFCoilOptProblem.constraint_value
        constraint[:] = function(vector, parameterisation, lcfs, min_distance)
        if grad.size > 0:
            grad[:] = approx_derivative(
                function,
                vector,
                f0=constraint,
                args=(parameterisation, lcfs, min_distance),
                bounds=[0, 1],
            )
        return constraint

    def optimise(self, x0=None):
        """
        Run the optimisation problem.
        """
        return super().optimise(x0)


# %%[markdown]

# We need to define some `Designers` and `Builders` for our various `Components`.

# %%


class PlasmaDesigner(Designer):
    param_cls = PlasmaDesignerParams

    def run(self):
        return self._build_wire(self.params)

    @staticmethod
    def _build_wire(params):
        return JohnerLCFS(
            var_dict={
                "r_0": {"value": params.R_0.value},
                "z_0": {"value": params.z_0.value},
                "a": {"value": params.R_0.value / params.A.value},
                "kappa_u": {"value": params.kappa_u.value},
                "kappa_l": {"value": params.kappa_l.value},
                "delta_u": {"value": params.delta_u.value},
                "delta_l": {"value": params.delta_l.value},
                "phi_u_pos": {"value": params.phi_pos_u.value, "lower_bound": 0.0},
                "phi_u_neg": {"value": params.phi_neg_u.value, "lower_bound": 0.0},
                "phi_l_pos": {"value": params.phi_pos_l.value, "lower_bound": 0.0},
                "phi_l_neg": {
                    "value": params.phi_neg_l.value,
                    "lower_bound": 0.0,
                    "upper_bound": 90,
                },
            }
        ).create_shape()


class PlasmaBuilder(Builder):
    """
    Our PlasmaBuilder
    """

    param_cls = None

    def __init__(self, wire, build_config):
        super().__init__(None, build_config)
        self.wire = wire

    def build(self) -> Component:
        """
        Run the full build of the Plasma
        """
        xz = self.build_xz()
        return Plasma(
            self.component_tree(
                xz=[xz],
                xy=[Component("")],
                xyz=[self.build_xyz(xz.shape)],
            )
        )

    def build_xz(self):
        """
        Build the xz Component of the Plasma
        """
        return PhysicalComponent("LCFS", BluemiraFace(self.wire))

    def build_xyz(self, lcfs):
        """
        Build the xyz Component of the Plasma
        """
        shape = revolve_shape(lcfs, degree=359)
        return PhysicalComponent("LCFS", shape)


class TFCoilDesigner(Designer):

    param_cls = None

    def __init__(self, plasma_lcfs, params, build_config):
        super().__init__(params, build_config)
        self.lcfs = plasma_lcfs
        self.parameterisation_cls = get_class_from_module(
            self.build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )

    def run(self) -> GeometryParameterisation:

        parameterisation = self.parameterisation_cls(
            var_dict={
                "x1": {"value": 3.0, "fixed": True},
                "x2": {"value": 15, "lower_bound": 12},
            }
        )
        my_tf_coil_opt_problem = MyTFCoilOptProblem(
            parameterisation,
            self.lcfs,
            optimiser=Optimiser(
                "SLSQP", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6}
            ),
            min_distance=1.0,
        )
        return my_tf_coil_opt_problem.optimise()


class TFCoilBuilder(Builder):
    """
    Our TF Coil builder.
    """

    param_cls = TFCoilBuilderParams

    def __init__(self, params, centreline):
        super().__init__(params, {})
        self.centreline = centreline

    def make_tf_wp_xs(self):
        """
        Make a wire for the cross-section of the winding pack in xy.
        """
        width = 0.5 * self.params.tf_wp_width.value
        depth = 0.5 * self.params.tf_wp_depth.value
        wire = make_polygon(
            {
                "x": [-width, width, width, -width],
                "y": [-depth, -depth, depth, depth],
                "z": 0.0,
            },
            closed=True,
        )
        return wire

    def build(self) -> ComponentManager:
        """
        Run the full build for the TF coils.
        """
        return TFCoil(
            self.component_tree(
                xz=[self.build_xz()],
                xy=[Component("")],
                xyz=[self.build_xyz()],
            )
        )

    def build_xz(self):
        """
        Build the xz Component of the TF coils.
        """
        inner = offset_wire(self.centreline, -0.5 * self.params.tf_wp_width.value)
        outer = offset_wire(self.centreline, 0.5 * self.params.tf_wp_width.value)
        return PhysicalComponent("Winding pack", BluemiraFace([outer, inner]))

    def build_xyz(self):
        """
        Build the xyz Component of the TF coils.
        """
        wp_xs = self.make_tf_wp_xs()
        wp_xs.translate((self.centreline.bounding_box.x_min, 0, 0))
        volume = sweep_shape(wp_xs, self.centreline)
        return PhysicalComponent("Winding pack", volume)


# %%[markdown]

# Now let us run our setup.

# %%

build_config = {
    "params": {},
    "Plasma": {
        "Designer": {
            "params": {
                "R_0": {
                    "value": 9.0,
                    "unit": "m",
                    "source": "Input",
                    "long_name": "Major radius",
                },
                "z_0": {
                    "value": 0.0,
                    "unit": "m",
                    "source": "Input",
                    "long_name": "Reference vertical coordinate",
                },
                "A": {
                    "value": 3.1,
                    "unit": "dimensionless",
                    "source": "Input",
                    "long_name": "Aspect ratio",
                },
                "kappa_u": {
                    "value": 1.6,
                    "unit": "dimensionless",
                    "source": "Input",
                    "long_name": "Upper elongation",
                },
                "kappa_l": {
                    "value": 1.8,
                    "unit": "dimensionless",
                    "source": "Input",
                    "long_name": "Lower elongation",
                },
                "delta_u": {
                    "value": 0.4,
                    "unit": "dimensionless",
                    "source": "Input",
                    "long_name": "Upper triangularity",
                },
                "delta_l": {
                    "value": 0.4,
                    "unit": "dimensionless",
                    "source": "Input",
                    "long_name": "Lower triangularity",
                },
                "phi_neg_u": {"value": 0, "unit": "degree", "source": "Input"},
                "phi_pos_u": {"value": 0, "unit": "degree", "source": "Input"},
                "phi_neg_l": {"value": 0, "unit": "degree", "source": "Input"},
                "phi_pos_l": {"value": 0, "unit": "degree", "source": "Input"},
            },
        },
    },
    "TF Coil": {
        "Designer": {
            "runmode": "run",
            "param_class": "PrincetonD",
            "params": {
                # "R_0": {"value": 9.0, "unit": "m"},
                "tf_wp_width": {
                    "value": 0.6,
                    "unit": "m",
                    "source": "Input",
                    "long_name": "Width of TF coil winding pack",
                },
                "tf_wp_depth": {
                    "value": 0.8,
                    "unit": "m",
                    "source": "Input",
                    "long_name": "Depth of TF coil winding pack",
                },
            },
        },
    },
    # "PF coils": {},
}

# TODO improve build config manipulation
plasma_params = PlasmaDesignerParams.from_dict(
    {**build_config["params"], **build_config["Plasma"]["Designer"].pop("params")}
)

plasma_designer = PlasmaDesigner(plasma_params, build_config["Plasma"])
plasma_wire = plasma_designer.execute()

plasma_builder = PlasmaBuilder(plasma_wire, build_config["Plasma"])
plasma = plasma_builder.build()

tf_coil_params = TFCoilBuilderParams.from_dict(
    {**build_config["params"], **build_config["TF Coil"]["Designer"].pop("params")}
)

tf_coil_designer = TFCoilDesigner(
    plasma.lcfs(), None, build_config["TF Coil"]["Designer"]
)
tf_parameterisation = tf_coil_designer.execute()

tf_coil_builder = TFCoilBuilder(tf_coil_params, tf_parameterisation.create_shape())
tf_coil = tf_coil_builder.build()
