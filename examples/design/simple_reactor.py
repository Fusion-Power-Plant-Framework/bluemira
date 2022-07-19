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
from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem, minimise_length
from bluemira.geometry.parameterisations import PrincetonD
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

# %%[markdown]

# Let's set up some parameters that we're going to use in our `ReactorDesign`.

# %%

params = ParameterFrame.from_list(
    [
        ["Name", "Reactor name", "MyExample", "dimensionless", None, "Input", None],
        ["R_0", "Major radius", 9.0, "m", None, "Input", None],
        # Plasma parameters
        ["z_0", "Reference vertical coordinate", 0.0, "m", None, "Input", None],
        ["A", "Aspect ratio", 3.1, "dimensionless", None, "Input", None],
        ["kappa_u", "Upper elongation", 1.6, "dimensionless", None, "Input", None],
        ["kappa_l", "Lower elongation", 1.8, "dimensionless", None, "Input", None],
        ["delta_u", "Upper triangularity", 0.4, "dimensionless", None, "Input", None],
        ["delta_l", "Lower triangularity", 0.4, "dimensionless", None, "Input", None],
        ["phi_neg_u", "", 0, "degree", None, "Input", None],
        ["phi_pos_u", "", 0, "degree", None, "Input", None],
        ["phi_neg_l", "", 0, "degree", None, "Input", None],
        ["phi_pos_l", "", 0, "degree", None, "Input", None],
        # TF coil parameters
        ["tf_wp_width", "Width of TF coil winding pack", 0.6, "m", None, "Input", None],
        ["tf_wp_depth", "Depth of TF coil winding pack", 0.8, "m", None, "Input", None],
    ]
)
# fmt: on


# %%[markdown]

# We need to define some `Builder`s for our various `Components`.

# %%


class PlasmaBuilder(Builder):
    """
    Our PlasmaBuilder
    """

    _name = "PlasmaComponent"
    _required_params = [
        "R_0",
        "A",
        "z_0",
        "kappa_u",
        "kappa_l",
        "delta_u",
        "delta_l",
        "phi_neg_u",
        "phi_pos_u",
        "phi_pos_l",
        "phi_neg_l",
    ]

    def __init__(self, params):
        self.wire = self._build_wire(params)

    @staticmethod
    def _build_wire(params):
        return JohnerLCFS(
            var_dict={
                "r_0": {"value": params["R_0"]},
                "z_0": {"value": params["z_0"]},
                "a": {"value": params["R_0"] / params["A"]},
                "kappa_u": {"value": params["kappa_u"]},
                "kappa_l": {"value": params["kappa_l"]},
                "delta_u": {"value": params["delta_u"]},
                "delta_l": {"value": params["delta_l"]},
                "phi_u_pos": {"value": params["phi_pos_u"], "lower_bound": 0.0},
                "phi_u_neg": {"value": params["phi_neg_u"], "lower_bound": 0.0},
                "phi_l_pos": {"value": params["phi_pos_l"], "lower_bound": 0.0},
                "phi_l_neg": {
                    "value": params["phi_neg_l"],
                    "lower_bound": 0.0,
                    "upper_bound": 90,
                },
            }
        ).create_shape()

    def reinitialise(self, params, lcfs_wire: BluemiraWire) -> None:
        """
        Hopefully not going to be a problem for much longer. Please ignore.
        """
        self.wire = lcfs_wire
        return super().reinitialise(params)

    def build(self) -> Component:
        """
        Run the full build of the Plasma
        """
        component = super().build()
        component.add_child(self.build_xz())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the xz Component of the Plasma
        """
        return Component(
            "xz", children=[PhysicalComponent("LCFS", BluemiraFace(self.wire))]
        )

    def build_xyz(self):
        """
        Build the xyz Component of the Plasma
        """
        lcfs = self.build_xz().get_component("xz").get_component("LCFS").shape
        shape = revolve_shape(lcfs, degree=359)
        return Component("xyz", children=[PhysicalComponent("LCFS", shape)])


class TFCoilBuilder(Builder):
    """
    Our TF Coil builder.
    """

    _name = "TFCoilComponent"
    _required_params = ["tf_wp_width", "tf_wp_depth"]

    def __init__(self, params, centreline):
        self._params = params
        self.centreline = centreline

    def reinitialise(self, params, centreline) -> None:
        """
        Hopefully not going to be a problem for much longer. Please ignore.
        """
        super().reinitialise(params)
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

    def build(self) -> Component:
        """
        Run the full build for the TF coils.
        """
        component = super().build()
        component.add_child(self.build_xz())
        component.add_child(self.build_xyz())

        return component

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

# Now let us run our setup.

# %%

R_0 = 9.0


build_config = {
    "Plasma": {
        "runmode": "run",
        "class": "",
        "local_params": {"radius": 4.0},
    },
    "TF coils": {
        "runmode": "run",  # ["read", "read", "mock"]
        "param_class": PrincetonD,
        "params": {"R_0": R_0},
    },
    "PF coils": {},
}


my_reactor = Component("My Simple Reactor")

plasma_builder = PlasmaBuilder(params)
my_reactor.add_child(plasma_builder.build())

lcfs = (
    my_reactor.get_component("PlasmaComponent")
    .get_component("xz")
    .get_component("LCFS")
    .shape.boundary[0]
)

parameterisation = PrincetonD(
    var_dict={
        "x1": {"value": 3.0, "fixed": True},
        "x2": {"value": 15, "lower_bound": 12},
    }
)
my_tf_coil_opt_problem = MyTFCoilOptProblem(
    parameterisation,
    lcfs,
    optimiser=Optimiser("SLSQP", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6}),
    min_distance=1.0,
)
tf_centreline = my_tf_coil_opt_problem.optimise().create_shape()
tf_coil_builder = TFCoilBuilder(params, tf_centreline)
my_reactor.add_child(tf_coil_builder.build())
my_reactor.show_cad()
