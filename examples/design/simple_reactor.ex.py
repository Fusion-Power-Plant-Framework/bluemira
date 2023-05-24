# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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
A simple user-facing reactor example, showing some of the building blocks, and how to
combine them.
"""

# %%
from dataclasses import dataclass
from typing import Dict

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import EmptyFrame, Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager, Reactor
from bluemira.base.reactor_config import ReactorConfig
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import GeometryOptimisationProblem, minimise_length
from bluemira.geometry.parameterisations import GeometryParameterisation
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

# %% [markdown]
#
# # Simplistic Reactor Design
#
# This example show hows to set up a simple reactor, consisting of a plasma and
# a single TF coil.
# The TF coil will be optimised such that its length is minimised,
# whilst maintaining a minimum distance to the plasma.
#
# To do this we'll run through how to set up the parameters for the build,
# how to define the `Builder`s and `Designer`s (including the optimisation problem)
# for the plasma and TF coil,
# and how to run the build with configurable parameters.
#
# Firstly we need to define the parameters we're going to use in our reactor design for
# each component.


# %%
@dataclass
class PlasmaDesignerParams(ParameterFrame):
    """Plasma Designer ParameterFrame"""

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
    """TF Coil Builder ParameterFrame"""

    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]


# %% [markdown]
#
# To manage access to properties of the components we need some `ComponentManagers`


# %%
class Plasma(ComponentManager):
    """Plasma manager"""

    def lcfs(self):
        """Get separatrix"""
        return (
            self.component().get_component("xz").get_component("LCFS").shape.boundary[0]
        )


class TFCoil(ComponentManager):
    """TF Coil manager"""

    def wp_volume(self):
        """Get winding pack volume"""
        return (
            self.component()
            .get_component("xyz")
            .get_component("Winding pack")
            .shape.volume()
        )


# %% [markdown]
#
# We then need a reactor in which to store the components.
# Notice that the typing of the components here is the relevant `ComponentManager`


# %%
class MyReactor(Reactor):
    """Reactor container"""

    plasma: Plasma
    tf_coil: TFCoil


# %% [markdown]
#
# Now we want to define a way to optimise the TF coil shape.
# We want to minimise the length of the TF coil, constraining the optimiser such that
# the any part of the coil is always a minimum distance away from the plasma.
#
# Further information on geometry can be found in the
# [geometry tutorial](../geometry/geometry_tutorial.ex.py) and information about
# geometry optimisation can be found in the
# [geometry optimisation tutorial](../geometry/optimisation_tutorial.ex.py).


# %%
class MyTFCoilOptProblem(GeometryOptimisationProblem):
    """
    A simple geometry optimisation problem for the TF coil current centreline

    Here we:

    minimise: length
        subject to:
            min_distance_to_LCFS >= min_distance
    """

    def __init__(
        self,
        geometry_parameterisation: GeometryParameterisation,
        lcfs: BluemiraWire,
        optimiser: Optimiser,
        min_distance: float,
    ):
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
    def constraint_value(
        vector: np.ndarray,
        parameterisation: GeometryParameterisation,
        lcfs: BluemiraWire,
        min_distance: float,
    ):
        """
        The constraint evaluation function
        """
        parameterisation.variables.set_values_from_norm(vector)
        shape = parameterisation.create_shape()
        return min_distance - distance_to(shape, lcfs)[0]

    @staticmethod
    def f_constraint(
        constraint: np.ndarray,
        vector: np.ndarray,
        grad: np.ndarray,
        parameterisation: GeometryParameterisation,
        lcfs: BluemiraWire,
        min_distance: float,
        ad_args=None,
    ):
        """
        Constraint function
        """
        tffunction = MyTFCoilOptProblem.constraint_value
        constraint[:] = tffunction(vector, parameterisation, lcfs, min_distance)
        if grad.size > 0:
            grad[:] = approx_derivative(
                tffunction,
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


# %% [markdown]
#
# We need to define some `Designers` and `Builders` for our various `Components`.
#
# Firstly the plasma.
# The plasma designer will, using its `ParameterFrame`, evaluate a `JohnerLCFS`
# geometry parameterisation, returning a wire representing the plasma's
# last-closed-flux-surface (LCFS).
#
# In this case `PlasmaDesigner` has some required parameters but `PlasmaBuilder` does
# not


# %%
class PlasmaDesigner(Designer):
    """Design a plasma's LCFS using a Johner parameterisation."""

    param_cls = PlasmaDesignerParams

    def run(self) -> GeometryParameterisation:
        """Build the LCFS, returning a closed wire defining its outline."""
        return self._build_wire(self.params)

    @staticmethod
    def _build_wire(params: PlasmaDesignerParams) -> GeometryParameterisation:
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
        )


class PlasmaBuilder(Builder):
    """Build the 3D geometry of a plasma from a given LCFS."""

    param_cls = None

    def __init__(self, wire: BluemiraWire, build_config: Dict):
        super().__init__(None, build_config)
        self.wire = wire

    def build(self) -> Component:
        """
        Run the full build of the Plasma
        """
        xz = self.build_xz()
        return self.component_tree(
            xz=[xz],
            xy=[Component("")],
            xyz=[self.build_xyz(xz.shape)],
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build a view of the plasma in the toroidal (xz) plane.

        This generates a ``PhysicalComponent``, whose shape is a face.
        """
        component = PhysicalComponent("LCFS", BluemiraFace(self.wire))
        component.display_cad_options.color = BLUE_PALETTE["PL"]
        component.display_cad_options.transparency = 0.5
        return component

    def build_xyz(self, lcfs: BluemiraFace) -> PhysicalComponent:
        """
        Build the 3D (xyz) Component of the plasma by revolving the given face
        360 degrees.
        """
        shape = revolve_shape(lcfs, degree=360)
        component = PhysicalComponent("LCFS", shape)
        component.display_cad_options.color = BLUE_PALETTE["PL"]
        component.display_cad_options.transparency = 0.5
        return component


# %% [markdown]
#
# And now the TF Coil, in this instance for simplicity we are only making one TF coil.
#
# The TF coil designer is finding the given geometry parameterisation given a string
# in the `build_config` which should point to a class.
# The parameterisation is then fed into the optimisation problem we made earlier.
# Finally when the designer is executed the optimisation problem is run to generate
# the centreline wire of the coil.
#
# The TF coil builder then is passed the centreline from the designer to create the
# Component and therefore the CAD of the TF coil.
# If more TF coils were to be required the build_xyz of `TFCoilBuilder` would need to
# be modified.
#
# Notice that only `TFCoilBuilder` has required parameters in this case.


# %%
class TFCoilDesigner(Designer):
    """TF coil Designer"""

    param_cls = None  # This designer takes no parameters

    def __init__(self, plasma_lcfs, params, build_config):
        super().__init__(params, build_config)
        self.lcfs = plasma_lcfs
        self.parameterisation_cls = get_class_from_module(
            self.build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )

    def run(self) -> GeometryParameterisation:
        """TF coil run method"""
        parameterisation = self.parameterisation_cls(
            var_dict=self.build_config["var_dict"]
        )
        my_tf_coil_opt_problem = MyTFCoilOptProblem(
            parameterisation,
            self.lcfs,
            optimiser=Optimiser(
                "SLSQP", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6}
            ),
            min_distance=1.0,  # the coil must be >= 1 meter from the LCFS
        )
        return my_tf_coil_opt_problem.optimise()


class TFCoilBuilder(Builder):
    """
    Build a 3D model of a TF Coil from a given centre line
    """

    param_cls = TFCoilBuilderParams

    def __init__(self, params, centreline):
        super().__init__(params, {})
        self.centreline = centreline

    def make_tf_wp_xs(self) -> BluemiraWire:
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
        return self.component_tree(
            xz=[self.build_xz()],
            xy=[Component("")],
            xyz=[self.build_xyz()],
        )

    def build_xz(self) -> PhysicalComponent:
        """
        Build the xz Component of the TF coils.
        """
        inner = offset_wire(self.centreline, -0.5 * self.params.tf_wp_width.value)
        outer = offset_wire(self.centreline, 0.5 * self.params.tf_wp_width.value)
        return PhysicalComponent("Winding pack", BluemiraFace([outer, inner]))

    def build_xyz(self) -> PhysicalComponent:
        """
        Build the xyz Component of the TF coils.
        """
        wp_xs = self.make_tf_wp_xs()
        wp_xs.translate((self.centreline.bounding_box.x_min, 0, 0))
        volume = sweep_shape(wp_xs, self.centreline)
        return PhysicalComponent("Winding pack", volume)


# %% [markdown]
#
# Now let us setup our build configuration.
# This could be stored as a JSON file and read in but for simplicity it is all written
# here.
# Notice there are no 'global' parameters as neither of the components share a variable.

# %%
build_config = {
    # This reactor has no global parameters, but this key would usually
    # be used to set parameters that are shared between components
    "params": {},
    "Plasma": {
        "designer": {
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
        "params": {},
        "designer": {
            "runmode": "run",
            "param_class": "PrincetonD",
            "var_dict": {
                "x1": {"value": 3.0, "fixed": True},
                "x2": {"value": 15, "lower_bound": 12},
            },
        },
        "builder": {
            "params": {
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
}

# %% [markdown]
#
# Now we set up our ParameterFrames

# %%

reactor_config = ReactorConfig(build_config, EmptyFrame)


# %% [markdown]
#
# We create our plasma

# %%
plasma_designer = PlasmaDesigner(
    reactor_config.params_for("Plasma", "designer"),
    reactor_config.config_for("Plasma", "designer"),
)
plasma_parameterisation = plasma_designer.execute()

plasma_builder = PlasmaBuilder(
    plasma_parameterisation.create_shape(),
    reactor_config.config_for("Plasma"),
)
plasma = Plasma(plasma_builder.build())

# %% [markdown]
#
# We create our TF coil

# %%
tf_coil_designer = TFCoilDesigner(
    plasma.lcfs(), None, reactor_config.config_for("TF Coil", "designer")
)
tf_parameterisation = tf_coil_designer.execute()

tf_coil_builder = TFCoilBuilder(
    reactor_config.params_for("TF Coil", "builder"),
    tf_parameterisation.create_shape(),
)
tf_coil = TFCoil(tf_coil_builder.build())

# %% [markdown]
#
# Finally we add the components to the reactor and show the CAD

# %%
reactor = MyReactor("Simple Example", n_sectors=1)

reactor.plasma = plasma
reactor.tf_coil = tf_coil

reactor.show_cad(n_sectors=1)
reactor.show_cad("xz")
