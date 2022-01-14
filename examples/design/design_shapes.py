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
A basic tutorial for configuring and running a design with parameterised shapes.
"""

# %%
from bluemira.base.design import Design
from bluemira.geometry.optimisation import GeometryOptimisationProblem

# %%[markdown]
# # Configuring and Running a Simple Shape-Based Design
#
# This example shows how to set up and run a shape-based Design that uses a parameterised
# shape as-is to represent a plasma surface, and solves a GeometryOptimisationProblem
# to represent a fictitious TF coil centerline.
#
# ## Defining the GeometryOptimisationProblem
#
# First we have to consider the optimisation problem that we would like to solve as part
# of our Design. In this case we would like to optimise the TF Coil centerline by making
# it as long as possible. This can be performed by solving an unconstrained
# GeometryOptimisationProblem that uses the negative length as the objective function
# (since minimising the negative length will maximise the value of the absolute length).
# Such an optimisation problem can be defined as below.

# %%
class MaximiseLength(GeometryOptimisationProblem):
    """
    A simple geometry optimisation problem that minimises length without constraints.
    """

    def calculate_length(self, x):
        """
        Calculate the length of the GeometryParameterisation.

        Result is negative as we're maximising rather than minimising. Note that most
        real life problems will minimise.
        """
        self.update_parameterisation(x)
        return -self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        """
        Objective function is the length of the parameterised shape.
        """
        length = self.calculate_length(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=length
            )

        return length


# %%[markdown]
# ## Configuring the Design
#
# The Design is configured by passing in a dictionary that provides the build stages to
# be run, and the way that each of those build stages should be set up. In particular
# in this case we define the following:
#
# - The Plasma build stage:
#   - uses the `MakeParameterisedShape` Builder class.
#   - parameterises the plasma shape using the `JohnerLCFS` GeometryParameterisation
#     class.
#   - maps the R_0 and A Design parameters to the r_0 and a shape parameters.
#   - labels the resulting component as "Shape".
#
# - The TF Coils build stage:
#   - uses the `MakeOptimisedShape` Builder class.
#   - parameterises the TF Coils centerline shape using the `PrincetonD`
#     GeometryParameterisation class.
#   - maps the r_tf_in_centre and r_tf_out_centre Design parameters to the x1 and x2
#     shape parameters. The dz shape parameter is set to 0 and fixed. The x1 shape
#     parameter is also fixed, while the x2 shape parameter is allowed to vary in the
#     optimisation, with an adjusted lower bound of 14.
#   - uses the `MaximiseLength` GeometryOptimisationProblem class to define the design
#     problem that will be solved as part of the TF Coil build stage.
#   - labels the resulting component as "Shape".


# %%
build_config = {
    "Plasma": {
        "class": "MakeParameterisedShape",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "label": "Shape",
    },
    "TF Coils": {
        "class": "MakeOptimisedShape",
        "param_class": "PrincetonD",
        "variables_map": {
            "x1": {
                "value": "r_tf_in_centre",
                "fixed": True,
            },
            "x2": {
                "value": "r_tf_out_centre",
                "lower_bound": 14.0,
            },
            "dz": {
                "value": 0.0,
                "fixed": True,
            },
        },
        "problem_class": MaximiseLength,
        "label": "Shape",
    },
}


# %%[markdown]
# ## Parameterising the Design
#
# The Design is parameterised by mapping the required parameter names to their initial
# values. All Designs must have a Name, and in this case the other required parameters
# are defined by the mapped variables in the `build_config`.

# %%
params = {
    "Name": "Shape Design Example",
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
    "r_tf_in_centre": (5.0, "Input"),
    "r_tf_out_centre": (15.0, "Input"),
}

# %%[markdown]
# ## Running the Design
#
# The Design object is defined as run as below:

# %%
design = Design(params, build_config)
component = design.run()

# %%[markdown]
# ## Visualising the Results
#
# The result of the design is a Component object that represents a tree of outputs from
# the different build stages. The resulting Component tree from this Design can be
# printed out.

# %%
print(component.tree())

# %%[markdown]
# It is also possible to plot the nodes in the tree that are defined with shapes (known
# as PhysicalComponents).

# %%
component.plot_2d()

# %%[markdown]
# We can also inspect the properties of the individual build stages by extracting them
# from the Design. This example shows how the resulting parameters from the solution of
# the TF Coils design problem can be extracted. Note that we have maximised the value of
# x2 without constraint, so it has found the upper bound as defined on that variable.
# All other variables have stayed as originally defined as they were set to be fixed.

# %%
tf_builder = design.get_builder("TF Coils")
tf_design_problem: GeometryOptimisationProblem = tf_builder.design_problem
print(tf_design_problem.parameterisation.variables)
