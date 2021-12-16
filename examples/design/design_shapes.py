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

from bluemira.base.design import Design
from bluemira.geometry.optimisation import GeometryOptimisationProblem


# Make an example optimisation problem to be solved in the design.


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


# Set up the configuration for the design.

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
params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
    "r_tf_in_centre": (5.0, "Input"),
    "r_tf_out_centre": (15.0, "Input"),
}

# Create our design object and run it to get the resulting component.

design = Design(params, build_config)
component = design.run()

# Plot our resulting component.

component.plot_2d()

# Get the TF design problem and print out the resulting parameterisation

tf_builder = design.get_builder("TF Coils")
tf_design_problem: GeometryOptimisationProblem = tf_builder.design_problem
print(tf_design_problem.parameterisation.variables)
