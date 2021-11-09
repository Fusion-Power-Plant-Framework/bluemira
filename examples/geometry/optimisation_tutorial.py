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
A quick tutorial on the optimisation of geometry in bluemira
"""

import numpy as np

from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import set_random_seed


set_random_seed(134365475)


class MyProblem(GeometryOptimisationProblem):
    """
    A simple geometry optimisation problem
    """

    def calculate_length(self, x):
        """
        Calculate the length of the GeometryParameterisation
        """
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        """
        Signature for an objective function.

        If we use a gradient-based optimisation algorithm and we don't how to calculate
        the gradient, we can approximate it numerically.

        Note that this is not particularly robust in some cases... Probably best to
        calculate the gradients analytically, or use a gradient-free algorithm.
        """
        length = self.calculate_length(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=length
            )

        return length


# Here we solve the problem with a gradient-based optimisation algorithm (SLSQP)

parameterisation_1 = PrincetonD(
    {
        "x1": {"lower_bound": 2, "value": 4, "upper_bound": 6},
        "x2": {"lower_bound": 10, "value": 14, "upper_bound": 16},
        "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5},
    }
)
# Here we're minimising the length, and we can work out that the dz variable will not
# affect the optimisation, so let's just fix at some value and remove it from the problem
parameterisation_1.fix_variable("dz", value=0)
slsqp_optimiser = Optimiser(
    "SLSQP", opt_conditions={"max_eval": 100, "ftol_abs": 1e-12, "ftol_rel": 1e-12}
)
problem = MyProblem(parameterisation_1, slsqp_optimiser)
problem.solve()

# Here we're minimising the length, within the bounds of our PrincetonD parameterisation,
# so we'd expect that x1 goes to its upper bound, and x2 goes to its lower bound.

print(
    f"x1: value: {parameterisation_1.variables['x1'].value}, upper_bound: {parameterisation_1.variables['x1'].upper_bound}"
)
print(
    f"x2: value: {parameterisation_1.variables['x2'].value}, lower_bound: {parameterisation_1.variables['x2'].lower_bound}"
)

# Now let's do the same with an optimisation algorithm that doesn't require gradients
parameterisation_2 = PrincetonD()
# Let's leave the dz variable in there this time...
cobyla_optimiser = Optimiser(
    "COBYLA",
    opt_conditions={
        "ftol_rel": 1e-3,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
problem = MyProblem(parameterisation_2, cobyla_optimiser)
problem.solve()

# Again, let's check it's found the correct result:

print(
    f"x1: value: {parameterisation_1.variables['x1'].value}, upper_bound: {parameterisation_1.variables['x1'].upper_bound}"
)
print(
    f"x2: value: {parameterisation_1.variables['x2'].value}, lower_bound: {parameterisation_1.variables['x2'].lower_bound}"
)


# Now let's include a relatively arbitrary constraint:
# We're going to minimise length again, but with a constraint that says that we don't
# want the length to be below some arbitrary value of 50.
# There are much better ways of doing this, but this is to demonstrate the use of an
# inequality constraint.


class MyConstrainedProblem(MyProblem):
    """
    Now, a constraint is added in
    """

    def __init__(self, parameterisation, optimiser, ineq_con_tolerances):
        super().__init__(parameterisation, optimiser)
        self.optimiser.add_ineq_constraints(self.f_ineq_constraints, ineq_con_tolerances)
        self.some_arg_value = 50

    def my_constraint(self, x):
        """
        Constraints are satisfied if the return value(s) are negative
        """
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        return np.array([self.some_arg_value - length])

    def f_ineq_constraints(self, constraint, x, grad):
        """
        Signature for an inequality constraint.

        If we use a gradient-based optimisation algorithm and we don't how to calculate
        the gradient, we can approximate it numerically.

        Note that this is not particularly robust in some cases... Probably best to
        calculate the gradients analytically, or use a gradient-free algorithm.
        """
        constraint[:] = self.my_constraint(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(self.my_constraint, x, constraint)

        return constraint


parameterisation_3 = PrincetonD()
slsqp_optimiser2 = Optimiser(
    "SLSQP",
    opt_conditions={
        "ftol_rel": 1e-3,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
problem = MyConstrainedProblem(parameterisation_3, slsqp_optimiser2, 1e-6 * np.ones(1))
problem.solve()

# Both x1 and x2 are free variables and between them they should be create a PrincetonD
# shape of length exactly 50 (as the bounds on these variables surely allow it).
# As we are minimising length, we'd expect to see a function value of 50 here (+/- the
# tolerances)... but we don't!

print(f"Theoretical optimum: {problem.some_arg_value-1e-6}")
print(f"Length: {parameterisation_3.create_shape().length}")
print(f"n_evals: {problem.optimiser.n_evals}")

# This is because we're using numerical gradients and jacobians for our objective and
# inequality constraint functions. This can be faster than other approaches, but is less
# robust and also less likely to find the best solution.

# Let's try a few different optimisers, noting:
#       The different termination conditions we can play with and their effect
#       The fact that ISRES is a stochastic optimiser, and its results will vary if we
#       don't always reset the random seed value.

parameterisation_4 = PrincetonD()
cobyla_optimiser2 = Optimiser(
    "COBYLA",
    opt_conditions={
        "ftol_rel": 1e-6,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
problem = MyConstrainedProblem(parameterisation_4, cobyla_optimiser2, 1e-6 * np.ones(1))
problem.solve()

print(f"Theoretical optimum: {problem.some_arg_value-1e-6}")
print(f"Length with COBYLA: {parameterisation_4.create_shape().length}")
print(f"n_evals: {problem.optimiser.n_evals}")

parameterisation_5 = PrincetonD()
irses_optimiser = Optimiser(
    "ISRES",
    opt_conditions={
        "ftol_rel": 1e-12,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
problem = MyConstrainedProblem(parameterisation_5, irses_optimiser, 1e-6 * np.ones(1))
problem.solve()

print(f"Theoretical optimum: {problem.some_arg_value-1e-6}")
print(f"Length with ISRES: {parameterisation_5.create_shape().length}")
print(f"n_evals: {problem.optimiser.n_evals}")


# Horses for courses folks... YMMV. Best thing you can do is specify your optimisation
# problem intelligently, using well-behaved objective and constraint functions, and smart
# bounds. Trying out different optimisers doesn't hurt. There's a trade-off between speed
# and accuracy. If you can't work out the analytical gradients, numerical gradients are a
# questionable approach, but do work well (fast) on some problems.
