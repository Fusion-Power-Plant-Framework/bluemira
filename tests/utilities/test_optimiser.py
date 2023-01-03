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

import numpy as np

from bluemira.utilities.optimiser import Optimiser


def f_rosenbrock(x, grad):
    """
    The rosenbrock function has a single optimum at:
        f(a, a**2) = 0
    """
    a = 1
    b = 100
    value = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    if grad.size > 0:
        # Here we can calculate the gradient of the objective function
        # NOTE: Gradients and constraints must be assigned in-place
        grad[0] = -2 * a + 4 * b * x[0] ** 3 - 4 * b * x[0] * x[1] + 2 * x[0]
        grad[1] = 2 * b * (x[1] - x[0] ** 2)

    return float(value)


def f_simple(x, grad):
    value = np.sum(x)
    if grad.size > 0:
        grad[:] = np.ones(5)

    return float(value)


def f_ineq_constraint(constraint, x, grad):
    constraint[:] = 0.5 - x[0]
    if grad.size > 0:
        grad[0, :] = [-1, 0, 0, 0, 0]
    return constraint


def f_ineq_constraints(constraint, x, grad):
    constraint[0] = (x[0] + x[1]) - 4
    constraint[1] = (-2 * x[0] + x[1]) - 2

    if grad.size > 0:
        # Again, if we can easily calculate the gradients.. we should!
        grad[0, :] = np.array([1, 1, 0, 0, 0])
        grad[1, :] = np.array([-2, 1, 0, 0, 0])

    return constraint


def f_eq_constraint(constraint, x, grad):
    constraint[:] = x[2] - 3.14
    if grad.size > 0:
        grad[0, :] = [0, 0, 1, 0, 0]
    return constraint


def f_eq_constraints(constraint, x, grad):
    constraint[0] = x[3] - 3.15
    constraint[1] = x[4] - 3.16
    if grad.size > 0:
        grad[0, :] = [0, 0, 0, 1, 0]
        grad[1, :] = [0, 0, 0, 0, 1]
    return constraint


class TestOptimiser:
    def test_rosenbrock(self):
        optimiser = Optimiser(
            "SLSQP",
            2,
            opt_conditions={"xtol_abs": 1e-12, "max_eval": 1000},
        )
        optimiser.set_objective_function(f_rosenbrock)
        optimiser.set_lower_bounds([0, 0])
        optimiser.set_upper_bounds([3, 3])
        result = optimiser.optimise([0.0, 0.0])
        np.testing.assert_allclose(result, np.array([1.0, 1.0]))

    def test_constraints(self):
        optimiser = Optimiser(
            "SLSQP",
            5,
            opt_conditions={"xtol_abs": 1e-12, "max_eval": 1000},
        )
        tol = 1e-6
        optimiser.set_objective_function(f_simple)
        optimiser.set_lower_bounds([0, 0, -1, -1, -1])
        optimiser.set_upper_bounds([3, 3, 4, 4, 4])
        optimiser.add_ineq_constraints(f_ineq_constraint, tolerance=tol)
        optimiser.add_ineq_constraints(f_ineq_constraints, tolerance=tol * np.ones(2))
        optimiser.add_eq_constraints(f_eq_constraint, tolerance=tol)
        optimiser.add_eq_constraints(f_eq_constraints, tolerance=tol * np.ones(2))
        # NOTE: Convergence only guaranteed for feasible starting point!
        result = optimiser.optimise(x0=np.array([0.6, 0.5, 3.14, 3.15, 3.16]))
        np.testing.assert_allclose(result, np.array([0.5, 0.0, 3.14, 3.15, 3.16]))
