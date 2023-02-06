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
Optimiser API tutorial
"""

# %%
from pprint import pprint

import numpy as np

from bluemira.utilities.optimiser import Optimiser, approx_derivative

# %% [markdown]
#
# # Optimisation example


# %%
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

    return value


results = {}
for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    optimiser = Optimiser(
        algorithm, 2, opt_conditions={"ftol_rel": 1e-22, "ftol_abs": 1e-12}
    )
    optimiser.set_objective_function(f_rosenbrock)
    optimiser.set_lower_bounds([-2, -2])
    optimiser.set_upper_bounds([3, 3])
    result = optimiser.optimise([0.5, -0.5])
    results[algorithm] = {
        "x": result,
        "f(x)": optimiser.optimum_value,
        "n_evals": optimiser.n_evals,
    }

print("Rosenbrock results:")
pprint(results)


# %% [markdown]
# They all get pretty close to the optimum here.
#
# The SLSQP algorithm which leverages the gradient we (in this case not so painfully)
# derived, does better here. It finds the optimum exactly, and in very few iterations.
#
# Now let's add in a constraint


# %%
def f_constraint(constraint, x, grad):
    """
    Let's say that we only want to search the space in which some combinations of
    variables are not allowed. We can't implement this using just bounds, hence we need
    to add some constraints in.

    All we're effectively doing here is chopping the search space rectangle, and saying
    that:
        x1 + x2 < 3
        x2 - 2x1 > 1
    """
    constraint[0] = (x[0] + x[1]) - 3
    constraint[1] = (-2 * x[0] + x[1]) + 1

    if grad.size > 0:
        # Again, if we can easily calculate the gradients.. we should!
        grad[0, :] = np.array([1, 1])
        grad[1, :] = np.array([-2, 1])

    return constraint


results = {}
for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    optimiser = Optimiser(
        algorithm,
        2,
        opt_conditions={"ftol_rel": 1e-22, "ftol_abs": 1e-12, "max_eval": 1000},
    )
    optimiser.set_objective_function(f_rosenbrock)
    optimiser.set_lower_bounds([-2, -2])
    optimiser.set_upper_bounds([3, 3])
    optimiser.add_ineq_constraints(f_constraint, tolerance=1e-6 * np.ones(2))
    result = optimiser.optimise([0.5, -0.5])
    results[algorithm] = {
        "x": result,
        "f(x)": optimiser.optimum_value,
        "n_evals": optimiser.n_evals,
    }

print("Constrained Rosenbrock results")
pprint(results)


# %% [markdown]
# So SLSQP and COBYLA do fine here, because there is only one minimum and it is a problem
# well suited to these algorithms. Note that the optimum complies with the constraints,
# so these algorithms actually perform better with the constraints (there is less space
# to search, and more rules and gradients to leverage).
#
# ISRES probably won't do so well here on average. Note that every time you run ISRES,
# you will get different results unless you set the same random seed.
#
# It's usually wise to set a max_eval termination condition when doing optimisations,
# otherwise they can take a very long time... and may never converge.
#
# What about a strongly multi-modal function with no easy analytical gradient?


# %%
def f_eggholder(x):
    """
    The multi-dimensional Eggholder function. It is strongly multi-modal.

    For the 2-D case bounded at +/- 512, the optimum is at:
        f(512, 404.2319..) = -959.6407..
    """
    f_x = 0
    for i in range(len(x) - 1):
        f_x += -(x[i + 1] + 47) * np.sin(np.sqrt(abs(x[i + 1] + 0.5 * x[i] + 47))) - x[
            i
        ] * np.sin(np.sqrt(abs(x[0] - x[i + 1] - 47)))
    return f_x


def f_eggholder_objective(x, grad):
    """
    Our little wrapper to interface with the optimiser (which needs a grad
    argument).
    """
    value = f_eggholder(x)

    if grad.size > 0:
        # Here, SLSQP needs to know what the gradient of the objective function is..
        # Seeing as we are lazy, we're going to approximate it.
        # This is not particularly robust, and can cause headaches.
        grad[:] = approx_derivative(f_eggholder, x, f0=value)

    return value


results = {}
for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    optimiser = Optimiser(
        algorithm,
        2,
        opt_conditions={"ftol_rel": 1e-22, "ftol_abs": 1e-12, "max_eval": 10000},
    )
    optimiser.set_objective_function(f_eggholder_objective)
    optimiser.set_lower_bounds([-512, -512])
    optimiser.set_upper_bounds([512, 512])
    result = optimiser.optimise([0, 0])
    results[algorithm] = {
        "x": result,
        "f(x)": optimiser.optimum_value,
        "n_evals": optimiser.n_evals,
    }

print("Eggholder results:")
pprint(results)

# %% [markdown]
# SLSQP and COBYLA are local optimisation algorithms, and converge rapidly on a local
# minimum. ISRES is a stochastic global optimisation algorithm, and keeps looking for
# longer, finding a much better minimum, but caps out at the maximum number of
# evaluations (usually).
