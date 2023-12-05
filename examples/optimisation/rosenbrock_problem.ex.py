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
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Constrained Rosenbrock Optimisation Problem
"""


# %% [markdown]
# # Constrained Rosenbrock Optimisation Problem
# Let's solve the unconstrained minimization problem:
#
# $$ \min_{x \in \mathbb{R}^2} (a-x_1)^2 + b(x_2-x_1^2)^2 \tag{1}$$
#
# for parameters $a = 1$, $b = 100$.
#
# This problem expects a minimum at $ x = ( a, a^2 ) $.
#

# %%
import time

import numpy as np

from bluemira.optimisation import optimise


def f_rosenbrock(x, a, b):
    """The Rosenbrock function."""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def df_rosenbrock(x, a, b):
    """Gradient of the Rosenbrock function."""
    grad = np.zeros(2)
    grad[0] = -2 * a + 4 * b * x[0] ** 3 - 4 * b * x[0] * x[1] + 2 * x[0]
    grad[1] = 2 * b * (x[1] - x[0] ** 2)
    return grad


a = 1
b = 100

for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    t1 = time.time()
    result = optimise(
        lambda x: f_rosenbrock(x, a, b),
        x0=np.array([0.5, -0.5]),
        algorithm=algorithm,
        df_objective=lambda x: df_rosenbrock(x, a, b),
        opt_conditions={"ftol_rel": 1e-12, "ftol_abs": 1e-12},
        keep_history=False,
        bounds=([-2, -2], [3, 3]),
    )
    t2 = time.time()
    print(f"{algorithm}: {result}, time={t2 - t1:.3f} seconds")

# %% [markdown]
# They all get pretty close to the optimum here.
#
# The SLSQP algorithm which leverages the gradient we (in this case not so painfully)
# derived, does better here. It finds the optimum exactly, and in very few iterations.
#
# COBYLA does not directly use the analytical gradient we provided it, but does a
# reasonable (and deterministic job).
#
# ISRES is a stochastic optimiser, and will perform differently each time, that is
# unless one sets the random seed. It is unlikely ever to find the exact optimum in
# this simple problem, but it will get reasonably close.
#
# Now let's add in a couple of simple constraints:
#
# $$ \min_{x \in \mathbb{R}^2} (a-x_1)^2 + b(x_2-x_1^2)^2 \tag{1}$$
#
# subject to
#
# $$ x_1 + x_2 \le 3 \tag{2} $$
# $$ x_2 -2x_1 \ge 1 \tag{3} $$
#
# for parameters $a = 1$, $b = 100$.

# %%


def f_constraint(x):
    """
    Let's say that we only want to search the space in which some combinations of
    variables are not allowed. We can't implement this using just bounds, hence we need
    to add some constraints in.

    All we're effectively doing here is chopping the search space rectangle, and saying
    that:
        x1 + x2 < 3
        x2 - 2x1 > 1
    """
    constraint = np.zeros(2)
    constraint[0] = (x[0] + x[1]) - 3
    constraint[1] = (-2 * x[0] + x[1]) + 1
    return constraint


def df_constraint(x):  # noqa: ARG001
    """Constraint Jacobian"""
    jac = np.zeros((2, 2))
    jac[0, 0] = 1
    jac[0, 1] = 1
    jac[1, 0] = -2
    jac[1, 1] = 1
    return jac


for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    t1 = time.time()
    result = optimise(
        lambda x: f_rosenbrock(x, a, b),
        x0=np.array([0.5, -0.5]),
        algorithm=algorithm,
        df_objective=lambda x: df_rosenbrock(x, a, b),
        opt_conditions={"ftol_rel": 1e-12, "ftol_abs": 1e-12, "max_eval": 1000},
        keep_history=False,
        bounds=([-2, -2], [3, 3]),
        ineq_constraints=[
            {
                "f_constraint": f_constraint,
                "df_constraint": df_constraint,
                "tolerance": 1e-6 * np.ones(2),
            },
        ],
    )
    t2 = time.time()
    print(f"{algorithm}: {result}, time={t2 - t1:.3f} seconds")

# %% [markdown]
# SLSQP and COBYLA do fine here, because there is only one minimum and it is a problem
# well suited to these algorithms. Note that the optimum complies with the constraints,
# so these algorithms actually perform better with the constraints (there is less space
# to search, and more rules and gradients to leverage).
#
# ISRES probably won't do so well here on average.
#
# It's usually wise to set a max_eval termination condition when doing optimisations,
# otherwise they can take a very long time... and may never converge. A default of
# max_eval = 2000 is set for you, but you can easily override it.
