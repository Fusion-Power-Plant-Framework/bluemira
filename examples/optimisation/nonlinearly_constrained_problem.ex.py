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
Non-Linearly Constrained Optimisation Problem
"""

# %% [markdown]
# # Non-Linearly Constrained Optimisation Problem
# Let's solve the non-linearly constrained minimization problem:
#
# $$ \min_{x \in \mathbb{R}^2} \sqrt{x_2} \tag{1}$$
#
# subject to
#
# $$ x_2 \ge 0 \tag{2} $$
# $$x_2 \ge (a_1x_1 + b_1)^3 \tag{3} $$
# $$ x_2 \ge (a_2 x_1 + b_2)^3 \tag{4} $$
#
# for parameters $a_1 = 2$, $b_1 = 0$, $a_2 = -1$, $b_2 = 1$.
#
# This problem expects a minimum at $ x = ( \frac{1}{3}, \frac{8}{27} ) $.
#
# This example is ripped straight from the
# [NLOpt docs](https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/#example-nonlinearly-constrained-problem).
#

# %% [markdown]
# ## Using the `optimise` Function
# Let's perform this optimisation,
# utilising bluemira's `optimise` function.

# %%
from typing import List, Tuple

import numpy as np

from bluemira.optimisation import optimise


def f_objective(x: np.ndarray) -> float:
    """Objective function to minimise."""
    return np.sqrt(x[1])


def df_objective(x: np.ndarray) -> np.ndarray:
    """Gradient of the objective function."""
    return np.array([0.0, 0.5 / np.sqrt(x[1])])


def f_constraint(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Inequality constraint function."""
    return (a * x[0] + b) ** 3 - x[1]


def df_constraint(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Inequality constraint gradient."""
    return np.array([3 * a * (a * x[0] + b) * (a * x[0] + b), -1.0])


result = optimise(
    f_objective,
    x0=np.array([1, 1]),
    algorithm="SLSQP",
    df_objective=df_objective,
    opt_conditions={"xtol_rel": 1e-10, "max_eval": 1000},
    keep_history=True,
    bounds=(np.array([-np.inf, 0]), np.array([np.inf, np.inf])),
    ineq_constraints=[
        {
            "f_constraint": lambda x: f_constraint(x, 2, 0),
            "df_constraint": lambda x: df_constraint(x, 2, 0),
            "tolerance": np.array([1e-8]),
        },
        {
            "f_constraint": lambda x: f_constraint(x, -1, 1),
            "df_constraint": lambda x: df_constraint(x, -1, 1),
            "tolerance": np.array([1e-8]),
        },
    ],
)
for x in result.history:
    print(x)
print(result)

# %% [markdown]
# ## Using the `OptimisationProblem` Class
# Alternatively, we can take a class-based approach to defining this
# optimisation problem, using the `OptimisationProblem` base class.

# %%
from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.typing import ConstraintT


class NonLinearConstraintOP(OptimisationProblem):
    """Optimisation problem with non-linear constraints."""

    def __init__(self, a1: float, a2: float, b1: float, b2: float):
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2

    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimise."""
        return np.sqrt(x[1])

    def df_objective(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the objective function."""
        return np.array([0.0, 0.5 / np.sqrt(x[1])])

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The lower and upper bounds of the optimisation parameters.

        Each set of bounds must be convertible to a numpy array of
        floats. If the lower or upper bound is a scalar value, that
        value is set as the bound for each of the optimisation
        parameters.
        """
        return np.array([-np.inf, 0]), np.array([np.inf, np.inf])

    def ineq_constraints(self) -> List[ConstraintT]:
        """The inequality constraints on the optimisation."""
        return [
            {
                "f_constraint": lambda x: self.f_constraint(x, self.a1, self.b1),
                "df_constraint": lambda x: self.df_constraint(x, self.a1, self.b1),
                "tolerance": np.array([1e-8]),
            },
            {
                "f_constraint": lambda x: self.f_constraint(x, self.a2, self.b2),
                "df_constraint": lambda x: self.df_constraint(x, self.a2, self.b2),
                "tolerance": np.array([1e-8]),
            },
        ]

    def f_constraint(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Inequality constraint function."""
        return (a * x[0] + b) ** 3 - x[1]

    def df_constraint(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Inequality constraint gradient."""
        return np.array([3 * a * (a * x[0] + b) * (a * x[0] + b), -1.0])


opt_problem = NonLinearConstraintOP(2, -1, 0, 1)
result = opt_problem.optimise(
    x0=np.array([1, 1]),
    algorithm="SLSQP",
    opt_conditions={"xtol_rel": 1e-10, "max_eval": 1000},
    keep_history=False,
)
print(result)
