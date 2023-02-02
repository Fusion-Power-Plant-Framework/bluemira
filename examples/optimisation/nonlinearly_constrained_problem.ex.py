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

# %%
import numpy as np

from bluemira.optimisation import optimise


def f_objective(x):
    return np.sqrt(x[1])


def df_objective(x):
    return np.array([0.0, 0.5 / np.sqrt(x[1])])


def f_constraint(x, a, b):
    return (a * x[0] + b) ** 3 - x[1]


def df_constraint(x, a, b):
    return np.array([3 * a * (a * x[0] + b) * (a * x[0] + b), -1.0])


result = optimise(
    f_objective,
    x0=np.array([1, 1]),
    algorithm="SLSQP",
    df_objective=df_objective,
    opt_conditions={"xtol_rel": 1e-4, "max_eval": 3000},
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
