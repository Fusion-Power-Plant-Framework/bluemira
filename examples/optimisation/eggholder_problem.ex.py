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
Constrained Rosenbrock Optimisation Problem
"""


# %% [markdown]
# # Eggholder Optimisation Problem
# Let's solve the unconstrained minimization problem:
#
# $$ \min_{x \in \mathbb{R}^2} -(x_2 + 47) \sin{\sqrt{\lvert\frac{x_2}{2}+x_1+47\rvert}} -x_1 \sin{\sqrt{\lvert x_1-x_2-47\rvert}} \tag{1}$$
#
# for parameters $a = 1$, $b = 100$.
#
# For the 2-D case, bounded at $\pm 512$, this function expects a minimum
# at $ x = (512, 404.2319..) = -959.6407..$.
#


# %%
import time

import numpy as np

from bluemira.optimisation import optimise


def f_eggholder(x):
    """
    The multi-dimensional Eggholder function. It is strongly multi-modal.
    """
    f_x = 0
    for i in range(len(x) - 1):
        f_x += -(x[i + 1] + 47) * np.sin(np.sqrt(abs(x[i + 1] + 0.5 * x[i] + 47))) - x[
            i
        ] * np.sin(np.sqrt(abs(x[0] - x[i + 1] - 47)))
    return f_x
