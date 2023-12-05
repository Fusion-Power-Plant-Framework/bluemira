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
# # Eggholder Optimisation Problem
# Let's solve the unconstrained minimization problem:
#
# $$ \min_{x \in \mathbb{R}^2} -(x_2 + 47) \sin{\sqrt{\lvert\frac{x_2}{2}+x_1+47\rvert}}
# -x_1 \sin{\sqrt{\lvert x_1-x_2-47\rvert}} \tag{1}$$
#
# For the 2-D case, bounded at $\pm 512$, this function expects a minimum
# at $ x = (512, 404.2319..) = -959.6407..$.
#
# Here, we're too lazy to come up with an analytical gradient, but let's see what we
# can do without.
#
# For the algorithms that require a gradient (e.g. SLSQP), one is automatically
# estimated for you, but you should not rely on this too heavily!

# %%
import time

import matplotlib.pyplot as plt
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


results = {}
for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    t1 = time.time()
    result = optimise(
        f_eggholder,
        df_objective=None,
        x0=np.array([0.0, 0.0]),
        algorithm=algorithm,
        opt_conditions={"ftol_rel": 1e-12, "ftol_abs": 1e-12, "max_eval": 10000},
        bounds=([-512, -512], [512, 512]),
        keep_history=True,
    )
    results[algorithm] = result
    t2 = time.time()
    print(f"{algorithm}: {result}, time={t2 - t1:.3f} seconds")

# %% [markdown]
# SLSQP and COBYLA are local optimisation algorithms, and converge rapidly on a local
# minimum. ISRES is a stochastic global optimisation algorithm, and keeps looking for
# longer, finding a much better minimum, but caps out at the maximum number of
# evaluations (usually).


# %%
# %matplotlib inline

n = 500
x = y = np.linspace(-512, 512, n)
xx, yy = np.meshgrid(x, y, indexing="ij")
zz = np.zeros((n, n))
for i, xi in enumerate(x):
    for j, yi in enumerate(y):
        zz[i, j] = f_eggholder([xi, yi])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0.0)

for algorithm in ["SLSQP", "COBYLA", "ISRES"]:
    result = results[algorithm]
    ax.plot(*result.x, zs=result.f_x, marker="o", color="red")
    ax.text(*result.x, result.f_x, algorithm)

ax.set_title("Eggholder function")
ax.set_xlabel("\n\nx")
ax.set_ylabel("\n\ny")
ax.set_zlabel("\n\nz")
ax.view_init(elev=0, azim=-45)
plt.show()
