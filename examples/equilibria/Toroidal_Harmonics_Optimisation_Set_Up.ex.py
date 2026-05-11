# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
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
Usage of the 'brute_force_toroidal_harmonic_approximation' function.
"""

# %% [markdown]
# # Example of using Toroidal Harmonic Constraints in a Coil Current Optimisation
#
# This example illustrates the usage of the bluemira
# brute_force_toroidal_harmonic_approximation function to create Toroidal
# Harmonic (TH) constraints to be used in a coil current optimisation
# problem for a double null DEMO-like tokamak.
#

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)

# %%
# Data from EQDSK file
# Using a double null DEMO-like equilibria here
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


eq_name = "DN-DEMO_eqref.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(
    eq_name, from_cocos=3, qpsi_positive=False, force_symmetry=True
)

# Plot equilibrium
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()


# %%[markdown]
# ## Setup

# Find TH approximation of coilset contribution to the core
# plasma region.
# %%
# Information needed for TH Approximation
psi_norm = 0.95
R_0, Z_0 = eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

# toroidal harmonic approximation
result = toroidal_harmonic_approximation(
    eq=eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
)

# %% [markdown]
# In this example, the TH approximation that gives the closest fit to
# the bluemira coilset psi we are trying to preserve requires the following
# 6 poloidal modes:
# %%
# print info and plot
print(f"Cos modes used = {result.cos_m}")
print(f"Sin modes used = {result.sin_m}")
# plot to compare th approx psi to bm psi
f, ax = plot_toroidal_harmonic_approximation(
    eq=eq, th_params=th_params, result=result, psi_norm=psi_norm
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
plt.show()
