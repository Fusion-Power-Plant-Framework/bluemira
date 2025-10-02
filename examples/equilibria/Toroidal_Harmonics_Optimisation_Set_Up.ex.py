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
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    brute_force_toroidal_harmonic_approximation,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem._tikhonov import (
    TikhonovCurrentCOP,  # noqa: PLC2701
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
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

# using brute force
result = brute_force_toroidal_harmonic_approximation(
    eq=eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
    cl=True,
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
    eq=eq, th_params=th_params, result=result, psi_norm=psi_norm, cl=True
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
plt.show()
# %% [markdown]
# ## Use in Optimisation Problem
# We can use the amplitudes for each of our approximation poloidal modes
# as constraints or targets. In the following example, we do both.
# In this example, we are merely attempting to preserve an equilibrium
# solution using only TH. Other constraints and/or targets could be
# used in conjunction with the TH.
# %%
# Create a constraint
th_constraint = ToroidalHarmonicConstraint(
    ref_harmonics_cos=result.cos_m,
    ref_harmonics_sin=result.sin_m,
    ref_harmonics_cos_amplitudes=result.cos_amplitudes_from_psi_fit,
    ref_harmonics_sin_amplitudes=result.sin_amplitudes_from_psi_fit,
    constraint_type="equality",
    th_params=th_params,
)
# Ensure control coils are set to those that can be used in the toroidal
# harmonic approximation
eq.coilset.control = list(th_params.th_coil_names)

# Show the constraint region
f, ax = plt.subplots()
th_constraint.plot(ax=ax)
eq.plot(ax=ax)

# %%
# Run the optimisation
th_current_opt_eq = deepcopy(eq)

current_opt_problem = TikhonovCurrentCOP(
    th_current_opt_eq,
    targets=MagneticConstraintSet([th_constraint]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[th_constraint],
)

program = PicardIterator(
    th_current_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.0,
    maxiter=30,
)
program()


# %%
# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

eq.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

th_current_opt_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium")
plt.show()

# %%
# Print coilset currents
print(f"Original currents = {eq.coilset.current}")
print(f"Optimised currents = {th_current_opt_eq.coilset.current}")

# %% [markdown]
# Plot comparison to see how much the psi has changed.
# %%
original_FS = (  # noqa: N816
    eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
)
approx_FS = th_current_opt_eq.get_flux_surface(psi_norm)  # noqa: N816

total_psi_diff = np.abs(eq.psi() - th_current_opt_eq.psi()) / np.max(
    np.abs(th_current_opt_eq.psi())
)

f, ax = plt.subplots()
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
ax.plot(
    approx_FS.x,
    approx_FS.z,
    color="red",
    label="Approximate FS after \noptimising using TH",
)
ax.plot(
    original_FS.x,
    original_FS.z,
    color="blue",
    label="Original equilibrium FS \nfrom Bluemira",
)
im = ax.contourf(eq.grid.x, eq.grid.z, total_psi_diff, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title(
    "Absolute relative difference between total psi and TH approximation psi", y=1.05
)
ax.legend(bbox_to_anchor=(1.1, 1.05))
eq.coilset.plot(ax=ax)
plt.show()
# %% [markdown]
# Fit metric
#      The fit metric we use for the LCFS comparison is as follows:
#         fit metric value = total area within one but not both LCFSs /
#                                      (input LCFS area + approximation LCFS area)
# %%
fit_metric = fs_fit_metric(original_FS, approx_FS)
print(f"fit metric = {fit_metric}")
