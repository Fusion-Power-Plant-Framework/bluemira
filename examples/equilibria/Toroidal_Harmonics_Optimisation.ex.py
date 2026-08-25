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
Example of using toroidal harmonic constraints in a
coil current optimisation.
"""

# %% [markdown]
# # Example of using Toroidal Harmonic Constraints in a Coil Current Optimisation
#
# This example illustrates the usage of the bluemira
# toroidal_harmonic_approximation function to create
# Toroidal Harmonic (TH) constraints to be used in a
# coil current optimisation problem for a single null
# DEMO-like equilibrium.

# %%
# Imports
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.analysis import EqAnalysis
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqSubplots,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find_legs import LegFlux
from bluemira.equilibria.optimisation.constraints import (
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem._tikhonov import (  # noqa: PLC2701
    TikhonovCurrentCOP,
)
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

# %%
# Get equilibrium data and plot
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

ref_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)

f, ax = plt.subplots()
ref_eq.plot(ax)
ref_eq.coilset.plot(ax)
plt.show()

# %% [markdown]
# Find the TH approximation of the coilset contribution to the
# core plasma region using the toroidal_harmonic_approximation function.

# %%
# Setup grid for TH approximation
psi_norm = 0.95
R_0, Z_0 = ref_eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=ref_eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)

# TH approximation
th_result = toroidal_harmonic_approximation(
    eq=ref_eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
)

# %% [markdown]
# We can see the TH modes selected by our approximation function
# and plot to compare the TH approximation of the coilset psi to the
# bluemira coilset psi.
# %%
# Info and plot
print(f"Cos modes used = {th_result.cos_m}")
print(f"Sin modes used = {th_result.sin_m}")
print(f"Error in approx = {th_result.error}")

# Plot to compare th approx psi to bm psi
f, ax = plot_toroidal_harmonic_approximation(
    eq=ref_eq, th_params=th_params, result=th_result, psi_norm=psi_norm
)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
ref_eq.coilset.plot(ax)
plt.show()

# %% [markdown]
# We can now make a TH constraint that we will use to hold the
# coilset contribution to the flux in the plasma region fixed in
# a coil current optimisation problem. This constraint uses
# the TH amplitudes from our approximation.
# %%
th_constraint = ToroidalHarmonicConstraint(
    th_result=th_result,
    constraint_type="equality",
)
# Ensure the control coils are set appropriately
# - could within the TH approx. region must have
# their currents held fixed
ref_eq.coilset.control = list(th_params.th_coil_names)

# Plot the constraint region
f, ax = plt.subplots()
th_constraint.plot(ax=ax)
ref_eq.coilset.plot(ax=ax)
ref_eq.plot(ax=ax)
ax.set_title("TH approximation region")
plt.show()

# %% [markdown]
# Now we can set up constraints for the new leg positions.
# In this example we will move the inner leg
# while holding the outer leg fixed.
# %%
# Leg movement
lcfs = ref_eq.get_LCFS()
x_bdry, z_bdry = lcfs.x, lcfs.z
arg_inner = np.argmin(x_bdry)

legs = LegFlux(ref_eq).get_legs()

# Get isoflux points along the outer leg to hold
# this fixed
out_leg_x = legs["lower_outer"][0].x[0::5]
out_leg_z = legs["lower_outer"][0].z[0::5]

# Set isoflux points that we will use to move the inner leg
in_leg_x = np.array([5.22, 4.17])
in_leg_z = np.array([-6.1, -6.37])

ref_lcfs = ref_eq.get_LCFS()
arg_inner = np.argmin(ref_lcfs.x)


inner_leg = IsofluxConstraint(
    in_leg_x, in_leg_z, ref_lcfs.x[arg_inner], ref_lcfs.z[arg_inner], tolerance=1e-3
)
outer_leg = IsofluxConstraint(
    out_leg_x, out_leg_z, ref_lcfs.x[arg_inner], ref_lcfs.z[arg_inner], tolerance=1e-3
)

# Plot to show the inner and outer leg constraints
f, ax = plt.subplots()
ref_eq.plot(ax)
inner_leg.plot(ax)
outer_leg.plot(ax)
ax.set_title("Isoflux points used for leg shaping")
plt.show()

# %% [markdown]
# Now we can perform the optimisation. We use the leg positions as
# magnetic targets, and the TH amplitudes as constraints.
#


# %%
# Perform the optimisation
# Use leg positions as magnetic targets, and
# TH amplitudes as constraint

# NOTE: If you are experimenting with changes to the optimisation
# problem, you may need to modify some of the optimiser
# or iterator settings. Our diagnostic plotting options can help
# you choose appropriate values.

th_move_legs_eq = deepcopy(ref_eq)

th_move_legs = TikhonovCurrentCOP(
    eq=th_move_legs_eq,
    targets=MagneticConstraintSet([
        inner_leg,
        outer_leg,
    ]),
    constraints=[
        th_constraint,
    ],
    gamma=1e-8,
)

program = PicardIterator(
    th_move_legs,
    fixed_coils=True,
    convergence=DudsonConvergence(limit=1e-3),
    relaxation=0.2,
    maxiter=30,
    check_constraints=True,
)

program()

# Plot side by side comparison between starting equilibrium
# and equilibrium after optimisation
f, axs = plt.subplots(1, 2)
ref_eq.plot(axs[0])
inner_leg.plot(axs[0])
outer_leg.plot(axs[0])
axs[0].set_title("Starting equilibrium")

th_move_legs_eq.plot(axs[1])
inner_leg.plot(axs[1])
outer_leg.plot(axs[1])
axs[1].set_title("Equilibrium after optimisation")

f.suptitle("Comparison of equilibrium before and after optimisation", y=0.85)
plt.show()
# %% [markdown]
# We can compare the plasma and coilset psi before and after the
# leg movement. Our aim was to hold fixed the coilset contribution
# to the flux within the 0.95 flux surface region.

# %%
diag_ops = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_ABS_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
)
eq_analysis = EqAnalysis(
    input_eq=th_move_legs_eq, diag_ops=diag_ops, reference_eq=ref_eq
)
eq_analysis.plot_compare_psi()
plt.show()
