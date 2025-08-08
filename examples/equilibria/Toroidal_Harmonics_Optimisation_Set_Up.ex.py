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
Usage of the 'toroidal_harmonic_approximation' function.
"""

# %% [markdown]
# # toroidal_harmonic_approximation Function
#
# This example illustrates the usage of the bluemira
# toroidal_harmonics_approximation function to create Toroidal
# Harmonic (TH) constraints to be used in a coil current optimisation
# problem for a double null DEMO-like tokamak.
#
# For more details about the inner workings of this function,
# see Toroidal_Approximation_Explained.ex.py.

# %% [markdown]
# ## Imports

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    toroidal_harmonic_approximation,
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

# %% [markdown]
# ## Inputs
#
# ### Required
#
# - eq = Our chosen bluemira equilibrium
#
# ### Optional
#
# - th_params: dataclass containing necessary parameters for use in TH approximation.
#   'None' will default to using the toroidal_grid_and_coil_setup function with the
#   input equilibrium to generate the th_params
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - psi_norm: 'None' will default to LCFS, otherwise choose the desired
#   normalised psi value of a closed flux surface that contains the core plasma
# - plot: Whether or not to plot the results

# %%
# Information needed for TH Approximation
psi_norm = 0.95
th_params, Am_cos, Am_sin, degree, fit_metric, approx_total_psi, approx_coilset_psi = (
    toroidal_harmonic_approximation(eq=eq, psi_norm=psi_norm, acceptable_fit_metric=0.01)
)

# %% [markdown]
# ## Outputs
#
# ### Results for use in optimisation
# - th_params: dataclass containing necessary parameters for use in TH approximation
# - Am_cos: TH cos amplitudes for required number of degrees
# - Am_sin: TH sin amplitudes for required number of degrees
#
# ### Informative outputs
#
# - degree: number of degrees required for a TH approx with the desired fit metric
# - fit_metric_value: fit metric achieved
# - approx_total_psi: the total psi obtained using the TH approximation
# - approx_coilset_psi: the coilset psi obtained using the TH approximation

# %%
# Print the outputs from the toroidal_harmonic_approximation function
print(f"Coils used in TH approximation: \n {th_params.th_coil_names}")

# %%
print(
    f"Number of degrees required for a TH approx with the desired fit metric: {degree}"
)
# %%
print(f"TH cos coefficients/amplitudes for {degree} degrees: \n{Am_cos}")
print(f"TH sin coefficients/amplitudes for {degree} degrees: \n{Am_sin}")
# %%
print(f"Fit metric achieved: {fit_metric}")

# %% [markdown]
# ## Use in Optimisation Problem
#
# Now we will use the approximation to set up constraints for an optimisation problem.
# We use a Tikhonov coilset optimisation problem with TH as constraints.
# This will try to minimise the sum of the currents squared while constraining the coil
# contribution to the core psi.
# Note: there is a limit to how many equality constraints you can use, so if you
# encounter any associated errors then try setting constraint_type="inequality".
# This will create 2 inequality constraints that are equivalent to the equality
# constraint.
#
# In this example we use an equality constraint.
# %%
# Use results of the toroidal harmonic approximation to create a set of coil constraints
th_constraint = ToroidalHarmonicConstraint(
    ref_harmonics_cos=Am_cos,
    ref_harmonics_sin=Am_sin,
    th_params=th_params,
    tolerance=1e-3,
    constraint_type="equality",
)

# Make sure we only optimise with coils outside the sphere containing the core plasma by
# setting control coils using the list of appropriate coils
eq.coilset.control = list(th_params.th_coil_names)

# %%
# Add an x point constraint
lcfs = eq.get_LCFS()
x_bdry, z_bdry = lcfs.x, lcfs.z
xp_idx = np.argmin(z_bdry)
x_point = FieldNullConstraint(
    x_bdry[xp_idx],
    z_bdry[xp_idx],
    tolerance=1e-1,
)

# %% [markdown]
# We are aiming to move the inner leg of the divertor, and we will use isoflux
# constraints to do so. Here we set up these isoflux constraints and
# show them plotted on our equilibria.
# %%
# Want to move the inner leg
# Create isoflux constraints for the target inner leg
arg_inner = np.argmin(x_bdry)

# Define points to use for the isoflux constraints
inner_leg_points_x = np.array([
    6.5,
    7.0,
    7.5,
])

inner_leg_points_z = np.array([
    6.25,
    5.95,
    5.7,
])

outer_legs_x_unmoved = np.array([
    9.7,
    9.89,
    10.1,
])

outer_legs_z_unmoved = np.array([6.5, 7.0, 7.5])

# Create the necessary isoflux constraints for the inner and outer legs, for
# the upper and lower divertors.
DN_unmoved_outer_leg_upper = IsofluxConstraint(
    outer_legs_x_unmoved,
    outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)

DN_unmoved_outer_leg_lower = IsofluxConstraint(
    outer_legs_x_unmoved,
    -outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)

DN_inner_leg_upper = IsofluxConstraint(
    inner_leg_points_x,
    inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)

DN_inner_leg_lower = IsofluxConstraint(
    inner_leg_points_x,
    -inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)

# Plot the isoflux points and the starting equilibrium for reference
f, ax = plt.subplots()
eq.plot(ax=ax)
DN_inner_leg_upper.plot(ax=ax)
DN_inner_leg_lower.plot(ax=ax)
DN_unmoved_outer_leg_upper.plot(ax=ax)
DN_unmoved_outer_leg_lower.plot(ax=ax)
plt.show()

# %% [markdown]
# Now we perform the Tikhonov coilset optimisation using our toroidal
# harmonic and isoflux leg constraints.

# %%
# OPTIMISATION

# Define the constraints to use in our optimisation problem
constraints = [
    th_constraint,
    x_point,
    DN_inner_leg_upper,
    DN_inner_leg_lower,
]

algorithm = "COBYLA"
# %%
# Make a copy of the equilibria
th_eq = deepcopy(eq)
# Set up a coilset optimisation problem using the toroidal harmonic constraints
th_con_len_opt = TikhonovCurrentCOP(
    th_eq,
    targets=MagneticConstraintSet([
        DN_inner_leg_upper,
        DN_inner_leg_lower,
        DN_unmoved_outer_leg_upper,
        DN_unmoved_outer_leg_lower,
    ]),
    gamma=1e-4,
    opt_algorithm=algorithm,
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=constraints,
)
# Find the optimised coilset
_ = th_con_len_opt.optimise()

# Update plasma - one solve
th_eq.solve()

# %% [markdown]
# We should not need to solve the GS equation while optimising if the TH approximation
# is sufficiently good, but we can have a look at what happens.

# %%
# Make a copy of the equilibria
th_current_opt_eq = deepcopy(eq)

# Set up a coilset optimisation problem using the toroidal harmonic constraint
current_opt_problem = TikhonovCurrentCOP(
    th_current_opt_eq,
    targets=MagneticConstraintSet([
        DN_inner_leg_upper,
        DN_inner_leg_lower,
        DN_unmoved_outer_leg_upper,
        DN_unmoved_outer_leg_lower,
    ]),
    gamma=1e-4,
    opt_algorithm=algorithm,
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=constraints,
)

diagnostic_plotting = PicardDiagnosticOptions(plot=PicardDiagnostic.EQ)
# SOLVE
program = PicardIterator(
    th_current_opt_eq,
    current_opt_problem,
    diagnostic_plotting=diagnostic_plotting,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.1,
    maxiter=30,
)
program()

# %%
# Plot the two approches
f, (ax_1, ax_2) = plt.subplots(1, 2)

th_eq.plot(ax=ax_1)
ax_1.set_title("Coils Optimised")

th_current_opt_eq.plot(ax=ax_2)
ax_2.set_title("Coils Optimised while GS solved")
plt.show()

# %% [markdown]
# Now we want to plot the starting equilibrium and the optimised
# equilibrium after using our TH constraints. We also show the
# isoflux points in purple, which are used to alter the shape of the divertor legs.

# %%
f, (ax_1, ax_2) = plt.subplots(1, 2)

eq.plot(ax=ax_1)
DN_inner_leg_upper.plot(ax=ax_1)
DN_inner_leg_lower.plot(ax=ax_1)
DN_unmoved_outer_leg_upper.plot(ax=ax_1)
DN_unmoved_outer_leg_lower.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

th_current_opt_eq.plot(ax=ax_2)
DN_inner_leg_upper.plot(ax=ax_2)
DN_inner_leg_lower.plot(ax=ax_2)
DN_unmoved_outer_leg_upper.plot(ax=ax_2)
DN_unmoved_outer_leg_lower.plot(ax=ax_2)
ax_2.set_title("TH Used to Optimise Leg Shaping")
plt.show()


# %% [markdown]
# Now we create a difference plot to show the absolute relative difference in total psi
# between the starting equilibrium and the optimised equilibrium. We also show the
# flux surface for each equilibrium for the psi_norm chosen - in this example it is
# set to 0.95.

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
# Compare starting equilibrium and optimised equilibrium using fit metric to see how
# "good" our approximation is.
# Fit metric value = total area within one but not both FSs /
#                    (input FS area + approximation FS area)
# A value of 0 would correspond to no differences, a value
# of 1 would correspond to no overlap between the flux surfaces.

# %%
fit_metric = fs_fit_metric(original_FS, approx_FS)
print(f"fit metric = {fit_metric}")
