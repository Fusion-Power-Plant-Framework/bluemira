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
# This example illustrates the input and output of the
# bluemira toroidal harmonics approximation function
# (toroidal_harmonic_approximation) which can be used
# in coilset current and position optimisation for conventional aspect ratio tokamaks.

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.display.plotter import Zorder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    toroidal_harmonic_approximation,
)
from bluemira.equilibria.optimisation.problem import (
    MinimalCurrentCOP,
)
from bluemira.equilibria.optimisation.problem._tikhonov import (
    TikhonovCurrentCOP,  # noqa: PLC2701
)
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %pdb

# %%
# Data from EQDSK file
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

# eq_name = "eqref_OOB.json"
# eq_name = Path(EQDATA, eq_name)
# eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)
eq_name = "DN-DEMO_eqref.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=3, qpsi_positive=False)

# Plot
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# eq.coilset.control = [
#     "PF_1",
#     "PF_2",
#     "PF_3",
#     "PF_4",
#     "PF_5",
#     "PF_6",
# ]
print(eq.coilset.get_control_coils().name)

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
#   input equilibrium to get the th_params
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - psi_norm: 'None' will default to LCFS, otherwise choose the desired
#   normalised psi value of a closed flux surface that containe the core plasma
# - plot: Whether or not to plot the results

# %%
# Information needed for TH Approximation
psi_norm = 0.95
th_params, Am_cos, Am_sin, degree, fit_metric, approx_total_psi = (
    toroidal_harmonic_approximation(eq=eq, psi_norm=psi_norm, acceptable_fit_metric=0.01)
)

# %% [markdown]
# ## Outputs
#
# ### Results for use in optimisation
# - th_params: dataclass containing necessary parameters for use in TH approximation.
# - Am_cos: TH amplitudes for required number of degrees
# - Am_sin: TH amplitudes for required number of degrees
#
# ### Informative outputs
#
# - degree: number of degrees required for a TH approx with the desired fit metric
# - fit_metric_value: fit metric achieved
# - approx_total_psi: the total psi obtained using the TH approximation

# %%
# Print the outputs from the toroidal_harmonic_approximation function
print(th_params.th_coil_names)
print(Am_cos)
print(Am_sin)
print(degree)
print(fit_metric)

# %%
# Plot the approx total psi and bluemira total psi
psi = approx_total_psi
psi_original = eq.psi()
levels = np.linspace(np.amin(psi), np.amax(psi), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(
    th_params.R, th_params.Z, psi, levels=levels, cmap="viridis", zorder=Zorder.PSI.value
)
plot.contour(
    eq.grid.x,
    eq.grid.z,
    psi_original,
    levels=levels,
    cmap="plasma",
    zorder=Zorder.PSI.value,
)
plt.show()

# %%
# psi = eq.psi()
# levels = np.linspace(np.amin(psi), np.amax(psi), 50)
# plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
# plot.set_title("approx_total_psi")
# plot.contour(
#     eq.grid.x, eq.grid.z, psi, levels=levels, cmap="viridis", zorder=Zorder.PSI.value
# )
# plt.show()

# %%
from bluemira.equilibria.find import find_flux_surf
from bluemira.geometry.coordinates import Coordinates

o_points, x_points = eq.get_OX_points()

f_s = find_flux_surf(
    eq.x,
    eq.z,
    eq.psi(),
    psi_norm,
    o_points=o_points,
    x_points=x_points,
)
og_lcfs = Coordinates({"x": f_s[0], "z": f_s[1]})
f_s = find_flux_surf(
    th_params.R,
    th_params.Z,
    psi,
    psi_norm,
    o_points=o_points,
    x_points=x_points,
)
approx_lcfs = Coordinates({"x": f_s[0], "z": f_s[1]})
plt.plot(og_lcfs.x, og_lcfs.z)
plt.plot(approx_lcfs.x, approx_lcfs.z)
plt.show()

# %% [markdown]
# ## Use in Optimisation Problem
#
# Now we will use the approximation to set up constraints for an optimisation problem.
# We use minimal current coilset optimisation problem with TH as the only constraints.
# This will try to minimise the sum of the currents squared while constraining the coil
# contribution to the core psi.

# %%
# Use results of the toroidal harmonic approximation to create a set of coil constraints
th_constraint = ToroidalHarmonicConstraint(
    ref_harmonics_cos=Am_cos,
    ref_harmonics_sin=Am_sin,
    th_params=th_params,
    tolerance=None,
    constraint_type="inequality",
)
th_constraint_inverted = ToroidalHarmonicConstraint(
    ref_harmonics_cos=Am_cos,
    ref_harmonics_sin=Am_sin,
    th_params=th_params,
    tolerance=None,
    invert=True,
    constraint_type="inequality",
)

th_constraint_equal = ToroidalHarmonicConstraint(
    ref_harmonics_cos=Am_cos,
    ref_harmonics_sin=Am_sin,
    th_params=th_params,
    tolerance=None,
    invert=False,
    constraint_type="equality",
)

# Make sure we only optimise with coils outside the sphere containing the core plasma by
# setting control coils using the list of appropriate coils
eq.coilset.control = list(th_params.th_coil_names)
print(eq.coilset.control)
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

constraints = [th_constraint, th_constraint_inverted]  # , x_point]


# %%
# # Make a copy of the equilibria
# th_eq = deepcopy(eq)
# # Set up a coilset optimisation problem using the toroidal harmonic constraint
# th_con_len_opt = MinimalCurrentCOP(
#     eq=th_eq,
#     coilset=th_eq.coilset,
#     max_currents=6.0e8,
#     constraints=constraints,
# )
# # Find the optimised coilset
# _ = th_con_len_opt.optimise()

# # Update plasma - one solve
# th_eq.solve()


# %%
# # We should not need to solve the GS equation while optimising if the TH approximation
# # is sufficiently good, but we can have a look at what happens.
# th_eq_solved = deepcopy(eq)
# th_con_len_opt = MinimalCurrentCOP(
#     eq=th_eq_solved,
#     coilset=th_eq_solved.coilset,
#     max_currents=6.0e8,
#     constraints=constraints,
# )

# # SOLVE
# program = PicardIterator(
#     th_eq_solved,
#     th_con_len_opt,
#     fixed_coils=True,
#     convergence=DudsonConvergence(5.0e-3),
#     relaxation=0.1,
#     maxiter=100,
#     plot=False,
# )
# _ = program()

# %%
# # Plot the two approches
# f, (ax_1, ax_2) = plt.subplots(1, 2)

# th_eq.plot(ax=ax_1)
# ax_1.set_title("Coils Optimised")

# th_eq_solved.plot(ax=ax_2)
# ax_2.set_title("Coils Optimised while GS solved")
# plt.show()


# %%
# TODO trying isoflux points for leg shaping for use in TikhonovCurrentCOP

arg_inner = np.argmin(x_bdry)

# %%
# # x_leg = np.array([5.0, 5.5, 6.0, 7.0, 7.5, 8.3, 9.1, 9.9, 10.7, 11.5])

# # z_leg = np.array([-8.0, -7.5, -7.1, -6.4, -6.05, -6.1, -6.5, -6.9, -7.3, -7.7])

# x_leg = np.array([5.8, 6.3, 6.8, 7.3, 7.8, 8.1, 8.3, 8.5, 8.7, 8.9])
# z_leg = np.array([-7.2, -6.8, -6.5, -6.2, -5.8, -5.9, -6.3, -6.6, -7.0, -7.4])

# # x_leg = np.array([
# #     7.0 - 1.2,
# #     7.2 - 1.2,
# #     7.4 - 1.2,
# #     7.6 - 1.2,
# #     7.8 - 1.2,
# #     8.1 - 1.0,
# #     8.3 - 1.0,
# #     8.5 - 1.0,
# #     8.7 - 1.0,
# #     8.9 - 1.0,
# # ])
# # z_leg = np.array([-5.55, -5.54, -5.53, -5.52, -5.51, -5.9, -6.5, -6.9, -7.4, -7.8])

# isoflux_SN_keep_legs_same = IsofluxConstraint(
#     x_leg,My power cut out earlier and my wifi has been a bit funny so far so
#     z_leg,
#     x_bdry[arg_inner],
#     z_bdry[arg_inner],
#     tolerance=1e-6,  # Difficult to choose...
# )


# f, ax = plt.subplots()
# eq.plot(ax=ax)
# isoflux_SN_keep_legs_same.plot(ax=ax)
# eq.coilset.plot(ax=ax)

# %%
# x_lfs = np.array([1.45, 2.0, 2.53, 2.90, 3.43, 4.28]) + 7.0
# z_lfs = np.array([5.2, 5.9, 6.2, 6.4, 6.6, 6.9]) + 1.1
# x_hfs = np.array([
#     5.0,
#     5.5,
#     6.0,
#     6.6,
#     7.5,
# ])
# z_hfs = np.array([8.0, 7.5, 7.1, 6.6, 6.2])

# x_legs = np.concatenate([x_lfs, x_hfs])
# z_legs = np.concatenate([-z_lfs, -z_hfs])

# legs_isoflux = IsofluxConstraint(
#     x_legs,
#     z_legs,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-6,  # -1 * np.mean(eq.psi(x_legs, z_legs)) * 1e-2,
# )


# f, ax = plt.subplots()
# eq.plot(ax=ax)
# legs_isoflux.plot(ax=ax)
# eq.coilset.plot(ax=ax)

# %%
# Double null leg constraints attempt


# x_lfs = np.array([1.86, 2.24, 2.53, 2.90, 3.43, 4.28, 5.80, 6.70]) + 7.5
# z_lfs = np.array([4.80, 5.38, 5.84, 6.24, 6.60, 6.76, 6.71, 6.71]) + 0.9
# x_hfs = np.array([1.42, 1.06, 0.81, 0.67, 0.62, 0.62, 0.64, 0.60]) + 7.5
# z_hfs = np.array([4.80, 5.09, 5.38, 5.72, 6.01, 6.65, 6.82, 7.34]) + 0.9

# x_lfs = np.array([9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5]) - 0.5
# z_lfs = np.array([5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.8]) + 0.5
# x_hfs = np.array([6, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75]) + 0.5
# z_hfs = np.array([6.6, 6.4, 6.2, 6.0, 5.8, 5.6, 5.5, 5.3]) + 0.5


# x_lfs = np.array([1.86, 2.24, 2.53, 2.90, 3.43, 4.28, 5.80, 6.70]) + 7.5
# z_lfs = np.array([4.80, 5.38, 5.84, 6.24, 6.60, 6.76, 6.71, 6.71]) + 1.1
x_lfs = 17.5 - np.array([
    6.5,
    6.8,
    7.0,
    7.5,
    7.75,
])
z_lfs = np.array([8.0, 7.5, 7.1, 6.4, 6.05])
x_hfs = (
    np.array([
        5.0,
        5.5,
        6.0,
        7.0,
        7.5,
    ])
    + 0.5
)
z_hfs = np.array([8.0, 7.5, 7.1, 6.4, 6.05])

x_legs_lfs1 = np.concatenate([x_lfs[:2], x_lfs[:2]])
x_legs_hfs1 = np.concatenate([x_hfs[:2], x_hfs[:2]])
x_legs_lfs2 = np.concatenate([x_lfs[2:], x_lfs[2:]])
x_legs_hfs2 = np.concatenate([x_hfs[2:], x_hfs[2:]])
z_legs_lfs1 = np.concatenate([z_lfs[:2], -z_lfs[:2]])
z_legs_hfs1 = np.concatenate([-z_hfs[:2], z_hfs[:2]])
z_legs_lfs2 = np.concatenate([z_lfs[2:], -z_lfs[2:]])
z_legs_hfs2 = np.concatenate([-z_hfs[2:], z_hfs[2:]])
# x_legs = np.concatenate([x_lfs, x_lfs])
# z_legs = np.concatenate([z_lfs, -z_lfs])

double_null_legs_isoflux_hfs_1 = IsofluxConstraint(
    x_legs_hfs1,
    z_legs_hfs1,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)
double_null_legs_isoflux_lfs_1 = IsofluxConstraint(
    x_legs_lfs1,
    z_legs_lfs1,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

double_null_legs_isoflux_hfs_2 = IsofluxConstraint(
    x_legs_hfs2,
    z_legs_hfs2,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)
double_null_legs_isoflux_lfs_2 = IsofluxConstraint(
    x_legs_lfs2,
    z_legs_lfs2,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

f, ax = plt.subplots()
eq.plot(ax=ax)
double_null_legs_isoflux_lfs_1.plot(ax=ax)
double_null_legs_isoflux_hfs_1.plot(ax=ax)
double_null_legs_isoflux_lfs_2.plot(ax=ax)
double_null_legs_isoflux_hfs_2.plot(ax=ax)
eq.coilset.plot(ax=ax)

# %%
# DOUBLE NULL JUST INNER LEG
# Unmoved legs
inner_leg_points_x_unmoved = np.array([
    5.5,
    6.0,
    6.5,
    7.5,
    8.0,
])

inner_leg_points_z_unmoved = np.array([8.0, 7.5, 7.1, 6.4, 6.05])

# Moving legs
inner_leg_points_x = np.array([
    5.5,
    6.0,
    6.5,
    7.5,
    8.0,
])

inner_leg_points_z = np.array([6.25, 6.20, 6.05, 5.65, 5.5])

outer_legs_x_unmoved = np.array([9.2, 9.5, 9.7, 9.89, 10.1])

outer_legs_z_unmoved = np.array([5.5, 6.0, 6.5, 7.0, 7.5])

DN_unmoved_outer_leg_upper = DN_inner_leg_upper = IsofluxConstraint(
    outer_legs_x_unmoved,
    outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

DN_unmoved_outer_leg_lower = DN_inner_leg_upper = IsofluxConstraint(
    outer_legs_x_unmoved,
    -outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

DN_inner_leg_upper = IsofluxConstraint(
    inner_leg_points_x,
    inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

DN_inner_leg_lower = IsofluxConstraint(
    inner_leg_points_x,
    -inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-2,
)

f, ax = plt.subplots()
eq.plot(ax=ax)
DN_inner_leg_upper.plot(ax=ax)
DN_inner_leg_lower.plot(ax=ax)
DN_unmoved_outer_leg_upper.plot(ax=ax)
DN_unmoved_outer_leg_lower.plot(ax=ax)
plt.show()
# %%

# TODO tidy up and play around with leg constraints, try reproducing first image from
# diagram in teams, can look at some eudemo divertor papers to see proposed shapes
# for single nulls

# %%
# # FORCE

# eq_forced = deepcopy(eq)
# eq_forced.coilset["PF_2"].current = 3e6
# eq_forced.coilset["PF_5"].current = 3e6
# eq_forced.coilset["PF_3"].current = -20e6
# eq_forced.coilset["PF_4"].current = -20e6

# plt.contourf(eq_forced.grid.x, eq_forced.grid.z, eq_forced.plasma.psi())
# plt.show()

# plt.contourf(
#     eq_forced.grid.x,
#     eq_forced.grid.z,
#     eq_forced.coilset.psi(eq_forced.grid.x, eq_forced.grid.z),
# )
plt.show()

# %%
# f, ax = plt.subplots()
# eq_forced.plot(ax=ax)
# eq_forced.coilset.plot(ax=ax)
# # double_null_legs_isoflux.plot(ax=ax)

# %%
# DOUBLE NULL OPTIMISATION

# constraints = [th_constraint, th_constraint_inverted]#, double_null_legs_isoflux]

constraints = [
    th_constraint_equal,
    DN_inner_leg_upper,
    DN_inner_leg_lower,
    # double_null_legs_isoflux_hfs_1,
    # double_null_legs_isoflux_lfs_1,
    # double_null_legs_isoflux_hfs_2,
    # double_null_legs_isoflux_lfs_2
]

# constraints = [th_constraint,
#                th_constraint_inverted,
#                #double_null_legs_isoflux_hfs_1,
#                 #double_null_legs_isoflux_lfs_1,
#                #double_null_legs_isoflux_hfs_2,
#                 #double_null_legs_isoflux_lfs_2
#               ]


th_current_opt_eq = deepcopy(eq)

current_opt_problem = TikhonovCurrentCOP(
    th_current_opt_eq.coilset,
    th_current_opt_eq,
    targets=MagneticConstraintSet([
        # double_null_legs_isoflux_hfs_1,
        # double_null_legs_isoflux_lfs_1,
        # double_null_legs_isoflux_hfs_2,
        # double_null_legs_isoflux_lfs_2,
        DN_inner_leg_upper,
        DN_inner_leg_lower,
        DN_unmoved_outer_leg_upper,
        DN_unmoved_outer_leg_lower,
        # x_point,
    ]),
    gamma=1e-8,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6, "eps": 1e-4},
    # opt_parameters={"initial_step": 0.03},
    max_currents=3e10,
    constraints=constraints,
)

# current_opt_problem = MinimalCurrentCOP(
#      eq=th_current_opt_eq,
#      coilset=th_current_opt_eq.coilset,
#      max_currents=6.0e100,
#      constraints=constraints,
# )

program = PicardIterator(
    th_current_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.1,
    plot=True,
)

# _ = current_opt_problem.optimise()
# th_current_opt_eq.solve()


# %%
# SINGLE NULL OPTIMISATION

# constraints = [th_constraint, th_constraint_inverted]#, double_null_legs_isoflux]

# constraints = [th_constraint_equal,
#                double_null_legs_isoflux_hfs_1,
#                double_null_legs_isoflux_lfs_1,
# #                double_null_legs_isoflux_hfs_2,
# #                double_null_legs_isoflux_lfs_2]

# constraints = [th_constraint,
#                th_constraint_inverted,]
#                #double_null_legs_isoflux_hfs_1,
#                #double_null_legs_isoflux_lfs_1,
#                #double_null_legs_isoflux_hfs_2,
#                #double_null_legs_isoflux_lfs_2]


# th_current_opt_eq = deepcopy(eq)

# current_opt_problem = TikhonovCurrentCOP(
#     th_current_opt_eq.coilset,
#     th_current_opt_eq,
#     targets=MagneticConstraintSet([legs_isoflux]),
#     gamma=0.0,
#     opt_algorithm="COBYLA",
#     opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
#     opt_parameters={},
#     max_currents=3e10,
#     constraints=constraints,
# )

# current_opt_problem = MinimalCurrentCOP(
#      eq=th_current_opt_eq,
#      coilset=th_current_opt_eq.coilset,
#      max_currents=6.0e100,
#      constraints=constraints,
# )

# program = PicardIterator(
#     th_current_opt_eq,
#     current_opt_problem,
#     fixed_coils=True,
#     convergence=DudsonConvergence(1e-3),
#     relaxation=0.1,
#     plot=False,
# )

# _ = current_opt_problem.optimise()
# th_current_opt_eq.solve()

# %%
# _ = current_opt_problem.optimise()
# th_current_opt_eq.plot()

# %%
# th_current_opt_eq.plot()

# %%
# program.iterate_once()

# %%
#     RUUUUUUUUUUUUUNNNNNNNNN


program()


# %%
f, ax = plt.subplots()
th_current_opt_eq.plot(ax=ax)
# double_null_legs_isoflux_lfs_1.plot(ax=ax)
# double_null_legs_isoflux_hfs_1.plot(ax=ax)
# double_null_legs_isoflux_lfs_2.plot(ax=ax)
# double_null_legs_isoflux_hfs_2.plot(ax=ax)

th_current_opt_eq.coilset.plot(ax=ax)
DN_inner_leg_upper.plot(ax=ax)
DN_inner_leg_lower.plot(ax=ax)
# %%
# Plot the two approches
f, (ax_1, ax_2) = plt.subplots(1, 2)

eq.plot(ax=ax_1)
DN_inner_leg_upper.plot(ax=ax_1)
DN_inner_leg_lower.plot(ax=ax_1)
DN_unmoved_outer_leg_upper.plot(ax=ax_1)
DN_unmoved_outer_leg_lower.plot(ax=ax_1)
# double_null_legs_isoflux_lfs_1.plot(ax=ax_1)
# double_null_legs_isoflux_hfs_1.plot(ax=ax_1)
# double_null_legs_isoflux_lfs_2.plot(ax=ax_1)
# double_null_legs_isoflux_hfs_2.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

th_current_opt_eq.plot(ax=ax_2)
DN_inner_leg_upper.plot(ax=ax_2)
DN_inner_leg_lower.plot(ax=ax_2)
DN_unmoved_outer_leg_upper.plot(ax=ax_2)
DN_unmoved_outer_leg_lower.plot(ax=ax_2)
# double_null_legs_isoflux_lfs_1.plot(ax=ax_2)
# double_null_legs_isoflux_hfs_1.plot(ax=ax_2)
# double_null_legs_isoflux_lfs_2.plot(ax=ax_2)
# double_null_legs_isoflux_hfs_2.plot(ax=ax_2)
ax_2.set_title("TH")
plt.show()

# %%
print(f"currents after optimisation: \n{th_current_opt_eq.coilset.current}\n")
print(f"original equilibrium currents: \n{eq.coilset.current}\n ")
print(
    f"difference between optimised and original currents: \n{th_current_opt_eq.coilset.current - eq.coilset.current}\n"
)

# %%
plt.contourf(
    th_current_opt_eq.grid.x, th_current_opt_eq.grid.z, th_current_opt_eq.plasma.psi()
)
plt.show()
plt.contourf(
    th_current_opt_eq.grid.x,
    th_current_opt_eq.grid.z,
    th_current_opt_eq.coilset.psi(th_current_opt_eq.grid.x, th_current_opt_eq.grid.z),
)
plt.show()

# %%
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    HarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    coil_toroidal_harmonic_amplitude_matrix,
)

a_mat_cos, a_mat_sin = coil_toroidal_harmonic_amplitude_matrix(
    eq.coilset,
    th_params,
    max_degree=degree,
)
value = 0.0
scale = 1.0
test_constraint = HarmonicConstraintFunction(
    a_mat_cos, a_mat_sin, Am_cos, Am_sin, value, scale
)

# %%
res = test_constraint.f_constraint(eq.coilset.get_control_coils().current)

# %%
res
