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
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,  # noqa: PLC2701
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
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

# eq.force_symmetry = True

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

o_points, x_points = eq.get_OX_points()

# %%
# # original PF 2 and 5
# print("original")
# print(eq.coilset["PF_2"].z)
# print(eq.coilset["PF_5"].z)

# # moved PF 2 and 5
# # eq.coilset["PF_1"].z =
# # eq.coilset["PF_6"].z =
# eq.coilset["PF_1"].x = 10.0
# eq.coilset["PF_6"].x = 10.0

# print("updated")
# print(eq.coilset["PF_1"].z)
# print(eq.coilset["PF_6"].z)

# print(eq.coilset)

# f, ax = plt.subplots()
# eq.plot(ax)
# eq.coilset.plot(ax)
# plt.show()

# # %%

# # Add an x point constraint
# lcfs = eq.get_LCFS()
# x_bdry, z_bdry = lcfs.x, lcfs.z

# arg_inner = np.argmin(x_bdry)

# sof_xbdry = np.array(eq.get_LCFS().x)[::5]
# sof_zbdry = np.array(eq.get_LCFS().z)[::5]

# isoflux_lcfs = IsofluxConstraint(
#     sof_xbdry,
#     sof_zbdry,
#     sof_xbdry[0],
#     sof_zbdry[0],
#     tolerance=1e-3,
#     constraint_value=0.25,  # Difficult to choose...
# )


# x_lfs = 17.5 - np.array([
#     6.0,
#     6.8,
#     7.2,
#     7.75,
#     8.0,
# ])


# z_lfs = np.array([8.0, 7.5, 7.1, 6.4, 6.05])


# x_hfs = (
#     np.array([
#         5.0,
#         5.5,
#         6.0,
#         7.0,
#         7.5,
#     ])
#     + 0.5
# )
# z_hfs = np.array([8.0, 7.5, 7.1, 6.4, 6.05])

# x_legs_lfs1 = np.concatenate([x_lfs[:2], x_lfs[:2]])
# x_legs_hfs1 = np.concatenate([x_hfs[:2], x_hfs[:2]])
# x_legs_lfs2 = np.concatenate([x_lfs[2:], x_lfs[2:]])
# x_legs_hfs2 = np.concatenate([x_hfs[2:], x_hfs[2:]])
# z_legs_lfs1 = np.concatenate([z_lfs[:2], -z_lfs[:2]])
# z_legs_hfs1 = np.concatenate([-z_hfs[:2], z_hfs[:2]])
# z_legs_lfs2 = np.concatenate([z_lfs[2:], -z_lfs[2:]])
# z_legs_hfs2 = np.concatenate([-z_hfs[2:], z_hfs[2:]])


# double_null_legs_isoflux_hfs_1 = IsofluxConstraint(
#     x_legs_hfs1,
#     z_legs_hfs1,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-2,
# )
# double_null_legs_isoflux_lfs_1 = IsofluxConstraint(
#     x_legs_lfs1,
#     z_legs_lfs1,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-2,
# )

# double_null_legs_isoflux_hfs_2 = IsofluxConstraint(
#     x_legs_hfs2,
#     z_legs_hfs2,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-2,
# )
# double_null_legs_isoflux_lfs_2 = IsofluxConstraint(
#     x_legs_lfs2,
#     z_legs_lfs2,
#     ref_x=x_bdry[arg_inner],
#     ref_z=z_bdry[arg_inner],
#     tolerance=1e-2,
# )
# f, ax = plt.subplots()
# eq.plot(ax=ax)
# isoflux_lcfs.plot(ax=ax)
# double_null_legs_isoflux_lfs_1.plot(ax=ax)
# double_null_legs_isoflux_hfs_1.plot(ax=ax)
# double_null_legs_isoflux_lfs_2.plot(ax=ax)
# double_null_legs_isoflux_hfs_2.plot(ax=ax)
# eq.coilset.plot(ax=ax)

# # %%
# magnetic_targets = MagneticConstraintSet([
#     double_null_legs_isoflux_hfs_1,
#     double_null_legs_isoflux_lfs_1,
#     double_null_legs_isoflux_hfs_2,
#     double_null_legs_isoflux_lfs_2,
#     isoflux_lcfs,
# ])
# # Unconstrained tikhonov
# unconstrained_eq = deepcopy(eq)
# unconstrained_cop = UnconstrainedTikhonovCurrentGradientCOP(
#     unconstrained_eq.coilset, unconstrained_eq, magnetic_targets, gamma=1e-8
# )
# unconstrained_iterator = PicardIterator(
#     unconstrained_eq,
#     unconstrained_cop,
#     fixed_coils=True,
#     plot=False,
#     relaxation=0.3,
#     convergence=DudsonConvergence(1e-6),
#     maxiter=100,
# )

# unconstrained_iterator()


# f, ax = plt.subplots()
# unconstrained_eq.plot(ax)
# unconstrained_eq.coilset.plot(ax)
# isoflux_lcfs.plot(ax=ax)
# double_null_legs_isoflux_lfs_1.plot(ax=ax)
# double_null_legs_isoflux_hfs_1.plot(ax=ax)
# double_null_legs_isoflux_lfs_2.plot(ax=ax)
# double_null_legs_isoflux_hfs_2.plot(ax=ax)
# plt.show()


# %%
# Information needed for TH Approximation
psi_norm = 1.0
th_params, Am_cos, Am_sin, degree, fit_metric, approx_total_psi = (
    toroidal_harmonic_approximation(
        eq=eq, psi_norm=psi_norm, acceptable_fit_metric=0.001
    )
)

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


# %%
# TODO trying isoflux points for leg shaping for use in TikhonovCurrentCOP

arg_inner = np.argmin(x_bdry)


# %%
# Double null leg constraints attempt
x_lfs = 17.5 - np.array([
    6.5,
    6.8,
    7.0,
    # 7.5,
    # 7.75,
])
z_lfs = np.array([8.0, 7.5, 7.1])  # , 6.4, 6.05])
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

# Moving legs
inner_leg_points_x_5_points_separatrix = np.array([
    5.5,
    6.0,
    6.5,
    7.5,
    8.0,
])

inner_leg_points_z_5_points_separatrix = np.array([6.25, 6.20, 6.05, 5.65, 5.5])

# # Original good legs
inner_leg_points_x = (
    np.array([
        # 5.5,
        # 6.0,
        7.0,
        7.5,
        8.0,
    ])
    - 0.5  # original good legs
)

inner_leg_points_z = (
    np.array([
        # 6.25,
        # 6.20,
        6.05,
        5.75,
        5.5,
    ])
    + 0.2  # usual one to get graph in my teams chat
)

# inner_leg_points_x = (
#     np.array([
#         # 5.5,
#         # 6.0,
#         6.5,
#         7.0,
#         7.5,
#     ])
#     + 0.3
# )

# inner_leg_points_z = (
#     np.array([
#         # 6.25,
#         # 6.20,
#         5.7,
#         5.6,
#         5.5,
#     ])
#     + 0.1
# )


# fmt: off
outer_legs_x_unmoved = np.array([#9.2,
                                 #9.5,
                                 9.7,
                                 9.89,
                                 10.1])

outer_legs_z_unmoved = np.array([#5.5,
                                 #6.0,
                                 6.5,
                                 7.0,
                                 7.5])

# fmt: on
DN_unmoved_outer_leg_upper = IsofluxConstraint(
    outer_legs_x_unmoved,
    outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-6,
)

DN_unmoved_outer_leg_lower = IsofluxConstraint(
    outer_legs_x_unmoved,
    -outer_legs_z_unmoved,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-6,
)

DN_inner_leg_upper = IsofluxConstraint(
    inner_leg_points_x,
    inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-6,
)

DN_inner_leg_lower = IsofluxConstraint(
    inner_leg_points_x,
    -inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-6,
)

f, ax = plt.subplots()
eq.plot(ax=ax)
DN_inner_leg_upper.plot(ax=ax)
DN_inner_leg_lower.plot(ax=ax)
DN_unmoved_outer_leg_upper.plot(ax=ax)
DN_unmoved_outer_leg_lower.plot(ax=ax)
plt.show()


# %%
# DOUBLE NULL OPTIMISATION

# constraints = [th_constraint, th_constraint_inverted]#, double_null_legs_isoflux]


# Moving inner leg
constraints = [
    th_constraint_equal,
    x_point,
    DN_inner_leg_upper,
    DN_inner_leg_lower,
    # DN_unmoved_outer_leg_lower,
    # DN_unmoved_outer_leg_upper,
]

# # Moving outer leg
# constraints = [
#     th_constraint_equal,
#     x_point,
#     # double_null_legs_isoflux_hfs_1,
#     # double_null_legs_isoflux_hfs_2,
#     double_null_legs_isoflux_lfs_1,
#     double_null_legs_isoflux_lfs_2,
# ]


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
    gamma=1e-4,  # inner leg good value for optimisation
    # gamma=1e-6,
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
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
    maxiter=30,
)

# _ = current_opt_problem.optimise()
# th_current_opt_eq.solve()


#     RUUUUUUUUUUUUUNNNNNNNNN


program()


# %%

original_FS = eq.get_LCFS() if psi_norm == 1.0 else eq.get_flux_surface(psi_norm)
approx_FS = th_current_opt_eq.get_flux_surface(psi_norm)


# Moving Inner Leg

f, ax = plt.subplots()
th_current_opt_eq.plot(ax=ax)
th_current_opt_eq.coilset.plot(ax=ax)
DN_inner_leg_upper.plot(ax=ax)
DN_inner_leg_lower.plot(ax=ax)
DN_unmoved_outer_leg_upper.plot(ax=ax)
DN_unmoved_outer_leg_lower.plot(ax=ax)
plt.show()

# Plot the two approches
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


# Moving Outer Leg

# f, ax = plt.subplots()
# th_current_opt_eq.plot(ax=ax)
# double_null_legs_isoflux_lfs_1.plot(ax=ax)
# double_null_legs_isoflux_hfs_1.plot(ax=ax)
# double_null_legs_isoflux_lfs_2.plot(ax=ax)
# double_null_legs_isoflux_hfs_2.plot(ax=ax)

# th_current_opt_eq.coilset.plot(ax=ax)


# # Plot the two approches
# f, (ax_1, ax_2) = plt.subplots(1, 2)

# eq.plot(ax=ax_1)
# double_null_legs_isoflux_lfs_1.plot(ax=ax_1)
# double_null_legs_isoflux_hfs_1.plot(ax=ax_1)
# double_null_legs_isoflux_lfs_2.plot(ax=ax_1)
# double_null_legs_isoflux_hfs_2.plot(ax=ax_1)
# ax_1.set_title("Starting Equilibrium")

# th_current_opt_eq.plot(ax=ax_2)
# double_null_legs_isoflux_lfs_1.plot(ax=ax_2)
# double_null_legs_isoflux_hfs_1.plot(ax=ax_2)
# double_null_legs_isoflux_lfs_2.plot(ax=ax_2)
# double_null_legs_isoflux_hfs_2.plot(ax=ax_2)
# ax_2.set_title("TH")
# plt.show()


print(f"currents after optimisation: \n{th_current_opt_eq.coilset.current}\n")
print(f"original equilibrium currents: \n{eq.coilset.current}\n ")
print(
    f"difference between optimised and original currents: \n{th_current_opt_eq.coilset.current - eq.coilset.current}\n"
)

# %%
# # TODO psi diff plot
# TODO add in the 2nd abs in other difference plots in other notebooks

total_psi_diff = np.abs(eq.psi() - th_current_opt_eq.psi()) / np.max(
    np.abs(th_current_opt_eq.psi())
)

f, ax = plt.subplots()
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
ax.plot(approx_FS.x, approx_FS.z, color="red", label="Approximate FS from TH")
ax.plot(original_FS.x, original_FS.z, color="blue", label="FS from Bluemira")
im = ax.contourf(eq.grid.x, eq.grid.z, total_psi_diff, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title("Absolute relative difference between total psi and TH approximation psi")
ax.legend(loc="upper right")
eq.coilset.plot(ax=ax)
plt.show()

# %%
