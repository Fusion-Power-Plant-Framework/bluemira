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
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem import (
    MinimalCurrentCOP,
)
from bluemira.equilibria.optimisation.problem._tikhonov import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,  # noqa: PLC2701
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

# %%
eq.force_symmetry = True
eq.force_symmetry

# %%
eq.get_OX_points()
R_0 = eq._o_points[0].x
Z_0 = eq._o_points[0].z
th_params_upper = toroidal_harmonic_grid_and_coil_setup(
    eq=eq, R_0=9.2, Z_0=2.8, radius=0.01
)

th_params_lower = toroidal_harmonic_grid_and_coil_setup(
    eq=eq, R_0=9.2, Z_0=-2.8, radius=0.01
)


# %%
# Information needed for TH Approximation

psi_approx_upper, Am_cos_upper, Am_sin_upper = toroidal_harmonic_approximate_psi(  # noqa: N806
    eq=eq, th_params=th_params_upper, max_degree=5
)
psi_approx_lower, Am_cos_lower, Am_sin_lower = toroidal_harmonic_approximate_psi(  # noqa: N806
    eq=eq, th_params=th_params_lower, max_degree=5
)

# %%
# Plot the approx total psi and bluemira total psi
psi_upper = psi_approx_upper
psi_lower = psi_approx_lower
psi_original = eq.coilset.psi(eq.grid.x, eq.grid.z)
levels = np.linspace(np.amin(psi_upper), np.amax(psi_upper), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(
    th_params_upper.R,
    th_params_upper.Z,
    psi_upper,
    levels=levels,
    cmap="viridis",
    zorder=Zorder.PSI.value,
)
plot.contour(
    th_params_lower.R,
    th_params_lower.Z,
    psi_lower,
    levels=levels,
    cmap="viridis",
    zorder=Zorder.PSI.value,
)
levels = np.linspace(np.amin(psi_original), np.amax(psi_original), 70)
plot.contour(
    eq.grid.x,
    eq.grid.z,
    psi_original,
    levels=levels,
    cmap="plasma",
    zorder=Zorder.PSI.value,
)
lcfs = eq.get_LCFS()
plot.plot(lcfs.x, lcfs.z)
eq.coilset.plot(ax=plot)
plt.show()


# %% [markdown]
# ## Use in Optimisation Problem


# %%
# Use results of the toroidal harmonic approximation to create a set of coil constraints
# th_constraint = ToroidalHarmonicConstraint(
#     ref_harmonics_cos=Am_cos,
#     ref_harmonics_sin=Am_sin,
#     th_params=th_params,
#     tolerance=None,
#     constraint_type="inequality",
# )
# th_constraint_inverted = ToroidalHarmonicConstraint(
#     ref_harmonics_cos=Am_cos,
#     ref_harmonics_sin=Am_sin,
#     th_params=th_params,
#     tolerance=None,
#     invert=True,
#     constraint_type="inequality",
# )

th_constraint_equal = ToroidalHarmonicConstraint(
    ref_harmonics_cos=Am_cos_upper,
    ref_harmonics_sin=Am_sin_upper,
    th_params=th_params_upper,
    tolerance=None,
    invert=False,
    constraint_type="equality",
)

# Make sure we only optimise with coils outside the sphere containing the core plasma by
# setting control coils using the list of appropriate coils
eq.coilset.control = list(th_params_upper.th_coil_names)
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
# Constraint and target setup

sof_xbdry = np.array(eq.get_LCFS().x)[::15]
sof_zbdry = np.array(eq.get_LCFS().z)[::15]

isoflux_lcfs = IsofluxConstraint(
    sof_xbdry,
    sof_zbdry,
    sof_xbdry[0],
    sof_zbdry[0],
    tolerance=1e-3,
    constraint_value=0.25,  # Difficult to choose...
)

constraints = [
    th_constraint_equal,
    # DN_inner_leg_upper,
    # DN_inner_leg_lower,
    x_point,
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

magnetic_targets = MagneticConstraintSet([
    double_null_legs_isoflux_lfs_1,
    double_null_legs_isoflux_lfs_2,
    # DN_inner_leg_upper,
    # DN_inner_leg_lower,
    # DN_unmoved_outer_leg_upper,
    # DN_unmoved_outer_leg_lower,
    # x_point,
    isoflux_lcfs,
])

# %%
# Unconstrained double null optimisation

unconstrained_eq = deepcopy(eq)
unconstrained_cop = UnconstrainedTikhonovCurrentGradientCOP(
    unconstrained_eq.coilset, unconstrained_eq, magnetic_targets, gamma=1e-8
)
unconstrained_iterator = PicardIterator(
    unconstrained_eq,
    unconstrained_cop,
    fixed_coils=True,
    plot=False,
    relaxation=0.3,
    convergence=DudsonConvergence(1e-6),
)

unconstrained_iterator()


# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

eq.plot(ax=ax_1)
eq.coilset.plot(ax=ax_1)
# DN_inner_leg_upper.plot(ax=ax_1)
# DN_inner_leg_lower.plot(ax=ax_1)
# DN_unmoved_outer_leg_upper.plot(ax=ax_1)
# DN_unmoved_outer_leg_lower.plot(ax=ax_1)


isoflux_lcfs.plot(ax=ax_1)
double_null_legs_isoflux_lfs_1.plot(ax=ax_1)
double_null_legs_isoflux_hfs_1.plot(ax=ax_1)
double_null_legs_isoflux_lfs_2.plot(ax=ax_1)
double_null_legs_isoflux_hfs_2.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

unconstrained_eq.plot(ax=ax_2)
eq.coilset.plot(ax=ax_2)
# DN_inner_leg_upper.plot(ax=ax_2)
# DN_inner_leg_lower.plot(ax=ax_2)
# DN_unmoved_outer_leg_upper.plot(ax=ax_2)
# DN_unmoved_outer_leg_lower.plot(ax=ax_2)
isoflux_lcfs.plot(ax=ax_2)
# # double_null_legs_isoflux_lfs_1.plot(ax=ax_2)
# # double_null_legs_isoflux_hfs_1.plot(ax=ax_2)
# # double_null_legs_isoflux_lfs_2.plot(ax=ax_2)
# # double_null_legs_isoflux_hfs_2.plot(ax=ax_2)
# ax_2.set_title("TH")
plt.show()


# %%
# DOUBLE NULL OPTIMISATION


th_current_opt_eq = deepcopy(unconstrained_eq)

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
    gamma=1e-12,
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    opt_parameters={},
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


#     RUUUUUUUUUUUUUNNNNNNNNN


program()


# %%
# f, ax = plt.subplots()
# th_current_opt_eq.plot(ax=ax)
# # double_null_legs_isoflux_lfs_1.plot(ax=ax)
# # double_null_legs_isoflux_hfs_1.plot(ax=ax)
# # double_null_legs_isoflux_lfs_2.plot(ax=ax)
# # double_null_legs_isoflux_hfs_2.plot(ax=ax)

# th_current_opt_eq.coilset.plot(ax=ax)
# DN_inner_leg_upper.plot(ax=ax)
# DN_inner_leg_lower.plot(ax=ax)
# %%
# Plot the two approches
f, (ax_1, ax_2, ax_3) = plt.subplots(1, 3)

eq.plot(ax=ax_1)
eq.coilset.plot(ax=ax_1)
# DN_inner_leg_upper.plot(ax=ax_1)
# DN_inner_leg_lower.plot(ax=ax_1)
# DN_unmoved_outer_leg_upper.plot(ax=ax_1)
# DN_unmoved_outer_leg_lower.plot(ax=ax_1)
double_null_legs_isoflux_lfs_1.plot(ax=ax_1)
double_null_legs_isoflux_hfs_1.plot(ax=ax_1)
double_null_legs_isoflux_lfs_2.plot(ax=ax_1)
double_null_legs_isoflux_hfs_2.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

unconstrained_eq.plot(ax=ax_2)
eq.coilset.plot(ax=ax_2)
# DN_inner_leg_upper.plot(ax=ax_2)
# DN_inner_leg_lower.plot(ax=ax_2)
# DN_unmoved_outer_leg_upper.plot(ax=ax_2)
# DN_unmoved_outer_leg_lower.plot(ax=ax_2)
isoflux_lcfs.plot(ax=ax_2)
double_null_legs_isoflux_lfs_1.plot(ax=ax_2)
double_null_legs_isoflux_hfs_1.plot(ax=ax_2)
double_null_legs_isoflux_lfs_2.plot(ax=ax_2)
double_null_legs_isoflux_hfs_2.plot(ax=ax_2)
ax_2.set_title("TH Unconstrained")


th_current_opt_eq.plot(ax=ax_3)
eq.coilset.plot(ax=ax_3)
# DN_inner_leg_upper.plot(ax=ax_3)
# DN_inner_leg_lower.plot(ax=ax_3)
# DN_unmoved_outer_leg_upper.plot(ax=ax_3)
# DN_unmoved_outer_leg_lower.plot(ax=ax_3)
double_null_legs_isoflux_lfs_1.plot(ax=ax_3)
double_null_legs_isoflux_hfs_1.plot(ax=ax_3)
double_null_legs_isoflux_lfs_2.plot(ax=ax_3)
double_null_legs_isoflux_hfs_2.plot(ax=ax_3)
ax_3.set_title("TH Constrained")
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
