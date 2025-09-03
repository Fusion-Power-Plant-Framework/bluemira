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
from bluemira.equilibria.analysis import EqAnalysis, MultiEqAnalysis, select_multi_eqs
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqSubplots,
    PsiPlotType,
)
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
    brute_force_toroidal_harmonic_approximation,
    optimisation_toroidal_harmonic_approximation,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem._tikhonov import TikhonovCurrentCOP

# %%
# Data from EQDSK file
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

# Comment in/out as required

# Double null
# eq_name = "DN-DEMO_eqref.json"
# eq_name = Path(EQDATA, eq_name)
# eq = Equilibrium.from_eqdsk(
#     eq_name, from_cocos=3, qpsi_positive=False, force_symmetry=True
# )

# # Single null
eq_name = "eqref_OOB.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)

# Plot equilibrium
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()


# %%
# Information needed for TH Approximation
# The acceptable fit metric value used here forces the approximation to use 10 degrees
psi_norm = 0.95


R_0, Z_0 = eq.effective_centre()
th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

# %%
# using optimisation fn :
result = optimisation_toroidal_harmonic_approximation(
    eq=eq, th_params=th_params, psi_norm=psi_norm, plot=False
)
print(f"cos degrees used = {result.cos_degrees}")
print(f"sin degrees used = {result.sin_degrees}")

plot_toroidal_harmonic_approximation(eq, th_params, result, psi_norm)

# %% using brute force
result = brute_force_toroidal_harmonic_approximation(
    eq=eq, th_params=th_params, psi_norm=psi_norm, tol=0.1, plot=False
)


# %%
# print(f"Combo used = {combo}")
print(f"cos degrees used = {result.cos_degrees}")
print(f"sin degrees used = {result.sin_degrees}")

raise ValueError


th_constraint_equal = ToroidalHarmonicConstraint(
    ref_harmonics_cos=result.cos_degrees,
    ref_harmonics_sin=result.sin_degrees,
    ref_harmonics_cos_amplitudes=result.cos_amplitudes,
    ref_harmonics_sin_amplitudes=result.sin_amplitudes,
    constraint_type="equality",
    th_params=th_params,
    tolerance=1e-3,
)

th_constraint_inequal = ToroidalHarmonicConstraint(
    ref_harmonics_cos=result.cos_degrees,
    ref_harmonics_sin=result.sin_degrees,
    ref_harmonics_cos_amplitudes=result.cos_amplitudes,
    ref_harmonics_sin_amplitudes=result.sin_amplitudes,
    constraint_type="inequality",
    th_params=th_params,
    tolerance=1e-3,
)

# %%
eq.coilset.control = list(th_params.th_coil_names)


# Add an x point constraint
lcfs = eq.get_LCFS()
x_bdry, z_bdry = lcfs.x, lcfs.z
xp_idx = np.argmin(z_bdry)
x_point = FieldNullConstraint(
    x_bdry[xp_idx],
    z_bdry[xp_idx],
    tolerance=1e-3,
)
arg_inner = np.argmin(x_bdry)


# Define points to use for theqe isoflux constraints
# NOTE not moving the legs at all here
inner_leg_points_x = (
    np.array([
        7.0,
        7.5,
        8.0,
    ])
    - 0.5
)

inner_leg_points_z = (
    np.array([
        6.15,
        5.8,
        5.5,
    ])
    + 0.5
)

outer_leg_points_x = np.array([
    8.5,
    8.8,
    9.05,
])

outer_leg_points_z = np.array([6.5, 7.0, 7.5])

# Create the necessary isoflux constraints

SN_unmoved_outer_leg_lower = IsofluxConstraint(
    outer_leg_points_x,
    -outer_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)


SN_unmoved_inner_leg_lower = IsofluxConstraint(
    inner_leg_points_x,
    -inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)

# Plot the isoflux points and the starting equilibrium
f, ax = plt.subplots()
eq.plot(ax=ax)
SN_unmoved_outer_leg_lower.plot(ax=ax)
SN_unmoved_inner_leg_lower.plot(ax=ax)
plt.show()

# %%
constraints = [
    th_constraint_equal,
    x_point,
    SN_unmoved_outer_leg_lower,
    SN_unmoved_inner_leg_lower,
]

algorithm = "COBYLA"
# algorithm = "SLSQP"


# %%
# Make copy of eq
th_eq = deepcopy(eq)

th_opt_unmoved_legs = TikhonovCurrentCOP(
    th_eq,
    targets=MagneticConstraintSet([
        SN_unmoved_outer_leg_lower,
        SN_unmoved_inner_leg_lower,
        x_point,
    ]),
    gamma=1e-4,
    opt_algorithm=algorithm,
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=constraints,
)
# Find the optimised coilseteq
_ = th_opt_unmoved_legs.optimise()

diag_ops = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
)
eq_analysis = EqAnalysis(input_eq=th_eq, reference_eq=eq)
_ = eq_analysis.plot_compare_psi(diag_ops=diag_ops)


# %%
# Update plasma - one solve
th_eq.solve()

eq_analysis = EqAnalysis(input_eq=th_eq, reference_eq=eq)
_ = eq_analysis.plot_compare_psi(diag_ops=diag_ops)


# %%

eq_dict = select_multi_eqs([eq, th_eq])
multi_analysis = MultiEqAnalysis(eq_dict)

# %%
# NOTE use analyse_plasma() to get the results to compare
eq_summary = eq.analyse_plasma()
print(
    eq_summary.tabulate(["Parameter", "value"], tablefmt="simple", value_label="start")
)
# %%
th_eq_summary = th_eq.analyse_plasma()
print(
    th_eq_summary.tabulate(
        ["Parameter", "value"], tablefmt="simple", value_label="post optimisation"
    )
)
# # %%
# # Plot physics parameters for the plasma core
# # Note that a list with the results is also output

core_results, ax = multi_analysis.plot_core_physics()
