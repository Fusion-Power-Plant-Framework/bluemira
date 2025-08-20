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
Testing TH COP
"""

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
from bluemira.equilibria.find_legs import LegFlux
from bluemira.equilibria.optimisation.constraints import (
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
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
from bluemira.optimisation import Algorithm

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

# Single null
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
psi_norm = 0.95

(
    error,
    combo,
    cos_degrees,
    sin_degrees,
    total_psi,
    vacuum_psi,
    cos_amplitudes,
    sin_amplitudes,
    th_params,
) = toroidal_harmonic_approximation(
    eq=eq,
    psi_norm=psi_norm,
    plot=True,
    max_error_value=0.5,  # RMS for psi in chosen flux surface
    tol=1e-2,
)

# %%
print(f"COS: {cos_degrees}")
print(f"SIN: {sin_degrees}")

# %%
print(f"COS: {cos_amplitudes}")
print(f"SIN: {sin_amplitudes}")

# %%
ref_lcfs = eq.get_LCFS()
legs = LegFlux(eq).get_legs()

# Make a copy of the equilibria
th_eq = deepcopy(eq)

# Use results of the toroidal harmonic approximation to create a set of coil constraints
th_constraint = ToroidalHarmonicConstraint(
    ref_harmonics_cos=cos_degrees,
    ref_harmonics_sin=sin_degrees,
    ref_harmonics_cos_amplitudes=cos_amplitudes,
    ref_harmonics_sin_amplitudes=sin_amplitudes,
    th_params=th_params,
    # tolerance=5e-5
    constraint_type="equality",
)

# X-point constraint
xp_idx = np.argmin(ref_lcfs.z)
x_point = FieldNullConstraint(
    ref_lcfs.x[xp_idx],
    ref_lcfs.z[xp_idx],
    # tolerance=5e-5,  # [T]
)

# Force constraints
force_constraints = CoilForceConstraints(
    eq.coilset,
    PF_Fz_max=450e6,
    CS_Fz_sum_max=300e6,
    CS_Fz_sep_max=250e6,
    # tolerance=5e-5,
)

leg_choice = "inner"

if leg_choice == "outer":
    leg_x = legs["lower_outer"][0].x
    leg_z = legs["lower_outer"][0].z
elif leg_choice == "inner":
    leg_x = legs["lower_inner"][0].x
    leg_z = legs["lower_inner"][0].z
elif leg_choice == "both":
    leg_x = np.append(legs["lower_inner"][0].x, legs["lower_outer"][0].x, axis=0)
    leg_z = np.append(legs["lower_inner"][0].z, legs["lower_outer"][0].z, axis=0)

leg_x = leg_x[0::5]
leg_z = leg_z[0::5]

arg_inner = np.argmin(ref_lcfs.x)
isoflux = IsofluxConstraint(
    leg_x,
    leg_z,
    ref_lcfs.x[arg_inner],
    ref_lcfs.z[arg_inner],
    # tolerance=1e-3,
    constraint_value=0.0,
)

# Set up a coilset optimisation problem using the spherical harmonic constraint
th_con_len_opt = MinimalCurrentCOP(
    eq=th_eq,
    max_currents=1e8,
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    constraints=[x_point, th_constraint, isoflux, force_constraints],
    opt_algorithm=Algorithm.SLSQP,
)

f, ax = plt.subplots()

eq.plot(ax=ax)
isoflux.plot(ax=ax)
x_point.plot(ax=ax)

# %%
result = th_con_len_opt.optimise()

# %%
diag_ops = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_REL_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
    # plot_mask=EqPlotMask.NONE,
    # interpolation_grid=InterpGrid.OVERLAP,
)
eq_analysis = EqAnalysis(input_eq=th_eq, reference_eq=eq)

_ = eq_analysis.plot_compare_psi(diag_ops=diag_ops)

# %%
eq_dict = select_multi_eqs([eq, th_eq])
multi = MultiEqAnalysis(eq_dict)

# %%
_ = multi.plot_core_physics()

# %%
_ = multi.plot_compare_profiles()

# %%
_ = multi.coilset_info_table()

# %%
start_i = eq.coilset.current / 1e6
opt_i = th_eq.coilset.current / 1e6
diff = np.abs(opt_i) - np.abs(start_i)
diff_percent = 100 * diff / start_i
diff_summed = np.sum(diff)

print(f"pre-opt currents: {start_i} MA")
print(f"post-opt currents: {opt_i} MA")
print(f"diff: {diff} MA")
print(f"sum of diff: {diff_summed} MA")
print(f"diff percent: {diff_percent} %")

# %%
th_eq.plot()
plt.show()

# %%
# One solve
th_eq.solve()
th_eq.plot()
plt.show()

# %%
eq_analysis = EqAnalysis(input_eq=th_eq, reference_eq=eq)
_ = eq_analysis.plot_compare_psi(diag_ops=diag_ops)

# %%
eq_dict = select_multi_eqs([eq, th_eq])
multi = MultiEqAnalysis(eq_dict)

# %%
_ = multi.plot_core_physics()

# %%
_ = multi.plot_compare_profiles()

# %%
# th_eq=deepcopy(eq)
# #Set up iterator
# program = PicardIterator(
#     th_eq,
#     th_con_len_opt,
#     fixed_coils=True,
#     convergence=DudsonConvergence(1e-3),
#     relaxation=0.1,
#     maxiter=50,
# )
# #solve
# program()

# %%
# diag_ops = EqDiagnosticOptions(
#     psi_diff=PsiPlotType.PSI_REL_DIFF,
#     split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
#     #plot_mask=EqPlotMask.NONE,
#     #interpolation_grid=InterpGrid.OVERLAP,
# )
# eq_analysis  = EqAnalysis(input_eq=th_eq, reference_eq=eq)
# _ = eq_analysis.plot_compare_psi(diag_ops=diag_ops)
