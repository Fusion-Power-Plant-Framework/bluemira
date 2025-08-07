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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    ToroidalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    coil_toroidal_harmonic_amplitude_matrix,
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)

# %%
# Data from EQDSK file
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


# Single null
eq_name = "eqref_OOB.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)

# Plot equilibrium
# f, ax = plt.subplots()
# eq.plot(ax=ax)
# eq.coilset.plot(ax=ax)
# plt.show()

# %%
# Set up grid points for use

# # Test the grid and coil setup using these params
# R_0, Z_0 = eq.effective_centre()
# th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

# # %%

# # Set up cos and sin degrees to use
# cos_degrees = np.array([0, 1, 2, 3])
# sin_degrees = np.array([2, 4])
# # test coil_toroidal_harmonic_amplitude_matrix
# Am_cos, Am_sin = coil_toroidal_harmonic_amplitude_matrix(
#     input_coils=eq.coilset,
#     th_params=th_params,
#     cos_degrees_chosen=cos_degrees,
#     sin_degrees_chosen=sin_degrees,
# )


# # %%
# # Test toroidal_harmonic_approximate_psi
# approx_coilset_psi, Am_cos2, Am_sin2 = toroidal_harmonic_approximate_psi(
#     eq=eq,
#     th_params=th_params,
#     cos_degrees_chosen=cos_degrees,
#     sin_degrees_chosen=sin_degrees,
# )

# %%
# Test toroidal_harmonic_approximation
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
    th_params2,
) = toroidal_harmonic_approximation(
    eq=eq,
    psi_norm=0.95,
    max_error_value=0.1,
    tol=0.005,
)  # , plot=True, max_error_value=0.1, tol=0.0055
# )

# %% making a constraint and a constraint function for the test
print("constraint function test")
ref_constraint_class = ToroidalHarmonicConstraint(
    ref_harmonics_cos=cos_degrees,
    ref_harmonics_sin=sin_degrees,
    ref_harmonics_cos_amplitudes=cos_amplitudes,
    ref_harmonics_sin_amplitudes=sin_amplitudes,
    th_params=th_params2,
    tolerance=None,
    constraint_type="equality",
)
ref_constraint_class.prepare(eq)
test_constraint_function = ToroidalHarmonicConstraintFunction(
    a_mat_cos=ref_constraint_class._args["a_mat_cos"],
    a_mat_sin=ref_constraint_class._args["a_mat_sin"],
    b_vec_cos=ref_constraint_class._args["b_vec_cos"],
    b_vec_sin=ref_constraint_class._args["b_vec_sin"],
    scale=ref_constraint_class._args["scale"],
    value=ref_constraint_class._args["value"],
)

cur_expand_mat = eq.coilset._opt_currents_expand_mat
ref_vector = cur_expand_mat @ np.ones(len(eq.coilset.name))
ref_result_cos = ref_constraint_class._args["a_mat_cos"] @ ref_vector
ref_result_sin = ref_constraint_class._args["a_mat_sin"] @ ref_vector
ref_amplitudes = np.append(
    ref_result_cos - ref_constraint_class._args["b_vec_cos"],
    ref_result_sin - ref_constraint_class._args["b_vec_sin"],
    axis=0,
)
test_returned_amplitudes = test_constraint_function.f_constraint(ref_vector)

print(ref_amplitudes)
print(test_returned_amplitudes)
for fc, res in zip(
    test_returned_amplitudes,
    ref_amplitudes,
    strict=False,
):
    print(fc)
    print(res)
    print(fc - res)
    assert fc == res

# %%
# # use coil_toroidal_harmonic_amplitude_matrix (same as in control_response)
# cos_arr, sin_arr = coil_toroidal_harmonic_amplitude_matrix(
#     input_coils=eq.coilset,
#     th_params=th_params2,
#     cos_degrees_chosen=cos_degrees,
#     sin_degrees_chosen=sin_degrees,
# )

# cos_evaluated, sin_evaluated = (
#     np.zeros(len(cos_arr)),
#     np.zeros(len(sin_arr)),
# )

# test_constraint_function = ToroidalHarmonicConstraintFunction(
#     a_mat_cos=cos_arr,
#     a_mat_sin=sin_arr,
#     b_vec_cos=cos_arr @ eq.coilset.current - cos_evaluated,
#     b_vec_sin=sin_arr @ eq.coilset.current - sin_evaluated,
#     scale=1e6,
#     value=0.0,
# )


# cur_expand_mat = eq.coilset._opt_currents_expand_mat
# ref_vector = cur_expand_mat @ np.ones(len(eq.coilset.name))
# ref_result_cos = cos_arr @ ref_vector
# ref_result_sin = sin_arr @ ref_vector
# ref_amplitudes = np.append(
#     ref_result_cos - (cos_arr @ eq.coilset.current - cos_evaluated),
#     ref_result_sin - (sin_arr @ eq.coilset.current - sin_evaluated),
#     axis=0,
# )
# test_returned_amplitudes = test_constraint_function.f_constraint(ref_vector)

# print(ref_amplitudes)
# print(test_returned_amplitudes)
# # for fc, res in zip(
# #     test_returned_amplitudes,
# #     ref_amplitudes,
# #     strict=False,
# # ):
# #     print(fc)
# #     print(res)
# #     print(fc - res)
# # assert fc == res
