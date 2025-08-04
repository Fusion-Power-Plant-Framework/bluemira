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

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    toroidal_harmonic_approximation,
)

# %%
# Data from EQDSK file
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

# Comment in/out as required

# Double null
eq_name = "DN-DEMO_eqref.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(
    eq_name, from_cocos=3, qpsi_positive=False, force_symmetry=True
)

# # Single null
# eq_name = "eqref_OOB.json"
# eq_name = Path(EQDATA, eq_name)
# eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)

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
# The acceptable fit metric value used here forces the approximation to use 10 degrees
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
    # tol=0.2, # Use this for SN
)
# Some notes:
# The default values work for DN
# For SN, use default max error and tol=0.2

# %%
# NOTE

# Removed the rest of this notebook. Can add own constraints here to play around with
# I will be updating all the example notebooks
