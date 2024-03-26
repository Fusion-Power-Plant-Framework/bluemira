# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
Usage of the 'spherical_harmonic_approximation' function.
"""

# %% [markdown]
# # spherical_harmonic_approximation Function
#
# This example illustrates the input and output of the
# Bluemira spherical harmonics approximation function
# (spherical_harmonic_approximation) which can be used
# in coilset current and position optimisation for spherical tokamaks.
# For an example of how spherical_harmonic_approximation is used
# please see, <TODO add link>

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    spherical_harmonic_approximation,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    spherical_harmonic_approximation,
)

plot_defaults()

# %pdb

# %%
# Data from EQDSK file
file_path = Path(
    get_bluemira_path("equilibria", subfolder="examples"), "SH_test_file.json"
)

# Plot
eq = Equilibrium.from_eqdsk(file_path.as_posix())
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# ## Inputs
#
# ### Required
#
# - eq = Our chosen Bluemira Equilibrium
#
# ### Optional
#
# - n_points: Number of desired collocation points
# - point_type: How the collocation points are distributed
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - r_t: typical length scale for spherical harmonic approximation
# - extra_info: set this to true if you wish to return additional
#               information and plot the results.

# %%
# Information needed for SH Approximation
(
    sh_coil_names,
    coil_current_harmonic_amplitudes,
    degree,
    fit_metric_value,
    approx_total_psi,
    r_t,
    sh_coilset_current,
) = spherical_harmonic_approximation(
    eq,
    n_points=20,
    point_type=PointType.ARC_PLUS_EXTREMA,
    acceptable_fit_metric=0.05,
    seed=15,
    plot=True,
)

# %% [markdown]
# ## Outputs
#
# spherical_harmonic_approximation outputs a dictionary of results
# that can be used in optimisation.
#
# ### Always output
#
# - "coilset", coilset to use with SH approximation
# - "r_t", typical length scale for spherical harmonic approximation
# - "harmonic_amplitudes", SH coefficients/amplitudes for required number of degrees
# - "max_degree", number of degrees required for a SH approx with the desired fit metric

# %%
print(sh_coil_names)

# %%
print(r_t)

# %%
print(coil_current_harmonic_amplitudes)

# %%
print(degree)

# %% [markdown]
# ### Output on request
#
# - "fit_metric_value", fit metric achieved
# - "approx_total_psi", the total psi obtained using the SH approximation

# %%
print(fit_metric_value)

# %%
psi = approx_total_psi
levels = np.linspace(np.amin(psi), np.amax(psi), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(eq.grid.x, eq.grid.z, psi, levels=levels, cmap="viridis", zorder=8)
plt.show()
