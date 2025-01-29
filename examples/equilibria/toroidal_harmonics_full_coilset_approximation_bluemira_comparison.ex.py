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
Calculate solution due to coilset as sum of toroidal harmonics, and compare to solution
from Bluemira.
"""
# %% [markdown]
# # Calculating the flux solution due to a coilset as a sum of toroidal harmonics
#
# This example illustrates the usage of the bluemira toroidal harmonics approximation
# function (`toroidal_harmonic_approximate_psi`) which can be used to approximate the
# magnetic flux due to a coilset. For full details and all equations, look in the
# notebook toroidal_harmonics_component_function_walkthrough_and_verification.ex.py.
# %% [markdown]
# ## Imports

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surf
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    toroidal_harmonic_approximate_psi,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.geometry.coordinates import Coordinates

# %%
# Get equilibrium
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

eq_name = "eqref_OOB.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)

# eq_name = "DN-DEMO_eqref.json"
# eq_name = Path(EQDATA, eq_name)
# eq = Equilibrium.from_eqdsk(eq_name, from_cocos=3, qpsi_positive=False)

# Plot the equilibrium
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %%
# Set the focus point
# Note: there is the possibility to experiment with the position of the focus, but
# the default is to use the plasma o point.
eq.get_OX_points()
R_0 = eq._o_points[0].x
Z_0 = eq._o_points[0].z

# %% [markdown]
# Use the `toroidal_harmonic_approximate_psi` function to use approximate the coilset
# contribution to the flux using toroidal harmonics. We need to provide this function
# with the equilibrium, eq, and the coordinates of the focus point, R_0 and Z_0. The
# function returns the psi_approx array. The default focus point is the plasma o point.

# %%
# Approximate psi and plot
psi_approx, R_approx, Z_approx = toroidal_harmonic_approximate_psi(
    eq=eq, R_0=R_0, Z_0=Z_0, max_degree=6
)

nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
plt.contourf(R_approx, Z_approx, psi_approx, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("TH Approximation for Coilset Psi")
plt.show()

# %% [markdown]
# Now we want to compare this approximation to the solution from bluemira.

# %%
# Want to compare to Bluemira coilset and find the fit metric

# Interpolation so we can compare psi over the same grid
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], eq.plasma.psi())
interpolated_plasma_psi = psi_func.ev(R_approx, Z_approx)

total_psi = psi_approx + interpolated_plasma_psi

# Find LCFS from TH approx
approx_eq = deepcopy(eq)
o_points, x_points = approx_eq.get_OX_points(total_psi)

f_s = find_flux_surf(
    R_approx, Z_approx, total_psi, 1.0, o_points=o_points, x_points=x_points
)
approx_LCFS = Coordinates({"x": f_s[0], "z": f_s[1]})
original_LCFS = eq.get_LCFS()

# Plot
plt.contourf(R_approx, Z_approx, total_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.plot(approx_LCFS.x, approx_LCFS.z, color="red", label="Approximate LCFS from TH")
plt.title("Total Psi using TH approximation for coilset psi")
plt.legend(loc="upper right")
plt.show()


# Obtain psi from Bluemira coilset
bm_coil_psi = np.zeros(np.shape(eq.grid.x))
for n in eq.coilset.name:
    bm_coil_psi = np.sum([bm_coil_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0)

# Plotting
plt.contourf(eq.grid.x, eq.grid.z, bm_coil_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Bluemira Coilset Psi")
plt.show()


# Interpolation to use same grid
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], bm_coil_psi)
interpolated_coilset_psi = psi_func.ev(R_approx, Z_approx)

# Plotting
plt.contourf(R_approx, Z_approx, interpolated_coilset_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Interpolated Bluemira Coilset Psi")
plt.show()

# Difference plot to compare TH approximation to Bluemira coil
coilset_psi_diff = np.abs(psi_approx - interpolated_coilset_psi) / np.max(
    interpolated_coilset_psi
)
f, ax = plt.subplots()
ax.plot(approx_LCFS.x, approx_LCFS.z, color="red", label="Approximate LCFS from TH")
ax.plot(original_LCFS.x, original_LCFS.z, color="blue", label="LCFS from Bluemira")
im = ax.contourf(R_approx, Z_approx, coilset_psi_diff, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title("Difference between coilset psi and TH approximation psi")
ax.legend(loc="upper right")
eq.coilset.plot(ax=ax)
plt.show()
# %% [markdown]
# We use a fit metric to evaluate the TH approximation.

# %%
# Fit metric to evaluate TH approximation
fit_metric_value = fs_fit_metric(original_LCFS, approx_LCFS)
print(f"fit metric value = {fit_metric_value}")

# %%
