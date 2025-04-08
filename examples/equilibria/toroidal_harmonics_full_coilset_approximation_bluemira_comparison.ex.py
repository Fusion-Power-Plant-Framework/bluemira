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
# # Example of using Toroidal Harmonic Approximation
#
# This example illustrates the inner workings of the bluemira toroidal harmonics
# approximation function (toroidal_harmonic_approximation) which can be used in
# coilset current and position optimisation for conventional aspect ratio tokamaks.
# For full details and all equations, look in the
# notebook toroidal_harmonics_component_function_walkthrough_and_verification.ex.py.
#
# ## Premise:
#
# - Our equilibrium solution will not change if the coilset contribution
# to the poloidal field (vacuum field) is kept the same in the region
# occupied by the core plasma (i.e. the region characterised by closed flux surfaces).
#
# - If we constrain the core plasma while altering (optimising) other aspects
#   of the magnetic configuration, then we will not need to re-solve for
#   the plasma equilibrium at each iteration.
#
# - We can decompose the vacuum field into Toroidal Harmonics (TH)
#   to create a minimal set of constraints for use in optimisation.
#
# There are benefits to using TH as a minimal set of constraints:
# - We can choose not to re-solve for the plasma equilibrium at each step, since the
# coilset contribution to the core plasma (within the LCFS) is constrained.
# - We have a minimal set of constraints (a set of harmonic amplitudes) for the core
# plasma contribution, which can reduce the dimensionality of the problem we are
# considering.
#
# We get the TH amplitudes/coefficients, $A(\tau, \sigma)$, from the following equations:
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} A_m^{\cos} \epsilon_m m! \sqrt{\frac{2}{\pi}}
# \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \cos(m \sigma) + A_m^
# {\sin}
# \epsilon_m m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}}
# \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \sin(m \sigma) $$
#
# where
#
# $$ A_m^{\cos, \sin} = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} factorial\_term \frac{\sinh(
# \tau_c)}
# {\Delta_c^{\frac{1}{2}}} P_{m - \frac{1}{2}}^{-1}(\cosh(\tau_c)) ^{\cos}_{\sin}(m
# \sigma_c) $$
#
# where
# - $A_m^{\cos, \sin}$ are coefficients for a single coil
# - subscript $c$ refers to a single coil
# - $I_c, \tau_c, \sigma_c$ are the coil current, and coil position in toroidal
# coordinates $(\tau, \sigma)$
# - $m$ is the poloidal mode number
# - $P_{\nu}^{\mu}$ is the associated Legendre function of the first kind of degree $\nu$
#  and order $\mu$
# - $\textbf{Q}_{\nu}^{\mu}$ is Olver's definition of the associated Legendre function of
# the second kind. See [here](https://dlmf.nist.gov/14) or F. W. J. Olver (1997b)
# Asymptotics and Special Functions. A. K. Peters, Wellesley, MA. for more information.
# - $\varepsilon_m = 1 $ for $m = 0$ and $\varepsilon_m = 2$ for $m \ge 1$
# - $ \Delta = \cosh(\tau) - \cos(\sigma) $
# - $ \Delta_c = \cosh(\tau_c) - \cos(\sigma_c) $
# - $ factorial\_term = \prod_{i=0}^{m-1} \left( 1 + \frac{1}{2(m-i)}\right) $
#
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
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.geometry.coordinates import Coordinates

# %% [markdown]
# Get equilibrium from EQDSK file and plot

# %%
# Get equilibrium
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

# eq_name = "eqref_OOB.json"
# eq_name = Path(EQDATA, eq_name)
# eq = Equilibrium.from_eqdsk(eq_name, from_cocos=7)

eq_name = "DN-DEMO_eqref.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=3, qpsi_positive=False)

# Plot the equilibrium
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# Set the focus point
#
# Note: there is the possibility to experiment with the position of the focus, but
# the default is to use the plasma o point.
# %%
# Set the focus point
eq.get_OX_points()
R_0 = eq._o_points[0].x
Z_0 = eq._o_points[0].z

# %% [markdown]
# Use the `toroidal_harmonic_grid_and_coil_setup` to obtain the necessary
# parameters required for the TH approximation, such as the R and Z coordinates of the
# TH grid.
#
# We then use the `toroidal_harmonic_approximate_psi` function to approximate the coilset
# contribution to the flux using toroidal harmonics. We need to provide this function
# with the equilibrium, eq, and the ToroidalHarmonicsParams dataclass, which contains
# necessary parameters for the TH approximation, such as the relevant coordinates
# and coil names for use in the approximation. The
# function returns the approx_coilset_psi array and the TH coefficient matrix A_m.
# The default focus point is the plasma o point.
# The white dot in the plot shows the focus point.
# %%
# Approximate psi and plot
th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

R_approx = th_params.R
Z_approx = th_params.Z

approx_coilset_psi, _, _ = toroidal_harmonic_approximate_psi(
    eq=eq, th_params=th_params, max_degree=5
)

nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
plt.contourf(R_approx, Z_approx, approx_coilset_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("TH Approximation for Coilset Psi")
plt.show()
# %% [markdown]
# We need to set up a mask to use when interpolating because we don't want
# to use interpolated values that are outside of the bluemira equilibria
# grid.

# %%
# Mask
min_grid_x, max_grid_x = np.min(eq.grid.x), np.max(eq.grid.x)
min_grid_z, max_grid_z = np.min(eq.grid.z), np.max(eq.grid.z)

R_mask = R_approx
R_mask = np.where(R_approx < min_grid_x, 0.0, 1.0)
R_mask = np.where(R_approx > max_grid_x, 0.0, 1.0)
Z_mask = Z_approx
Z_mask = np.where(Z_approx < min_grid_z, 0.0, 1.0)
Z_mask = np.where(Z_approx > max_grid_z, 0.0, 1.0)
mask = R_mask * Z_mask

# %% [markdown]
# Now we want to compare this approximation to the solution from bluemira.

# %%
# Plot total psi using approximate coilset psi from TH, and plasma psi from bluemira
# Interpolation so we can compare psi over the same grid
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], eq.plasma.psi())
interpolated_plasma_psi = psi_func.ev(R_approx, Z_approx)

total_psi = approx_coilset_psi + interpolated_plasma_psi
total_psi *= mask

# Find LCFS from TH approx
approx_eq = deepcopy(eq)
o_points, x_points = approx_eq.get_OX_points(total_psi)

# The fit metric is a measure of how 'good' the approximation is.
# Fit metric value = total area within one but not both FSs /
#                    (input FS area + approximation FS area)
psi_norm = 1.0
f_s = find_flux_surf(
    R_approx, Z_approx, total_psi, psi_norm, o_points=o_points, x_points=x_points
)
approx_LCFS = Coordinates({"x": f_s[0], "z": f_s[1]})
original_LCFS = eq.get_LCFS() if psi_norm == 1.0 else eq.get_flux_surface(psi_norm)

# Plot
plt.contourf(R_approx, Z_approx, total_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.plot(
    approx_LCFS.x,
    approx_LCFS.z,
    color="red",
    label="Approximate LCFS from TH approximation",
)
plt.title("Total Psi using TH approximation for coilset psi")
plt.legend(loc="upper right")
plt.show()

# %%
# Plot interpolated bluemira total psi
# Obtain psi from Bluemira coilset
bm_coil_psi = np.zeros(np.shape(eq.grid.x))
for n in eq.coilset.name:
    bm_coil_psi = np.sum([bm_coil_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0)


# Obtain total psi from Bluemira
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], eq.psi())
interpolated_bm_total_psi = psi_func.ev(R_approx, Z_approx)
interpolated_bm_total_psi *= mask

# Plotting
plt.contourf(R_approx, Z_approx, interpolated_bm_total_psi, levels=nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Interpolated Total Bluemira Psi")
plt.show()
# %%
# Plot interpolated bluemira coilset psi
# Interpolation to use same grid
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], bm_coil_psi)
interpolated_coilset_psi = psi_func.ev(R_approx, Z_approx)

# Plotting
plt.contourf(R_approx, Z_approx, interpolated_coilset_psi * mask, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Interpolated Bluemira Coilset Psi")
plt.show()
# %%
# Difference plot to compare TH approximation to Bluemira coilset psi
# We see zero difference in the core region, which we expect as we are constraining
# the flux in this region, and we see larger differences outside of the approximation
# region.
coilset_psi_diff = np.abs(approx_coilset_psi - interpolated_coilset_psi) / np.max(
    np.abs(interpolated_coilset_psi)
)
coilset_psi_diff_plot = coilset_psi_diff * mask
f, ax = plt.subplots()
ax.plot(approx_LCFS.x, approx_LCFS.z, color="red", label="Approximate LCFS from TH")
ax.plot(original_LCFS.x, original_LCFS.z, color="blue", label="LCFS from Bluemira")
im = ax.contourf(R_approx, Z_approx, coilset_psi_diff_plot, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title("Absolute relative difference between coilset psi and TH approximation psi")
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.0))
eq.coilset.plot(ax=ax)
plt.show()


# %%
# Difference plot to compare TH approximation to Bluemira total psi
total_psi_diff = np.abs(total_psi - interpolated_bm_total_psi) / np.max(
    np.abs(interpolated_bm_total_psi)
)
total_psi_diff_plot = total_psi_diff * mask
f, ax = plt.subplots()
ax.plot(approx_LCFS.x, approx_LCFS.z, color="red", label="Approx FS from TH")
# ax.plot(original_LCFS.x, original_LCFS.z, color="blue", label="FS from Bluemira")
im = ax.contourf(R_approx, Z_approx, total_psi_diff_plot, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title("Absolute relative difference between total psi and TH approximation psi")
ax.legend(loc="upper right")
eq.coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# We use a fit metric to evaluate the TH approximation.

# %%
# Fit metric to evaluate TH approximation
fit_metric_value = fs_fit_metric(original_LCFS, approx_LCFS)
print(f"fit metric value = {fit_metric_value}")

# %% [markdown]
# All of these TH functions are combined in the `toroidal_harmonic_approximation`
# function, which takes the equilibrium, eq, and the TH parameters, th_params, and
# approximates psi using TH. The function uses a fit metric to find the appropriate
# number of degrees to use for the approximation.
#
# Here is an example of using the function, setting plot to True outputs a graph of the
# difference in total psi between the TH approximation and bluemira.
# %%
(
    toroidal_harmonics_params,
    Am_cos,
    Am_sin,
    degree,
    fit_metric,
    approx_total_psi,
    approx_coilset_psi,
) = toroidal_harmonic_approximation(
    eq=eq, th_params=th_params, plot=True, psi_norm=psi_norm
)
