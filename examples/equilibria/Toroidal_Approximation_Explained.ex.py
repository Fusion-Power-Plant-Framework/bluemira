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
# For an example of how toroidal_harmonic_approximation is used
# to create toroidal harmonics constraints for use in a coilset optimisation problem,
# please see
# Toroidal_Harmonics_Optimisation_Set_Up.ex.py
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
# - We have a small, viable set of constraints (a set of harmonic amplitudes) for the
# core plasma contribution, which can reduce the dimensionality of the problem we are
# considering.
#
# %% [markdown]
# We get the TH amplitudes/coefficients, $A(\tau, \sigma)$, from the following equations:
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} A_m^{\cos} \epsilon_m m! \sqrt{\frac{2}{\pi}}
# \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \cos(m \sigma)
# \\+ A_m^{\sin} \epsilon_m m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}}
# \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \sin(m \sigma) $$
#
# where
#
# $$ A_m^{\cos, \sin} = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} \upsilon_{fact} \frac{\sinh(
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
# - The factorial term, $\upsilon_{fact}$ = 1 if $m = 0$, else $= \prod_{i=0}^{m-1}
# \left( 1 + \frac{1}{2(m-i)}\right) $
#
# We can then obtain the flux, $\psi$ by using $\psi = R A$.

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np

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
# Get equilibrium from EQDSK file

# %%
# Get equilibrium
EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")

eq_name = "DN-DEMO_eqref.json"
eq_name = Path(EQDATA, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=3, qpsi_positive=False)


# %% [markdown]
# Set the focus point
#
# Note: there is the possibility to experiment with the position of the focus, but
# the default is to use the effective centre of the plasma.
# %%
# Set the focus point
R_0, Z_0 = eq.effective_centre()

# %% [markdown]
# Use the `toroidal_harmonic_grid_and_coil_setup` to obtain the necessary
# parameters required for the TH approximation, such as the R and Z coordinates of the
<<<<<<< HEAD:examples/equilibria/Toroidal_Approximation_Explained.ex.py
# TH grid. We then plot the equilibrium and in orange we show the region over which
# we will use our toroidal harmonic approximation.

=======
# TH grid.
#
# We then use the `toroidal_harmonic_approximate_psi` function to approximate the coilset
# contribution to the flux using toroidal harmonics. We need to provide this function
# with the equilibrium, eq, and the ToroidalHarmonicsParams dataclass, which contains
# necessary parameters for the TH approximation, such as the relevant coordinates
# and coil names for use in the approximation. The
# function returns the approx_coilset_psi array and the TH coefficient matrix A_m.
# The default focus point is the effective centre of the plasma.
# The white dot in the plot shows the focus point.
>>>>>>> e5150c10 (work up until midday 7th aug - committing before moving back to toroidal-harmonics-constraints branch to tidy up before release):examples/equilibria/toroidal_harmonics_full_coilset_approximation_bluemira_comparison.ex.py
# %%
# Approximate psi and plot
th_params = toroidal_harmonic_grid_and_coil_setup(eq=eq, R_0=R_0, Z_0=Z_0)

R_approx = th_params.R
Z_approx = th_params.Z

# Plot the equilibrium and the region over which we approximate the flux using toroidal
# harmonics.
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
max_R = np.max(th_params.R)  # noqa: N816
min_R = np.min(th_params.R)  # noqa: N816
max_Z = np.max(th_params.Z)  # noqa: N816
min_Z = np.min(th_params.Z)  # noqa: N816
centre_R = (max_R - min_R) / 2 + min_R  # noqa: N816
centre_Z = (max_Z - np.abs(min_Z)) / 2  # noqa: N816
radius = (max_R - min_R) / 2

ax.add_patch(
    patch.Circle((centre_R, centre_Z), radius, ec="orange", fill=True, fc="orange")
)
plt.show()
# %% [markdown]
#
# We then use the `toroidal_harmonic_approximate_psi` function to approximate the coilset
# contribution to the flux using toroidal harmonics. This makes use of the equation for
# $A_m^{\cos, \sin}$ as displayed at the start of this notebook.
# We need to provide this function
# with the equilibrium, eq, and the ToroidalHarmonicsParams dataclass, which contains
# necessary parameters for the TH approximation, such as the relevant coordinates
# and coil names for use in the approximation. The
# function returns the approx_coilset_psi array and the TH coefficient matrices Am_cos
# and Am_sin.

# %%
approx_coilset_psi, _, _ = toroidal_harmonic_approximate_psi(
    eq=eq, th_params=th_params, max_degree=5
)


# %% [markdown]
# Now we want to compare this approximation for the coilset psi to the solution from
# bluemira.
# Since we assume the flux in the plasma region is kept fixed, we can calculate
# approximate total psi = bluemira plasma psi + toroidal harmonic coilset psi
# approximation.
#
# We also create a difference plot to compare our TH coilset psi approximation to the
# bluemira coilset psi.
# We can see zero difference in the core region, which we expect as we are constraining
# the flux in this region, and we see differences outside of the approximation
# region, but this is expected as we are only requiring the core region to be kept
# fixed.
# Since the plasma psi is kept fixed, the total psi difference plot would look
# the same as the coilset psi difference plot shown below.

# %%
# Approximate total psi = approximate coilset psi from TH + plasma psi from bluemira

plasma_psi = eq.plasma.psi(R_approx, Z_approx)

total_psi = approx_coilset_psi + plasma_psi

# Find LCFS from TH approx to add to plot
approx_eq = deepcopy(eq)
o_points, x_points = approx_eq.get_OX_points(total_psi)

# The fit metric is a measure of how 'good' the approximation is.
# Fit metric value = total area within one but not both FSs /
#                    (input FS area + approximation FS area)
psi_norm = 0.95
f_s = find_flux_surf(
    R_approx, Z_approx, total_psi, psi_norm, o_points=o_points, x_points=x_points
)
approx_fs = Coordinates({"x": f_s[0], "z": f_s[1]})
original_fs = (
    eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
)

# Difference in coilset psi between TH approximation and bluemira
bluemira_coilset_psi = eq.coilset.psi(R_approx, Z_approx)

coilset_psi_diff = np.abs(approx_coilset_psi - bluemira_coilset_psi) / np.max(
    np.abs(bluemira_coilset_psi)
)
coilset_psi_diff_plot = coilset_psi_diff
f, axs = plt.subplots(1, 2)

# Plot
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
axs0_plot = axs[0].contourf(R_approx, Z_approx, total_psi, nlevels, cmap=cmap)
axs[0].set_xlabel("R")
axs[0].set_ylabel("Z")
axs[0].plot(
    approx_fs.x,
    approx_fs.z,
    color="red",
    label="Approx. FS from TH",
)
axs[0].set_title("Total Psi using TH approximation for coilset psi")
axs[0].legend(loc="upper right")

axs[1].plot(approx_fs.x, approx_fs.z, color="red", label="Approx. FS from TH")
axs[1].plot(
    original_fs.x,
    original_fs.z,
    color="c",
    linestyle="dashed",
    label="LCFS from Bluemira",
)
axs1_plot = axs[1].contourf(
    R_approx, Z_approx, coilset_psi_diff_plot, levels=nlevels, cmap=cmap
)
f.colorbar(axs1_plot, ax=axs[1], fraction=0.05)
axs[1].set_title(
    "Absolute relative difference between coilset psi \nand TH approximation psi"
)
axs[1].legend(loc="upper right", bbox_to_anchor=(1.1, 1.0))
eq.coilset.plot(ax=axs[1])
axs[0].set_aspect("equal")
axs[1].set_aspect("equal")
plt.show()


# %% [markdown]
# We use a fit metric to evaluate the TH approximation.
# This involves
# finding a flux surface (here corresponding to a psi_norm=0.95) for our
# approximate total psi, and comparing to the equivalent flux surface from
# the bluemira total psi. The fit metric we use for this flux surface
# comparison is as follows: fit metric value = total area within one but not
# both flux surfaces / (bluemira flux surface area + approximation flux surface area).
# A fit metric of 0 would correspond to a perfect match between our approximation
# and bluemira, and a fit metric of 1 would correspond to no overlap between the
# approximation and bluemira flux surfaces.
# %%
# Fit metric to evaluate TH approximation
fit_metric_value = fs_fit_metric(original_fs, approx_fs)
print(f"fit metric value = {fit_metric_value}")

# %% [markdown]
# All of these TH functions are combined in the `toroidal_harmonic_approximation`
# function, which takes the equilibrium, eq, and the TH parameters, th_params, and
# approximates psi using TH. The function uses a fit metric to find the appropriate
# number of degrees to use for the approximation.
#
# Here is an example of using the function, setting plot to True outputs a graph of the
# difference in total psi between the TH approximation and bluemira. This graph
# is the same as the one we produced in this notebook above.
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
