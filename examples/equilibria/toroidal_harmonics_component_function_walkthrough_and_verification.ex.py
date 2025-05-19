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
An example of plotting the internal harmonics in toroidal coordinates about a focus
point and calculating solution due to a single wire as a sum of toroidal harmonics.
"""

# %% [markdown]
# # 1. Example of plotting the internal harmonics in toroidal coordinates

# This example starts by showing how to plot the individual cos and sin toroidal harmonic
# contributions about a focus point.

# %% [markdown]
# ## Imports

# %%
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils._coil import Coil  # noqa: PLC2701
from bluemira.equilibria.coils._grouping import CoilSet  # noqa: PLC2701
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501  # noqa: E501
    legendre_p,
    legendre_q,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.utilities.tools import cylindrical_to_toroidal, toroidal_to_cylindrical

# %% [markdown]
# First we need to set up the grid in cylindrical coordinates over which we will solve,
# and define the location of the focus point.
# We will convert this to toroidal coordinates.
#
#
# To convert from cylindrical coordinates $(R, z)$ to toroidal coordinates
# $(\tau, \sigma)$ about a focus point $(R_0, z_0)$ we have the following relations:
# $$ \tau = \ln\frac{d_{1}}{d_{2}} $$
# $$ \sigma = {sign}(z - z_{0}) \arccos\frac{d_{1}^2 + d_{2}^2 - 4 R_{0}^2}{2 d_{1}
# d_{2}}  $$
# where
# $$ d_{1}^2 = (R + R_{0})^2 + (z - z_{0})^2 $$
# $$ d_{2}^2 = (R - R_{0})^2 + (z - z_{0})^2 $$
#
# The domains for the toroidal coordinates are $0 \le \tau < \infty$ and $-\pi < \sigma
# \le \pi$.

# %%
# Set up grid over which to solve
r = np.linspace(0, 6, 100)
z = np.linspace(-6, 6, 100)
R, Z = np.meshgrid(r, z)

# Define focus point
R_0 = 3.0
Z_0 = 0.0

# Convert to toroidal coordinates
tau, sigma = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R, Z=Z)

# %% [markdown]
# Now we want to calculate and plot the internal harmonics in toroidal coordinates about
# the focus.
#
# We use the following equations
# $$ \psi_{sin} = R \sqrt{\cosh (\tau) - \cos (\sigma)} \textbf{Q}_{m - \frac{1}{2}}^1
# (\cosh \tau) \sin (m \sigma) $$
#
# $$ \psi_{cos} = R \sqrt{\cosh (\tau) - \cos (\sigma)} \textbf{Q}_{m - \frac{1}{2}}^1
# (\cosh \tau) \cos (m \sigma) $$
#
#
# $\textbf{Q}_{\nu}^{\mu}$ is Olver's definition of the associated Legendre function
# of the second kind. See [here](https://dlmf.nist.gov/14) or F. W. J. Olver
# (1997b) Asymptotics and Special Functions. A. K. Peters, Wellesley, MA. for more
# information.
# Here we have degree $\nu = 1$, and half integer order $\mu = m - \frac{1}{2}$.
#
# We evaluate these contributions to the harmonics about the focus and plot.
# %%
# Calculate and plot individual contributions from toroidal harmonics

nu = np.arange(0, 5)

# Setting up plots
fig_sin, axs_sin = plt.subplots(1, len(nu))
fig_sin.suptitle("sin")
fig_sin.supxlabel("R")
fig_sin.supylabel("Z")
fig_cos, axs_cos = plt.subplots(1, len(nu))
fig_cos.suptitle("cos")
fig_cos.supxlabel("R")
fig_cos.supylabel("Z")
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]

for i_nu in range(len(nu)):
    foo = (
        R
        * np.sqrt(np.cosh(tau) - np.cos(sigma))
        * legendre_q(nu[i_nu] - 1 / 2, 1, np.cosh(tau))
    )
    psi_sin = foo * np.sin(nu[i_nu] * sigma)
    psi_cos = foo * np.cos(nu[i_nu] * sigma)
    axs_sin[i_nu].contour(R, Z, psi_sin, levels=nlevels, cmap=cmap)
    axs_sin[i_nu].title.set_text(f"m = {i_nu}")
    axs_cos[i_nu].contour(R, Z, psi_cos, levels=nlevels, cmap=cmap)
    axs_cos[i_nu].title.set_text(f"m = {i_nu}")


# %% [markdown]
# # 2. Calculating the flux solution due to a single wire as a sum of toroidal harmonics
#
# This example now illustrates how the magnetic flux due to a single wire can be
# calculated by using toroidal harmonics about a focus point.


# %% [markdown]
# We want to calculate the psi field due to a circular wire using toroidal
# harmonics.
#
# The relevant equations for psi in toroidal coordinates, $(\tau, \sigma)$, are
#
# $$ \psi = R A $$
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} \sum_{\cos, \sin} A_m^{\cos, \sin} \epsilon_m
# m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^1(\cosh \tau)_
# {\sin}^{\cos} (m \sigma)  $$
#
# $$ A_m^{\cos, \sin} = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} \prod_{i=0}^{m-1} \left( 1 +
# \frac{1}{2(m-i)}\right) \frac{\sinh(\tau_c)}
# {\Delta_c^{\frac{1}{2}}} P_{m - \frac{1}{2}}^{-1}(\cosh(\tau_c))_{\sin}^{\cos}(m
# \sigma_c) $$
#
# Here, $m$ is the poloidal mode number, and we have the following
# $$ \Delta = \cosh \tau - \cos \sigma, $$
# $$ \Delta_c = \cosh \tau_c - \cos \sigma_c $$
# $$ \epsilon_0 = 1, \epsilon_{m\ge1}=2$$
# $P_{\nu}^{\mu}$ is the associated Legendre function of the first kind, of degree $\nu$
# and order $\mu$. $\textbf{Q}_{\nu}^{\mu}$ is Olver's definition of the associated
# Legendre function of the second kind. See [here](https://dlmf.nist.gov/14) or F. W. J.
# Olver
# (1997b) Asymptotics and Special Functions. A. K. Peters, Wellesley, MA. for more
# information.
#
# The following image shows the psi field we are expecting to obtain at the end of this
# example.
# %%
single_wire_image = Image.open("single_wire_output_image.png")
display(single_wire_image)
# %% [markdown]
# First we define the location in cylindrical coordinates of the focus $(R_0, z_0)$ and
# of the wire $(R_c, z_c)$, and the current in the wire, $I_c$.
#
# We then convert the location of the wire from cylindrical $(R_c, z_c)$ to toroidal
# coordinates $(\tau_c, \sigma_c)$ using the coordinate transform relations from above.
#
# We need the location of the wire in toroidal coordinates. We also need to
# approximate $\tau$ at the focus instead of using coordinate transform functions
# as this would result in divide by zero errors. We use these values to create a
# grid in toroidal coordinates. Once we have the grid in toroidal coordinates, we convert
# this to cylindrical coordinates for use later.
#
# We want the coil to be located outside of this grid
# otherwise there will be large differences between the Bluemira psi and the TH
# approximation of psi in the region near the coil.
#
#
# %%
# Wire location
R_c = 5.0
Z_c = 1.0
I_c = 5e6

# Focus of toroidal coordinates
R_0 = 3.0
Z_0 = 0.2

# Location of wire in toroidal coordinates
tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_c, Z=Z_c)

# Using approximate value for d2_min to avoid infinities
# Approximating tau at the focus instead of using coordinate transform functions
# (in order to avoid divide by 0 errors)
# These approximate values come from the toroidal coordinate transform functions
# given at the start of this notebook.
# We have that $\tau = \ln \frac{d_1}{d_2}$, so we can see that the maximum value
# of tau occurs when $d_1$ is at its largest value and when $d_2$ is at its smallest
# value.
# We have that $d_1^2 = (R + R_0)^2 + (z - z_0)^2$, and so $d_1$ is largest when R = R_0
# and z = z_0, and this gives $d_1 = 2 * R_0$
# We have that $d_2 = \\sqrt((R - R_0)^2 + (z - z_0)^2)$, and so $d_2$ is smallest when
# R = R_0 and z = z_0. However this would give $d_2 = 0$ and this would cause divide by
# zero errors. To avoid this, we set d2_min to be equal to a small number instead of 0.

# Change the value of tau_offset to change the extent of the toroidal harmonic
# approximation. If the toroidal approximation grid contains the coil, you will
# notice large differences in psi in the region near the coil between the bluemira
# solution and the TH approximation. The importance of this becomes clear in the full
# coilset approximation. For more information on the full coilset approximation, see
# toroidal_harmonics_full_coilset_approximation_bluemira_comparison.ex.py.
tau_offset = 0.05

d2_min = 0.05
tau_max = np.log(2 * R_0 / d2_min)
n_tau = 200
tau = np.linspace(tau_c + tau_offset, tau_max, n_tau)
n_sigma = 150
sigma = np.linspace(-np.pi, np.pi, n_sigma)

# Create grid in toroidal coordinates
tau, sigma = np.meshgrid(tau, sigma)

# Convert to cylindrical coordinates
R, Z = toroidal_to_cylindrical(R_0=R_0, z_0=Z_0, tau=tau, sigma=sigma)


# %% [markdown]
# Now we want to calculate the following coefficients
# $$ A_m = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} factorial\_term \frac{\sinh(\tau_c)}
# {\Delta_c^{\frac{1}{2}}} P_{m - \frac{1}{2}}^{-1}(\cosh(\tau_c)) $$
# $$ A_m^{sin} = A_m \sin(m \sigma_c) $$
# $$ A_m^{cos} = A_m \cos(m \sigma_c) $$
# which are functions of the poloidal mode number, $m$, and the coil position,
# where $$ \Delta_c = \cosh(\tau_c) - \cos(\sigma_c) $$
# and $$ factorial\_term = \prod_{i=0}^{m-1} \left( 1 + \frac{1}{2(m-i)}\right) $$
#

# %%
# Useful combinations
Delta = np.cosh(tau) - np.cos(sigma)
Deltac = np.cosh(tau_c) - np.cos(sigma_c)

# Calculate coefficients
m_max = 5
Am_cos = np.zeros(m_max + 1)
Am_sin = np.zeros(m_max + 1)

for m in range(m_max + 1):
    factorial_term = 1 if m == 0 else np.prod(1 + 0.5 / np.arange(1, m + 1))
    A_m = (
        (MU_0 * I_c / 2 ** (5 / 2))
        * factorial_term
        * (np.sinh(tau_c) / np.sqrt(Deltac))
        * legendre_p(m - 1 / 2, 1, np.cosh(tau_c))
    )
    Am_cos[m] = A_m * np.cos(m * sigma_c)
    Am_sin[m] = A_m * np.sin(m * sigma_c)

# %% [markdown]
# Now we can use the following
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} A_m^{\cos} \epsilon_m m! \sqrt{\frac{2}{\pi}}
# \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \cos(m \sigma) + A_m^
# {\sin}
# \epsilon_m m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}}
# \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \sin(m \sigma) $$

# along with $$\psi = R A$$
# to calculate the solution and plot the psi graph. Here we have that
# $ \epsilon_0 = 1$ and $\epsilon_{m\ge 1} = 2$.

# %%
epsilon = 2 * np.ones(m_max + 1)
epsilon[0] = 1
A = np.zeros_like(R)

for m in range(m_max + 1):
    A += Am_cos[m] * epsilon[m] * factorial(m) * np.sqrt(2 / np.pi) * np.sqrt(
        Delta
    ) * legendre_q(m - 1 / 2, 1, np.cosh(tau)) * np.cos(m * sigma) + Am_sin[m] * epsilon[
        m
    ] * factorial(m) * np.sqrt(2 / np.pi) * np.sqrt(Delta) * legendre_q(
        m - 1 / 2, 1, np.cosh(tau)
    ) * np.sin(m * sigma)

psi_th_approx = R * A
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
plt.contour(R, Z, psi_th_approx, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("TH Approximation of Psi")
# %%[markdown]
# Now we can compare this approximation for psi with the solution from Bluemira.

# %%
# Comparison to single Bluemira coil
coil = Coil(R_c, Z_c, current=I_c, dx=0.1, dz=0.1, ctype="PF", name="PF_0")

coilset = CoilSet(coil)
grid = Grid(np.min(R), np.max(R), np.min(Z), np.max(Z), 150, 200)

bm_coil_psi = np.zeros(np.shape(R))
for n in coilset.name:
    bm_coil_psi = np.sum([bm_coil_psi, coilset[n].psi(grid.x, grid.z)], axis=0)

nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
plt.contour(grid.x, grid.z, bm_coil_psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Bluemira Coil Psi")

# %% [markdown]
# We interpolate the solution from Bluemira to compare over the same grid as that used
# for the toroidal harmonics approximation.

# %%
# Interpolation to compare over same grid
psi_func = RectBivariateSpline(grid.x[:, 0], grid.z[0, :], bm_coil_psi)
interpolated_coilset_psi = psi_func.ev(R, Z)
plt.contour(R, Z, interpolated_coilset_psi, levels=nlevels, cmap=cmap)
plt.plot(R_c, Z_c, marker="o", markersize=10, label="Coil")
plt.title("Bluemira Coilset Psi Interpolated onto TH grid")
plt.legend(loc="upper left")
plt.show()

# Difference plot to compare TH approximation to Bluemira coil
coilset_psi_diff = psi_th_approx - interpolated_coilset_psi
im = plt.contourf(R, Z, coilset_psi_diff, levels=nlevels, cmap=cmap, zorder=8)
plt.colorbar(mappable=im)
plt.plot(R_c, Z_c, marker="o", markersize=10, label="Coil")
plt.title("Difference in coilset psi between TH approximation and Bluemira")
plt.legend(loc="upper left")
plt.show()

# %%
