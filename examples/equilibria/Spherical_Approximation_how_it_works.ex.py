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
An example that shows how the Spherical Harmonic Approximation works
"""

# %% [markdown]
# # Example of using Spherical Harmonic Approximation
#
# This example illustrates the inner workings of the Bluemira spherical harmonics
# approximation function (spherical_harmonic_approximation) which can be used in
# coilset current and position optimisation for spherical tokamaks.
# For an example of how spherical_harmonic_approximation is used
# please see, spherical_harmonic_approximation_basic_function_check.
# please see, spherical_harmonic_approximation_basic_function_check.
#
# ## Premise:
#
# - Our equilibrium solution will not change if the coilset contribution
# to the poloidal field (vacuum field) is kept the same in the region
# occupied by the core plasma (i.e. the region characterised by closed fux surfaces).
#
# - If we constrain the core plasma while altering (optimising) other aspects
#   of the magnetic configuration, then we will not need to re-solve for
#   the plasma equilibrium at each iteration.
#
# - We can decompose the vacuum field into Spherical Harmonics (SH)
#   to create a minimal set of constraints for use in optimisation.

# %% [markdown]
# ### Imports

# %%
from pathlib import Path

import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import lpmv

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    coil_harmonic_amplitude_matrix,
    coils_outside_lcfs_sphere,
    collocation_points,
    harmonic_amplitude_marix,
    lcfs_fit_metric,
)

# %pdb

# %% [markdown]
# ### Equilibria and Coilset Data from File
#
# Note: We cannot use coils that are within the sphere containing the LCFS
# for our approximation. The maximum radial distance of the LCFS is used
# as a limit (orange shaded area in plot below).
# If you have coils in this region then we need to specify a list of the
# coil names that are outside of the radial limit
# (this is done automatically in the spherical_harmonic_approximation class).

# %%
# Data from EQDSK file
file_path = Path(
    get_bluemira_path("equilibria", subfolder="examples"), "SH_test_file.json"
)

eq = Equilibrium.from_eqdsk(file_path)

# Get the necessary boundary locations and length scale
# for use in spherical harmonic approximations.

# Starting LCFS
original_LCFS = eq.get_LCFS()

# Names of coils located outside of the sphere containing the LCFS
sh_coil_names, bdry_r = coils_outside_lcfs_sphere(eq)

# Typical length scale
r_t = bdry_r

# Plasma boundary x and z locations from data file
x_bdry = original_LCFS.x
z_bdry = original_LCFS.z
max_bdry_r = np.max(np.linalg.norm([x_bdry, z_bdry], axis=0))

# Plot
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
ax.set_xlim(np.min(eq.grid.x), np.max(eq.grid.z))
eq.coilset.plot(ax=ax)
ax.set_xlim(np.min(eq.grid.x), np.max(eq.grid.z))
max_circ = patch.Circle((0, 0), max_bdry_r, ec="orange", fill=True, fc="orange")
ax.add_patch(max_circ)
plt.show()

# %% [markdown]
# ### Find Vacuum Psi
#
# Vacuum (coil) contribution to the poloidal flux =
#       total flux - contribution from plasma

# %%
# Poloidal magnetic flux
total_psi = eq.psi()

# Psi contribution from plasma
plasma_psi = eq.plasma.psi(eq.grid.x, eq.grid.z)
# Psi contribution from plasma
plasma_psi = eq.plasma.psi(eq.grid.x, eq.grid.z)

# Calculate psi contribution from the vacuum, i.e.,
# from coils located outside of the sphere containing LCFS
vacuum_psi = np.zeros(np.shape(eq.grid.x))
for n in sh_coil_names:
    vacuum_psi = np.sum([vacuum_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0)

# Calculate psi contribution from coils not used in the SH approx
non_sh_coil_cont = eq.coilset.psi(eq.grid.x, eq.grid.z) - vacuum_psi
# Calculate psi contribution from the vacuum, i.e.,
# from coils located outside of the sphere containing LCFS
vacuum_psi = np.zeros(np.shape(eq.grid.x))
for n in sh_coil_names:
    vacuum_psi = np.sum([vacuum_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0)

# Calculate psi contribution from coils not used in the SH approx
non_sh_coil_cont = eq.coilset.psi(eq.grid.x, eq.grid.z) - vacuum_psi

# Plot
x_plot = eq.grid.x
z_plot = eq.grid.z
nlevels = 50
cmap = "viridis"

levels1 = np.linspace(np.amin(total_psi), np.amax(total_psi), nlevels)
levels2 = np.linspace(np.amin(plasma_psi), np.amax(plasma_psi), nlevels)
levels3 = np.linspace(np.amin(vacuum_psi), np.amax(vacuum_psi), nlevels)

plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
plot2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=1)
plot3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1)

plot1.set_title("Total")
plot2.set_title("Plasma")
plot3.set_title("Vacuum")

plot1.contour(x_plot, z_plot, total_psi, levels=levels1, cmap=cmap, zorder=8)
plot2.contour(x_plot, z_plot, plasma_psi, levels=levels2, cmap=cmap, zorder=8)
plot3.contour(x_plot, z_plot, vacuum_psi, levels=levels3, cmap=cmap, zorder=8)
plot3.contour(
    x_plot,
    z_plot,
    non_sh_coil_cont,
    levels=levels3,
    cmap=cmap,
    zorder=8,
    linestyles="dashed",
)

plt.show()

# %% [markdown]
# ### Flux Function at collocation points within LCFS
#
# The steps are as follows:
#
#
# 1. Select collocation points within the LCFS for the chosen equilibrium.
#
#
# 2. Calculate psi at the collocation points using interpolation of
#    the equilibrium vacuum psi.
#
#
# 3. Construct matrix from harmonic amplitudes and fit to psi at collocation points.
#
# #### 1. Collocation points
#
# There are currently four options for collocation point distribution:
# - 'arc' = equispaced points on an arc of fixed radius
# - 'arc_plus_extrema' = 'arc' plus the min and max points
#    of the LCFS in the x- and z-directions (4 points total)
# - 'random'
# - 'random_plus_extrema'
#
# N.B., the SH degree that you calculate up to should be
# less than the number of collocation points.
#
# In the plot below the collocations points are shown as
# purple dots, and the LCFS is indicated by a red line.

# %%
# Number of desired collocation points excluding extrema (always 4 or +4 automatically)
n = 20

# Create the set of collocation points for the harmonics
collocation = collocation_points(
    n,
    original_LCFS,
    PointType.ARC_PLUS_EXTREMA,
    seed=15,
)

# Plot

x_plot = eq.grid.x
z_plot = eq.grid.z
nlevels = 50
levels = np.linspace(np.amin(total_psi), np.amax(total_psi), nlevels)
cmap = "viridis"
plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
plot1.contour(x_plot, z_plot, vacuum_psi, levels=levels, cmap=cmap, zorder=8)
plot1.scatter(collocation.x, collocation.z, color="purple")
plot1.plot(x_bdry, z_bdry, color="red")
plt.show()
print("number of collocation points = ", len(collocation.x))

# %% [markdown]
# #### 2. Flux function at collocation points
# N.B. linear interpolation is default.

# %%
# Set up with gridded values from chosen equilibrium
psi_func = RectBivariateSpline(eq.grid.x[:, 0], eq.grid.z[0, :], vacuum_psi)

# Evaluate at collocation points
collocation_psivac = psi_func.ev(collocation.x, collocation.z)

# %% [markdown]
# #### 3. Construct and fit harmonic amplitudes matrix
#
# Construct matrix from harmonic amplitudes called 'harmonics2collocation'
# (rows = collocation, columns = degrees).
#
# For each collocation point (r, $\theta$) and degree calculate the following:
#
# $$ A_{l} = \frac{ r_{collocation}^{(l+1)} }{ r_{typical}^{l} }
#           \frac{ P^{1}_{l}cos(\theta) }{ \sqrt(l(l+1)) } $$
#
# Fit to psi at collocation points using 'np.linalg.lstsq'.
# This function uses least-squares to find the vector x that
# approximately solves the equation a @ x = b.
# Where 'a' is our coefficient matrix 'harmonics2collocation',
# and 'b' is our dependant variable, a.k.a, the psi values
# at the collocation points found by interpolation 'collocation_psivac'.
#
# Result is 'psi_harmonic_amplitudes', a.k.a., the coefficients necessary
# to represent the vacuum psi using a sum of harmonics up to the selected degree.
#
# In our case, the maximum degree of harmonics to calculate up to
# is set to be equal to number of collocation points - 1.

# %%
# max_degree is set in the bluemira SH code but we need it here
max_degree = len(collocation.x) - 1

# Construct matrix from harmonic amplitudes for flux function at collocation points
harmonics2collocation = harmonic_amplitude_marix(collocation.r, collocation.theta, r_t)

# Fit harmonics to match values at collocation points
psi_harmonic_amplitudes, _, _, _ = np.linalg.lstsq(
    harmonics2collocation, collocation_psivac, rcond=None
)

# %% [markdown]
# ### Selecting the required degrees for the approximation
#
# Choose the maximum number of degrees to calculate up to
# in order to achieve a appropriate SH approximation.
# Below we set plot_max_degree = 7 as a test.
# You can change the number to see what will happen to
# the plotted results.
#
# In the next section, we will calculate a metric that can be used
# to find an appropriate number of degrees to use.

# %%
# Select the maximum degree to use in the approximation
plot_max_degree = 7

if plot_max_degree > max_degree:
    print("You are trying to plot more degrees then you calculated.")

# Plot: overplot or difference - set to true or false to change plot
plot_diff = False

# Spherical Coords
r = np.sqrt(eq.x**2 + eq.z**2)
theta = np.arctan2(eq.x, eq.z)

# First harmonic is constant
approx_psi_vac_data = psi_harmonic_amplitudes[0] * np.ones(np.shape(total_psi))

for degree in np.arange(1, plot_max_degree + 1):
    approx_psi_vac_data = approx_psi_vac_data + (
        psi_harmonic_amplitudes[degree]
        * eq.x
        * (r / r_t) ** degree
        * lpmv(1, degree, np.cos(theta))
        / np.sqrt(degree * (degree + 1))
    )

# Plot
x_plot = eq.grid.x
z_plot = eq.grid.z
nlevels = 50
levels = np.linspace(np.amin(total_psi), np.amax(total_psi), nlevels)
cmap = "viridis"
plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
plot2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=1)
plot3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1)
plot1.set_title("Vacuum Psi")
plot3.set_title("SH Psi")
plot1.contour(x_plot, z_plot, vacuum_psi, levels=levels, cmap=cmap, zorder=8)
plot1.plot(x_bdry, z_bdry, color="red")
if plot_diff:
    plot2.contour(
        x_plot,
        z_plot,
        vacuum_psi - approx_psi_vac_data,
        levels=levels,
        cmap=cmap,
        zorder=8,
    )
else:
    plot2.contour(
        x_plot, z_plot, approx_psi_vac_data, levels=levels, cmap=cmap, zorder=8
    )
    plot2.contour(
        x_plot,
        z_plot,
        vacuum_psi,
        levels=levels,
        cmap=cmap,
        zorder=8,
        linestyles="dashed",
    )
plot2.plot(x_bdry, z_bdry, color="red")
plot3.contour(x_plot, z_plot, approx_psi_vac_data, levels=levels, cmap=cmap, zorder=8)
plot3.plot(x_bdry, z_bdry, color="red")
plt.show()

# %% [markdown]
# ### Calculate Coil Currents and Find the Appropriate Degree to calculate up to
#
# psi_harmonic_amplitudes can be written as a function of the current distribution
# from the coils outside of the sphere that contains the LCFS.
# As we already have the value of these coefficients,
# we can use them to calculate the coil currents.
#
# We can then calculate an approximate value of the vacuum psi,
# and compare the LCFS from our starting equilibria to the LCFS
# of our the equilibria calculated from our SH approximation.
# This will allow us to choose the maximum degree necessary
# for an appropriate approximation.
#
# The steps are as follows:
#
# 1. Construct matrix from harmonic amplitudes and calculate
#   the necessary coil currents using the previously calculated
#   'psi_harmonic_amplitudes'.
#
#     - Construct matrix from harmonic amplitudes called
#       'currents2harmonics' (rows = degrees, columns = coils).
#       For each coil location (r, $\theta$) and degree calculate the following:
#       $$ A_{l} = \frac{\mu_{0}}{2} (\frac{r_{typical}}{r_{coil}})^{l} sin(\theta)
#       \frac{ P^{1}_{l}cos(\theta) }{ \sqrt(l(l+1)) } $$
#
#     - Calculate the coil currents using 'np.linalg.lstsq'.
#       This function uses least-squares to find the vector x
#       that approximately solves the equation a @ x = b.
#       Where 'a' is our coefficient matrix 'currents2harmonics',
#       and 'b' is our dependant variable,
#       a.k.a, the psi harmonic amplitudes .
#
#
# 2. Calculate the total value of psi, find the XZ coordinates of the associated LCFS
# and compare to the original LCFS.
#
#     - The fit metric we use for the LCFS comparison is as follows:
#         fit metric value = total area within one but not both LCFSs /
#                                      (input LCFS area + approximation LCFS area)

# %%
# Set min to save some time
min_degree = 2
# Choose acceptable value for fit metric
# 0 = good, 1 = bad
acceptable = 0.05

for degree in np.arange(min_degree, max_degree):  # + 1):
    # Construct matrix from harmonic amplitudes for coils
    currents2harmonics = coil_harmonic_amplitude_matrix(
        eq.coilset,
        degree,
        r_t,
        sh_coil_names,
    )

    # Calculate necessary coil currents
    currents, _, _, _ = np.linalg.lstsq(
        currents2harmonics[1:, :], (psi_harmonic_amplitudes[1:degree]), rcond=None
    )

    # Calculate the coefficients (amplitudes) of spherical harmonics
    # for use in optimising equilibria.
    coil_current_harmonic_amplitudes = currents2harmonics[1:, :] @ currents

    # Set currents in coilset
    for n, i in zip(sh_coil_names, currents):
        eq.coilset[n].current = i

    # Calculate the approximate Psi contribution from the coils
    # (including the non control coils)
    coilset_approx_psi = eq.coilset.psi(eq.grid.x, eq.grid.z)

    # We only wish to plot the contribution from the coils used in the approximation
    coilset_approx_psi_plot = np.zeros(np.shape(eq.grid.x))
    for n in sh_coil_names:
        coilset_approx_psi_plot = np.sum(
            [coilset_approx_psi_plot, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0
        )

    # Total
    approx_total_psi = coilset_approx_psi + plasma_psi
    eq.get_OX_points(approx_total_psi, force_update=True)

    # Get LCFS for approximation
    approx_LCFS = eq.get_LCFS(psi=approx_total_psi)

    # Compare staring equilibrium to new approximate equilibrium
    fit_metric_value = lcfs_fit_metric(original_LCFS, approx_LCFS)

    if fit_metric_value <= acceptable:
        print("fit metric = ", fit_metric_value, "degree required = ", degree)
        break
    elif degree == max_degree:
        print(
            "Oh no you need to use more degrees! Add some more collocation points"
            " please."
        )
        print("fit metric = ", fit_metric_value)

# Plot

x_plot = eq.grid.x
z_plot = eq.grid.z

nlevels = 50
levels = np.linspace(np.amin(total_psi), np.amax(total_psi), nlevels)
cmap = "viridis"

plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
plot2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=1)
plot3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1)

plot1.set_title("Vacuum Psi")
plot2.set_title("SH Approx Psi")
plot3.set_title("Coilset Approx Psi")

plot1.contour(x_plot, z_plot, vacuum_psi, levels=levels, cmap=cmap, zorder=8)
plot1.plot(x_bdry, z_bdry, color="red")

plot2.contour(x_plot, z_plot, approx_psi_vac_data, levels=levels, cmap=cmap, zorder=8)
plot2.plot(x_bdry, z_bdry, color="red")

plot3.contour(
    x_plot, z_plot, coilset_approx_psi_plot, levels=levels, cmap=cmap, zorder=8
)
plot3.plot(x_bdry, z_bdry, color="red")

plot1.set_xlim(np.min(x_plot), np.max(x_plot))
plot2.set_xlim(np.min(x_plot), np.max(x_plot))
plot3.set_xlim(np.min(x_plot), np.max(x_plot))

plt.show()

# %% [markdown]
# ### Compare the difference
#
# Compare the difference between the coilset contribution from our
# starting equilibria and our approximations.
# There should be minimal differences between the psi values inside the LCFS.
# The psi values outside the LCFS would be allowed to vary during optimisation.
#
# In the plot below, we include our original LCFS (solid red line)
# and the LCFS found using our approximation (dashed blue line).
#
# The coil current harmonic amplitudes used in the approximation
# can be used as constraints.

# %%
x_lcfs = approx_LCFS.x
z_lcfs = approx_LCFS.z

# Plot

x_plot = eq.grid.x
z_plot = eq.grid.z

nlevels = 50
levels = np.linspace(np.amin(total_psi), np.amax(total_psi), nlevels)
cmap = "viridis"

plot1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=1)
plot2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, colspan=1)

plot1.set_title("Vacuum - SH Approx Psi")
plot2.set_title("Vacuum - Coilset Approx Psi")

plot1.contour(
    x_plot, z_plot, vacuum_psi - approx_psi_vac_data, levels=levels, cmap=cmap, zorder=8
)
plot1.plot(x_bdry, z_bdry, color="red")

plot2.contour(
    x_plot,
    z_plot,
    vacuum_psi - coilset_approx_psi_plot,
    levels=levels,
    cmap=cmap,
    zorder=8,
)
plot2.plot(x_bdry, z_bdry, color="red")
plot2.plot(x_lcfs, z_lcfs, color="blue", linestyle="dashed")

plt.show()
