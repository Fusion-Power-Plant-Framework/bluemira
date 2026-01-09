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
# # Example of using a Spherical Harmonic Approximation
#
# This example illustrates how to set up the bluemira spherical harmonics
# approximation class (SphericalHarmonicApproximation) which can be used as
# a core-coil decoupling approach to coilset current and position
# optimisation for spherical tokamaks.
#
# ### Premise
#
# Our equilibrium (Grad-Shafranov) solution, which determines the plasma shape,
# will not change if the coilset contribution to the poloidal field is kept
# the same in the region occupied by the core plasma (i.e. the region
# characterised by closed fux surfaces). If we constrain the coilset conribution
# to the core, the plasma should remain fixed while we optimise other aspects of
# the magnetic configuration.
#
# We can decompose the coilset contibution to the poiloidal field into
# Spherical Harmonics (SH) and use the associated amplitudes as a set of constraints.

# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.display.plotter import Zorder
from bluemira.equilibria.analysis import (
    EqAnalysis,
    MultiEqAnalysis,
    select_multi_eqs,
)
from bluemira.equilibria.coils import Coil, CoilSet, symmetrise_coilset
from bluemira.equilibria.diagnostics import (
    EqDiagnosticOptions,
    EqSubplots,
    FluxSurfaceType,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    SphericalHarmonicApproximation,
    SphericalHarmonicsParams,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.problem import (
    TikhonovCurrentCOP,
)
from bluemira.equilibria.profiles import BetaIpProfile, DoublePowerFunc
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %% [markdown]
# ## Equilibria and Coilset Set Up
#
# Below we set up a test equilibrium to use in this example, it is MAST-U like.
#
# First we set up the coilset class, then the grid, and finally the equilibrium
# profile information, before combining it all to set up an Equilibrium object to
# use as a starting point.
#
# Use of the Spherical Harmonic Approximation requires that you have a valid
# Grad-Shafranov solution, and so we do an initial optimisation using a set of
# magnetic targets to achieve the desired plasma boundary shape.

# %%
# Coilset set up
masty = {
    "xc": [
        0.19475,
        0.24849975,
        0.42625037,
        0.60125023,
        0.8432501,
        0.96224999,
        1.92205048,
        1.32175004,
        1.55675006,
        1.56500018,
        1.71500003,
        1.35444999,
        0.24849975,
        0.42625037,
        0.60125023,
        0.8432501,
        0.96224999,
        1.92205048,
        1.32175004,
        1.55675006,
        1.56500018,
        1.71500003,
        1.35444999,
    ],
    "zc": [
        0.0,
        1.22402805,
        1.57249993,
        1.735000015,
        1.982000055,
        1.4936999649999998,
        1.9499999849999998,
        1.467700005,
        1.4676999450000001,
        1.0956499,
        0.352150035,
        0.943414985,
        -1.22402805,
        -1.57249993,
        -1.735000015,
        -1.982000055,
        -1.4936999649999998,
        -1.9499999849999998,
        -1.467700005,
        -1.4676999450000001,
        -1.0956499,
        -0.352150035,
        -0.943414985,
    ],
    "dxc": [
        0.0,
        0.007000,
        0.036750,
        0.036750,
        0.036750,
        0.036750,
        0.022050,
        0.036750,
        0.036750,
        0.065000,
        0.065000,
        0.036750,
        0.007000,
        0.036750,
        0.036750,
        0.036750,
        0.036750,
        0.022050,
        0.036750,
        0.036750,
        0.065000,
        0.065000,
        0.036750,
    ],
    "dzc": [
        1.581,
        0.192378,
        0.036750,
        0.022050,
        0.022050,
        0.022050,
        0.044100,
        0.022050,
        0.022050,
        0.058500,
        0.058500,
        0.052750,
        0.192378,
        0.036750,
        0.022050,
        0.022050,
        0.022050,
        0.044100,
        0.022050,
        0.022050,
        0.058500,
        0.058500,
        0.052750,
    ],
    "coil_names": [
        "Solenoid",
        "PXU",
        "D1U",
        "D2U",
        "D3U",
        "DpU",
        "D5U",
        "D6U",
        "D7U",
        "P4U",
        "P5U",
        "P6U",
        "PXL",
        "D1L",
        "D2L",
        "D3L",
        "DpL",
        "D5L",
        "D6L",
        "D7L",
        "P4L",
        "P5L",
        "P6L",
    ],
    "current": [
        5000,
        1500,
        8300,
        -1400,
        4800,
        -1700,
        900,
        -300,
        -500,
        -2700,
        -7100,
        20,
        1500,
        8300,
        -1400,
        4800,
        -1700,
        900,
        -300,
        -500,
        -2700,
        -7100,
        20,
    ],
    "coil_types": [
        "CS",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
    ],
    "n_turns": [
        234,
        42,
        35,
        23,
        23,
        23,
        24,
        27,
        23,
        23,
        23,
        23,
        42,
        35,
        23,
        23,
        23,
        24,
        27,
        23,
        23,
        23,
        23,
    ],
}
coils = []
for (
    xi,
    zi,
    dxi,
    dzi,
    typei,
    namei,
) in zip(
    masty["xc"],
    masty["zc"],
    masty["dxc"],
    masty["dzc"],
    masty["coil_types"],
    masty["coil_names"],
    strict=False,
):
    coil = Coil(
        xi,
        zi,
        current=5000 if namei == "Solenoid" else 0,
        dx=dxi,
        dz=dzi,
        ctype=typei,
        name=namei,
    )
    coils.append(coil)
coilset = CoilSet(*coils)

# Set control coils
coilset.control = [
    "PXU",
    "D1U",
    "D2U",
    "D3U",
    "DpU",
    "D5U",
    "D6U",
    "D7U",
    "P4U",
    "P5U",
    "P6U",
    "PXL",
    "D1L",
    "D2L",
    "D3L",
    "DpL",
    "D5L",
    "D6L",
    "D7L",
    "P4L",
    "P5L",
    "P6L",
]

# Make symmetric
coilset = symmetrise_coilset(coilset)

# %%
# Grid set up
grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)

# Profile params
I_p = 6e5  # A
R_0 = 0.85  # m
A = 1.3
B_0 = 0.588  # T
betap = 0.3

# Profile set up
profiles = BetaIpProfile(
    betap=betap,
    I_p=I_p,
    R_0=R_0,
    B_0=B_0,
    shape=DoublePowerFunc([1.8, 1.2]),
)

# Equilibrium set up
eq = Equilibrium(coilset, grid, profiles, psi=None)


# Plot starting eq and coilset
_, ax = plt.subplots()
eq.plot(ax=ax)
coilset.plot(ax=ax)
plt.show()

# %%
# x-point constraints
x_xp = 0.6
z_xp = 1.1
x_point_u = FieldNullConstraint(x_xp, z_xp, tolerance=1e-6)
x_point_l = FieldNullConstraint(x_xp, -z_xp, tolerance=1e-6)

# isoflux constraints
x_omp, z_omp = 1.4, 0.0
x_imp, z_imp = 0.35, 0.0
x_p2 = [
    x_xp,
    x_imp,
    x_omp,
    1.2,
    1.2,
    0.85,
    0.75,
    x_imp,
    x_imp,
    x_imp,
    x_imp,
    0.85,
    0.75,
    0.45,
    0.45,
]
z_p2 = [
    -z_xp,
    z_imp,
    z_omp,
    0.7,
    -0.7,
    1.7,
    1.6,
    0.2,
    0.1,
    -0.1,
    -0.2,
    -1.7,
    -1.6,
    -1.8,
    1.8,
]
isoflux = IsofluxConstraint(
    x_p2,
    z_p2,
    x_xp,
    z_xp,
    tolerance=1e-6,
    constraint_value=0.0,
)

# Create set to use as magnetic targets
constraints_set = MagneticConstraintSet([isoflux, x_point_u, x_point_l])

# Plot constraints over set up eq
_, ax = plt.subplots()
eq.plot(ax=ax)
constraints_set.plot(ax=ax)
coilset.plot(ax=ax)
plt.show()

# %%
# Set up Current Optimisation Problem (COP)
current_opt_problem = TikhonovCurrentCOP(
    eq,
    targets=MagneticConstraintSet([isoflux]),
    gamma=5e-7,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    opt_parameters={},
    max_currents=1e6,
    constraints=[isoflux, x_point_u, x_point_l],
)

# Run using Picard
program = PicardIterator(
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1.0e-4),
    relaxation=0.0,
    maxiter=30,
)
_ = program()

# Plot the result
f, ax = plt.subplots()
eq.plot(ax=ax)
constraints_set.plot(ax=ax)
coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# ## Using the Spherical Harmonic (SH) Approximation

# %% [markdown]
# ## Input Parameters
#
# ### Required
#
# - eq : Our chosen bluemira equilibrium
#
# ### Optional SphericalHarmonicsParams
#
# Parameters used in the spherical harmonic class are constrained
# in the input dataclass, if these are not chosen then the default
# values will be set.
#
# - n_points: Number of desired collocation points
# - point_type: How the collocation points are distributed
# - grid_num: The number of points in x-direction and z-direction,
#   to use with grid point distribution
# - psi_norm: 'None' will default to LCFS, otherwise choose the desired
#   normalised $\psi$ value of a closed flux surface that contain the core plasma
# - seed: Seed value to use with random point distribution
# - gamma_max: Maximum value of regularisation parameter to use in optimisation,
#  range of 0 to gamma_max is used
# - amplitude_variation_thresh: Maximum value for harmonic amplitude
# coefficient of variation, this is the threshold for significant harmonic selection
# - plot_find_significant_degrees: Whether or not to plot the results
#
# Note: gamma_max and amplitude_variation_thresh will need to be tuned for
# different device equilibria, pleas use plot_find_significant_degrees=True
# to find appropriate values.

# %%
# Set control coils - we are choosing to only use divertor coils
# here as we will be using the SH to constrain the core plasma,
# while changing the seperatix leg shape later in this notebook.
eq.coilset.control = [
    "D1U",
    "D2U",
    "D3U",
    "DpU",
    "D5U",
    "D6U",
    "D7U",
    "D1L",
    "D2L",
    "D3L",
    "DpL",
    "D5L",
    "D6L",
    "D7L",
]

# Choose setup parameters for SH approximation
params = SphericalHarmonicsParams(
    n_points=50,
    point_type=PointType.GRID_POINTS,
    grid_num=None,
    psi_norm=None,
    seed=None,
    gamma_max=1e-5,
    amplitude_variation_thresh=1e-2,
    plot_find_significant_degrees=True,
)

# %% [markdown]
# ## What happens in the SH appoximation class set up?
#
# The external coil contribution to the poloidal field within a core plasma region
# can be approximated using SH expansion. Consider an imaginary closed surface,
# centered at the origin of our modelled power plant coordinate system, which contains
# the core plasma to be constrained. The solution within this sphere due to the
# external axisymmetric current distribution written as,
#
# $$A(r,\theta) = \sum_{l=1}^{\infty} A_{l} (\frac{r}{r_{0}})^{l} \frac{P^{m=1}_{l}
# (\cos{\theta})}{\sqrt{l(l+1)}}$$
#
# where $r$ is distance from the origin, $\theta$ is poloidal angle, and $r{_t}$
# is a constant equal to a typical lengthscale, e.g., the radius of the sphere
# enclosing the plasma. $ A(r,\theta)$ is the scalar field $= \psi/R$ where $\psi$
# is the flux function, and R is the major radial coordinate ($=r \sin{\theta}$).
# The functions $P^{1}_{l}(\cos{\theta})$ are the associated Legendre polynomials
# of degree $l$ and order $m=1$. $A{_l}$ is the SH amplitude for a given degree.
#
# ### Get Approximation Region and Optimisable Coils
#
# We want to preserve the coil contribution to the poloidal $\psi$ inside the region
# of closed flux surfaces, i.e., the core plasma region.
#
# First we find our approximation region, which is a circular area
# (in the poloidal plane) that contains the core plasma, and is centred
# at (0, 0) on our cartesian grid coordinates.
# Note that, by default, the core region radial limit is set to contain
# the furthest radial extent of the Last Closed Flux Surface
# (LCFS, with a normalised psi of ~ 1.0) but users can choose an alternative
# closed flux surface by setting the normalised psi value input parameter.
#
# We cannot use coils that are within the sphere containing the core plasma
# for our optimisation, their contribution must remain unchanged, and so
# coils within this region are removed from our list of SH optimisable coils.
#
# ### Determine SH Approximation for the Coilset Contribution
#
# We want to approximate the coilset contribution to the poloidal $\psi$ within the
# chosen closed flux surface.
#
# We first select a set of collocation points within the chosen flux surface,
# these are locations within the core plasma at which we will sample $\psi$ and
# use to fit our approximation. Exactly how these points are spaced can be set using
# the associated input parameters. We then separate the $\psi$ contributions into
# 'plasma', 'coilset' and 'excluded coil' and obtain their values at the
# collocation points using the Bluemira $\psi$ functions for coils and plasma.
#
# Then we can construct our $\psi$ harmonic amplitude
# matrix using the following equation,
#
# $$ A_{l} = \frac{ r^{(l+1)} }{ r_{t}^{l} }
#           \frac{ P^{1}_{l}cos(\theta) }{ \sqrt(l(l+1)) } $$
#
# where $A_{l}$ is the harmonic amplitude for degree $l$,
# each collocation point had position ($r$, $\theta$),
# $r_{t}$ is a typical length scale (we use the radius of the chosen
# flux surface at the outer midplane), $P_{l} \cos{\theta}$ are Legendre polynomials of
# degree $l$ and order $m=1$.
#
# In an ideal world, we would use an infinite number of harmonics and achieve a perfect
# representation of $\psi$ rather than an approximation. However, we wish to have small
# set of constraints and so we choose to determine which harmonics
# have a more significant contribution to the approximation.
# To this end, we optimise for the harmonic amplitude
# values using Lasso as the objective function to be minimised.
#
# Lasso regularisation is equivalent to Ordinary Least Squares (OLS) with a penalty term
# $\gamma$. The aim is to reduce some of the values to zero, in order to select only the
# important features and ignore the less important ones. The objective function
# to minimise is given by,
#
# $$ ||Ax - b||^{2} + \gamma \sum_{j=1}^{p} |x_{j}| $$
#
# Where, in this particular case, $A$ is our harmonic matrix,
# $b$ is the vector of $\psi$ values at the collocation points,
# and $x$ is the vector of harmonic amplitudes
# necessary to represent $\psi$ using a sum of selected harmonics.
# A larger $\gamma$ imposes a greater penalty on the absolute value
# of magnitude of x, $\gamma = 0$ is equivalent to just using OLS.
#
# We scan over a range of $\gamma$ values (from 0 to a maximum value set
# in the input params) and run the optimisation for each.
# We then calculate the coefficient of variation for the set of optimised
# amplitudes for each degree. If this is sufficiently low (threshold
# can be set in input parameters) then the harmonic is considered
# significant and is included in our approximation (note that the amplitude
# values of the $\gamma=0$ optimisation are used,
# there should only be a tiny difference between amplitude values for different
# $\gamma$s if the harmonic is significant).

# %%
# Get SH approximation
sh_approx = SphericalHarmonicApproximation(
    eq,
    params,
)

# %% [markdown]
# ## Outputs
#
# The result of the approximation are constrained in the SphericalHarmonicsResult class.
#
# ### Results for use in optimisation
#
# - "coil_names", names of the coils that can be used with SH approximation
# - "amplitudes", SH amplitudes for required degrees
# - "degrees", selected degrees, the most significant degrees
# are found using optimisation within SphericalHarmonicApproximation
# -  "r_t", approximation length scale
#
# ### Informative outputs
#
# - "fit_metric_value", fit metric achieved - this is a comparison of the original
# closed flux surface to the flux surface obtained with the SH approximated $\psi$,
# smaller values mean a better fit
# - "sh_approx_coilset_psi", the coilset psi obtained using the SH approximation
# - "sh_approx_currents_coilset_psi", the coilset psi obtained using
# the approximated currents
# - "original_fs", coordinates of the plasma boundary from
# the approximation input equilibria
# - "approx_fs", coordinates of the plasma boundary found using the approximation

# %%
# Plot SH approximation
ax = sh_approx.plot()
for d, amp in zip(sh_approx.result.degrees, sh_approx.result.amplitudes, strict=False):
    print(f"degree {d} amplitude: {amp}")
print(f"fit metric for chosen flux surface: {sh_approx.result.fit_metric_value}")

# %%
# Psi from SH approximation
psi1 = sh_approx.result.sh_approx_coilset_psi
# Psi from coil currents found using the SH approximation
psi2 = sh_approx.result.sh_approx_currents_coilset_psi
# Psi from original coil currents
psi3 = sh_approx.psi_contributions.coilset_psi

# Add the excluded coils contribution
psi1 += sh_approx.psi_contributions.excluded_coil_psi
psi2 += sh_approx.psi_contributions.excluded_coil_psi
psi3 += sh_approx.psi_contributions.excluded_coil_psi

# Add the plasma contribution
psi1 += sh_approx.psi_contributions.plasma_psi
psi2 += sh_approx.psi_contributions.plasma_psi
psi3 += sh_approx.psi_contributions.plasma_psi

x_plot = eq.grid.x
z_plot = eq.grid.z
nlevels = 50
cmap = "viridis"
levels = np.linspace(np.amin(psi3), np.amax(psi3), nlevels)
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("SH coilset psi")
ax2.set_title("Psi from SH currents")
ax3.set_title("Orignial coilset psi")
ax1.set_aspect("equal")
ax2.set_aspect("equal")
ax3.set_aspect("equal")

ax1.contour(x_plot, z_plot, psi1, levels=levels, cmap=cmap, zorder=Zorder.PSI.value)
ax2.contour(x_plot, z_plot, psi2, levels=levels, cmap=cmap, zorder=Zorder.PSI.value)
ax3.contour(x_plot, z_plot, psi3, levels=levels, cmap=cmap, zorder=Zorder.PSI.value)

for ax in [ax1, ax2, ax3]:
    ax.plot(sh_approx.result.original_fs.x, sh_approx.result.original_fs.z, color="red")
    ax.plot(sh_approx.result.approx_fs.x, sh_approx.result.approx_fs.z, color="blue")
plt.show()

# %% [markdown]
# ## Building the SH Constraint
#
# The vector of harmonic amplitudes determined above can be written as a function
# of the current distribution outside of the sphere of approximation, enabling them
# to be used as constraints in a coilset optimisation.
#
# In Bluemira we can optimise the coil current $I_{c}$ and
# position ($r_{c}$, $\theta_{c}$). We use the following relation
# to create a control matrix for use in our constraint function,
#
# $$ A_{l} = \frac{\mu_{0}}{2} (\frac{r_{t}}{r_{c}})^{l} \sin\theta_{c}
# \frac{ P_{l}cos(\theta_{c}) }{ \sqrt(l(l+1)) } \times I_{c}$$
#
#
# where $A_{l}$ is the harmonic amplitude for degree $l$,
# $r_{t}$ is typical length scale, $P_{l} \cos{\theta}$
# are Legendre polynomials of degree $l$ and order $m=1$.

# %%
# Use results of the spherical harmonic approximation to create a set of coil constraints
sh_constraint = SphericalHarmonicConstraint(sh_approximation_result=sh_approx.result)
# Make sure we only optimise with coils outside the sphere containing the core plasma by
# setting control coils using the list of appropriate coils
eq.coilset.control = sh_approx.result.coil_names

# %% [markdown]
# ## Use in Optimisation Problem
#
# Now we will use the approximation to set up constraints for an optimisation problem.
# We use minimal current coilset optimisation problem with SH as the only constraints.
# This will try to minimise the sum of the currents squared while constraining the coil
# contribution to the core psi.

# %%
# x-point constraints
x_xp = 0.6
z_xp = 1.1
x_point_u = FieldNullConstraint(x_xp, z_xp, tolerance=1e-5)
x_point_l = FieldNullConstraint(x_xp, -z_xp, tolerance=1e-5)

# isoflux constraints
x_imp = 0.35
z_imp = 0.0

x_leg = [
    1.4,
    1.05,
    0.95,
    0.85,
    0.75,
    1.4,
    1.05,
    0.95,
    0.85,
    0.75,
]
z_leg = [
    1.95,
    1.82,
    1.76,
    1.7,
    1.6,
    -1.95,
    -1.82,
    -1.76,
    -1.7,
    -1.6,
]

x_leg = [
    1.4,
    0.75,
    1.4,
    0.75,
]
z_leg = [
    1.95,
    1.6,
    -1.95,
    -1.6,
]

isoflux = IsofluxConstraint(
    x_leg,
    z_leg,
    x_imp,
    z_imp,
    tolerance=1e-5,
    constraint_value=0.0,
)

constraints_set = MagneticConstraintSet([isoflux])

f, ax = plt.subplots()
eq.plot(ax=ax)
constraints_set.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %%
# Make a copy of the equilibria
sh_eq = deepcopy(eq)
# Set up a coilset optimisation problem using the spherical harmonic constraint
sh_tik_opt = TikhonovCurrentCOP(
    eq=sh_eq,
    targets=MagneticConstraintSet([sh_constraint]),
    gamma=1e-8,
    opt_algorithm="SLSQP",
    constraints=[x_point_u, x_point_l, isoflux],
)

# %%
# SOLVE
program = PicardIterator(
    sh_tik_opt,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-6),
    relaxation=0.0,
    maxiter=100,
)
program()

# %%
f, ax = plt.subplots()

sh_eq.plot(ax=ax)
sh_eq.coilset.plot(ax=ax)
constraints_set.plot(ax=ax)
ax.set_title("Coils Optimised")
plt.show()

# %%
# Give the starting eq and optimised eq some labels for plotting.
reference_eq = deepcopy(eq)
reference_eq.label = "Start Eq."
sh_eq.label = "Opt. Eq."

# Diagnostic settings for looking at the psi difference
diag_ops_1 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_REL_DIFF,
    split_psi_plots=EqSubplots.XZ,
)
# Diagnostic settings for looking at the relative psi difference,
# with the plasma and coilset psi contributions plotted separately.
# We have also added a mask so that we only plot values from inside
# the reference LCFS.
diag_ops_2 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_REL_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
    # plot_mask=EqPlotMask.IN_REF_LCFS,
)

# Create an EqAnalysis object for our optimised eq,
# with the starting eq as a reference.
eq_analysis = EqAnalysis(
    input_eq=sh_eq,
    reference_eq=reference_eq,
    diag_ops=diag_ops_1,
)

# %%
eq_analysis.plot_compare_psi()

# %%
equilibria_dictionary = select_multi_eqs(
    equilibrium_input=[reference_eq, sh_eq],
    equilibrium_names=["Ref.", "Opt."],
    control_coils=sh_eq.coilset.control,
)
# Now create the analysis class for multiple equilibria
multi_analysis = MultiEqAnalysis(equilibria_dictionary)

# %%
# Plot a selected flux surface from each equilibria
ax = multi_analysis.plot_compare_flux_surfaces(flux_surface=FluxSurfaceType.SEPARATRIX)

# %%
# Print a table comparing coilset information,
# the equilibria can have different coilsets.
# Note: when we defined multi_analysis, we set the control_coils to be PF type coils,
# so only coils with that type are printed for each equilibria.

# coilset_table = multi_analysis.coilset_info_table()

# %%
# Default value_type in coilset comparison table is coil current,
# but we can also choose from: x-position, z-position, coil field, and coil force.
# coilset_table = multi_analysis.coilset_info_table(value_type=CSData.B)
