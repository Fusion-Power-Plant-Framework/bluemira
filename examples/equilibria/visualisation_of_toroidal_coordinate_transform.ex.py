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
A notebook to demonstrate conversion between cylindrical and toroidal coordinate systems
using the Bluemira functions 'cylindrical_to_toroidal' and 'toroidal_to_cylindrical'.
"""

# %% [markdown]

# # Example of toroidal coordinate transform
#
# This notebook demonstrates conversion between cylindrical and toroidal coordinate
# systems using the Bluemira functions 'cylindrical_to_toroidal' and
# 'toroidal_to_cylindrical'.
#
# We denote toroidal coordinates by $(\tau, \sigma, \phi)$ and cylindrical coordinates by
#  $(R, z, \phi)$.
#

# %% [markdown]
# Imports
# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image

from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics import (
    cylindrical_to_toroidal,
    toroidal_to_cylindrical,
)

# %%
toroidal_image = Image.open("images/toroidal-coordinates-diagram-wolfram.png")
display(toroidal_image)

# %% [markdown]
# This diagram is taken from
# [Wolfram MathWorld](https://mathworld.wolfram.com/ToroidalCoordinates.html) and shows a
#  toroidal coordinate system. It uses $(u, v, \phi)$ whereas we use $(\tau, \sigma,
# \phi)$.
#
# In toroidal coordinates, surfaces of constant $\tau$ are non-intersecting tori of
# different radii, and surfaces of constant $\sigma$ are non-concentric spheres of
# different radii which intersect the focal ring.
# %% [markdown]
#
# We are working in the poloidal plane, so we set $\phi = 0$, and so are looking at a
# bipolar coordinate system. We are transforming about a focus $(R_0, z_0)$ in the
# poloidal plane.
#
# Here, curves of constant $\tau$ are non-intersecting circles of different radii that
# surround the focus and curves of constant $\sigma$ are non-concentric circles
# which intersect at the focus.
#
# To transform from toroidal coordinates to cylindrical coordinates about the focus in
# the poloidal plant $(R_0, z_0)$, we have the following relations:
#
# $$R = R_0 \frac{\sinh\tau}{\cosh\tau - \cos\sigma}$$
# $$z - z_0 = R_0 \frac{\sin\tau}{\cosh\tau - \cos\sigma}$$
#
# where we have $0 \le \tau < \infty$ and $-\pi < \sigma \le \pi$.
#
# The inverse transformations are given by:
#
# $$ \tau = \ln\frac{d_1}{d_2}$$
# $$ \sigma = \text{sign}(z - z_0) \arccos\frac{d_1^2 + d_2^2 - 4 R_0^2}{2 d_1 d_2}$$
#
# where we have
#
# $$ d_1^2 = (R + R_0)^2 + (z - z_0)^2$$
# $$ d_2^2 = (R - R_0)^2 + (z - z_0)^2$$


# %% [markdown]
# # Converting a unit circle
# We will start with an example of converting a unit circle in cylindrical coordinates to
#  toroidal coordinates and then converting back to cylindrical.
# This unit circle will be centered at the point (2,0) in the poloidal plane.
# %%
# Create a unit circle in cylindrical coordinates centered at (2,0) and plot
theta = np.linspace(-np.pi, np.pi, 100)
x = 2 + np.cos(theta)
y = np.sin(theta)
plt.plot(x, y)
plt.title("Unit circle centered at (2,0) in cylindrical coordinates")
plt.xlabel("R")
plt.ylabel("z")
plt.axis("square")
plt.show()

# Convert to toroidal coordinates and plot
tau_sig_list = cylindrical_to_toroidal(R=x, R_0=2, Z=y, z_0=0)
tau = tau_sig_list[0]
sigma = tau_sig_list[1]
plt.plot(tau, sigma)
plt.title("Unit circle converted to toroidal coordinates")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\sigma$")
plt.show()

# Convert back to cylindrical coordinates and plot
rzlist = toroidal_to_cylindrical(R_0=2, z_0=0, tau=tau, sigma=sigma)
rs = rzlist[0]
zs = rzlist[1]
plt.plot(rs, zs)
plt.title("Unit circle centered at (2,0) converted back to cylindrical coordinates")
plt.xlabel("R")
plt.ylabel("z")
plt.axis("square")


# %% [markdown]
# # Curves of constant $\tau$ and $\sigma$
# When plotting in cylindrical coordinates, curves of constant $\tau$ correspond to
# non-intersecting circles that surround the focus $(R_0, z_0)$, and curves of constant
# $\sigma$ correspond to non-concentric circles that intersect at the focus.
# 1. Curves of constant $\tau$ plotted in both cylindrical and toroidal coordinates:
# %%
# Define the focus point
R_0 = 1
z_0 = 0

# Create array of 6 tau values, 6 curves of constant tau will be plotted
tau = np.linspace(0.5, 2, 6)
sigma = np.linspace(-np.pi, np.pi, 200)

rlist = []
zlist = []
# Plot the curve in cylindrical coordinates for each constant value of tau
for t in tau:
    rzlist = toroidal_to_cylindrical(R_0=R_0, z_0=z_0, sigma=sigma, tau=t)
    rlist.append(rzlist[0])
    zlist.append(rzlist[1])
    plt.plot(rzlist[0], rzlist[1])

plt.axis("square")
plt.xlabel("R")
plt.ylabel("z")
plt.title(r"$\tau$ isosurfaces: curves of constant $\tau$ in cylindrical coordinates")
plt.show()

# Now convert to toroidal coordinates and plot. The curves of constant tau are now
# straight lines
taulist = []
sigmalist = []
for i in range(len(rlist)):
    tausiglist = cylindrical_to_toroidal(R_0=R_0, z_0=z_0, R=rlist[i], Z=zlist[i])
    taulist.append(tausiglist[0])
    sigmalist.append(tausiglist[1])
    plt.plot(tausiglist[0], tausiglist[1])

plt.xlabel(r"$\tau$")
plt.ylabel(r"$\sigma$")
plt.title(r"$\tau$ isosurfaces: curves of constant $\tau$ in toroidal coordinates")
plt.show()


# %% [markdown]
# 2. Curves of constant $\sigma$ plotted in both cylindrical and toroidal coordinates:
# %%
# Define the focus point
R_0 = 1
z_0 = 0

# Create array of 6 sigma values, 6 curves of constant sigma will be plotted
sigma = np.linspace(0.5, np.pi / 2, 6)
tau = np.linspace(0, 5, 200)

rlist = []
zlist = []
# Plot the curve in cylindrical coordinates for each constant value of sigma
for s in sigma:
    rzlist = toroidal_to_cylindrical(R_0=R_0, z_0=z_0, sigma=s, tau=tau)
    rlist.append(rzlist[0])
    zlist.append(rzlist[1])
    plt.plot(rzlist[0], rzlist[1])
plt.axis("square")
plt.xlabel("R")
plt.ylabel("z")
plt.title(
    r"$\sigma$ isosurfaces: curves of constant $\sigma$ in cylindrical coordinates"
)
plt.show()

# Now convert to toroidal coordinates and plot. The curves of constant sigma are now
# straight lines
taulist = []
sigmalist = []
for i in range(len(rlist)):
    tausiglist = cylindrical_to_toroidal(R_0=R_0, z_0=z_0, R=rlist[i], Z=zlist[i])
    taulist.append(tausiglist[0])
    sigmalist.append(tausiglist[1])
    plt.plot(tausiglist[0], tausiglist[1])

plt.xlabel(r"$\tau$")
plt.ylabel(r"$\sigma$")
plt.title(r"$\sigma$ isosurfaces: curves of constant $\sigma$ in toroidal coordinates")
plt.show()

# %%
