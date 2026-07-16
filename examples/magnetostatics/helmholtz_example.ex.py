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
Simple HelmholzCage example with different current sources.
"""

# %% [markdown]
# # Simple HelmholtzCage example
# ## Introduction
#
# In this example we will build some HelmholtzCages with different types of current
# sources.

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.display import plot_2d
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, offset_wire
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource

# %% [markdown]
# # Set up some geometry and key parameters

# %%
n_TF = 12
current = 20e6
breadth = 0.5
depth = 0.5
radius = 6
x_c = 9
z_c = 0

circle = make_circle(radius, center=(x_c, 0, z_c), axis=(0, 1, 0))
inner_circle = make_circle(radius - breadth, center=(x_c, 0, z_c), axis=(0, 1, 0))
outer_circle = make_circle(radius + breadth, center=(x_c, 0, z_c), axis=(0, 1, 0))
tf_xz_shape = BluemiraFace([outer_circle, inner_circle])

# %% [markdown]
# # Make a Biot-Savart filament (which needs to be properly discretised)

# %%
n_filaments_x = 2
n_filaments_y = 3
fil_radius = 0.5 * (breadth + depth) / (n_filaments_x * n_filaments_y)

filaments = []
filaments = []
dx_offsets = np.linspace(-breadth / 2, breadth / 2, n_filaments_x)
dy_offsets = np.linspace(-depth / 2, depth / 2, n_filaments_y)

for dx in dx_offsets:
    for dy in dy_offsets:
        new_loop = offset_wire(circle, dx)
        new_loop.translate(vector=(0, dy, 0))
        coordinates = new_loop.discretise(ndiscr=50)
        coordinates.close()
        filaments.append(coordinates)

biotsavart_circuit = BiotSavartFilament(
    filaments, radius=fil_radius, current=current / (n_filaments_x * n_filaments_y)
)

biotsavart_circuit.plot()
plt.show()

# %% [markdown]
# # Make an analytical circuit with a rectangular cross-section comprised
# # of several trapezoidal prism elements

# %%
coordinates = circle.discretise(ndiscr=100, byedges=True)
analytical_circuit1 = ArbitraryPlanarRectangularXSCircuit(
    coordinates, breadth=breadth, depth=depth, current=current
)

analytical_circuit1.plot()
plt.show()

# %% [markdown]
# # Make an analytical circuit of a circle arc with a rectangular cross-section

# %%
analytical_circuit2 = CircularArcCurrentSource(
    [x_c, 0, z_c],
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    breadth=breadth,
    depth=depth,
    radius=radius,
    dtheta=360,
    current=current,
)

analytical_circuit2.plot()
plt.show()

# %% [markdown]
# # Pattern the three circuits into HelmholtzCages

# %%
biotsavart_tf_cage = HelmholtzCage(biotsavart_circuit, n_TF=n_TF)
analytical_tf_cage1 = HelmholtzCage(analytical_circuit1, n_TF=n_TF)
analytical_tf_cage2 = HelmholtzCage(analytical_circuit2, n_TF=n_TF)

# Let's look at the circular arc cage
analytical_tf_cage2.plot()
plt.show()

# %% [markdown]
# # Calculate the fields in the x-y and x-z planes for the three different
# # source terms

# %%
nx, ny = 50, 50
x = np.linspace(0, 18, nx)
y = np.linspace(-18, 0, ny)
xx1, yy = np.meshgrid(x, y, indexing="ij")

biotsavart_xy_fields = biotsavart_tf_cage.field(xx1, yy, np.zeros_like(xx1))
analytical_xy_fields = analytical_tf_cage1.field(xx1, yy, np.zeros_like(xx1))
analytical_xy_fields2 = analytical_tf_cage2.field(xx1, yy, np.zeros_like(xx1))

biotsavart_xy_fields = np.sqrt(np.sum(biotsavart_xy_fields**2, axis=0))
analytical_xy_fields = np.sqrt(np.sum(analytical_xy_fields**2, axis=0))
analytical_xy_fields2 = np.sqrt(np.sum(analytical_xy_fields2**2, axis=0))

nx, nz = 50, 51
x = np.linspace(0, 18, nx)
z = np.linspace(-10, 10, nz)
xx, zz = np.meshgrid(x, z, indexing="ij")

biotsavart_xz_fields = biotsavart_tf_cage.field(xx, np.zeros_like(xx), zz)
analytical_xz_fields = analytical_tf_cage1.field(xx, np.zeros_like(xx), zz)
analytical_xz_fields2 = analytical_tf_cage2.field(xx, np.zeros_like(xx), zz)

# Note we are going to look at |B|, not B_T
biotsavart_xz_fields = np.sqrt(np.sum(biotsavart_xz_fields**2, axis=0))
analytical_xz_fields = np.sqrt(np.sum(analytical_xz_fields**2, axis=0))
analytical_xz_fields2 = np.sqrt(np.sum(analytical_xz_fields2**2, axis=0))

# %% [markdown]
# # Calculate the toroidal field ripple in the x-z plane

# %%

analytical_xz2_fields = analytical_tf_cage2.field(xx, np.zeros_like(xx), zz)
analytical_xz2_fields = np.sqrt(np.sum(analytical_xz2_fields**2, axis=0))
ripple = analytical_tf_cage2.ripple(xx, np.zeros_like(xx), zz)[1]

# We're going to mask the ripple as it can become non-sensical outside the TF
# coil region (0 - 0) / (0 + 0)
ripple_masked = np.ma.masked_outside(ripple, 0, 30.0)
# %% [markdown]
#
# # Let's visualise the results


# %%


def plot_2d_cage_results(xz_fields, label="$B$ [T]"):
    """
    Plot utility for contours in 2-D projections in matplotlib.
    """
    b_max = np.amax(xz_fields)
    levels = np.linspace(0, b_max, 20)
    f, ax = plt.subplots()
    cm = ax.contourf(xx, zz, xz_fields, levels=levels, cmap="magma", zorder=-1)
    plot_2d(
        tf_xz_shape,
        options=PlotOptions(face_options={"color": "g", "alpha": 0.5}),
        ax=ax,
        show=False,
    )
    f = plt.gcf()
    cb0 = f.colorbar(cm)
    cb0.ax.set_title(label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")


def plot_3d_cage_results(cage, xz_fields, xy_fields):
    """
    Plot utility for contours in 3-D projections in matplotlib.
    """
    b_max = max(np.amax(xz_fields), np.amax(xy_fields))
    levels = np.linspace(0, b_max, 20)

    cage.plot()
    ax = plt.gca()

    cm = ax.contourf(
        xx1,
        yy,
        xy_fields,
        zdir="z",
        levels=levels,
        offset=0,
        alpha=0.8,
        zorder=-100,
        cmap="magma",
    )

    xz_fields_masked = np.ma.masked_where(zz < 0, xz_fields)
    ax.contourf(
        xx,
        xz_fields_masked,
        zz,
        zdir="y",
        levels=levels,
        offset=0,
        alpha=0.8,
        zorder=-100,
        cmap="magma",
    )
    f = plt.gcf()
    cb0 = f.colorbar(cm, shrink=0.46)
    cb0.ax.set_title("$B$ [T]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_ylabel("z [m]")
    plt.show()


# %%
plot_2d_cage_results(ripple_masked, label=r"$\delta_{TF}$ [%]")
plt.show()

# %% [markdown]
# # Compare the toroidal field from the current sources in 2-D and
# # along the midplane with the analytical 1/r model

# %%
plot_2d_cage_results(analytical_xz2_fields)
ax = plt.gca()
one_over_r_2d = analytical_tf_cage1.field(x_c, 0, 0)[1] * x_c * 1 / xx
ax.contour(
    xx, zz, one_over_r_2d, levels=20, cmap="viridis", linestyles="dashed", zorder=10
)
plt.show()

x_line = np.linspace(2.0, 18, 200)
b_t_trapezoids = analytical_tf_cage1.field(
    x_line, np.zeros_like(x_line), np.zeros_like(x_line)
)[1]
b_t_circular_arc = analytical_tf_cage2.field(
    x_line, np.zeros_like(x_line), np.zeros_like(x_line)
)[1]
one_over_r = analytical_tf_cage1.field(x_c, 0, 0)[1] * x_c * 1 / x_line
f, ax = plt.subplots()
ax.plot(x_line, b_t_trapezoids, label="Trapezoidal Prism")
ax.plot(
    x_line,
    b_t_circular_arc,
    label="Circular Arc",
)
ax.plot(x_line, one_over_r, label="1/r", linestyle="dashed")

ax.plot([2.5, 2.5], [-6, 27], color="k", ls="-.", alpha=0.5)
ax.plot([3.5, 3.5], [-6, 27], color="k", ls="-.", alpha=0.5)
ax.plot([15.5, 15.5], [-6, 27], color="k", ls="-.", alpha=0.5)
ax.plot([14.5, 14.5], [-6, 27], color="k", ls="-.", alpha=0.5)
ax.set_xlabel("x [m]")
ax.set_ylim([-5, 25])
ax.set_ylabel("$B_{T}$ [T]")
ax.legend()
plt.show()

# %% [markdown]
# # Plot the two cages and the results in the two planes

# %%
plot_3d_cage_results(biotsavart_tf_cage, biotsavart_xz_fields, biotsavart_xy_fields)
plot_3d_cage_results(analytical_tf_cage1, analytical_xz_fields, analytical_xy_fields)
plot_3d_cage_results(analytical_tf_cage2, analytical_xz_fields2, analytical_xy_fields2)
