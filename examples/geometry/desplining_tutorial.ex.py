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
A desplining tutorial for users.
"""

# %% [markdown]
# # Desplining Tutorial
# ## Why do we need desplining in the first place?
#
# Technically, we do not always need to. Splines are obviously the better
# representation for smooth curved geometries.
#
# But some post-processing tools cannot handle splines properly, so sometimes
# we need to get rid of them. For example, when converting our CAD geometry to
# CSG geometry for OpenMC runs, we need simpler representations of the spline
# surfaces.
#
# Here, by *desplining*, we mean approximating a spline using a number of
# straight line segments. In other words, we simply discretise the boundaries
# of the XZ faces using a user-defined discretisation value, rebuild the XZ
# faces using these discretised boundaries, and then revolve them to create
# the XYZ faces.
#
# In this tutorial, we will go through this desplining process with a simple
# example. However, this process is not free from problems, so we will also
# look at some of these problems and how we can deal with them.
#
# Note that the faces used in this example do not represent any real reactor
# faces. They are simply a toy model created to demonstrate the desplining
# process.
#
# ## Imports
#
# Let's start out by importing all the basic objects, and some typical tools
# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import FacePlotter, plot_2d
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.despliner import despline_xz_face
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_common,
    interpolate_bspline,
    make_polygon,
    repair_gaps_between_faces,
    repair_overlapping_geos,
)
from bluemira.geometry.wire import BluemiraWire

# %% [markdown]
# Do not run this example for freecad backend
# %%
current_backend = os.getenv("BLUEMIRA_GEOMETRY_BACKEND", "cadquery").lower()

if current_backend != "cadquery":
    bluemira_warn(
        f"Skipping example: requires cadquery, current backend is {current_backend}"
    )
    sys.exit(0)  # Exit gracefully

# %% [markdown]
# ## Helper Functions
# The following helper functions will be useful throughout this tutorial.
# %%


def line(a, b):
    """Create a straight line between two points in the XZ plane.

    Parameters
    ----------
    a, b : tuple[float, float]
        Start and end points given as (x, z) coordinates.

    Returns
    -------
    BluemiraWire
        Straight wire connecting the two points.
    """
    return make_polygon({
        "x": [a[0], b[0]],
        "y": [0, 0],
        "z": [a[1], b[1]],
    })


def spline(p):
    """Create a B-spline through a set of points in the XZ plane.

    Parameters
    ----------
    p : array-like
        Points defining the spline, given as (x, z) coordinates.

    Returns
    -------
    BluemiraWire
        Interpolated B-spline wire.
    """
    p = np.array(p)

    return interpolate_bspline(
        Coordinates({
            "x": p[:, 0],
            "y": np.zeros(len(p)),
            "z": p[:, 1],
        })
    )


def plot_face(face, ax, color, alpha=0.65, *, show_vertices: bool = True):
    """Plot a face in the XZ plane.

    Parameters
    ----------
    face : BluemiraFace
        Face to plot.
    ax : matplotlib.axes.Axes
        Axes on which to plot the face.
    color : str
        Face colour.
    alpha : float, optional
        Face transparency.
    show_vertices : bool, optional
        Whether to show the boundary vertices.
    """
    q = FacePlotter()
    q.options.view = "xz"

    q.options.face_options = {
        "color": color,
        "alpha": alpha,
    }

    q.options.wire_options = {
        "color": "black",
        "linewidth": 1,
    }

    q.plot_2d(face, ax=ax, show=False)

    if show_vertices:
        for boundary in face.boundary:
            plot_2d(boundary.vertexes.T, ax=ax, show=False)


def visualise_desplining(original, desplined):
    """Visualise the original and desplined XZ faces side by side.

    Parameters
    ----------
    original : BluemiraFace
        Original face containing spline edges.
    desplined : BluemiraFace
        Face after desplining.
    """
    _, ax = plt.subplots(1, 2, figsize=(8, 4))

    plot_face(original, ax[0], "cornflowerblue")
    ax[0].set_title("Original")

    plot_face(desplined, ax[1], "salmon")
    ax[1].set_title("Desplined")

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("X")
        a.set_ylabel("Z")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Toy Model - 1 (One Single face)
#
# Let's make a simple XZ face whose boundary contains spline
# edges. We can use some of the helper functions above for that.
#
# First, let's make the XZ face from scratch
# and then create the face.
# %%
# Full right boundary of xz face
p = [
    (1.00, 0.00),
    (0.94, 0.50),
    (0.90, 1.00),
    (0.94, 1.50),
    (1.00, 2.00),
]

# Left boundary of xz face
p_left = [
    (0.00, 0.00),
    (-0.06, 0.50),
    (-0.10, 1.00),
    (-0.06, 1.50),
    (0.00, 2.00),
]

xz_face_1 = BluemiraFace(
    BluemiraWire([
        line((0, 0), (1, 0)),
        spline(p),
        line((1, 2), (0, 2)),
        spline(p_left[::-1]),
    ]),
    label="xz face 1",
)

# %% [markdown]
# ### Desplining Face 1
#
# Now let us think about how to despline Face 1 for, say, a neutronics
# run in the future. We can despline its XZ face using a chosen discretisation
# value and visualise the original and desplined geometries.
# %%
face_1_desplined = despline_xz_face(xz_face_1, 15)

visualise_desplining(
    xz_face_1,
    face_1_desplined,
)

# %% [markdown]
# ## Toy Model - 2 (Two faces touching each other)
#
# Let's make the problem a little harder. So far, we have looked at desplining
# a single face. Now let's consider two faces that are touching each
# other and share part of their boundary.
#
# Ideally, we would like to have the shared part of the boundary discretised
# in exactly the same way for both faces, so that their shared boundaries
# still match exactly after desplining.
#
# But since we are discretising the entire boundary (or boundaries) of each XZ
# face, rather than the shared part separately, specifying the same
# discretisation value for both faces does not necessarily guarantee this.
#
# Also, if the boundaries only partially touch, the shared part may represent
# only a small portion of the boundary of each face. In that case, even
# using the same discretisation value can result in different discretisations
# along the shared part.
#
# Let's see what happens in such a case. First create the second face
# %%

# Right boundary of face 2
p_right = [
    (1.60, 0.50),
    (1.54, 0.75),
    (1.50, 1.00),
    (1.54, 1.25),
    (1.60, 1.50),
]

# Small face touches only the middle part
p_shared = [
    (0.94, 0.50),
    (0.91, 0.75),
    (0.90, 1.00),
    (0.91, 1.25),
    (0.94, 1.50),
]


xz_face_2 = BluemiraFace(
    BluemiraWire([
        spline(p_shared),
        line(p_shared[-1], (1.60, 1.50)),
        spline(p_right[::-1]),
        line((1.60, 0.50), p_shared[0]),
    ]),
    label="xz face 2",
)

face_2_desplined = despline_xz_face(
    xz_face_2,
    10,
)

# %% [markdown]
# Let's visualise what happened
# %%
# Inspect if there is an overlapping region
overlap = boolean_common(face_1_desplined, face_2_desplined)

fig = plt.figure(figsize=(16, 6))

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1, 1, 0.5],
    wspace=0.28,
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax_zoom = fig.add_subplot(gs[2])


# Original
plot_face(xz_face_1, ax0, "cornflowerblue")
plot_face(xz_face_2, ax0, "salmon")
ax0.set_title("Original")


# Overlap after desplining
plot_face(face_1_desplined, ax1, "cornflowerblue")
plot_face(face_2_desplined, ax1, "salmon")

if overlap:
    for region in overlap:
        plot_face(region, ax1, "gold", alpha=1.0)

ax1.set_title("Overlap after desplining")


# Small overlap pocket
q = FacePlotter()
q.options.view = "xz"

q.options.face_options = {
    "color": "gold",
    "alpha": 1.0,
}

q.options.show_wires = False

if overlap:
    for region in overlap:
        q.plot_2d(region, ax=ax_zoom, show=False)

ax_zoom.set_title("Overlap\n(zoomed)", fontsize=8)
ax_zoom.set_aspect("equal")

ax_zoom.set_xlim(0.88, 1.02)
ax_zoom.set_ylim(0.65, 1.35)

ax_zoom.set_xticks([])
ax_zoom.set_yticks([])


# Main plots
for a in [ax0, ax1]:
    a.set_aspect("equal")
    a.set_xlabel("X")
    a.set_ylabel("Z")
    a.set_xlim(-0.5, 1.7)
    a.set_ylim(-0.1, 2.1)


plt.show()

# %% [markdown]
# ### Overlaps detected!
#
# Now we can clearly see that the different discretisations have introduced
# an overlap between the two faces.
#
# So, how do we solve this?
#
# One possible solution would be to directly cut one face with the other.
# For this simple toy model, the resulting boundary still looks reasonably
# well-behaved. However, for more complex geometries, a direct cut can produce
# undesirable features along the shared boundary.
#
# To avoid relying on a direct cut, our approach is to first fuse the
# geometries and then cut them. Let's see how this works.
#
# #### Option 1: Repair the desplined XZ face 2, giving face 1 priority
#
# We can choose to keep the desplined XZ face 1 as it is and
# repair the desplined XZ face 2 around it. In other words,
# face 1 gets priority, and face 2 is modified to remove the overlap.
#
# For the repair, we first fuse the two desplined XZ faces and then cut the
# desplined XZ face 1 from the fused geometry. This gives us a
# repaired XZ face for face 2 whose boundary now follows that of
# face 1 along the overlapping region.
# %%
face_2_repaired = repair_overlapping_geos(
    face_2_desplined,
    [face_1_desplined],
)

overlap = boolean_common(
    face_1_desplined,
    face_2_desplined,
)

fig = plt.figure(figsize=(16, 6))

gs = fig.add_gridspec(
    1,
    4,
    width_ratios=[1, 1, 0.18, 1],
    wspace=0.28,
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax_zoom = fig.add_subplot(gs[2])
ax2 = fig.add_subplot(gs[3])


# Original
plot_face(xz_face_1, ax0, "cornflowerblue")
plot_face(xz_face_2, ax0, "salmon")
ax0.set_title("Original")


# Overlap after desplining
plot_face(face_1_desplined, ax1, "cornflowerblue")
plot_face(face_2_desplined, ax1, "salmon")

if overlap:
    for region in overlap:
        plot_face(region, ax1, "gold", alpha=1.0)

ax1.set_title("Overlap after desplining")


# Small overlap pocket
q = FacePlotter()
q.options.view = "xz"

q.options.face_options = {
    "color": "gold",
    "alpha": 1.0,
}

q.options.show_wires = False

if overlap:
    for region in overlap:
        q.plot_2d(region, ax=ax_zoom, show=False)

ax_zoom.set_title("Overlap\n(zoomed)", fontsize=8)
ax_zoom.set_aspect("equal")

ax_zoom.set_xlim(0.88, 1.02)
ax_zoom.set_ylim(0.65, 1.35)

ax_zoom.set_xticks([])
ax_zoom.set_yticks([])


# Repaired
plot_face(face_1_desplined, ax2, "cornflowerblue")
plot_face(face_2_repaired, ax2, "mediumseagreen")
ax2.set_title("Repaired")


# Main plots
for a in [ax0, ax1, ax2]:
    a.set_aspect("equal")
    a.set_xlabel("X")
    a.set_ylabel("Z")
    a.set_xlim(-0.5, 1.7)
    a.set_ylim(-0.1, 2.1)

plt.show()

# %% [markdown]
# #### Option 2: Repair the desplined XZ face 1, giving face 2 priority
#
# Alternatively, we can keep the desplined XZ face 2 as it is and
# repair the desplined XZ face 1 around it. In this case,
# face 2 gets priority, and face 1 is modified to remove the overlap.
#
# As before, we first fuse the two desplined XZ faces and then cut the
# desplined XZ face 2 from the fused geometry. This gives us a
# repaired XZ face 1 whose boundary now follows that of
# face 2 along the overlapping region.
# %%
# %%
face_1_repaired = repair_overlapping_geos(
    face_1_desplined,
    [face_2_desplined],
)

fig = plt.figure(figsize=(16, 6))

gs = fig.add_gridspec(
    1,
    4,
    width_ratios=[1, 1, 0.18, 1],
    wspace=0.28,
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax_zoom = fig.add_subplot(gs[2])
ax2 = fig.add_subplot(gs[3])


# Original
plot_face(xz_face_1, ax0, "cornflowerblue")
plot_face(xz_face_2, ax0, "salmon")
ax0.set_title("Original")


# Overlap after desplining
plot_face(face_1_desplined, ax1, "cornflowerblue")
plot_face(face_2_desplined, ax1, "salmon")

if overlap:
    for region in overlap:
        plot_face(region, ax1, "gold", alpha=1.0)

ax1.set_title("Overlap after desplining")


# Small overlap pocket
q = FacePlotter()
q.options.view = "xz"

q.options.face_options = {
    "color": "gold",
    "alpha": 1.0,
}

q.options.show_wires = False

if overlap:
    for region in overlap:
        q.plot_2d(region, ax=ax_zoom, show=False)

ax_zoom.set_title("Overlap\n(zoomed)", fontsize=8)
ax_zoom.set_aspect("equal")
ax_zoom.set_xlim(0.88, 1.02)
ax_zoom.set_ylim(0.65, 1.35)
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])


# Repaired: face 2 gets priority
plot_face(
    face_1_repaired,
    ax2,
    "mediumseagreen",
)

plot_face(
    face_2_desplined,
    ax2,
    "salmon",
)

ax2.set_title("Repaired")


# Main plots
for a in [ax0, ax1, ax2]:
    a.set_aspect("equal")
    a.set_xlabel("X")
    a.set_ylabel("Z")
    a.set_xlim(-0.5, 1.7)
    a.set_ylim(-0.1, 2.1)

plt.show()

# %% [markdown]
# ### Which option to choose?
#
# For now, this totally depends on the user and which face they want to
# give priority to. In the neutronics module examples, we will discuss this in
# more detail and look at how to decide which face should be repaired.
# %%

# %% [markdown]
# ## Toy Model - 3 (Two faces touching each other, and desplining creates a gap)
#
# So far, we have seen how different discretisations of a shared boundary can
# create an overlap between two faces. But overlap is not the only problem
# we can get.
#
# Depending on the geometry and how the shared boundary is discretised, the
# opposite can also happen: the two desplined boundaries can move away from
# each other and create a gap.
#
# Let's create another simple toy model to look at such a case. As before, the
# two faces initially touch each other exactly. We will then despline them
# differently and see how a gap can appear between them.
#
# ### faces 3 and 4
# %%
# Shared boundary
p_shared_3 = [
    (1.00, 0.50),
    (1.06, 0.75),
    (1.10, 1.00),
    (1.06, 1.25),
    (1.00, 1.50),
]

# Left boundary of XZ face 3
p_left_3 = [
    (0.15, 0.00),
    (0.09, 0.50),
    (0.05, 1.00),
    (0.09, 1.50),
    (0.15, 2.00),
]

# Right boundary of XZ face 4
p_right_4 = [
    (1.60, 0.50),
    (1.66, 0.75),
    (1.70, 1.00),
    (1.66, 1.25),
    (1.60, 1.50),
]


# XZ face 3
face_3 = BluemiraFace(
    BluemiraWire([
        line((0.15, 0.00), (1.00, 0.00)),
        line((1.00, 0.00), (1.00, 0.50)),
        spline(p_shared_3),
        line((1.00, 1.50), (1.00, 2.00)),
        line((1.00, 2.00), (0.15, 2.00)),
        spline(p_left_3[::-1]),
    ]),
    label="face 3",
)


# XZ face 4
face_4 = BluemiraFace(
    BluemiraWire([
        line((1.00, 0.50), (1.60, 0.50)),
        spline(p_right_4),
        line((1.60, 1.50), (1.00, 1.50)),
        spline(p_shared_3[::-1]),
    ]),
    label="face 4",
)

# %% [markdown]
# Despline, and plot comparisons
# %%
face_3_desplined = despline_xz_face(
    face_3,
    20,
)

face_4_desplined = despline_xz_face(
    face_4,
    60,
)

overlap_3_4 = boolean_common(
    face_3_desplined,
    face_4_desplined,
)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))


# Original: touching exactly
plot_face(
    face_3,
    ax[0],
    "cornflowerblue",
    show_vertices=False,
)
plot_face(
    face_4,
    ax[0],
    "salmon",
    show_vertices=False,
)

ax[0].set_title("Original")


# After different desplining
plot_face(
    face_3_desplined,
    ax[1],
    "cornflowerblue",
    show_vertices=False,
)
plot_face(
    face_4_desplined,
    ax[1],
    "salmon",
    show_vertices=False,
)
# Highlight overlap in gold
if overlap_3_4:
    for region in overlap_3_4:
        plot_face(
            region,
            ax[1],
            "gold",
            alpha=1.0,
            show_vertices=False,
        )
ax[1].set_title("Gaps and overlaps after desplining")

# Zoomed-in gap
plot_face(
    face_3_desplined,
    ax[2],
    "cornflowerblue",
    alpha=1.0,
    show_vertices=False,
)
plot_face(
    face_4_desplined,
    ax[2],
    "salmon",
    alpha=1.0,
    show_vertices=False,
)

ax[2].set_title("Gap (zoomed in)")


# Main views
for a in ax[:2]:
    a.set_xlim(-0.1, 1.8)
    a.set_ylim(-0.1, 2.1)

# Zoom around shared boundary
ax[2].set_xlim(0.95, 1.13)
ax[2].set_ylim(0.65, 1.35)


for a in ax:
    a.set_aspect("equal")
    a.set_xlabel("X")
    a.set_ylabel("Z")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Sewing as a solution for gaps
#
# So, what do we do when desplining creates a gap as well as an overlap?
#
# In this case, after fixing overlaps, we can use sewing. The idea is to sew
# neighbouring boundaries that lie within a specified tolerance, so that the
# two desplined XZ faces recover a matching boundary where the gap was introduced.
#
# Let's see how this works for our two desplined faces. First, run the
# overlap fixer as before.
# %%
face_4_repaired = repair_overlapping_geos(
    face_4_desplined,
    [face_3_desplined],
)
# %% [markdown]
# Now, sew, and plot comparisons
# %%

xz_repaired_3, xz_repaired_4 = repair_gaps_between_faces(
    [
        face_3_desplined,
        face_4_repaired,
    ],
    tolerance=1e-2,
)
_, ax = plt.subplots(1, 4, figsize=(20, 5))

# Before repairing
plot_face(
    face_3_desplined,
    ax[0],
    "cornflowerblue",
    show_vertices=True,
)
plot_face(
    face_4_repaired,
    ax[0],
    "salmon",
    show_vertices=True,
)
ax[0].set_title("Before repairing")

# Before repairing - zoomed in
plot_face(
    face_3_desplined,
    ax[1],
    "cornflowerblue",
    show_vertices=True,
)
plot_face(
    face_4_repaired,
    ax[1],
    "salmon",
    show_vertices=True,
)
ax[1].set_title("Before repairing (zoomed in)")
ax[1].set_xlim(0.9, 1.2)
ax[1].set_ylim(0.75, 1.2)

# After repairing
plot_face(
    xz_repaired_3,
    ax[2],
    "cornflowerblue",
    show_vertices=True,
)
plot_face(
    xz_repaired_4,
    ax[2],
    "salmon",
    show_vertices=True,
)
ax[2].set_title("After repairing")

# After repairing - zoomed in
plot_face(
    xz_repaired_3,
    ax[3],
    "cornflowerblue",
    show_vertices=True,
)
plot_face(
    xz_repaired_4,
    ax[3],
    "salmon",
    show_vertices=True,
)
ax[3].set_title("After repairing (zoomed in)")
ax[3].set_xlim(0.9, 1.2)
ax[3].set_ylim(0.75, 1.2)

for a in ax:
    a.set_aspect("equal")
    a.set_xlabel("X")
    a.set_ylabel("Z")

plt.tight_layout()
plt.show()

# %% [markdown]
# We also see that the desplining features are retained after sewing, while
# the gap between the neighbouring faces is removed.
# %%
