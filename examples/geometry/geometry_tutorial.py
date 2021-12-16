# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                    J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A geometry tutorial for users.
"""

# %% [markdown]

## Introduction

# Geometry is not plasma physics, but it isn't trivial either. Chances are most of
# your day-to-day interaction with bluemira will revolve around geometry in some form
# or another. Puns intended.

# There a few basic concepts you need to familiarise yourself with:
# * Basic objects: [`BluemiraWire`, `BluemiraFace`, `BluemiraShell`, `BluemiraSolid`]
# * Matryoshka structure
# * Geometry creation
# * Geometry operations

## Imports

# Let's start out by importing all the basic objects, and some typical tools

# %%

from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import (
    make_circle,
    make_polygon,
    make_bspline,
    revolve_shape,
    extrude_shape,
    sweep_shape,
    boolean_cut,
    boolean_fuse,
)

# %%[markdown]

## Make a cylinder

# There are many ways to make a cylinder, but perhaps the simplest way is as follows:
# * Make a circular Wire
# * Make a Face from that Wire
# * Extrude that Face along a vector, to make a Solid

# %%

circle_wire = make_circle(
    radius=5,
    center=(0, 0, 0),
    axis=(0, 0, 1),
    start_angle=0,
    end_angle=360,
    label="my_wire",
)
circle_face = BluemiraFace(circle_wire, label="my_face")
cylinder = extrude_shape(circle_face, vec=(0, 0, 10), label="my_solid")
