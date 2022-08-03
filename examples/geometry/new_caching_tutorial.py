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

# %%[markdown]
# ## Introduction

# Geometry is not plasma physics, but it isn't trivial either. Chances are most of
# your day-to-day interaction with bluemira will revolve around geometry in some form
# or another. Puns intended.

# There a few basic concepts you need to familiarise yourself with:
# * Basic objects: [`BluemiraWire`, `BluemiraFace`, `BluemiraShell`, `BluemiraSolid`]
# * Basic properties
# * Matryoshka structure
# * Geometry creation
# * Geometry modification
# * Geometry operations

# ## Imports

# Let's start out by importing all the basic objects, and some typical tools

# %%
import numpy as np

# Some display functionality
from bluemira.display import plot_2d, show_cad
from bluemira.display.displayer import DisplayCADOptions

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.tools import (
    make_face,
    make_polygon,
    extrude_shape,
    boolean_fuse,
    serialize_shape,
    deserialize_shape
)

# Basic objects
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane

from bluemira.geometry.parameterisations import (
    PictureFrame,
    PolySpline,
    PrincetonD,
    TripleArc,
)

wire1 = make_polygon(
    [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]
    ],
    label="wire1"
)
wire2 = make_polygon(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
)

wire_fuse = boolean_fuse([wire1, wire2])
print(wire_fuse)

wire1 = make_polygon(
    [
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ],
    label="wire1"
)
wire2 = make_polygon(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ],
)

wire_fuse = boolean_fuse([wire1, wire2])
print(wire_fuse)
print(wire_fuse.is_closed())


wire1 = make_polygon(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0.5, 1, 0]
     ],
    label="wire1"
)
wire2 = make_polygon(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ],
)




# wire1 = make_polygon(
#     [[0, 1, -1], [0, 0, 1], [0, 0, -1]],
#     label="wire1"
# )
# wire2 = make_polygon(
#     [
#         [1, 0, 0],
#         [1, 1, 0],
#         [0, 0, 0],
#     ],
# )

from bluemira.geometry.coordinates import Coordinates
square_points = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
]

points2 = [
    (1.0, 1.5, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.5, 0.0),
]

points = square_points
points.append(square_points[0])
coord = Coordinates(points)
wire1 = make_polygon(coord.points[0:4], label="wire1")

coord2 = Coordinates(points2)
wire2 = make_polygon(coord2.points, label="wire2")

from bluemira.display.plotter import FacePlotter, WirePlotter
import bluemira.display as display
from bluemira.geometry.placement import BluemiraPlacement

# wplotter = WirePlotter()
# wplotter.options.show_points = True
# wplotter.options.wire_options = {'color': 'black', 'linewidth': 2.5, 'zorder': 20}
# wplotter.options.view = BluemiraPlacement(label="xyz")
# ax = wplotter.plot_2d(wire1, show=False)
# wplotter.plot_2d(wire2, ax = ax)
#
# # wire = BluemiraWire([wire1, wire2], label="wire")
#
# from bluemira.geometry.parameterisations import TripleArc, PictureFrame
# curve = TripleArc().create_shape()
#
# wplotter = WirePlotter()
# wplotter.options.wire_options = {'color': 'black', 'linewidth': 2.5, 'zorder': 20}
# wplotter.plot_2d(curve)
#
# p = PictureFrame()
# # wire = p.create_shape()
# # wplotter.plot_2d(wire)
#
# p.adjust_variable("x1", value=4, lower_bound=4, upper_bound=5)
# p.adjust_variable("x2", value=16, lower_bound=14, upper_bound=18)
# p.adjust_variable(
#     "z1",
#     value=8,
#     lower_bound=5,
#     upper_bound=15,
# )
# p.adjust_variable(
#     "z2",
#     value=-8,
#     lower_bound=-15,
#     upper_bound=-5,
# )
# p.adjust_variable("ri", value=0, lower_bound=0, upper_bound=2)
# p.adjust_variable("ro", value=0, lower_bound=0, upper_bound=5)
# wire = p.create_shape()
# wplotter.plot_2d(wire)

base = np.array([0,0,0])
# create a random axis. A constant value has been added to avoid [0,0,0]
axis = np.array([1,0,0])
plane = BluemiraPlane(base, axis)
lx = 20
ly = 10
bmface = plane.to_face(lx, ly)

fplotter = FacePlotter()

fplotter.plot_3d(bmface)

buffer = cadapi.serialize_shape(bmface.shape)
face = cadapi.deserialize_shape(buffer)

bmbuffer = serialize_shape(bmface)
print(bmbuffer)
bmface1 = deserialize_shape(bmbuffer)