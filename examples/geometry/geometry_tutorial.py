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
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid

# Some useful tools
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    make_bspline,
    make_circle,
    make_polygon,
    revolve_shape,
    save_as_STEP,
    sweep_shape,
)

# Basic objects
from bluemira.geometry.wire import BluemiraWire

# %%[markdown]

# ## Make a cylinder

# There are many ways to make a cylinder, but perhaps the simplest way is as follows:
# * Make a circular Wire
# * Make a Face from that Wire
# * Extrude that Face along a vector, to make a Solid

# %%
# Note that we are going to give these geometries some labels, which
# we might use later.
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

# %%[markdown]

# ## Simple properties and representations

# %%
# Let's start off with some simple properties
print(f"Circle length: {circle_wire.length} m")
print(f"Circle area: {circle_face.area} m^2")
print(f"Cylinder volume: {cylinder.volume} m^3")

# You can also just print or repr these objects to get some useful info
print(cylinder)

# %%[markdown]
# ## Display

# Geometry objects can be displayed via `show_cad`, and the appearance
# of said objects customised by specifying `color` and `transparency`.

# %%
show_cad(cylinder, DisplayCADOptions(color="blue", transparency=0.1))

# %%[markdown]
# ## Matryoshka structure

# Bluemira geometries are structured in a commonly used "Matryoshka" or
# "Russian doll"-like structure.

# Solid -> Shell -> Face -> Wire

# These are accessible via the boundary attribute, so, in general, the boundary
# of a Solid is a Shell or set of Shells, and a Shell will have a set of Faces, etc.

# Let's take a little peek under the hood of our cylinder

# %%
print(f"Our cylinder is a BluemiraSolid: {isinstance(cylinder, BluemiraSolid)}")

i, j, k = 0, 0, 0  # This is just to facilitate comprehension
for i, shell in enumerate(cylinder.boundary):
    print(f"Shell: {i}.{j}.{k} is a BluemiraShell: {isinstance(shell, BluemiraShell)}")
    for j, face in enumerate(shell.boundary):
        print(f"Face: {i}.{j}.{k} is a BluemiraFace: {isinstance(face, BluemiraFace)}")
        for k, wire in enumerate(face.boundary):
            print(
                f"Wire: {i}.{j}.{k} is a BluemiraWire: {isinstance(wire, BluemiraWire)}"
            )

# %%[markdown]
# OK, so a cylinder is pretty simple, but more complicated shapes
# will follow the same pattern.

# It does go deeper than this, but that is outside the intended
# user-realm.

# %%[markdown]
# ## Geometry creation

# Let's get familiar with some more ways of making geometries. We've
# looked at circle already, but what else is out there:
# * polygons
# * splines
# * a bit of everything

# %%
# Polygon
theta = np.linspace(0, 2 * np.pi, 6)
x = 5 * np.cos(theta)
y = np.zeros(6)
z = 5 * np.sin(theta)

points = np.array([x, y, z])
pentagon = make_polygon(points)

plot_2d(pentagon)

# %%[markdown]
# Polygons are good for things with straight lines.
# Circles you've met already.
# For everything else, there's splines.

# Say you have a weird shape, that you might calculate via a equation.
# It's not a good idea to make a polygon with lots of very small sides
# for this. It's computationally expensive, and it will look ugly.

# %%
# Spline

x = np.linspace(0, 10, 1000)
y = 0.5 * np.sin(x) + 3 * np.cos(x) ** 2
z = np.zeros(1000)

points = np.array([x, y, z])
spline = make_bspline(points)
points = np.array([x, y + 3, z])
polygon = make_polygon(points)

show_cad(
    [spline, polygon], [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")]
)

# %%
# To get an idea of why polygons are bad / slow / ugly, try:
vector = (0, 0, 1)
show_cad(
    [extrude_shape(spline, vector), extrude_shape(polygon, vector)],
    [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")],
)


# %%
# There is nothing stopping you from combining different primitives, though!

radius = 2
part_circle = make_circle(radius=radius, start_angle=0, end_angle=270)

points = np.array([[radius, 0, 0], [0, 0, -radius], [0, 0, 0]])
closure = make_polygon(points)

my_shape = BluemiraWire([part_circle, closure])

# Let's just check we got that right...
print(f"My shape is closed: {my_shape.is_closed()}")

show_cad(BluemiraFace(my_shape))

# %%[markdown]
# ## Geometry operations: Part 1
# Making 3-D shapes from 2-D shapes

# You can:
# * extrude a shape `extrude_shape`, as we did with our cylinder
# * revolve a shape `revolve_shape`
# * sweep a shape `sweep_shape`

# %%
# Make a hollow cylinder, by revolving a rectangle
points = np.array([[4, 5, 5, 4], [0, 0, 0, 0], [2, 2, 3, 3]])
rectangle = BluemiraFace(make_polygon(points, closed=True))

hollow_cylinder = revolve_shape(
    rectangle, base=(0, 0, 0), direction=(0, 0, 1), degree=360
)

show_cad(hollow_cylinder)

# %%
# Sweep a profile along a path

points = np.array([[4.5, 4.5], [0, 3], [2.5, 2.5]])
straight_line = make_polygon(points)
quarter_turn = make_circle(center=(3, 3, 2.5), axis=(0, 0, 1), radius=1.5, end_angle=90)
path = BluemiraWire([straight_line, quarter_turn])
solid = sweep_shape(rectangle.boundary[0], path)
show_cad(solid)

# %%[markdown]
# ## Geometry operations: Part 2
# Making 3-D shapes from 3-D shapes

# Boolean operations often come in very useful when making CAD.
# * You can join geometries together with `boolean_fuse`
# * You can cut geometries from one another with `boolean_cut`

# %%
points = np.array(
    [
        [0, 2, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 3],
    ]
)

box_1 = BluemiraFace(make_polygon(points, closed=True))
box_1 = extrude_shape(box_1, (0, 2, 0))

points = np.array(
    [
        [1, 3, 3, 1],
        [0, 0, 0, 0],
        [0, 0, 2, 2],
    ]
)

box_2 = BluemiraFace(make_polygon(points, closed=True))
box_2 = extrude_shape(box_2, (0, 1, 0))

fused_boxes = boolean_fuse([box_1, box_2])

show_cad(fused_boxes)

cut_box_1 = boolean_cut(box_1, box_2)[0]

show_cad(cut_box_1)

# %%[markdown]
# ## Modification of existing geometries

# Now we're going to look at some stuff that we can do to change
# geometries we've already made.
# * Rotate
# * Translate
# * Scale

# %%
# Let's save a deepcopy of a shape before modifying
new_cut_box_1 = cut_box_1.deepcopy()

new_cut_box_1.rotate(base=(0, 0, 0), direction=(0, 1, 0), degree=45)
new_cut_box_1.translate((0, 3, 0))
new_cut_box_1.scale(3)
blue_red_options = [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")]
show_cad([cut_box_1, new_cut_box_1], options=blue_red_options)

# %%[markdown]
# ## Exporting geometry

# At present, only the STEP Assembly format is supported
# for exporting geometry.

# %%
# Try saving any shape or group of shapes created above
# as a STEP assembly

my_shapes = [cut_box_1]
# Modify this file path to where you want to save the data.
my_file_path = "my_tutorial_assembly.STP"
save_as_STEP(my_shapes, filename=my_file_path, scale=1)
