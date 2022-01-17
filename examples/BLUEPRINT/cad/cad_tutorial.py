# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
A little tutorial on how to make CAD in BLUEPRINT.

The cad module is basically a wrapper around an existing python interface to
a much lower level library (OCE).

With it, you can convert geometry objects (Loop, Shell) into CAD objects,
using terminology familiar to those who have used CAD programs in the past.

You are meant to step your way through each line of this script, and
introspect the objects you are creating along the way.
Feel free to change parameters!
"""

# %%
import os

from bluemira.base.file import get_bluemira_path
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    extrude,
    make_axis,
    make_circle,
    make_face,
    revolve,
    save_as_STEP,
    show_CAD,
    translate_shape,
)
from BLUEPRINT.geometry.loop import Loop

# %%[markdown]
# Let's say you want to make some 3-D shapes to impress your boss.
# You will need:
# 1.  An idea of what it you want to make
# 2.  An idea of how it is you will make it
# 3.  Some helpful tools in order to get the job done
#
# The basic idea behind any 3-D CAD is to start with some primitives (points,
# lines, splines, etc.) to make 2-D objects, to then make 3-D objects
#
# A lot of this module simplifies out the first two steps, leaving you to worry
# about what you want to make.
#
# ## MAKE A CUBE
#
# Let's say we want to make a cube. Here's one way of doing it:
# 1.  make a square of size L
# 2.  extrude the square by length L
#
# ### Step 1: make a square
#
# For this we use a geometry object: Loop, which is a collection of coordinates

# %%
square = Loop(x=[2, 4, 4, 2, 2], z=[2, 2, 4, 4, 2])

print(f"square.x: {square.x}")
print(f"square.z: {square.z}")

# %%[markdown]
# Good, so these are the same as specified. But what about the y-dimension?

# %%
print(f"square.y: {square.y}")

# %%[markdown]
# It is auto-populated to an array of zeros. This kind of thing is important
# if you care where your cube is going to be.
#
# Now, we need to make a 2-D CAD representation of the square. This kind of
# object we will call a "face". "make_face" takes a Loop object

# %%
face = make_face(square)

# %%[markdown]
# Now let's say you want to look at your square face.

# %%
show_CAD(face)

# %%[markdown]
# ### Step 2: extrude the square
#
# We use the extrude function for this, and there are different ways of
# specifying the extrusion
#
# 2.1: specifying the length and axis

# %%
cube1 = extrude(face, length=2, axis="y")

# %%[markdown]
# 2.1: with the "vec" argument,

# %%
cube2 = extrude(face, vec=[0, 2, 0])

# %%[markdown]
# Let's check that these produced the same result...

# %%
show_CAD(cube1, cube2)

# %%[markdown]
# Huh? only one cube? They are on top of each other!
# So let's move one away a little bit

# %%
cube2 = translate_shape(cube2, [4, 0, 0])

show_CAD(cube1, cube2)

# %%[markdown]
# ## MAKE A TORUS
#
# Let's say we want to make a torus. Here's one way of doing it:
# 1.  make a circle of radius R2, at centre (0, R1)
# 2.  revolve the circle by 360 degrees

# %%
R1 = 9
R2 = 3
angle = 360

# %%[markdown]
# ### Step 1: make a circle
#
# For this we have to proceed a little differently, as making a circle with
# lots of individual points (like in a Loop object) isn't very good for CAD.
# We use a direct implementation of a circle in OCC/OCE. This directly gives
# us a face object

# %%
face = make_circle(centre=[R1, 0, 0], direction=[0, 1, 0], radius=R2)

# %%[markdown]
# (note the 3-D coordinate interface)

# %%
show_CAD(face)

# %%[markdown]
# ## Step 2: revolve the circle
#
# But... about what? We need to make an axis object

# %%
axis = make_axis([0, 0, 0], [0, 0, 1])  # about the z-axis

torus = revolve(face, axis)

show_CAD(torus)

# %%[markdown]
# ## BOOLEAN OPERATIONS
#
# Let's say your boss is really impressed by lots of CAD
# You're going to need to stick your CAD bits together...
#
# Let's take our torus, make a copy, move that to the side a little, and stick
# them together... to make a doublet torus shape

# %%
torus2 = translate_shape(torus, [0, 0, 2.5])

doublet = boolean_fuse(torus, torus2)

show_CAD(doublet)

# %%[markdown]
# What about the opposite result?

# %%
cutlet = boolean_cut(torus, torus2)

show_CAD(cutlet)

# %%[markdown]
# What about more complex shapes?
#
# ## MAKE A SPLINY SHAPE
#
# For this we're going to load some Loop shapes from files

# %%
path = get_bluemira_path("BLUEPRINT/cad/test_data", subfolder="tests")
name = "plasmaloop.json"
filename = os.sep.join([path, name])

plasma = Loop.from_file(filename)

# %%[markdown]
# Let's have a look

# %%
plasma.plot()

# %%[markdown]
# OK, but how many points are we dealing with here?

# %%
print(f"number of points in plasma: {len(plasma)}")

# %%[markdown]
# That's starting to be a lot... what does it mean in practice?

# %%
plasma.plot(points=True)

# %%[markdown]
# When we make a face from a Loop, it draws lines between all the individual
# points. Let's try it:

# %%
face = make_face(plasma)

# %%[markdown]
# Just extrude it a little bit to see a bit more of what is going on...

# %%
plasma_block = extrude(face, vec=[0, 5, 0])

show_CAD(plasma_block)

# %%[markdown]
# OK, so that's some really nasty CAD... Lines everywhere. Large object/file size
# What can we do about it?
#
# In general, for curvy shapes, Bezier splines are much better for CAD than
# lots of points

# %%
face = make_face(plasma, spline=True)
show_CAD(face)

# %%
plasma_block2 = extrude(face, vec=[0, 5, 0])
plasma_block2 = translate_shape(plasma_block2, [7, 0, 0])
show_CAD(plasma_block, plasma_block2)

# %%[markdown]
# We can save the CAD boundary representation (BRep) objects as STP files:

# Note that if you are using WSL on Windows, the part is going to be saved in your
# Ubuntu environment
# %%
path = os.getcwd()
filename = os.sep.join([path, "plasma_block_test"])
save_as_STEP(plasma_block2, filename)

# %%[markdown]
# You can check it was saved by typing: `explorer.exe .` in your ubuntu terminal, and
# navigating to the file path above.

# You can view the file in any CAD program, like FreeCAD..
# %%[markdown]
# OK, now: have some fun:
#
# ## FREEFORM or CORED PLASMA
#
# Play around as you like, or:
# 1.  Revolve the 2-D plasma profile to make a half-plasma
# 2.  Take your previous torus shape (or make a new one) and hollow out the
#     plasma
