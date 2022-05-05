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
A little tutorial on how to make geometry in BLUEPRINT and how to use
the most basic building blocks: Loop and Shell.

You are meant to step your way through each line of this script, and
introspect the objects you are creating along the way.
Feel free to change parameters!
"""

# %%[markdown]
# # Geometry Tutorial

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print
from BLUEPRINT.geometry.boolean import (
    boolean_2d_common,
    boolean_2d_difference,
    boolean_2d_union,
    simplify_loop,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell

plt.close("all")

# %%[markdown]
# ## BACKGROUND
#
# Geometry matters. It is not plasma physics, but it is not trivial.
# It is the source of most errors in BLUEPRINT, because of the number of
# funny edge and corner cases, and because when you parameterise lots of
# different things in very different ways there is a combinatorial explosion of
# geometrical possibilities.
#
# The CAD in BLUEPRINT is not smart. It doesn't know when there are clashes.
# We try to avoid clashes as much as possible in 2-D first.
#
# A bit of background on BLUEPRINT coordinates...
#
# The coordinate system is right-handed and centred at (0, 0, 0).
#
# See e.g. https://mathworld.wolfram.com/Right-HandedCoordinateSystem.html
#
# Or enjoy some of my finest ASCII art instead:
#
# ```
#   z(+) y (+)
#   ^   &
#   |  /
#   |/
#   |-----------> x (+)
# ```
#
# The default plane in BLUEPRINT is the x-z plane.
# The default axis of revolution is the z-axis.
# Rotations are counter-clockwise
# The unit of length is the [metre]. The units in BLUEPRINT are SI (..mostly..)
#
# So, for example, a plasma shape would have a 2-D cross-section specified in m
# on the x-z plane, and rotated about the z-axis to form a torus.

# %%[markdown]
#
# ## INTRODUCTION TO THE LOOP
#
# Problem: we need to store geometry information in a coherent way.
# Solution: the Loop class
#
# *  It is an intelligent collection of coordinates (ordered set of vertices)
# *  With methods to modify the coordinates, or use them in other ways
#
#
# Let's make a triangle:

# %%
triangle = Loop(x=[1, 2, 0], y=[2**0.5, 0, 0])

print(f"triangle.x: {triangle.x}")
print(f"triangle.y: {triangle.y}")

# %%[markdown]
# Good, so these coordinates are the same as specified... Or are they??
#
# *  No.. They have been re-ordered! The Loop is now counter-clockwise
# *  Also, the coordinates are now floats, not integers.
#
#
# These details are important when dealing with various operations, and it is
# important to understand that Loops are by default counter-clockwise (ccw) and
# have underlying numpy arrays of floats for coordinates.
#
# But what about the z-dimension?

# %%
print(f"triangle.z: {triangle.z}")

# %%[markdown]
# It is auto-populated to an array of zeros. This kind of thing is important
# if you care where things are in 3-D.
#
# You can get a quick summary of the Loop by printing it to the console

# %%
print(triangle)

# %%[markdown]
# That doesn't look great.. does it?

# %%
bluemira_print(str(triangle))  # :)

# %%[markdown]
# So let's look at the triangle, and its points

# %%
triangle.plot(points=True)

# %%[markdown]
# The defaults are a red line for open loops, a black line for closed loops, and the
# Loop is filled with a polygon face if it is closed.
# Notice that the triangle is an open Loop as it only has two segments.

# %%
print(f"Is my triangle a closed loop?: {triangle.closed}")

# %%[markdown]
# It is usually desirable to work with closed Loops. These are used frequently
# in BLUEPRINT to describe space reservation of things in 2-D.
#
# Unlike ccw, Loops are not forced to be closed Loops.

# %%
triangle.close()
print(triangle)

# %%[markdown]
# Notice it is now of length = 4, and not 3 as before. The first and end points
# are coincident.
#
# Now, let's plot the triangle again, and make it look a little different

# %%
triangle.plot(edgecolor="k", facecolor="grey")

# %%[markdown]
# Let's see what else this thing can do:

# %%
bluemira_print(
    "Summary of the triangle loop:\n"
    f"|   area: {triangle.area:.2f} m\n"
    f"|   perimeter: {triangle.length:.2f} m\n"
    f"|   centroid: {triangle.centroid}"
)

# %%[markdown]
# ## ACCESSING COORDINATES
#
# The underlying Loop coordinates can be accessed in a variety of ways.
#
# For full coordinate arrays, via access to the xyz attribute

# %%
print(triangle.xyz)

# %%[markdown]
# For full single coordinates, via attribute access or dictionary call

# %%
print(triangle.x)
print(triangle["x"])

# %%[markdown]
# For individual vertices, via indexing and slicing

# %%
print(f"First point in triangle: {triangle[0]}")
print(f"First two points in triangle: {triangle[:2]}")

# %%[markdown]
# Note that when we access single points, we always get the 3 dimensions
# If we only want the "most important" 2 dimensions, then we have to do this:

# %%
print(f"First point x-y: {triangle.d2.T[0]}")
print(f"First two points x-y: {triangle.d2.T[:2]}")

# %%[markdown]
# The d2 attribute accesses the two important coordinates in which the Loop was
# specified. So x-y for x-y Loops and y-z for y-z Loops, etc.

# %%[markdown]
# ## INSTANTIATING A LOOP
#
# There are other ways to instantiate a Loop:
# 1.  from a dictionary
# 2.  from a numpy array
# 3.  from a file

# %%[markdown]
# ### Instantiating a loop from a dictionary

# %%
dictionary = {"x": [1, 2, 3], "y": 4.4573, "z": [0, 0, 2]}

# %%[markdown]
# Notice that if we give a single value to one of the dimensions, the Loop will
# assume that it is on e.g. x-z plane, at an offset of y from the 0-x-z plane.

# %%
dict_loop = Loop.from_dict(dictionary)

# %%[markdown]
# ### Instantiating a loop from a numpy array

# %%
coordinates = np.array(
    [
        [0, 0, 2, 3, 4, 5, 6, 7, 5, 4, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 4, 5, 6, 6, 7, 5, 4, 3, 2, 2],
    ]
)

array_loop = Loop.from_array(coordinates)

# %%[markdown]
# When plotting Loops, it is also nice sometimes to specify the matplotlib
# Axes object onto which they are plotted.
f, ax = plt.subplots()

array_loop.plot(ax)
ax.set_title("A loop from a numpy array")

# %%[markdown]
# ### Instantiating a loop from a file
#
# This is mostly useful for debugging, save Loops if they are nasty geometry
# edge cases, or passing information to people. Loops are saved in a JSON
# format, because it is widely-used and human readable.
#
# First we need to get the folder where some Loops are stored:

# %%
path = get_bluemira_path("BLUEPRINT/cad/test_data", subfolder="tests")

print(path)

# %%[markdown]
# Then we need to pick a Loop

# %%
name = "plasmaloop.json"
filename = os.sep.join([path, name])

# %%[markdown]
# And instantiate a Loop from it.

# %%
plasma_loop = Loop.from_file(filename)

f, ax = plt.subplots()
plasma_loop.plot(ax, edgecolor="r", facecolor="pink")

# %%[markdown]
# ## TRANSFORMING LOOPS
#
# Obviously, we want to be able to move this stuff around

# %%
f, ax = plt.subplots()
plasma_loop.translate([4, 0, 0])
plasma_loop.plot(ax, edgecolor="r", facecolor="pink")

# %%[markdown]
# The plasma_loop has been permanently moved. What if we want a copy?

# %%
plasma_loop2 = plasma_loop.translate([4, 0, 0], update=False)

# %%[markdown]
# Now let's rotate it (about the y-axis, clockwise by 30 degrees)

# %%
plasma_loop2.rotate(theta=-30, p1=[0, 0, 0], p2=[0, 1, 0])

# %%
f, ax = plt.subplots()
plasma_loop.plot(ax, edgecolor="b", facecolor="grey")
plasma_loop2.plot(ax, edgecolor="r", facecolor="pink")

# %%[markdown]
# Alright, but what about 3-D? Sure, why not

# %%
loop = plasma_loop.rotate(
    theta=45, p1=[0.1, 0.254, 0.74], p2=[0.4, 0.2, 0.1], update=False
)

loop.plot(facecolor="green")

# %%[markdown]
# ---
# **NOTE**
# matplotlib is not great at 3-D stuff.. some homebrew hacks make this
# work, but if you want to break it you won't have to try very hard.
#
# ---

# %%[markdown]
# ## WORKING WITH LOOPS
#
# There are lots of things we want to do with geometry.. below are some helpful methods

# %%[markdown]
# ### Offsetting

# %%
f, ax = plt.subplots()
plasma_loop.plot(ax)

# %%[markdown]
# Outwards

# %%
f, ax = plt.subplots()
plasma_loop.plot(ax)
for offset_size, color in zip([0.2, 0.4, 0.6, 1], ["r", "orange", "g", "b"]):
    offset_plasma = plasma_loop.offset(offset_size)
    offset_plasma.plot(ax, edgecolor=color, fill=False)

# %%[markdown]
# Inwards

# %%
f, ax = plt.subplots()
plasma_loop.plot(ax)
for offset_size, color in zip([0.2, 0.4, 0.6, 1], ["r", "orange", "g", "b"]):
    offset_plasma = plasma_loop.offset(-offset_size)
    offset_plasma.plot(ax, edgecolor=color, fill=False)

# %%[markdown]
# That last one looks a little funny... welcome to geometry!
# Loops are not yet smart enough to detect that they are self-intersecting..
# Be careful. There are ways of dealing with this, see e.g. below.

# %%
clean_offset = simplify_loop(offset_plasma)

f, ax = plt.subplots()
offset_plasma.plot(ax, facecolor="r")
clean_offset.plot(ax, facecolor="b")
ax.set_xlim([11.64, 11.74])
ax.set_ylim([3.7, 3.76])

# %%[markdown]
# ### Boolean operations
#
# Let's pick up with some fresh plasma loops

# %%
loop1 = Loop.from_file(filename)

loop2 = loop1.translate([4, 0, 0], update=False)

# %%[markdown]
# We can join Loops with boolean union operations

# %%
union = boolean_2d_union(loop1, loop2)[0]
f, ax = plt.subplots()
union.plot(ax, facecolor="b")

# %%[markdown]
# We can subtract loops with boolean difference operations
# We can intersect the loops with boolean common operations
#
# Note that in both the cases there can be multiple answers!
# (in this case there is only one, but we are expecting a list anyway (hence [0])

# %%
diff = boolean_2d_difference(loop1, loop2)[0]

common = boolean_2d_common(loop1, loop2)[0]

f, ax = plt.subplots()
diff.plot(ax, facecolor="r")
common.plot(ax, facecolor="b")

# %%[markdown]
# OK, enough about Loops.

# %%[markdown]
# ## INTRODUCTION TO THE SHELL
#
# In BLUEPRINT, a Shell is Loop within a Loop. It is a subset of the Polygon
# with holes problem.
#
# Let's make one
# We have to specify an inner Loop and an outer Loop
# It's not super-smart, so if you want to break it, again, you can.

# %%
shell = Shell(clean_offset, plasma_loop)

# %%[markdown]
# Shells work in much the same way as Loops

# %%
f, ax = plt.subplots()
shell.plot(ax, linewidth=8, edgecolor="k", facecolor="orange")

# %%[markdown]
# ---
# **NOTE**
# matplotlib kwargs will usually work.. usually
#
# ---
#
# WE can transform them too

# %%
rotated_shell = shell.rotate(
    theta=234, p1=[0, 23, 5], p2=[23.5, 423, np.pi], update=False
)

rotated_shell.plot(edgecolor="k", facecolor="b")
