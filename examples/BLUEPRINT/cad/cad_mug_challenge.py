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
A more advanced CAD tutorial to make a mug.
"""

# %%


import os

from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    extrude,
    make_axis,
    make_circle,
    revolve,
    save_as_STEP,
    show_CAD,
)

# %%[markdown]
# If you've completed the previous tutorial and played around a little making freeform
# shapes, consider this challenge: modelling a CAD mug.
# A basic mug with handle can be made in CAD using the same tools covered in the tutorial
# , namely:
# ```
# make_circle()
# extrude()
# make_axis()
# revolve()
# booleon_fuse()
# boolean_cut()
# ```
# In other words, a simple mug can be created by correctly building the same cylinders
# and tori from tokamak CAD. Try to make a mug in the box below using the CAD functions
# above.
#
# # MAKE A MUG

# %%


# This line will create a 2D circular base for your mug.
# You can use this as a starting point or delete the line to start from scratch.
base = make_circle(centre=[0, 0, 0], direction=[0, 0, 1], radius=0.05)

# Your code here.

show_CAD(base)


# %%[markdown]
# # WRITE A CAD FUNCTION
#
# Generally, it's good practice to functionise your code. That way, it's easy to create a
# similar object later by varying the function inputs, rather than writing out the same
# thing again. But what are the key parameters of a mug?
#
# Below is an example of function to create a CAD mug with parameters set by keyword
# arguements. For a circular mug, the radius of the base is the key parameter, along with
# height. The thickness of the mug is also set here as a fraction of radius and height.
# However, the mug handle is hard coded in shape relative to the mug radius and height.
#
# See if you can modify the code below to add parameters allowing customisation of the
# handle.

# %%


def CAD_mug(dimensions=(0.10, 0.10), thicknesses=(0.10, 0.10)):
    """
    Plots a basic mug in CAD. Dimensions are the diameter and height of the mug in m.
    Thicknesses is the width of the walls/base as a fraction of the mug diameter/height.
    """
    # Set parameters.
    diameter = dimensions[0]
    radius = diameter / 2
    height = dimensions[1]
    wall_thickness = thicknesses[0] * diameter
    base_thickness = thicknesses[1] * height

    # Create mug vessel.
    outer = make_circle(centre=[0, 0, 0], direction=[0, 0, 1], radius=radius)
    outer = extrude(outer, vec=[0, 0, height])
    inner = make_circle(
        centre=[0, 0, base_thickness],
        direction=[0, 0, 1],
        radius=radius - wall_thickness,
    )
    inner = extrude(inner, vec=[0, 0, height - base_thickness])
    vessel = boolean_cut(outer, inner)

    # Create handle.
    dist = 0.25 * height  # distance from base of mug to centre of lower handle joint
    face = make_circle(
        centre=[radius, 0, dist], direction=[1, 0, 0], radius=0.12 * radius
    )
    axis = make_axis([radius, 0, 0.5 * height], [0, -1, 0])
    handle = revolve(face, axis, angle=180)

    mug = boolean_fuse(vessel, handle)
    return mug


# %%


mug = CAD_mug(dimensions=(0.10, 0.11), thicknesses=(0.05, 0.10))
show_CAD(mug)

# %%

path = os.getcwd()
filename = os.sep.join([path, "mug_test"])
save_as_STEP(mug, filename)
