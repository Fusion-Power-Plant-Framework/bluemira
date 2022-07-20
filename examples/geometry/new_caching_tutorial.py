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
import bluemira.geometry.tools as geotools

# Basic objects
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace

# %%[markdown]

# ## Make a cylinder

# There are many ways to make a cylinder, but perhaps the simplest way is as follows:
# * Make a circular Wire
# * Make a Face from that Wire
# * Extrude that Face along a vector, to make a Solid

# %%
# Note that we are going to give these geometries some labels, which
# we might use later.
points = [[0,0,0], [1,0,0], [1,1,0]]
wire = cadapi.make_polygon(points)
bmwire = BluemiraWire(wire)

bmwire1 = geotools.make_polygon(points, "open_poly", False)
bmwire2 = geotools.make_polygon(points, "closed_poly", True)

#bmface1 = geotools.make_face(bmwire1)#
bmface2 = geotools.make_face(bmwire2)
bmshell = geotools.make_shell([bmface2])
