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
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

# import from freecad
import freecad
import Part
from FreeCAD import Base

# import numpy lib
import numpy

# import typing
from typing import Union


# Geometry creation
def make_polygon(points: Union[list, numpy.ndarray], closed: bool = False,
              placement=None) -> Part.Wire:
    """Make a polygon from a set of points.

    Args:
        points (Union[list, numpy.ndarray]): list of points. It can be given \
            as a list of 3D tuples, a 3D numpy array, or similar.
        closed (bool, optional): if True, the first and last points will be\
            connected in order to form a closed polygon. Defaults to False.

    Returns:
        Part.Wire: a FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    wire = Part.makePolygon(pntslist)
    if closed and not wire.isClosed():
        # add a line that closes the wire
        line = Part.makePolygon([pntslist[-1], pntslist[0]])
        wire = Part.Wire([wire, line])
    if placement:
        wire.Placement = placement
    return wire
