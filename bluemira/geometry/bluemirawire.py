# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Wrapper for FreeCAD Part.Wire objects
"""

from __future__ import annotations

from typing import Union, List

# import from freecad
import freecad
import Part

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.tools import (
    discretize_by_edges, discretize, close_wire
)

# import from error
from bluemira.geometry.error import NotClosedWire


class BluemiraWire(BluemiraGeo):
    """Bluemira Wire class."""
    def __init__(
            self,
            boundary,
            label: str = "",
            lcar: Union[float, List[float]] = 0.1
    ):
        boundary_classes = [self.__class__, Part.Wire]
        super().__init__(boundary, label, lcar, boundary_classes)

    @property
    def Length(self):
        """float: total length of the shape."""

        # Note:the method is recursively implemented considering that
        # Part.Wire has a similar Length property.

        return self.single_wire.Length

    @property
    def Area(self):
        """float: total area of the shape."""

        # Note:the method is recursively implemented considering that
        # Part.Area has a similar Length property.

        return self.single_wire.Area

    @property
    def Volume(self):
        """float: total volume of the shape."""

        # Note:the method is recursively implemented considering that
        # Part.Wire has a similar Length property.

        return self.single_wire.Volume

    @property
    def single_wire(self):
        return Part.Wire(self.Wires)

    @property
    def Wires(self) -> List[Part.Wire]:
        """list(Part.Wire): list of wires of which the shape consists of."""

        # Note:the method is recursively implemented considering that
        # Part.Wire has a similar Wires property.

        wires = []
        for o in self.boundary:
            wires += o.Wires
        return wires

    def isClosed(self):
        """True if the shape is closed"""

        # Note: isClosed is also a function of Part.Wire.

        return self.single_wire.isClosed()

    def close_shape(self):
        """Close the shape with a LineSegment between shape's end and
            start point. This function modify the object boundary.
        """
        if not self.isClosed():
            closure = close_wire(self.single_wire)
            self.boundary.append(closure)

        # check that the new boundary is closed
        if not self.isClosed():
            raise NotClosedWire("The open boundary has not been closed correctly.")

    def discretize(self, ndiscr: int = 100, byedges: bool = False):

        """Discretize the wire in ndiscr equidistant points.
        If byedges is True, each edges is discretized separately using and approximated
        distance (wire.Length/ndiscr)."""

        if byedges:
            points = discretize_by_edges(self.single_wire, ndiscr)
        else:
            points = discretize(self.single_wire, ndiscr)
        return points
