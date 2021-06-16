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
from bluemira.geometry.bluemirageo import BluemiraGeo
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

    def _check_boundary(self, objs):
        """Check if objects objs can be used as boundaries"""
        if not hasattr(objs, '__len__'):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                return objs
        raise TypeError("Only {} objects can be used for {}".format(
            self._boundary_classes))

    @BluemiraGeo.boundary.setter
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)

    @property
    def shape(self):
        """Part.Wire: shape of the object as a single wire"""
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
        # This will help in recursive functions.

        return self.shape.isClosed()

    def close_shape(self):
        """Close the shape with a LineSegment between shape's end and
            start point. This function modify the object boundary.
        """
        if not self.isClosed():
            closure = close_wire(self.shape)
            self.boundary.append(closure)

        # check that the new boundary is closed
        if not self.isClosed():
            raise NotClosedWire("The open boundary has not been closed correctly.")

    def discretize(self, ndiscr: int = 100, byedges: bool = False):

        """Discretize the wire in ndiscr equidistant points.
        If byedges is True, each edges is discretized separately using and approximated
        distance (wire.Length/ndiscr)."""

        if byedges:
            points = discretize_by_edges(self.shape, ndiscr)
        else:
            points = discretize(self.shape, ndiscr)
        return points
