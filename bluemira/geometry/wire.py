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
from bluemira.geometry.bmbase import BluemiraGeo

from bluemira.geometry.freecadapi import (
    discretize_by_edges, discretize, close_wire, make_polygon
)

# import mathematical library
import numpy

# import from error
from bluemira.geometry.error import NotClosedWire


class BluemiraWire(BluemiraGeo):
    """Bluemira Wire class."""

    # # Necessary only if there are changes to the base attrs dictionary
    # attrs = {**BluemiraGeo.attrs}

    def __init__(
            self,
            boundary,
            label: str = "",
            lcar: Union[float, List[float]] = 0.1
    ):
        boundary_classes = [self.__class__, Part.Wire]
        super().__init__(boundary, label, lcar, boundary_classes)

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, Part.Wire):
                output = BluemiraWire(output)
            return output
        return wrapper

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
    def _shape(self) -> Part.Wire:
        """Part.Wire: shape of the object as a single wire"""
        return Part.Wire(self._wires)

    @property
    def _wires(self) -> List[Part.Wire]:
        """list(Part.Wire): list of wires of which the shape consists of."""

        wires = []
        for o in self.boundary:
            if isinstance(o, Part.Wire):
                wires += o.Wires
            else:
                wires += o._wires
        return wires

    def close_shape(self):
        """Close the shape with a LineSegment between shape's end and
            start point. This function modify the object boundary.
        """
        if not self.is_closed():
            closure = close_wire(self._shape)
            self.boundary.append(closure)

        # check that the new boundary is closed
        if not self.is_closed():
            raise NotClosedWire("The open boundary has not been closed correctly.")

    def discretize(self, ndiscr: int = 100, byedges: bool = False) -> numpy.ndarray:

        """Discretize the wire in ndiscr equidistant points.
        If byedges is True, each edges is discretized separately using and approximated
        distance (wire.Length/ndiscr)."""

        if byedges:
            points = discretize_by_edges(self._shape, ndiscr)
        else:
            points = discretize(self._shape, ndiscr)
        return points

    @staticmethod
    def make_polygon(points: Union[list, numpy.ndarray], closed: bool = False) -> \
            BluemiraWire:
        """Make a BluemiraWire polygon from a set of points. If closed is True,
        the wire will be forced to be closed."""
        return BluemiraWire(make_polygon(points, closed))
