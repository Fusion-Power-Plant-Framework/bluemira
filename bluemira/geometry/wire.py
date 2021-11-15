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
Wrapper for FreeCAD Part.Wire objects
"""

from __future__ import annotations

from typing import List

# import from bluemira
from bluemira.geometry.base import BluemiraGeo

import bluemira.geometry._freecadapi as _freecadapi

from bluemira.geometry._freecadapi import (
    discretize_by_edges,
    discretize,
    wire_closure,
    scale_shape,
    translate_shape,
    apiWire,
)

# import mathematical library
import numpy

# import from error
from bluemira.geometry.error import NotClosedWire, MixedOrientationWireError


class BluemiraWire(BluemiraGeo):
    """Bluemira Wire class."""

    # # Necessary only if there are changes to the base attrs dictionary
    # attrs = {**BluemiraGeo.attrs}

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [self.__class__, apiWire]
        super().__init__(boundary, label, boundary_classes)
        self._check_orientations()

        # connection variable with BLUEPRINT Loop
        self._bp_loop = None

    def _check_orientations(self):
        orientations = []
        for boundary in self.boundary:
            if isinstance(boundary, apiWire):
                orient = boundary.Orientation
            elif isinstance(boundary, self.__class__):
                orient = boundary._shape.Orientation
            orientations.append(orient)

        if orientations.count(orientations[0]) != len(orientations):
            raise MixedOrientationWireError(
                f"Cannot make a BluemiraWire from wires of mixed orientations: {orientations}"
            )
        return orientations

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, apiWire):
                output = BluemiraWire(output)
            return output

        return wrapper

    @property
    def _shape(self) -> apiWire:
        """apiWire: shape of the object as a single wire"""
        wires = self._wires
        if len(wires) == 1 and isinstance(wires[0], apiWire):
            return wires[0]

        return apiWire(self._wires)

    @property
    def _wires(self) -> List[apiWire]:
        """list(apiWire): list of wires of which the shape consists of."""
        wires = []
        for o in self.boundary:
            if isinstance(o, apiWire):
                for w in o.Wires:
                    wires += [apiWire(w.OrderedEdges)]
                if w.Orientation != wires[-1].Orientation:
                    wires[-1].reverse()
            else:
                wires += o._wires
        return wires

    def get_single_wire(self) -> BluemiraWire:
        """Get a single wire representing the object"""
        return BluemiraWire(self._shape)

    def __add__(self, other):
        """Add two wires"""
        output = None
        if isinstance(other, BluemiraWire):
            output = BluemiraWire([self, other])
        else:
            raise TypeError(f"{type(other)} is not an instance of BluemiraWire.")
        return output

    def close(self) -> None:
        """Close the shape with a line segment between shape's end and start point.
        This function modify the object boundary.
        """
        if not self.is_closed():
            closure = wire_closure(self._shape)
            if isinstance(self.boundary[0], apiWire):
                self.boundary.append(closure)
            else:
                self.boundary.append(BluemiraWire(closure))

        # check that the new boundary is closed
        if not self.is_closed():
            raise NotClosedWire("The open boundary has not been closed.")

    def discretize(
        self, ndiscr: int = 100, byedges: bool = False, dl: float = None
    ) -> numpy.ndarray:
        """Discretize the wire in ndiscr equidistant points or with a reference dl
        segment step.
        If byedges is True, each edges is discretized separately using an approximated
        distance (wire.Length/ndiscr) or the specified dl.

        Returns
        -------
        points:
            a numpy array with the x,y,z coordinates of the discretized points.
        """
        if byedges:
            points = discretize_by_edges(self._shape, ndiscr=ndiscr, dl=dl)
        else:
            points = discretize(self._shape, ndiscr=ndiscr, dl=dl)
        return points

    def scale(self, factor) -> None:
        """Apply scaling with factor to this object. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                scale_shape(o, factor)
            else:
                o.scale(factor)

    def translate(self, vector) -> None:
        """Translate this shape with the vector. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                translate_shape(o, vector)
            else:
                o.translate(vector)

    def change_plane(self, plane):
        """Apply a plane transformation to the wire"""
        for o in self.boundary:
            if isinstance(o, apiWire):
                _freecadapi.change_plane(o, plane._shape)
            else:
                o.change_plane(plane)
