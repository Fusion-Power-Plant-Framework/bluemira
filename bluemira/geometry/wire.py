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

from typing import Iterable, List, Optional

import bluemira.codes._freecadapi as cadapi
from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
import bluemira.codes._freecadapi as cadapi
from bluemira.codes.error import FreeCADError

# import from bluemira
from bluemira.geometry.base import BluemiraGeo, _Orientation
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import (
    GeometryError,
    MixedOrientationWireError,
    NotClosedWire,
)

__all__ = ["BluemiraWire"]


class BluemiraWire(BluemiraGeo):
    """Bluemira Wire class."""

    def __init__(self, shape, label: str = ""):
        shape_classes = [cadapi.apiWire]
        super().__init__(shape, label, shape_classes)

    def __add__(self, other):
        """Add two wires"""
        output = None
        if isinstance(other, BluemiraWire):
            wire = cadapi.apiWire([self._shape, other._shape])
            output = BluemiraWire(wire)
        else:
            raise TypeError(f"{type(other)} is not an instance of BluemiraWire.")
        return output

    def close(self) -> None:
        """
        Close the shape with a line segment between shape's end and start point.
        This function modifies self.
        """
        if not self.is_closed():
            closure = cadapi.wire_closure(self._shape)
            closed_wire = cadapi.apiWire([self._shape, closure])
            self.shape = closed_wire
        # check that the new boundary is closed
        if not self.is_closed():
            raise NotClosedWire("The open boundary has not been closed.")

    def discretize(
        self, ndiscr: int = 100, byedges: bool = False, dl: float = None
    ) -> Coordinates:
        """
        Discretize the wire in ndiscr equidistant points or with a reference dl
        segment step.
        If byedges is True, each edges is discretized separately using an approximated
        distance (wire.Length/ndiscr) or the specified dl.

        Returns
        -------
        points: Coordinates
            a np array with the x,y,z coordinates of the discretized points.
        """
        if byedges:
            points = cadapi.discretize_by_edges(self._shape, ndiscr=ndiscr, dl=dl)
        else:
            points = cadapi.discretize(self._shape, ndiscr=ndiscr, dl=dl)
        return Coordinates(points)

    def scale(self, factor) -> None:
        """
        Apply scaling with factor to this object. This function modifies the self
        object.
        """
        cadapi.scale_shape(self._shape, factor)

    def translate(self, vector) -> None:
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        cadapi.translate_shape(self._shape, vector)

    def rotate(
        self,
        base: tuple = (0.0, 0.0, 0.0),
        direction: tuple = (0.0, 0.0, 1.0),
        degree: float = 180,
    ):
        """
        Rotate this shape.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle
        """

        cadapi.rotate_shape(self._shape, base, direction, degree)

    def change_placement(self, placement):
        """Changes the object placement"""
        cadapi.change_placement(self._shape, placement._shape)

    def value_at(self, alpha: Optional[float] = None, distance: Optional[float] = None):
        """
        Get a point along the wire at a given parameterised length or length.

        Parameters
        ----------
        alpha: Optional[float]
            Parameterised distance along the wire length, in the range [0 .. 1]
        distance: Optional[float]
            Physical distance along the wire length

        Returns
        -------
        point: np.ndarray
            Point coordinates (w.r.t. BluemiraWire's BluemiraPlacement)
        """
        if alpha is None and distance is None:
            raise GeometryError("Must specify one of alpha or distance.")
        if alpha is not None and distance is not None:
            raise GeometryError("Must specify either alpha or distance, not both.")

        if distance is None:
            if alpha < 0.0:
                bluemira_warn(
                    f"alpha must be between 0 and 1, not: {alpha}, setting to 0.0"
                )
                alpha = 0
            elif alpha > 1.0:
                bluemira_warn(
                    f"alpha must be between 0 and 1, not: {alpha}, setting to 1.0"
                )
                alpha = 1.0
            distance = alpha * self.length

        return cadapi.wire_value_at(self._shape, distance)

    def parameter_at(self, vertex: Iterable, tolerance: float = EPS):
        """
        Get the parameter value at a vertex along a wire.

        Parameters
        ----------
        wire: apiWire
            Wire along which to get the parameter
        vertex: Iterable
            Vertex for which to get the parameter
        tolerance: float
            Tolerance within which to get the parameter

        Returns
        -------
        alpha: float
            Parameter value along the wire at the vertex

        Raises
        ------
        GeometryError:
            If the vertex is further away to the wire than the specified tolerance
        """
        try:
            return cadapi.wire_parameter_at(
                self._shape, vertex=vertex, tolerance=tolerance
            )
        except FreeCADError as e:
            raise GeometryError(e.args[0])

    def start_point(self) -> Coordinates:
        """
        Get the coordinates of the start of the wire.
        """
        return Coordinates(cadapi.start_point(self._shape))

    def end_point(self) -> Coordinates:
        """
        Get the coordinates of the end of the wire.
        """
        return Coordinates(cadapi.end_point(self._shape))

    @property
    def vertexes(self):
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def wires(self):
        return self

    @property
    def faces(self):
        return []

    @property
    def shells(self):
        return []

    @property
    def solids(self):
        return []
