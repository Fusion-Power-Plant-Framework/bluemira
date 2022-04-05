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

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes._freecadapi import (
    apiWire,
    change_placement,
    discretize,
    discretize_by_edges,
    end_point,
    rotate_shape,
    scale_shape,
    start_point,
    translate_shape,
    wire_closure,
    wire_parameter_at,
    wire_value_at,
)
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
        self._orientation = orientations[0]

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
        return self._create_wire()

    def _create_wire(self, check_reverse=True):
        wire = apiWire(self._wires)
        if check_reverse:
            return self._check_reverse(wire)
        else:
            return wire

    @property
    def _wires(self) -> List[apiWire]:
        """list(apiWire): list of wires of which the shape consists of."""
        wires = []
        for o in self.boundary:
            if isinstance(o, apiWire):
                for w in o.Wires:
                    wire = apiWire(w.OrderedEdges)
                    if self._orientation != _Orientation(wire.Orientation):
                        edges = []
                        for edge in wire.OrderedEdges:
                            edge.reverse()
                            edges.append(edge)
                        wire = apiWire(edges)
                    wires += [wire]
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
        """
        Close the shape with a line segment between shape's end and start point.
        This function modifies the object boundary.
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
            points = discretize_by_edges(self._shape, ndiscr=ndiscr, dl=dl)
        else:
            points = discretize(self._shape, ndiscr=ndiscr, dl=dl)
        return Coordinates(points)

    def scale(self, factor) -> None:
        """
        Apply scaling with factor to this object. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                scale_shape(o, factor)
            else:
                o.scale(factor)

    def translate(self, vector) -> None:
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                translate_shape(o, vector)
            else:
                o.translate(vector)

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
        for o in self.boundary:
            if isinstance(o, apiWire):
                rotate_shape(o, base, direction, degree)
            else:
                o.rotate(base, direction, degree)

    def change_placement(self, placement):
        """Changes the object placement"""
        for o in self.boundary:
            if isinstance(o, apiWire):
                change_placement(o, placement._shape)
            else:
                o.change_placement(placement)

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

        return wire_value_at(self.get_single_wire()._shape, distance)

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
            return wire_parameter_at(
                self.get_single_wire()._shape, vertex=vertex, tolerance=tolerance
            )
        except FreeCADError as e:
            raise GeometryError(e.args[0])

    def start_point(self) -> Coordinates:
        """
        Get the coordinates of the start of the wire.
        """
        return Coordinates(start_point(self._shape))

    def end_point(self) -> Coordinates:
        """
        Get the coordinates of the end of the wire.
        """
        return Coordinates(end_point(self._shape))
