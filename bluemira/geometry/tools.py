#  bluemira is an integrated inter-disciplinary design tool for future fusion
#  reactors. It incorporates several modules, some of which rely on other
#  codes, to carry out a range of typical conceptual fusion reactor design
#  activities.
#  #
#  Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                     J. Morris, D. Short
#  #
#  bluemira is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#  #
#  bluemira is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  Lesser General Public License for more details.
#  #
#  You should have received a copy of the GNU Lesser General Public
#  License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Useful functions for bluemira geometries.
"""
# import from freecadapi
from bluemira.geometry.base import BluemiraGeo
from . import _freecadapi

# import bluemira geometries
from .wire import BluemiraWire
from .face import BluemiraFace
from .shell import BluemiraShell
from .solid import BluemiraSolid

# import mathematical modules
import numpy as np

# import typing
from typing import Union


# # =============================================================================
# # Geometry creation
# # =============================================================================
def make_polygon(
    points: Union[list, np.ndarray], label: str = "", closed: bool = False
) -> BluemiraWire:
    """Make a polygon from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label: str, default = ""
        Object's label
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed polygon. Defaults to False.

    Returns
    -------
    wire: BluemiraWire
        a bluemira wire that contains the polygon
    """
    return BluemiraWire(_freecadapi.make_polygon(points, closed), label=label)


def make_bspline(
    points: Union[list, np.ndarray], label: str = "", closed: bool = False
) -> BluemiraWire:
    """Make a bspline from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label: str, default = ""
        Object's label
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed bspline. Defaults to False.

    Returns
    -------
    wire: BluemiraWire
        a bluemira wire that contains the bspline
    """
    return BluemiraWire(_freecadapi.make_bspline(points, closed), label=label)


def make_bezier(
    points: Union[list, np.ndarray], label: str = "", closed: bool = False
) -> BluemiraWire:
    """Make a bspline from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label: str, default = ""
        Object's label
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed bspline. Defaults to False.

    Returns
    -------
    wire: BluemiraWire
        a bluemira wire that contains the bspline
    """
    return BluemiraWire(_freecadapi.make_bezier(points, closed), label=label)


def make_circle(
    radius=1.0,
    center=[0.0, 0.0, 0.0],
    start_angle=0.0,
    end_angle=360.0,
    axis=[0.0, 0.0, 1.0],
    label: str = "",
) -> BluemiraWire:
    """
    Create a circle or arc of circle object with given parameters.

    Parameters
    ----------
    radius: float, default =1.0
        Radius of the circle
    center: Iterable, default = [0, 0, 0]
        Center of the circle
    start_angle: float, default = 0.0
        Start angle of the arc [degrees]
    end_angle: float, default = 360.0
        End angle of the arc [degrees]. If start_angle == end_angle, a circle is created,
        otherwise a circle arc is created
    axis: Iterable, default = [0, 0, 1]
        Normal vector to the circle plane. It defines the clockwise/anticlockwise
        circle orientation according to the right hand rule.
    label: str
        object's label

    Returns
    -------
    wire: BluemiraWire
        bluemira wire that contains the arc or circle
    """
    output = _freecadapi.make_circle(radius, center, start_angle, end_angle, axis)
    return BluemiraWire(output, label=label)


def make_circle_arc_3P(p1, p2, p3, label: str = ""):  # noqa: N802
    """
    Create an arc of circle object given three points.

    Parameters
    ----------
    p1: Iterable
        Starting point of the circle arc
    p2: Iterable
        Middle point of the circle arc
    p3: Iterable
        End point of the circle arc

    Returns
    -------
    wire: BluemiraWire
        bluemira wire that contains the arc or circle
    """
    # TODO: check what happens when the 3 points are in a line
    output = _freecadapi.make_circle_arc_3P(p1, p2, p3)
    return BluemiraWire(output, label=label)


def make_ellipse(
    center=[0.0, 0.0, 0.0],
    major_radius=2.0,
    minor_radius=1.0,
    major_axis=[1, 0, 0],
    minor_axis=[0, 1, 0],
    start_angle=0.0,
    end_angle=360.0,
    label: str = "",
):
    """
    Create an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center: Iterable, default = [0, 0, 0]
        Center of the ellipse
    major_radius: float, default = 2
        Major radius of the ellipse
    minor_radius: float, default = 2
        Minor radius of the ellipse (float). Default to 2.
    major_axis: Iterable, default = [1, 0, 0]
        Major axis direction
    minor_axis: Iterable, default = [0, 1, 0]
        Minor axis direction
    start_angle:  float, default = 0
        Start angle of the arc [degrees]
    end_angle: float, default = 360
        End angle of the arc [degrees].  if start_angle == end_angle, an ellipse is
        created, otherwise a ellipse arc is created
    label: str, default = ""
        Object's label

    Returns
    -------
    wire: BluemiraWire:
         Bluemira wire that contains the arc or ellipse
    """
    output = _freecadapi.make_ellipse(
        center,
        major_radius,
        minor_radius,
        major_axis,
        minor_axis,
        start_angle,
        end_angle,
    )
    return BluemiraWire(output, label=label)


def wire_closure(bmwire: BluemiraWire, label="closure") -> BluemiraWire:
    """Close this wire with a line segment

    Parameters
    ----------
    bmwire: BluemiraWire
        supporting wire for the closure
    label: str, default = ""
        Object's label

    Returns
    -------
        closure: BluemiraWire
            Closure wire
    """
    wire = bmwire._shape
    closure = BluemiraWire(_freecadapi.wire_closure(wire), label=label)
    return closure


# # =============================================================================
# # Shape operation
# # =============================================================================
def revolve_shape(
    shape,
    base: tuple = (0.0, 0.0, 0.0),
    direction: tuple = (0.0, 0.0, 1.0),
    degree: float = 180,
):
    """
    Apply the revolve (base, dir, degree) to this shape

    Parameters
    ----------
    shape: BluemiraGeo
        The shape to be revolved
    base: tuple (x,y,z), default = (0.0, 0.0, 0.0)
        Origin location of the revolution
    direction: tuple (x,y,z), default = (0.0, 0.0, 1.0)
        The direction vector
    degree: double, default = 180
        revolution angle

    Returns
    -------
    shape: BluemiraSolid
        the revolved shape.
    """
    solid = _freecadapi.revolve_shape(shape._shape, base, direction, degree)
    faces = solid.Faces
    bmfaces = []
    for face in faces:
        bmfaces.append(BluemiraFace._create(face))
    bmshell = BluemiraShell(bmfaces)
    bmsolid = BluemiraSolid(bmshell)
    return bmsolid


def extrude_shape(shape: BluemiraGeo, vec: tuple, label=None) -> BluemiraSolid:
    """
    Apply the extrusion along vec to this shape

    Parameters
    ----------
    shape: BluemiraGeo
        The shape to be extruded
    vec: tuple (x,y,z)
        The vector along which to extrude
    label: str, default = None
        label of the output shape

    Returns
    -------
    shape: BluemiraSolid
        The extruded shape.
    """
    if label is None:
        label = shape.label

    solid = _freecadapi.extrude_shape(shape._shape, vec)
    faces = solid.Faces
    bmfaces = []
    for face in faces:
        bmfaces.append(BluemiraFace._create(face))
    bmshell = BluemiraShell(bmfaces)
    bmsolid = BluemiraSolid(bmshell, label)
    return bmsolid


def distance_to(geo1: BluemiraGeo, geo2: BluemiraGeo):
    """Calculate the distance between two BluemiraGeos.

    Parameters
    ----------
    geo1: BluemiraGeo
        reference shape.
    geo2: BluemiraGeo
        target shape.

    Returns
    -------
    output: a tuple of two -> (dist, vectors)
        dist is the minimum distance (float value)
        vectors is a list of tuples corresponding to the nearest points
        between geo1 and geo2. The distance between those points
        is the minimum distance given by dist.
    """
    shape1 = geo1._shape
    shape2 = geo2._shape
    return _freecadapi.dist_to_shape(shape1, shape2)


# # =============================================================================
# # Save functions
# # =============================================================================
def save_as_STEP(shapes, filename="test", scale=1):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes: Iterable (BluemiraGeo, ...)
        List of shape objects to be saved
    filename: str, default = "test"
        Full path filename of the STP assembly
    scale: float, default = 1.0
        The scale in which to save the Shape objects
    """
    if not filename.endswith(".STP"):
        filename += ".STP"

    if not isinstance(shapes, list):
        shapes = [shapes]

    freecad_shapes = [s._shape for s in shapes]
    _freecadapi.save_as_STEP(freecad_shapes, filename, scale)
