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

from copy import deepcopy
from typing import Iterable, List, Union

import numba as nb
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.coordinates import Coordinates


def convert(apiobj, label=""):
    """Convert a FreeCAD shape into the corresponding BluemiraGeo object."""
    if isinstance(apiobj, cadapi.apiWire):
        output = BluemiraWire(apiobj, label)
    elif isinstance(apiobj, cadapi.apiFace):
        output = BluemiraFace._create(apiobj, label)
    elif isinstance(apiobj, cadapi.apiShell):
        output = BluemiraShell._create(apiobj, label)
    elif isinstance(apiobj, cadapi.apiSolid):
        output = BluemiraSolid._create(apiobj, label)
    else:
        raise ValueError(f"Cannot convert {type(apiobj)} object into a BluemiraGeo.")
    return output


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
    points = Coordinates(points).T
    return BluemiraWire(cadapi.make_polygon(points, closed), label=label)


def make_bspline(
    points: Union[list, np.ndarray],
    label: str = "",
    closed: bool = False,
) -> BluemiraWire:
    """
    Make a bspline from a set of points.

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
    points = Coordinates(points).T
    return BluemiraWire(cadapi.make_bspline(points, closed), label=label)


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
    points = Coordinates(points).T
    return BluemiraWire(cadapi.make_bezier(points, closed), label=label)


def make_circle(
    radius=1.0,
    center=(0.0, 0.0, 0.0),
    start_angle=0.0,
    end_angle=360.0,
    axis=(0.0, 0.0, 1.0),
    label: str = "",
) -> BluemiraWire:
    """
    Create a circle or arc of circle object with given parameters.

    Parameters
    ----------
    radius: float, default =1.0
        Radius of the circle
    center: Iterable, default = (0, 0, 0)
        Center of the circle
    start_angle: float, default = 0.0
        Start angle of the arc [degrees]
    end_angle: float, default = 360.0
        End angle of the arc [degrees]. If start_angle == end_angle, a circle is created,
        otherwise a circle arc is created
    axis: Iterable, default = (0, 0, 1)
        Normal vector to the circle plane. It defines the clockwise/anticlockwise
        circle orientation according to the right hand rule.
    label: str
        object's label

    Returns
    -------
    wire: BluemiraWire
        bluemira wire that contains the arc or circle
    """
    output = cadapi.make_circle(radius, center, start_angle, end_angle, axis)
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
    output = cadapi.make_circle_arc_3P(p1, p2, p3)
    return BluemiraWire(output, label=label)


def make_ellipse(
    center=(0.0, 0.0, 0.0),
    major_radius=2.0,
    minor_radius=1.0,
    major_axis=(1, 0, 0),
    minor_axis=(0, 1, 0),
    start_angle=0.0,
    end_angle=360.0,
    label: str = "",
):
    """
    Create an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center: Iterable, default = (0, 0, 0)
        Center of the ellipse
    major_radius: float, default = 2
        Major radius of the ellipse
    minor_radius: float, default = 2
        Minor radius of the ellipse (float). Default to 2.
    major_axis: Iterable, default = (1, 0, 0)
        Major axis direction
    minor_axis: Iterable, default = (0, 1, 0)
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
    output = cadapi.make_ellipse(
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
    """
    Close this wire with a line segment

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
    closure = BluemiraWire(cadapi.wire_closure(wire), label=label)
    return closure


def offset_wire(
    wire: BluemiraWire,
    thickness: float,
    join: str = "intersect",
    open_wire: bool = True,
    label: str = "",
) -> BluemiraWire:
    """
    Make a planar offset from a planar wire.

    Parameters
    ----------
    wire: BluemiraWire
        Wire to offset from
    thickness: float
        Offset distance. Positive values outwards, negative values inwards
    join: str
        Offset method. "arc" gives rounded corners, and "intersect" gives sharp corners
    open_wire: bool
        For open wires (counter-clockwise default) whether or not to make an open offset
        wire, or a closed offset wire that encompasses the original wire. This is
        disabled for closed wires.

    Returns
    -------
    wire: BluemiraWire
        Offset wire
    """
    return BluemiraWire(
        cadapi.offset_wire(wire._shape, thickness, join, open_wire), label=label
    )


# # =============================================================================
# # Shape operation
# # =============================================================================
def revolve_shape(
    shape,
    base: tuple = (0.0, 0.0, 0.0),
    direction: tuple = (0.0, 0.0, 1.0),
    degree: float = 180,
    label: str = "",
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
    shape: Union[BluemiraShell, BluemiraSolid]
        the revolved shape.
    """
    if degree > 360:
        bluemira_warn("Cannot revolve a shape by more than 360 degrees.")
        degree = 360

    if degree == 360:
        # We split into two separate revolutions of 180 degree and fuse them
        if isinstance(shape, BluemiraWire):
            shape = BluemiraFace(shape)._shape
            flag_shell = True
        else:
            shape = shape._shape
            flag_shell = False

        shape_1 = cadapi.revolve_shape(shape, base, direction, degree=180)
        shape_2 = deepcopy(shape_1)
        shape_2 = cadapi.rotate_shape(shape_2, base, direction, degree=180)
        result = cadapi.boolean_fuse([shape_1, shape_2])

        if flag_shell:
            result = result.Shells[0]

        return convert(result, label)

    return convert(cadapi.revolve_shape(shape._shape, base, direction, degree), label)


def extrude_shape(shape: BluemiraGeo, vec: tuple, label="") -> BluemiraSolid:
    """
    Apply the extrusion along vec to this shape

    Parameters
    ----------
    shape: BluemiraGeo
        The shape to be extruded
    vec: tuple (x,y,z)
        The vector along which to extrude
    label: str, default = ""
        label of the output shape

    Returns
    -------
    shape: BluemiraSolid
        The extruded shape.
    """
    if not label:
        label = shape.label

    return convert(cadapi.extrude_shape(shape._shape, vec), label)


def sweep_shape(profiles, path, solid=True, frenet=True, label=""):
    """
    Sweep a profile along a path.

    Parameters
    ----------
    profiles: BluemiraWire
        Profile to sweep
    path: BluemiraWire
        Path along which to sweep the profiles
    solid: bool
        Whether or not to create a Solid
    frenet: bool
        If true, the orientation of the profile(s) is calculated based on local curvature
        and tangency. For planar paths, should not make a difference.

    Returns
    -------
    swept: Union[BluemiraSolid, BluemiraShell]
        Swept geometry object
    """
    if not isinstance(profiles, Iterable):
        profiles = [profiles]

    profile_shapes = [p._shape for p in profiles]

    result = cadapi.sweep_shape(profile_shapes, path._shape, solid, frenet)

    return convert(result, label=label)


def distance_to(geo1: BluemiraGeo, geo2: BluemiraGeo):
    """
    Calculate the distance between two BluemiraGeos.

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
    return cadapi.dist_to_shape(shape1, shape2)


def circular_pattern(
    shape, origin=(0, 0, 0), direction=(0, 0, 1), degree=360, n_shapes=10
) -> List[BluemiraGeo]:
    """
    Make a equally spaced circular pattern of shapes.

    Parameters
    ----------
    shape: BluemiraGeo
        Shape to pattern
    origin: Iterable(3)
        Origin vector of the circular pattern
    direction: Iterable(3)
        Direction vector of the circular pattern
    degree: float
        Angle range of the patterning
    n_shapes: int
        Number of shapes to pattern

    Returns
    -------
    shapes: List[BluemiraGeo]
        List of patterned shapes, the first element is the original shape
    """
    angle = degree / n_shapes

    shapes = [shape]
    for i in range(1, n_shapes):
        new_shape = shape.deepcopy()
        new_shape.rotate(origin, direction, i * angle)
        shapes.append(new_shape)
    return shapes


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
    cadapi.save_as_STEP(freecad_shapes, filename, scale)


# ======================================================================================
# Signed distance functions
# ======================================================================================


@nb.jit(nopython=True, cache=True)
def _nb_dot_2D(v_1, v_2):
    """
    Numba 2-D dot product
    """
    return v_1[0] * v_2[0] + v_1[1] * v_2[1]


@nb.jit(nopython=True, cache=True)
def _nb_clip(val, a_min, a_max):
    """
    Numba 1-D clip
    """
    return a_min if val < a_min else a_max if val > a_max else val


@nb.jit(nopython=True, cache=True)
def _signed_distance_2D(point, polygon):
    """
    2-D function for the signed distance from a point to a polygon. The return value is
    negative if the point is outside the polygon, and positive if the point is inside the
    polygon.

    Parameters
    ----------
    point: np.ndarray(2)
        2-D point
    polygon: np.ndarray(n, 2)
        2-D set of point coordinates

    Returns
    -------
    signed_distance: float
        Signed distance value of the point to the polygon

    Notes
    -----
    Credit: Inigo Quilez (https://www.iquilezles.org/)
    """
    sign = -1.0
    point = np.asfarray(point)
    polygon = np.asfarray(polygon)
    n = len(polygon)

    d = _nb_dot_2D(point - polygon[0], point - polygon[0])

    for i in range(n - 1):
        j = i + 1
        e = polygon[j] - polygon[i]
        w = point - polygon[i]
        b = w - e * _nb_clip(_nb_dot_2D(w, e) / _nb_dot_2D(e, e), 0.0, 1.0)
        d_new = _nb_dot_2D(b, b)
        if d_new < d:
            d = d_new

        cond = np.array(
            [
                point[1] >= polygon[i][1],
                point[1] < polygon[j][1],
                e[0] * w[1] > e[1] * w[0],
            ]
        )
        if np.all(cond) or np.all(~cond):
            sign = -sign

    return sign * np.sqrt(d)


@nb.jit(nopython=True, cache=True)
def signed_distance_2D_polygon(subject_poly, target_poly):
    """
    2-D vector-valued signed distance function from a subject polygon to a target
    polygon. The return values are negative for points outside the subject polygon, and
    positive for points inside the subject polygon.

    Parameters
    ----------
    subject_poly: np.ndarray(n, 2)
        Subject polygon
    target_poly: np.ndarray(m, 2)
        Target polygon

    Returns
    -------
    signed_distance: np.ndarray(n)
        Signed distances from the subject polygon to the target polygon
    """
    m = len(subject_poly)
    d = np.zeros(m)

    for i in range(m):
        d[i] = _signed_distance_2D(subject_poly[i], target_poly)

    return d


def signed_distance(wire_1, wire_2):
    """
    Single-valued signed "distance" function between two wires. Will return negative
    values if wire_1 does not touch or intersect wire_2, 0 if there is one intersection,
    and a positive estimate of the intersection length if there are overlaps.

    Parameters
    ----------
    wire_1: BluemiraWire
        Subject wire
    wire_2: BluemiraWire
        Target wire

    Returns
    -------
    signed_distance: float
        Signed distance from wire_1 to wire_2

    Notes
    -----
    This is not a pure implementation of a distance function, as for overlapping wires a
    metric of the quantity of overlap is returned (a positive value). This nevertheless
    enables the use of such a function as a constraint in gradient-based optimisers.
    """
    d, vectors = distance_to(wire_1, wire_2)

    if d == 0.0:  # Intersections are exactly 0.0
        if len(vectors) <= 1:
            # There is only one intersection: the wires are touching but not overlapping
            return 0.0
        else:
            # There are multiple intersections: the wires are overlapping
            # For now, without boolean operations, get an estimate of the intersection
            # length
            length = 0
            for i in range(1, len(vectors)):
                p1 = vectors[i - 1][0]
                p2 = vectors[i][0]

                length += np.sqrt(
                    (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
                )

            # TODO: Use a boolean difference operation to get the lengths of the
            # overlapping wire segment(s)
            return length
    else:
        # There are no intersections, return minimum distance
        return -d


# ======================================================================================
# Boolean operations
# ======================================================================================
def boolean_fuse(shapes, label=""):
    """
    Fuse two or more shapes together. Internal splitter are removed.

    Parameters
    ----------
    shapes: Iterable (BluemiraGeo, ...)
        List of shape objects to be saved
    label: str
        Label for the resulting shape

    Returns
    -------
    merged_geo: BluemiraGeo
        Result of the boolean operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    if not isinstance(shapes, list):
        raise ValueError(f"{shapes} is not a list.")

    if len(shapes) < 2:
        raise ValueError("At least 2 shapes must be given")

    # check that all the shapes are of the same time
    _type = type(shapes[0])
    if not all(isinstance(s, _type) for s in shapes):
        raise ValueError(f"All instances in {shapes} must be of the same type.")

    api_shapes = [s._shape for s in shapes]
    try:
        merged_shape = cadapi.boolean_fuse(api_shapes)
        return convert(merged_shape, label)

    except Exception as e:
        raise GeometryError(f"Boolean fuse operation failed: {e}")


def boolean_cut(shape, tools):
    """
    Difference of shape and a given (list of) topo shape cut(tools)

    Parameters
    ----------
    shape: BluemiraGeo
        the reference object
    tools: Iterable
        List of BluemiraGeo shape objects to be used as tools.

    Returns
    -------
    cut_shape:
        Result of the boolean operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    apishape = shape._shape
    if not isinstance(tools, list):
        tools = [tools]
    apitools = [t._shape for t in tools]
    cut_shape = cadapi.boolean_cut(apishape, apitools)

    if isinstance(cut_shape, Iterable):
        return [convert(obj, shape.label) for obj in cut_shape]

    return convert(cut_shape, shape.label)
