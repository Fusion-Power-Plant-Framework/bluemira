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

import datetime
import inspect
import json
import os
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Type, Union

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull

import bluemira.mesh.meshing as meshing
from bluemira.base.components import Component, get_properties_from_components
from bluemira.base.constants import EPS
from bluemira.base.file import force_file_extension, get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo, GeoMeshable
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire


@cadapi.catch_caderr(GeometryError)
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


class BluemiraGeoEncoder(json.JSONEncoder):
    """
    JSON Encoder for BluemiraGeo.
    """

    def default(self, obj):
        """
        Override the JSONEncoder default object handling behaviour for BluemiraGeo.
        """
        if isinstance(obj, BluemiraGeo):
            return serialize_shape(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _reconstruct_function_call(signature, *args, **kwargs) -> dict:
    """
    Reconstruct the call of a function with inputs arguments and defaults.
    """
    data = {}

    # Inspect the function call and reconstruct defaults
    for i, key in enumerate(signature.parameters.keys()):
        if i < len(args):
            data[key] = args[i]
        else:
            if key not in kwargs:
                value = signature.parameters[key].default
                if value != inspect._empty:
                    data[key] = value
            else:
                data[key] = kwargs[key]

    # Catch any kwargs not in signature
    for k, v in kwargs.items():
        if k not in data:
            data[k] = v
    return data


def _make_debug_file(name) -> str:
    """
    Make a new file in the geometry debugging folder.
    """
    path = get_bluemira_path("generated_data/naughty_geometry", subfolder="")
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M")
    fmt_string = "{}-{}{}.json"
    name = fmt_string.format(name, timestamp, "")
    filename = os.path.join(path, name)

    i = 0
    while os.path.isfile(filename):
        i += 1
        increment = f"_{i}"
        name = fmt_string.format(name, timestamp, increment)
        filename = os.path.join(path, name)
    return filename


def log_geometry_on_failure(func):
    """
    Decorator for debugging of failed geometry operations.
    """
    signature = inspect.signature(func)
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cadapi.FreeCADError as error:
            data = _reconstruct_function_call(signature, *args, **kwargs)
            filename = _make_debug_file(func_name)

            # Dump the data in the file
            try:
                with open(filename, "w") as file:
                    json.dump(data, file, indent=4, cls=BluemiraGeoEncoder)

                bluemira_debug(
                    f"Function call {func_name} failed. Debugging information was saved to: {filename}"
                )
            except Exception:
                bluemira_warn(
                    f"Failed to save the failed geometry operation {func_name} to JSON."
                )

            raise error

    return wrapper


def fallback_to(fallback_func, exception):
    """
    Decorator for a fallback to an alternative geometry operation.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception:
                bluemira_warn(
                    f"{func.__name__} failed, falling back to {fallback_func.__name__}."
                )
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator


# # =============================================================================
# # Geometry creation
# # =============================================================================
def _make_vertex(point):
    """
    Make a vertex.

    Parameters
    ----------
    point: Iterable
        Coordinates of the point

    Returns
    -------
    vertex: apiVertex
        Vertex at the point
    """
    if len(point) != 3:
        raise GeometryError("Points must be of dimension 3.")

    return cadapi.apiVertex(*point)


def closed_wire_wrapper(drop_closure_point):
    """
    Decorator for checking / enforcing closures on wire creation functions.
    """

    def decorator(func):
        def wrapper(points, label="", closed=False):
            points = Coordinates(points)
            if points.closed:
                if closed is False:
                    bluemira_warn(
                        f"{func.__name__}: input points are closed but closed=False, defaulting to closed=True."
                    )
                closed = True
                if drop_closure_point:
                    points = Coordinates(points.points[:-1])
            wire = func(points, label=label, closed=closed)
            if closed:
                wire = cadapi.close_wire(wire)
            return BluemiraWire(wire, label=label)

        return wrapper

    return decorator


@closed_wire_wrapper(drop_closure_point=True)
def make_polygon(
    points: Union[list, np.ndarray], label: str = "", closed: bool = False
) -> BluemiraWire:
    """
    Make a polygon from a set of points.

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

    Notes
    -----
    If the input points are closed, but closed is False, the returned BluemiraWire will
    be closed.
    """
    return cadapi.make_polygon(points.T)


@closed_wire_wrapper(drop_closure_point=False)
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

    Notes
    -----
    If the input points are closed, but closed is False, the returned BluemiraWire will
    be closed.
    """
    return cadapi.make_bezier(points.T)


def make_bspline(
    poles, mults, knots, periodic, degree, weights, check_rational, label: str = ""
):
    """
    Builds a B-Spline by a lists of Poles, Mults, Knots

    Parameters
    ----------
    poles: Union[list, np.ndarray]
        list of poles.
    mults: Union[list, np.ndarray]
        list of integers for the multiplicity
    knots: Union[list, np.ndarray]
        list of knots
    periodic: bool
        Whether or not the spline is periodic (same curvature at start and end points)
    degree: int
        bspline degree
    weights: Union[list, np.ndarray]
        sequence of float
    check_rational: bool
        Whether or not to check if the BSpline is rational

    Returns
    -------
    wire: BluemiraWire
    """
    return BluemiraWire(
        cadapi.make_bspline(
            poles, mults, knots, periodic, degree, weights, check_rational
        ),
        label=label,
    )


def _make_polygon_fallback(points, label="", closed=False, **kwargs) -> BluemiraWire:
    """
    Overloaded function signature for fallback option from interpolate_bspline
    """
    return make_polygon(points, label, closed)


@fallback_to(_make_polygon_fallback, cadapi.FreeCADError)
@log_geometry_on_failure
def interpolate_bspline(
    points: Union[list, np.ndarray],
    label: str = "",
    closed: bool = False,
    start_tangent: Optional[Iterable] = None,
    end_tangent: Optional[Iterable] = None,
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
    start_tangent: Optional[Iterable]
        Tangency of the BSpline at the first pole. Must be specified with end_tangent
    end_tangent: Optional[Iterable]
        Tangency of the BSpline at the last pole. Must be specified with start_tangent

    Returns
    -------
    wire: BluemiraWire
        a bluemira wire that contains the bspline
    """
    points = Coordinates(points)
    return BluemiraWire(
        cadapi.interpolate_bspline(points.T, closed, start_tangent, end_tangent),
        label=label,
    )


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
    wire = bmwire.shape
    closure = BluemiraWire(cadapi.wire_closure(wire), label=label)
    return closure


def _offset_wire_discretised(
    wire,
    thickness,
    /,
    join: str = "intersect",
    open_wire: bool = True,
    label="",
    *,
    fallback_method="square",
    byedges=True,
    ndiscr=200,
    **fallback_kwargs,
) -> BluemiraWire:
    """
    Fallback function for discretised offsetting

    Raises
    ------
    GeometryError
        If the wire is not closed. This function cannot handle the offet of an open
        wire.
    """
    from bluemira.geometry._pyclipper_offset import offset_clipper

    if not wire.is_closed() and not open_wire:
        wire = wire.deepcopy()
        wire.close()

    if not wire.is_closed() and open_wire:
        raise GeometryError(
            "Fallback function _offset_wire_discretised cannot handle open wires."
        )

    coordinates = wire.discretize(byedges=byedges, ndiscr=ndiscr)

    result = offset_clipper(
        coordinates, thickness, method=fallback_method, **fallback_kwargs
    )
    return make_polygon(result, label=label, closed=True)


@fallback_to(_offset_wire_discretised, cadapi.FreeCADError)
@log_geometry_on_failure
def offset_wire(
    wire: BluemiraWire,
    thickness: float,
    /,
    join: str = "intersect",
    open_wire: bool = True,
    label: str = "",
    *,
    fallback_method="square",
    byedges=True,
    ndiscr=400,
    **fallback_kwargs,
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

    Other Parameters
    ----------------
    byedges: bool (default = True)
        Whether or not to discretise the wire by edges
    ndiscr: int (default = 200)
        Number of points to discretise the wire to
    fallback_method: str
        Method to use in discretised offsetting, will default to `square` as `round`
        is know to be very slow

    Notes
    -----
    If primitive offsetting failed, will fall back to a discretised offset
    implementation, where the fallback kwargs are used. Discretised offsetting is
    only supported for closed wires.

    Returns
    -------
    wire: BluemiraWire
        Offset wire
    """
    return BluemiraWire(
        cadapi.offset_wire(wire.shape, thickness, join, open_wire), label=label
    )


def convex_hull_wires_2d(
    wires: Sequence[BluemiraWire], ndiscr: int, plane="xz"
) -> BluemiraWire:
    """
    Perform a convex hull around the given wires and return the hull
    as a new wire.

    The operation performs discretisations on the input wires.

    Parameters
    ----------
    wires: Sequence[BluemiraWire]
        The wires to draw a hull around.
    ndiscr: int
        The number of points to discretise each wire into.
    plane: str
        The plane to perform the hull in. One of: 'xz', 'xy', 'yz'.
        Default is 'xz'.

    Returns
    -------
    hull: BluemiraWire
        A wire forming a convex hull around the input wires in the given
        plane.
    """
    if not wires:
        raise ValueError("Must have at least one wire to draw a hull around.")
    if plane == "xz":
        plane_idxs = (0, 2)
    elif plane == "xy":
        plane_idxs = (0, 1)
    elif plane == "yz":
        plane_idxs = (1, 2)
    else:
        raise ValueError(f"Invalid plane: '{plane}'. Must be one of 'xz', 'xy', 'yz'.")

    shape_discretizations = []
    for wire in wires:
        discretized_points = wire.discretize(byedges=True, ndiscr=ndiscr)
        shape_discretizations.append(getattr(discretized_points, plane))
    coords = np.hstack(shape_discretizations)

    hull = ConvexHull(coords.T)
    hull_coords = np.zeros((3, len(hull.vertices)))
    hull_coords[plane_idxs, :] = coords[:, hull.vertices]
    return make_polygon(hull_coords, closed=True)


# # =============================================================================
# # Shape operation
# # =============================================================================
def revolve_shape(
    shape,
    base: Iterable = (0.0, 0.0, 0.0),
    direction: Iterable = (0.0, 0.0, 1.0),
    degree: float = 180,
    label: str = "",
):
    """
    Apply the revolve (base, dir, degree) to this shape

    Parameters
    ----------
    shape: BluemiraGeo
        The shape to be revolved
    base: Iterable (x,y,z), default = (0.0, 0.0, 0.0)
        Origin location of the revolution
    direction: Iterable (x,y,z), default = (0.0, 0.0, 1.0)
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
            shape = BluemiraFace(shape).shape
            flag_shell = True
        else:
            shape = shape.shape
            flag_shell = False

        shape_1 = cadapi.revolve_shape(shape, base, direction, degree=180)
        shape_2 = deepcopy(shape_1)
        shape_2 = cadapi.rotate_shape(shape_2, base, direction, degree=-180)
        result = cadapi.boolean_fuse([shape_1, shape_2], remove_splitter=False)

        if flag_shell:
            result = result.Shells[0]

        return convert(result, label)

    return convert(cadapi.revolve_shape(shape.shape, base, direction, degree), label)


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

    return convert(cadapi.extrude_shape(shape.shape, vec), label)


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

    profile_shapes = [p.shape for p in profiles]

    result = cadapi.sweep_shape(profile_shapes, path.shape, solid, frenet)

    return convert(result, label=label)


def distance_to(
    geo1: Union[Iterable[float], BluemiraGeo], geo2: Union[Iterable[float], BluemiraGeo]
):
    """
    Calculate the distance between two BluemiraGeos.

    Parameters
    ----------
    geo1: Union[Iterable[float], BluemiraGeo]
        Reference shape. If an iterable of length 3, converted to a point.
    geo2: Union[Iterable[float], BluemiraGeo]
        Target shape. If an iterable of length 3, converted to a point.

    Returns
    -------
    dist: float
        Minimum distance
    vectors: List[Tuple]
        List of tuples corresponding to the nearest points between geo1 and geo2. The
        distance between those points is the minimum distance given by dist.
    """
    # Check geometry for vertices
    if isinstance(geo1, Iterable):
        shape1 = _make_vertex(geo1)
    else:
        shape1 = geo1.shape
    if isinstance(geo2, Iterable):
        shape2 = _make_vertex(geo2)
    else:
        shape2 = geo2.shape
    return cadapi.dist_to_shape(shape1, shape2)


def split_wire(wire: BluemiraWire, vertex: Iterable, tolerance: float = EPS):
    """
    Split a wire at a given vertex.

    Parameters
    ----------
    wire: apiWire
        Wire to be split
    vertex: Iterable
        Vertex at which to split the wire
    tolerance: float
        Tolerance within which to find the closest vertex on the wire

    Returns
    -------
    wire_1: Optional[BluemiraWire]
        First half of the wire. Will be None if the vertex is the start point of the wire
    wire_2: Optional[BluemiraWire]
        Last half of the wire. Will be None if the vertex is the start point of the wire

    Raises
    ------
    GeometryError:
        If the vertex is further away to the wire than the specified tolerance
    """
    wire_1, wire_2 = cadapi.split_wire(wire.shape, vertex, tolerance=tolerance)
    if wire_1:
        wire_1 = BluemiraWire(wire_1)
    if wire_2:
        wire_2 = BluemiraWire(wire_2)
    return wire_1, wire_2


def slice_shape(shape: BluemiraGeo, plane: BluemiraPlane):
    """
    Calculate the plane intersection points with an object

    Parameters
    ----------
    shape: Union[BluemiraWire, BluemiraFace, BluemiraSolid, BluemiraShell]
        obj to intersect with a plane
    plane: BluemiraPlane

    Returns
    -------
    Wire: Union[List[np.ndarray], None]
        returns array of intersection points
    Face, Solid, Shell: Union[List[BluemiraWire], None]
        list of intersections lines

    Notes
    -----
    Degenerate cases such as tangets to solid or faces do not return intersections
    if the shape and plane are acting at the Plane base.
    Further investigation needed.

    """
    _slice = cadapi.slice_shape(shape.shape, plane.base, plane.axis)

    if isinstance(_slice, np.ndarray) and _slice.size > 0:
        return _slice

    _slice = [convert(obj) for obj in _slice]

    if len(_slice) > 0:
        return _slice


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


def mirror_shape(
    shape: BluemiraGeo, base: tuple, direction: tuple, label=""
) -> BluemiraGeo:
    """
    Get a mirrored copy of a shape about a plane.

    Parameters
    ----------
    shape:
        Shape to mirror
    base:
        Mirror plane base
    direction:
        Mirror plane normal direction

    Returns
    -------
    mirrored_shape
        The mirrored shape
    """
    if np.isclose(np.linalg.norm(direction), EPS):
        raise GeometryError("Direction vector cannot have a zero norm.")
    return convert(cadapi.mirror_shape(shape.shape, base, direction), label=label)


# # =============================================================================
# # Save functions
# # =============================================================================
def save_as_STP(
    shapes: Union[BluemiraGeo, Iterable[BluemiraGeo]], filename: str, scale: float = 1
):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes: Iterable (BluemiraGeo, ...)
        List of shape objects to be saved
    filename: str
        Full path filename of the STP assembly
    scale: float, default = 1.0
        The scale in which to save the Shape objects
    """
    filename = force_file_extension(filename, [".stp", ".step"])

    if not isinstance(shapes, list):
        shapes = [shapes]

    freecad_shapes = [s.shape for s in shapes]
    cadapi.save_as_STP(freecad_shapes, filename, scale)


def save_cad(
    components: Union[Component, Iterable[Component]], filename: str, scale: float = 1
):
    """
    Save the CAD of a component (eg a reactor) or a list of components

    Parameters
    ----------
    components: Union[Component, Iterable[Component]]
        components to save
    filename: str
        Full path filename of the STP assembly
    scale: float, default = 1.0
        The scale in which to save the Shape objects
    """
    save_as_STP(get_properties_from_components(components, "shape"), filename, scale)


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

    api_shapes = [s.shape for s in shapes]
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
    apishape = shape.shape
    if not isinstance(tools, list):
        tools = [tools]
    apitools = [t.shape for t in tools]
    cut_shape = cadapi.boolean_cut(apishape, apitools)

    if isinstance(cut_shape, Iterable):
        return [convert(obj, shape.label) for obj in cut_shape]

    return convert(cut_shape, shape.label)


def point_inside_shape(point, shape):
    """
    Check whether or not a point is inside a shape.

    Parameters
    ----------
    point: Iterable(3)
        Coordinates of the point
    shape: BluemiraGeo
        Geometry to check with

    Returns
    -------
    inside: bool
        Whether or not the point is inside the shape
    """
    return cadapi.point_inside_shape(point, shape.shape)


def point_on_plane(point, plane, tolerance=D_TOLERANCE):
    """
    Check whether or not a point is on a plane.

    Parameters
    ----------
    point: Iterable
        Coordinates of the point
    plane: BluemiraPlane
        Plane to check
    tolerance: float
        Tolerance with which to check

    Returns
    -------
    point_on_plane: bool
        Whether or not the point is on the plane
    """
    return (
        abs(
            cadapi.apiVector(point).distanceToPlane(
                plane.shape.Position, plane.shape.Axis
            )
        )
        < tolerance
    )


# # =============================================================================
# # Serialize and Deserialize
# # =============================================================================
def serialize_shape(shape: BluemiraGeo):
    """
    Serialize a BluemiraGeo object.
    """
    type_ = type(shape)

    output = []
    if isinstance(shape, BluemiraGeo):
        sdict = {"label": shape.label, "boundary": output}
        for obj in shape.boundary:
            output.append(serialize_shape(obj))
            if isinstance(shape, GeoMeshable) and shape.mesh_options is not None:
                if shape.mesh_options.lcar is not None:
                    sdict["lcar"] = shape.mesh_options.lcar
                if shape.mesh_options.physical_group is not None:
                    sdict["physical_group"] = shape.mesh_options.physical_group
        return {str(type(shape).__name__): sdict}
    elif isinstance(shape, cadapi.apiWire):
        return cadapi.serialize_shape(shape)
    else:
        raise NotImplementedError(f"Serialization non implemented for {type_}")


def deserialize_shape(buffer: dict):
    """
    Deserialize a BluemiraGeo object obtained from serialize_shape.

    Parameters
    ----------
    buffer
        Object serialization as stored by serialize_shape

    Returns
    -------
        The deserialized BluemiraGeo object.
    """
    supported_types = [BluemiraWire, BluemiraFace, BluemiraShell]

    def _extract_mesh_options(shape_dict: dict):
        mesh_options = None
        if "lcar" in shape_dict:
            mesh_options = meshing.MeshOptions()
            mesh_options.lcar = shape_dict["lcar"]
        if "physical_group" in shape_dict:
            mesh_options = mesh_options or meshing.MeshOptions()
            mesh_options.physical_group = shape_dict["physical_group"]
        return mesh_options

    def _extract_shape(shape_dict: dict, shape_type: Type[BluemiraGeo]):
        label = shape_dict["label"]
        boundary = shape_dict["boundary"]

        temp_list = []
        for item in boundary:
            if issubclass(shape_type, BluemiraWire):
                for k in item:
                    if k == shape_type.__name__:
                        shape = deserialize_shape(item)
                    else:
                        shape = cadapi.deserialize_shape(item)
                    temp_list.append(shape)
            else:
                temp_list.append(deserialize_shape(item))

        mesh_options = _extract_mesh_options(shape_dict)

        shape = shape_type(label=label, boundary=temp_list)
        if mesh_options is not None:
            shape.mesh_options = mesh_options
        return shape

    for type_, v in buffer.items():
        for supported_types in supported_types:
            if type_ == supported_types.__name__:
                return _extract_shape(v, BluemiraWire)

        raise NotImplementedError(f"Deserialization non implemented for {type_}")


# # =============================================================================
# # shape utils
# # =============================================================================
def get_shape_by_name(shape: BluemiraGeo, name: str):
    """
    Search through the boundary of the shape and get any shapes with a label
    corresponding to the provided name. Includes the shape itself if the name matches
    its label.

    Parameters
    ----------
    shape: BluemiraGeo
        The shape to search for the provided name.
    name: str
        The name to search for.

    Returns
    -------
    shapes: List[BluemiraGeo]
        The shapes known to the provided shape that correspond to the provided name.
    """
    shapes = []
    if hasattr(shape, "label") and shape.label == name:
        shapes.append(shape)
    if hasattr(shape, "boundary"):
        for o in shape.boundary:
            shapes += get_shape_by_name(o, name)
    return shapes


# ======================================================================================
# Find operations
# ======================================================================================
def find_clockwise_angle_2d(base: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Find the clockwise angle between the 2D vectors ``base`` and
    ``vector`` in the range [0°, 360°).

    Parameters
    ----------
    base: np.ndarray[float, (2, N)]
        The vector to start the angle from.
    vector: np.ndarray[float, (2, N)]
        The vector to end the angle at.

    Returns
    -------
    angle: np.ndarray[float, (1, N)]
        The clockwise angle between the two vectors in degrees.
    """
    if not isinstance(base, np.ndarray) or not isinstance(vector, np.ndarray):
        raise TypeError(
            f"Input vectors must have type np.ndarray, found '{type(base)}' and "
            f"'{type(vector)}'."
        )
    if base.shape[0] != 2 or vector.shape[0] != 2:
        raise ValueError(
            f"Input vectors' axis 0 length must be 2, found shapes '{base.shape}' and "
            f"'{vector.shape}'."
        )
    det = base[1] * vector[0] - base[0] * vector[1]
    dot = np.dot(base, vector)
    # Convert to array here in case arctan2 returns a scalar
    angle = np.array(np.arctan2(det, dot))
    angle[angle < 0] += 2 * np.pi
    return np.degrees(angle)
