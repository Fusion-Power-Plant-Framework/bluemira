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
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

import freecad  # noqa (F401)
import Part
import FreeCAD
from FreeCAD import Base
import BOPTools
import BOPTools.SplitAPI
import BOPTools.GeneralFuseResult
import BOPTools.JoinAPI
import BOPTools.JoinFeatures
import BOPTools.ShapeMerge
import BOPTools.Utils
import BOPTools.SplitFeatures
import FreeCADGui

# import math lib
import numpy as np
import math

# import typing
from typing import List, Optional, Iterable, Union, Dict

# import errors and warnings
from bluemira.codes.error import FreeCADError
from bluemira.base.look_and_feel import bluemira_warn

from bluemira.base.constants import EPS
from bluemira.geometry.constants import MINIMUM_LENGTH

# import visualisation
from pivy import coin, quarter
from PySide2.QtWidgets import QApplication

apiWire = Part.Wire  # noqa (N816)
apiFace = Part.Face  # noqa (N816)
apiShell = Part.Shell  # noqa (N816)
apiSolid = Part.Solid  # noqa (N816)
apiShape = Part.Shape  # noqa (N816)
apiCompound = Part.Compound  # noqa (N816)

# ======================================================================================
# Array, List, Vector, Point manipulation
# ======================================================================================


def check_data_type(data_type):
    """
    Decorator to check the data type of the first parameter input (args[0]) of a
    function.

    Raises
    ------
    TypeError: If args[0] objects are not instances of data_type
    """

    def _apply_to_list(func):
        def wrapper(*args, **kwargs):
            output = []
            objs = args[0]
            is_list = isinstance(objs, list)
            if not is_list:
                objs = [objs]
                if len(args) > 1:
                    args = [objs, args[1:]]
                else:
                    args = [objs]
            if all(isinstance(o, data_type) for o in objs):
                output = func(*args, **kwargs)
                if not is_list:
                    output = output[0]
            else:
                raise TypeError(
                    f"Only {data_type} instances can be converted to {type(output)}"
                )
            return output

        return wrapper

    return _apply_to_list


@check_data_type(Base.Vector)
def vector_to_list(vectors):
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a list"""
    return [list(v) for v in vectors]


@check_data_type(Part.Point)
def point_to_list(points):
    """Converts a FreeCAD Part.Point or list(Part.Point) into a list"""
    return [[p.X, p.Y, p.Z] for p in points]


@check_data_type(Part.Vertex)
def vertex_to_list(vertexes):
    """Converts a FreeCAD Part.Vertex or list(Part.Vertex) into a list"""
    return [[v.X, v.Y, v.Z] for v in vertexes]


@check_data_type(Base.Vector)
def vector_to_numpy(vectors):
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a numpy array"""
    return np.array([np.array(v) for v in vectors])


@check_data_type(Part.Point)
def point_to_numpy(points):
    """Converts a FreeCAD Part.Point or list(Part.Point) into a numpy array"""
    return np.array([np.array([p.X, p.Y, p.Z]) for p in points])


@check_data_type(Part.Vertex)
def vertex_to_numpy(vertexes):
    """Converts a FreeCAD Part.Vertex or list(Part.Vertex) into a numpy array"""
    return np.array([np.array([v.X, v.Y, v.Z]) for v in vertexes])


# ======================================================================================
# Geometry creation
# ======================================================================================


def make_solid(shell: apiShell):
    """Make a solid from a shell."""
    return Part.makeSolid(shell)


def make_shell(faces: List[apiFace]):
    """Make a shell from faces."""
    return Part.makeShell(faces)


def make_compound(shapes):
    """
    Make an FreeCAD compound object out of many shapes

    Parameters
    ----------
    shapes: list of FreeCAD shape objects
        A set of objects to be compounded

    Returns
    -------
    compound: FreeCAD compound object
        A compounded set of shapes
    """
    return Part.makeCompound(shapes)


def make_polygon(points: Union[list, np.ndarray], closed: bool = False) -> Part.Wire:
    """
    Make a polygon from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed shape.

    Returns
    -------
    wire: Part.Wire
        a FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    wire = Part.makePolygon(pntslist)
    if closed:
        wire = close_wire(wire)
    return wire


def make_bezier(points: Union[list, np.ndarray], closed: bool = False) -> Part.Wire:
    """
    Make a bezier curve from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed shape.

    Returns
    -------
    wire: Part.Wire
        a FreeCAD wire that contains the bezier curve
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    bc = Part.BezierCurve()
    bc.setPoles(pntslist)
    wire = Part.Wire(bc.toShape())
    if closed:
        wire = close_wire(wire)
    return wire


def make_bspline(
    points: Union[list, np.ndarray],
    closed: bool = False,
    **kwargs,
) -> Part.Wire:
    """
    Make a bezier curve from a set of points.

    Parameters
    ----------
    points: Union[list, np.ndarray]
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    closed: bool, default = False
        if True, the first and last points will be connected in order to form a
        closed shape.

    Other Parameters
    ----------------
        knot sequence

    Returns
    -------
    wire: apiWire
        a FreeCAD wire that contains the bezier curve
    """
    # In this case, it is not really necessary to convert points in FreeCAD vector. Just
    # left for consistency with other methods.
    # TODO: Add support for start and end tangencies.. I tried but I don't think FreeCAD
    # wraps OCC enough here.
    pntslist = [Base.Vector(x) for x in points]
    bsc = Part.BSplineCurve()
    bsc.interpolate(pntslist, PeriodicFlag=closed, **kwargs)
    wire = apiWire(bsc.toShape())
    return wire


def make_circle(
    radius=1.0,
    center=[0.0, 0.0, 0.0],
    start_angle=0.0,
    end_angle=360.0,
    axis=[0.0, 0.0, 1.0],
):
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
        circle orientation according to the right hand rule. Default [0., 0., 1.].

    Returns
    -------
    wire: Part.Wire
        FreeCAD wire that contains the arc or circle
    """
    # TODO: check the creation of the arc when start_angle < end_angle
    output = Part.Circle()
    output.Radius = radius
    output.Center = Base.Vector(center)
    output.Axis = Base.Vector(axis)
    if start_angle != end_angle:
        output = Part.ArcOfCircle(
            output, math.radians(start_angle), math.radians(end_angle)
        )
    return Part.Wire(Part.Edge(output))


def make_circle_arc_3P(p1, p2, p3):  # noqa: N802
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
    wire: Part.Wire
        FreeCAD wire that contains the arc of circle
    """
    # TODO: check what happens when the 3 points are in a line
    arc = Part.ArcOfCircle(Base.Vector(p1), Base.Vector(p2), Base.Vector(p3))
    return Part.Wire(Part.Edge(arc))


def make_ellipse(
    center=[0.0, 0.0, 0.0],
    major_radius=2.0,
    minor_radius=1.0,
    major_axis=[1, 0, 0],
    minor_axis=[0, 1, 0],
    start_angle=0.0,
    end_angle=360.0,
):
    """
    Creates an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center: Iterable, default = [0, 0, 0]
        Center of the ellipse
    major_radius: float, default = 2
        the major radius of the ellipse
    minor_radius: float, default = 1
        the minor radius of the ellipse
    major_axis: Iterable, default = [1, 0, 0,]
        major axis direction
    minor_axis: Iterable, default = [0, 1, 0,]
        minor axis direction
    start_angle: float, default = 0.0
        Start angle of the arc [degrees]
    end_angle: float, default = 360.0
        End angle of the arc [degrees]. If start_angle == end_angle, an ellipse is
        created, otherwise an arc of ellipse is created

    Returns
    -------
    wire: Part.Wire
        FreeCAD wire that contains the ellipse or arc of ellipse
    """
    # TODO: check the creation of the arc when start_angle < end_angle
    s1 = Base.Vector(major_axis).normalize().multiply(major_radius)
    s2 = Base.Vector(minor_axis).normalize().multiply(minor_radius)
    center = Base.Vector(center)
    output = Part.Ellipse(s1, s2, center)

    start_angle = start_angle % 360.0
    end_angle = end_angle % 360.0

    if start_angle != end_angle:
        output = Part.ArcOfEllipse(
            output, math.radians(start_angle), math.radians(end_angle)
        )

    return Part.Wire(Part.Edge(output))


def _wire_is_planar(wire):
    """
    Check if a wire is planar.
    """
    try:
        face = Part.Face(wire)
    except Part.OCCError:
        return False
    return isinstance(face.Surface, Part.Plane)


def _wire_is_straight(wire):
    """
    Check if a wire is a straight line.
    """
    if len(wire.Edges) == 1:
        edge = wire.Edges[0]
        if len(edge.Vertexes) == 2:
            straight = dist_to_shape(edge.Vertexes[0], edge.Vertexes[1])[0]
            if np.isclose(straight, wire.Length, rtol=EPS, atol=1e-8):
                return True
    return False


def offset_wire(
    wire: apiWire, thickness: float, join: str = "intersect", open_wire: bool = True
) -> apiWire:
    """
    Make an offset from a wire.

    Parameters
    ----------
    wire: Part.Wire
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
    wire: Part.Wire
        Offset wire
    """
    if _wire_is_straight(wire):
        raise FreeCADError("Cannot offset a straight line.")

    if not _wire_is_planar(wire):
        raise FreeCADError("Cannot offset a non-planar wire.")

    if join == "arc":
        f_join = 0
    elif join == "intersect":
        f_join = 2
    else:
        # NOTE: The "tangent": 1 option misbehaves in FreeCAD
        raise FreeCADError(
            f"Unrecognised join value: {join}. Please choose from ['arc', 'intersect']."
        )

    if wire.isClosed() and open_wire:
        open_wire = False

    shape = apiShape(wire)
    try:
        wire = apiWire(shape.makeOffset2D(thickness, f_join, False, open_wire))
    except Base.FreeCADError as error:
        msg = "\n".join(
            [
                "FreeCAD was unable to make an offset of wire:",
                f"{error.args[0]['sErrMsg']}",
            ]
        )
        raise FreeCADError(msg)

    fix_wire(wire)
    return wire


# ======================================================================================
# Object properties
# ======================================================================================
def _get_api_attr(obj, prop):
    try:
        return getattr(obj, prop)
    except AttributeError:
        raise FreeCADError(f"FreeCAD object {obj} does not have an attribute: {prop}")


def length(obj) -> float:
    """Object's length"""
    return _get_api_attr(obj, "Length")


def area(obj) -> float:
    """Object's Area"""
    return _get_api_attr(obj, "Area")


def volume(obj) -> float:
    """Object's volume"""
    return _get_api_attr(obj, "Volume")


def center_of_mass(obj) -> np.ndarray:
    """Object's center of mass"""
    return _get_api_attr(obj, "CenterOfMass")


def is_null(obj):
    """True if obj is null"""
    return _get_api_attr(obj, "isNull")()


def is_closed(obj):
    """True if obj is closed"""
    return _get_api_attr(obj, "isClosed")()


def is_valid(obj):
    """True if obj is valid"""
    return _get_api_attr(obj, "isValid")()


def bounding_box(obj):
    """Object's bounding box"""
    box = _get_api_attr(obj, "BoundBox")
    return box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax


# ======================================================================================
# Wire manipulation
# ======================================================================================
def wire_closure(wire: Part.Wire):
    """Create a line segment wire that closes an open wire"""
    closure = None
    if not wire.isClosed():
        vertexes = wire.OrderedVertexes
        points = [v.Point for v in vertexes]
        closure = make_polygon([points[-1], points[0]])
    return closure


def close_wire(wire: Part.Wire):
    """
    Closes a wire with a line segment, if not already closed.
    A new wire is returned.
    """
    if not wire.isClosed():
        vertexes = wire.OrderedVertexes
        points = [v.Point for v in vertexes]
        wline = make_polygon([points[-1], points[0]])
        wire = Part.Wire([wire, wline])
    return wire


def discretize(w: Part.Wire, ndiscr: int = 10, dl: float = None):
    """
    Discretize a wire.

    Parameters
    ----------
    w : Part.Wire
        wire to be discretized.
    ndiscr : int
        number of points for the whole wire discretization.
    dl : float
        target discretization length (default None). If dl is defined,
        ndiscr is not considered.

    Returns
    -------
    output : list(numpy.ndarray)
        list of points.

    """
    # discretization points array
    output = []

    if dl is None:
        pass
    elif dl <= 0.0:
        raise ValueError("dl must be > 0.")
    else:
        # a dl is calculated for the discretisation of the different edges
        # NOTE: must discretise to at least two points.
        ndiscr = max(math.ceil(w.Length / dl + 1), 2)

    # discretization points array
    output = w.discretize(ndiscr)
    output = vector_to_numpy(output)
    return output


def discretize_by_edges(w: Part.Wire, ndiscr: int = 10, dl: float = None):
    """
    Discretize a wire taking into account the edges of which it consists of.

    Parameters
    ----------
    w : Part.Wire
        wire to be discretized.
    ndiscr : int
        number of points for the whole wire discretization. Final number of points
        can be slightly different due to edge discretization routine.
    dl : float
        target discretization length (default None). If dl is defined,
        ndiscr is not considered.

    Returns
    -------
    output : list(numpy.ndarray)
        list of points.
    """
    # discretization points array
    output = []

    if dl is None:
        # dl is calculated for the discretisation of the different edges
        dl = w.Length / float(ndiscr)
    elif dl <= 0.0:
        raise ValueError("dl must be > 0.")

    # edges are discretised taking into account their orientation
    # Note: OrderedEdges already return a list of edges that considers the edge in the
    # correct sequence and orientation. No need for tricks after the discretization.
    for e in w.OrderedEdges:
        pointse = list(discretize(Part.Wire(e), dl=dl))
        output += pointse[:-1]

    if w.isClosed():
        output += [output[0]]
    else:
        output += [pointse[-1]]

    output = np.array(output)
    return output


def dist_to_shape(shape1, shape2):
    """
    Find the minimum distance between two shapes

    Parameters
    ----------
    shape1:
        reference shape.
    shape2:
        target shape.

    Returns
    -------
    output:
        a tuple of two -> (dist, vectors)
        dist is the minimum distance (float value)
        vectors is a list of tuples corresponding to the nearest points (numpy.ndarray)
        between shape1 and shape2. The distance between those points is the minimum
        distance given by dist.
    """
    dist, solution, info = shape1.distToShape(shape2)
    vectors = []
    for v1, v2 in solution:
        vectors.append((vector_to_numpy(v1), vector_to_numpy(v2)))
    return dist, vectors


def wire_plane_intersect(wire, plane):
    """
    Calculate the intersection of a wire with a plane.

    Parameters
    ----------
    wire: apiWire
        The loop to calculate the intersection on
    plane: apiPlacement
        The plane to calculate the intersection with

    Returns
    -------
    inter: Union[np.array(3, n_intersections), None]
        The xyz coordinates of the intersections with the wire. Returns None if
        there are no intersections detected
    """
    plane = _placement_to_plane(plane)
    face = apiFace(plane)

    if not _wire_is_planar(wire):
        bluemira_warn(
            "You are intersecting a non-planar wire with a plane, cannot "
            "guarantee return type will be correct."
        )

    if wire.isCoplanar(face):
        raise FreeCADError(
            "Cannot intersect this wire with this plane: they are coplanar."
        )

    vertexes = wire.section([face]).Vertexes
    if vertexes:
        return vertex_to_numpy(vertexes)

    return None


# ======================================================================================
# Save functions
# ======================================================================================
def save_as_STEP(shapes, filename="test", scale=1):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes: (Shape, ..)
        Iterable of shape objects to be saved
    filename: str
        Full path filename of the STP assembly
    scale: float (default 1)
        The scale in which to save the Shape objects
    """
    if not filename.endswith(".STP"):
        filename += ".STP"

    if not isinstance(shapes, list):
        shapes = [shapes]

    if not all(not shape.isNull() for shape in shapes):
        raise FreeCADError("Shape is null.")

    compound = make_compound(shapes)

    if scale != 1:
        # scale the compound. Since the scale function modifies directly the shape,
        # a copy of the compound is made to avoid modification of the original shapes.
        compound = compound.copy().scale(scale)

    compound.exportStep(filename)


# # =============================================================================
# # Shape manipulations
# # =============================================================================
def scale_shape(shape, factor):
    """
    Apply scaling with factor to the shape

    Parameters
    ----------
    shape: FreeCAD Shape object
        The shape to be scaled
    factor: float
        The scaling factor

    Returns
    -------
    shape: the modified shape
    """
    return shape.scale(factor)


def translate_shape(shape, vector: tuple):
    """
    Apply scaling with factor to the shape

    Parameters
    ----------
    shape: FreeCAD Shape object
        The shape to be scaled
    vector: tuple (x,y,z)
        The translation vector

    Returns
    -------
    shape: the modified shape
    """
    return shape.translate(vector)


def rotate_shape(
    shape,
    base: tuple = (0.0, 0.0, 0.0),
    direction: tuple = (0.0, 0.0, 1.0),
    degree: float = 180,
):
    """
    Apply the rotation (base, dir, degree) to this shape

    Parameters
    ----------
    shape: FreeCAD Shape object
        The shape to be rotated
    base: tuple (x,y,z)
        Origin location of the rotation
    direction: tuple (x,y,z)
        The direction vector
    degree: float
        rotation angle

    Returns
    -------
    shape: the modified shape
    """
    return shape.rotate(base, direction, degree)


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
    shape: FreeCAD Shape object
        The shape to be revolved
    base: tuple (x,y,z)
        Origin location of the revolution
    direction: tuple (x,y,z)
        The direction vector
    degree: double
        revolution angle

    Returns
    -------
    shape:
        the revolved shape.
    """
    base = Base.Vector(base)
    direction = Base.Vector(direction)
    return shape.revolve(base, direction, degree)


def extrude_shape(shape, vec: tuple):
    """
    Apply the extrusion along vec to this shape

    Parameters
    ----------
    shape: FreeCAD Shape object
        The shape to be extruded
    vec: tuple (x,y,z)
        The vector along which to extrude

    Returns
    -------
    shape:
        The extruded shape.
    """
    vec = Base.Vector(vec)
    return shape.extrude(vec)


def _edges_tangent(edge_1, edge_2):
    """
    Check if two adjacent edges are tangent to one another.
    """
    angle = edge_1.tangentAt(edge_1.LastParameter).getAngle(
        edge_2.tangentAt(edge_2.FirstParameter)
    )
    return np.isclose(
        angle,
        0.0,
        rtol=1e-4,
        atol=1e-4,
    )


def _wire_edges_tangent(wire):
    """
    Check that all consecutive edges in a wire are tangent
    """
    if len(wire.Edges) <= 1:
        return True

    else:
        edges_tangent = []
        for i in range(len(wire.Edges) - 1):
            edge_1 = wire.Edges[i]
            edge_2 = wire.Edges[i + 1]
            edges_tangent.append(_edges_tangent(edge_1, edge_2))

    if wire.isClosed():
        # Check last and first edge tangency
        edges_tangent.append(_edges_tangent(wire.Edges[-1], wire.Edges[0]))

    return all(edges_tangent)


def _split_wire(wire):
    """
    Split a wire into two parts.
    """
    edges = wire.OrderedEdges
    if len(edges) == 1:
        # Only one edge in the wire, which we need to split
        edge = edges[0]
        p_start, p_end = edge.ParameterRange
        p_mid = 0.5 * (p_end - p_start)
        edges_1 = edge.Curve.toShape(p_start, p_mid)
        edges_2 = edge.Curve.toShape(p_mid, p_end)

    else:
        # We can just sub-divide the wire by its edges
        n_split = int(len(edges) / 2)
        edges_1, edges_2 = edges[:n_split], edges[n_split:]

    return apiWire(edges_1), apiWire(edges_2)


def sweep_shape(profiles, path, solid=True, frenet=True):
    """
    Sweep a a set of profiles along a path.

    Parameters
    ----------
    profiles: Iterable[apiWire]
        Set of profiles to sweep
    path: apiWire
        Path along which to sweep the profiles
    solid: bool
        Whether or not to create a Solid
    frenet: bool
        If true, the orientation of the profile(s) is calculated based on local curvature
        and tangency. For planar paths, should not make a difference.

    Returns
    -------
    swept: Union[Part.Solid, Part.Shell]
        Swept geometry object
    """
    if not isinstance(profiles, Iterable):
        profiles = [profiles]

    closures = [p.isClosed() for p in profiles]
    all_closed = sum(closures) == len(closures)
    none_closed = sum(closures) == 0

    if not all_closed and not none_closed:
        raise FreeCADError("You cannot mix open and closed profiles when sweeping.")

    if none_closed and solid:
        bluemira_warn(
            "You cannot sweep open profiles and expect a Solid result. Disabling this."
        )
        solid = False

    if not _wire_edges_tangent(path):
        raise FreeCADError(
            "Sweep path contains edges that are not consecutively tangent. This will produce unexpected results."
        )

    result = path.makePipeShell(profiles, True, frenet)

    solid_result = apiSolid(result)
    if solid:
        return solid_result
    else:
        return solid_result.Shells[0]


# ======================================================================================
# Boolean operations
# ======================================================================================
def boolean_fuse(shapes):
    """
    Fuse two or more shapes together. Internal splitter are removed.

    Parameters
    ----------
    shapes: Iterable
        List of FreeCAD shape objects to be fused together. All the objects in the
        list must be of the same type.

    Returns
    -------
    fuse_shape:
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

    try:
        if _type == Part.Wire:
            merged_shape = BOPTools.SplitAPI.booleanFragments(shapes, "Split")
            if len(merged_shape.Wires) > len(shapes):
                raise FreeCADError(
                    f"Fuse wire creation failed. Possible "
                    f"overlap or internal intersection of "
                    f"input shapes {shapes}."
                )
            else:
                merged_shape = merged_shape.fuse(merged_shape.Wires)
                merged_shape = Part.Wire(merged_shape.Wires)
                return merged_shape

        elif _type == Part.Face:
            merged_shape = shapes[0].fuse(shapes[1:])
            merged_shape = merged_shape.removeSplitter()
            if len(merged_shape.Faces) > 1:
                raise FreeCADError(
                    f"Boolean fuse operation on {shapes} gives more than one face."
                )
            return merged_shape.Faces[0]

        elif _type == Part.Solid:
            merged_shape = shapes[0].fuse(shapes[1:])
            merged_shape = merged_shape.removeSplitter()
            if len(merged_shape.Solids) > 1:
                raise FreeCADError(
                    f"Boolean fuse operation on {shapes} gives more than one solid."
                )
            return merged_shape.Solids[0]

        else:
            raise ValueError(
                f"Fuse function still not implemented for {_type} instances."
            )
    except Exception as e:
        raise FreeCADError(str(e))


def boolean_cut(shape, tools, split=True):
    """
    Difference of shape and a given (list of) topo shape cut(tools)

    Parameters
    ----------
    shape: FreeCAD shape
        the reference object
    tools: Iterable
        List of FreeCAD shape objects to be used as tools.
    split: bool
        If True, shape is split into pieces based on intersections with tools.

    Returns
    -------
    cut_shape:
        Result of the boolean operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    _type = type(shape)

    if not isinstance(tools, list):
        tools = [tools]

    cut_shape = shape.cut(tools)
    if split:
        cut_shape = BOPTools.SplitAPI.slice(cut_shape, tools, mode="Split")

    if _type == Part.Wire:
        output = cut_shape.Wires
    elif _type == Part.Face:
        output = cut_shape.Faces
    elif _type == Part.Shell:
        output = cut_shape.Shells
    elif _type == Part.Solid:
        output = cut_shape.Solids
    else:
        raise ValueError(f"Cut function not implemented for {_type} objects.")
    return output


# ======================================================================================
# Geometry healing
# ======================================================================================


def fix_wire(wire, precision=EPS, min_length=MINIMUM_LENGTH):
    """
    Fix a wire by removing any small edges and joining the remaining edges.

    Parameters
    ----------
    wire: apiWire
        Wire to fix
    precision: float
        General precision with which to work
    min_length: float
        Minimum edge length
    """
    wire.fix(precision, min_length, min_length)


# ======================================================================================
# Plane manipulations
# ======================================================================================

# BluemiraPlane wraps Base.Placement not Part.Plane. These conversions become useful..
# They are probably a bit broken...
def _placement_to_plane(placement):
    """
    Convert a FreeCAD Base.Placement to FreeCAD Part.Plane
    """
    plane = Part.Plane(placement.Base, Base.Vector(0.0, 0.0, 1.0))
    plane.rotate(placement)
    return plane


def _plane_to_placement(plane):
    """
    Convert a FreeCAD Part.Plane to FreeCAD Base.Placement
    """
    placement = Base.Placement(plane.Position, plane.Axis, 0)
    placement.Rotation = plane.Rotation
    return placement


def make_plane(base, axis, angle):
    """
    Make a FreeCAD Placement

    Parameters
    ----------
    base: Iterable
        a vector representing the Plane's position
    axis: Iterable
        normal vector to the Plane
    angle:
        rotation angle in degree
    """
    base = Base.Vector(base)
    axis = Base.Vector(axis)

    return Base.Placement(base, axis, angle)


def make_plane_3P(point_1, point_2, point_3):  # noqa: N802
    """
    Make a FreeCAD Placement from three points.

    Parameters
    ----------
    point_1: Iterable
        First point
    point_2: Iterable
        Second Point
    point_3: Iterable
        Third point

    Returns
    -------
    plane: Base.Placement
        The "plane"
    """
    plane = Part.Plane(Base.Vector(point_1), Base.Vector(point_2), Base.Vector(point_3))
    return _plane_to_placement(plane)


def move_plane(plane, vector):
    """
    Moves the FreeCAD Plane along the given vector

    Parameters
    ----------
    plane: FreeCAD plane
        the FreeCAD plane to be modified
    vector: Iterable
        direction along which the plane is moved

    Returns
    -------
    nothing:
        The plane is directly modified.
    """
    plane.move(Base.Vector(vector))


def change_plane(geo, plane):
    """
    Change the placement of a FreeCAD object

    Parameters
    ----------
    geo: FreeCAD object
        the object to be modified
    plane: FreeCAD plane
        the FreeCAD plane to be modified

    Returns
    -------
    nothing:
        The object is directly modified.
    """
    new_placement = geo.Placement.multiply(plane)
    new_base = plane.multVec(geo.Placement.Base)
    new_placement.Base = new_base
    geo.Placement = new_placement


def _colourise(
    node: coin.SoNode,
    options: Dict,
):
    if isinstance(node, coin.SoMaterial):
        rgb = options["color"]
        transparency = options["transparency"]
        node.ambientColor.setValue(coin.SbColor(*rgb))
        node.diffuseColor.setValue(coin.SbColor(*rgb))
        node.transparency.setValue(transparency)
    for child in node.getChildren() or []:
        _colourise(child, options)


def show_cad(
    parts: Union[Part.Shape, List[Part.Shape]],
    options: Optional[Union[Dict, List[Dict]]] = None,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[_PlotCADOptions, List[_PlotCADOptions]]]
        The options to use to display the parts.
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        from bluemira.display.displayer import get_default_options

        dict_options = get_default_options()
        options = [dict_options] * len(parts)

    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise FreeCADError(
            "If options for display are provided then there must be as many options as "
            "there are parts to display."
        )

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    if not hasattr(FreeCADGui, "subgraphFromObject"):
        FreeCADGui.setupWithoutGUI()

    doc = FreeCAD.newDocument()

    root = coin.SoSeparator()

    for part, option in zip(parts, options):
        new_part = part.copy()
        new_part.rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -90.0)
        obj = doc.addObject("Part::Feature")
        obj.Shape = new_part
        doc.recompute()
        subgraph = FreeCADGui.subgraphFromObject(obj)
        _colourise(subgraph, option)
        root.addChild(subgraph)

    viewer = quarter.QuarterWidget()
    viewer.setBackgroundColor(coin.SbColor(1, 1, 1))
    viewer.setTransparencyType(coin.SoGLRenderAction.SCREEN_DOOR)
    viewer.setSceneGraph(root)

    viewer.setWindowTitle("Bluemira Display")
    viewer.show()
    app.exec_()
