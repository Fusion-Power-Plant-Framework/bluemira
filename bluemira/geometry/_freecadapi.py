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

# import from freecad
import math

import freecad  # noqa: F401
import Part
from FreeCAD import Base

# import math lib
import numpy as np
import math

# import typing
from typing import Union

# import errors
from bluemira.geometry.error import GeometryError


# # =============================================================================
# # Array, List, Vector, Point manipulation
# # =============================================================================
def check_data_type(data_type):
    """Decorator to check the data type of the first parameter input (args[0]) of a
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


# # =============================================================================
# # Geometry creation
# # =============================================================================
def make_polygon(points: Union[list, np.ndarray], closed: bool = False) -> Part.Wire:
    """Make a polygon from a set of points.

    Parameters
    ----------
        points (Union[list, np.ndarray]): list of points. It can be given
            as a list of 3D tuples, a 3D numpy array, or similar.
        closed (bool, optional): if True, the first and last points will be
            connected in order to form a closed shape. Defaults to False.

    Returns
    -------
        Part.Wire: a FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    wire = Part.makePolygon(pntslist)
    if closed:
        wire = close_wire(wire)
    return wire


def make_bezier(points: Union[list, np.ndarray], closed: bool = False) -> Part.Wire:
    """Make a bezier curve from a set of points.

    Parameters
    ----------
        points (Union[list, np.ndarray]): list of points. It can be given
            as a list of 3D tuples, a 3D numpy array, or similar.
        closed (bool, optional): if True, the first and last points will be
            connected in order to form a closed shape. Defaults to False.

    Returns
    -------
        Part.Wire: a FreeCAD wire that contains the bezier curve
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
    points: Union[list, np.ndarray], closed: bool = False, **kwargs
) -> Part.Wire:
    """Make a bezier curve from a set of points.

    Parameters
    ----------
        points (Union[list, np.ndarray]): list of points. It can be given
            as a list of 3D tuples, a 3D numpy array, or similar.
        closed (bool, optional): if True, the first and last points will be
            connected in order to form a closed shape. Defaults to False.
        Parameters: (optional) knot sequence

    Returns
    -------
        Part.Wire: a FreeCAD wire that contains the bezier curve
    """
    # In this case, it is not really necessary to convert points in FreeCAD vector. Just
    # left for consistency with other methods.
    pntslist = [Base.Vector(x) for x in points]
    bsc = Part.BSplineCurve()
    bsc.interpolate(pntslist, PeriodicFlag=closed, **kwargs)
    wire = Part.Wire(bsc.toShape())
    return wire


def make_circle(
    radius=1.0,
    center=[0.0, 0.0, 0.0],
    startangle=0.0,
    endangle=360.0,
    axis=[0.0, 0.0, 1.0],
):
    """make_circle([radius, center, startangle, endangle, axis])

    Creates a circle or arc of circle object with given parameters.

    TODO: check the creation of the arc when startangle < endangle

    Parameters
    ----------
        radius: the radius of the circle (float). Default to 1.
        center: the center of the circle (list or numpy.array). Default [0., 0., 0.].
        startangle: start angle of the arc (in degrees). Default to 0.
        endangle: end angle of the arc (in degrees). Default to 360.
            if startangle and endangle are equal, a circle is created,
            if they are different an arc is created
        axis: normal vector to the circle plane (list or numpy.array).
            Default [0., 0., 1.])

    Returns
    -------
        Part.Wire: a FreeCAD wire that contains the arc or circle
    """

    output = Part.Circle()
    output.Radius = radius
    output.Center = Base.Vector(center)
    output.Axis = Base.Vector(axis)
    if startangle != endangle:
        output = Part.ArcOfCircle(
            output, math.radians(startangle), math.radians(endangle)
        )
    return Part.Wire(Part.Edge(output))


# # =============================================================================
# # Object's properties
# # =============================================================================
def length(obj) -> float:
    """Object's length"""
    prop = "Length"
    if hasattr(obj, prop):
        return getattr(obj, prop)
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def area(obj) -> float:
    """Object's Area"""
    prop = "Area"
    if hasattr(obj, prop):
        return getattr(obj, prop)
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def volume(obj) -> float:
    """Object's volume"""
    prop = "Volume"
    if hasattr(obj, prop):
        return getattr(obj, prop)
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def center_of_mass(obj) -> np.ndarray:
    """Object's center of mass"""
    prop = "CenterOfMass"
    if hasattr(obj, prop):
        # CenterOfMass returns a vector.
        return getattr(obj, prop)
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def is_null(obj):
    """True if obj is null"""
    prop = "isNull"
    if hasattr(obj, prop):
        return getattr(obj, prop)()
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def is_closed(obj):
    """True if obj is closed"""
    prop = "isClosed"
    if hasattr(obj, prop):
        return getattr(obj, prop)()
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


def bounding_box(obj):
    """Object's bounding box"""
    prop = "BoundBox"
    if hasattr(obj, prop):
        # FreeCAD BoundBox is a FreeCAD object. For the moment there is not a
        # complementary object in bluemira. Thus, this method will just return
        # (XMin, YMin, Zmin, XMax, YMax, ZMax)
        box = getattr(obj, prop)
        return box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax
    else:
        raise GeometryError(f"FreeCAD object {obj} has not property {prop}")


# # =============================================================================
# # Part.Wire manipulation
# # =============================================================================
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
    """Discretize a wire.

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
        ndiscr = math.ceil(w.Length / dl)

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
    # Note: this is a tricky part in Freecad. Reversed wires need a
    # reverse operation for the generated points and the list of generated
    # points for each edge.
    for e in w.OrderedEdges:
        pointse = list(discretize(Part.Wire(e), dl=dl))
        # if edge orientation is reversed, the generated list of points
        # must be reversed
        if e.Orientation == "Reversed":
            pointse.reverse()
        output += pointse[:-1]
    if w.isClosed():
        output += pointse[-1:]
    # if wire orientation is reversed, output must be reversed
    if w.Orientation == "Reversed":
        output.reverse()
    output = np.array(output)
    return output


def dist_to_shape(shape1, shape2):
    """Find the minimum distance between two shapes

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


# # =============================================================================
# # Save functions
# # =============================================================================
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
        raise GeometryError("Shape is null.")

    compound = make_compound(shapes)

    if scale != 1:
        # scale the compound. Since the scale function modifies directly the shape,
        # a copy of the compound is made to avoid modification of the original shapes.
        compound = compound.copy().scale(scale)

    # doc = FreeCAD.newDocument()
    # obj = FreeCAD.ActiveDocument.addObject("App::DocumentObject", "Test")
    #
    # freecad_comp = FreeCAD.ActiveDocument.addObject("Part::Feature")
    #
    # # link the solid to the object
    # freecad_comp.Shape = compound
    #
    # Part.export([freecad_comp], filename)

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
    degree: double
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


def make_compound(shapes):
    """
    Make an FreeCAD compound object out of many shapes

    Parameters
    ----------
    *shapes: list of FreeCAD shape objects
        A set of objects to be compounded

    Returns
    -------
    compound: FreeCAD compound object
        A compounded set of shapes
    """
    compound = Part.makeCompound(shapes)
    return compound
