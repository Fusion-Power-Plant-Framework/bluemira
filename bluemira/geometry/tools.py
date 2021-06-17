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
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

# import from freecad
import freecad
import Part
from FreeCAD import Base

# import numpy lib
import numpy

# import typing
from typing import Union


#########################################
# Array, List, Vector, Point manipulation
#########################################
def check_data_type(data_type):
    """Decorator to check the data type of the first parameter input (args[0] of a
    function.

    Raises
    ______
    TypeError
        If args[0] objects are not instances of data_type
    """

    def _apply_to_list(func):
        def wrapper(*args, **kwargs):
            output = []
            objs = args[0]
            is_list = isinstance(objs, list)
            if not is_list:
                objs = [objs]
            if all(isinstance(o, data_type) for o in objs):
                output = func(*args, **kwargs)
                if not is_list:
                    output = output[0]
            else:
                for o in objs:
                    print(type(o))
                raise TypeError("Only {} instances can be concerted to {}".format(
                    data_type, type(output)))
            return output

        return wrapper

    return _apply_to_list


@check_data_type(Base.Vector)
def vector_to_numpy(vectors):
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a numpy array"""
    return numpy.array([numpy.array(v) for v in vectors])


@check_data_type(Part.Point)
def point_to_numpy(points):
    """Converts a FreeCAD Part.Point or list(Part.Point) into a numpy array"""
    return numpy.array([numpy.array([p.X, p.Y, p.Z]) for p in points])


###################################
# Part.Wire manipulation
###################################
def wire_closure(wire: Part.Wire):
    """ Create a line segment wire that closes an open wire"""
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


def discretize(w: Part.Wire, ndiscr: int):
    """Discretize a wire.

    Parameters
    ----------
    w : Part.Wire
        wire to be discretized.
    ndiscr : int
        number of points for the whole wire discretization.

    Returns
    -------
    output : list(Base.Vector)
        list of Base.Vector points.

    """
    # discretization points array
    output = w.discretize(ndiscr)
    output = vector_to_numpy(output)
    return output


def discretize_by_edges(w: Part.Wire, ndiscr: int):
    """Discretize a wire taking into account the edges of which it consists of.

    Parameters
    ----------
    w : Part.Wire
        wire to be discretized.
    ndiscr : int
        number of points for the whole wire discretization.

    Returns
    -------
    output : list(Base.Vector)
        list of Base.Vector points.

    """
    # discretization points array
    output = []
    # a dl is calculated for the discretisation of the different edges
    dl = w.Length / float(ndiscr)
    # edges are discretised taking into account their orientation
    # Note: this is a tricky part in Freecad. Reversed wires need a
    # reverse operation for the generated points and the list of generated
    # points for each edge.
    for e in w.OrderedEdges:
        pointse = e.discretize(Distance=dl)
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
    output = vector_to_numpy(output)
    return output


###################################
# Geometry creation
###################################
def make_polygon(points: Union[list, numpy.ndarray], closed: bool = False) -> Part.Wire:
    """Make a polygon from a set of points.

    Args:
        points (Union[list, numpy.ndarray]): list of points. It can be given \
            as a list of 3D tuples, a 3D numpy array, or similar.
        closed (bool, optional): if True, the first and last points will be\
            connected in order to form a closed polygon. Defaults to False.

    Returns:
        Part.Wire: a FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    wire = Part.makePolygon(pntslist)
    if closed and not wire.isClosed():
        # add a line that closes the wire
        line = Part.makePolygon([pntslist[-1], pntslist[0]])
        wire = Part.Wire([wire, line])
    return wire
