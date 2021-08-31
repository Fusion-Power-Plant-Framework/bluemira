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

from __future__ import annotations

# import from freecad
import freecad
import Part
from FreeCAD import Base

# import numerical library
import numpy
import math

# import plotting library
import matplotlib.pyplot as plt

from typing import Union

import logging
module_logger = logging.getLogger(__name__)


class Shape():
    """This class is a container for geometrical shapes.
    
    It is limited to work with the FreeCAD object **Part.Wire** and it is the
    main object used by the meshing api to generate a mesh with gmsh.
    
    .. todo::
        * check if an implementation based on a tree structure, as made for \
        the Component class, is feasible.


    

    Returns
    -------

    """

    def __init__(
            self,
            boundary: Union[Part.Wire, Shape, list[Part.Wire, Shape]],
            label: str = "",
            lcar: Union[float, list[float]] = 0.1
    ):
        """Constructor for Shape

        Parameters
        ----------
        boundary : Union[Part.Wire, Shape, list[Part.Wire, Shape]]
            objects used for the definition of the geometrical shape.
        label: str, optional
            Defaults to "".
        lcar: Union[float, list[float]], optional
            characteristic length used for the generation of the mesh. \
                Defaults to 0.1

        Raises
        ------
        ValueError:
            In case boundary elements are not Part.Wire or Shape objects.

        Attributes
        ----------
        label: str, optional
            Defaults to "".
        lcar: Union[float, list[float]], optional
            characteristic length used for the generation of the mesh. \
                Defaults to 0.1

        """
        self.boundary = boundary
        self.label = label
        self.lcar = lcar

    @property
    def boundary(self):
        """ """
        return self._boundary

    @boundary.setter
    def boundary(self, objs):
        """

        Parameters
        ----------
        objs :
            

        Returns
        -------

        """
        self._boundary = None
        # if objs is not a list, it is converted into a list.
        if not hasattr(objs, '__len__'):
            objs = [objs]
        if (all(isinstance(o, Part.Wire) for o in objs) or
                all(isinstance(o, Shape) for o in objs)):
            self._boundary = objs
        else:
            [print("{} {}".format(type(o), o)) for o in objs]
            raise ValueError("Only Part.Wire or Shape objects can be used"
                             " as objs.")

    @property
    def Length(self):
        """float: total length of the shape perimiter."""

        # Note:the method is recursively implemented considering that
        # Part.Wire has a similar Length property.

        return sum([o.Length for o in self.boundary])

    @property
    def Wires(self):
        """list(Part.Wire): list of wires of which the shape consists of."""
        wires = []
        for o in self.boundary:
            wires += o.Wires
        return wires

    def getSingleWire(self):
        """Generate a single wire that represent the shape's boundary"""
        return Part.Wire(self.Wires)

    @property
    def allshapes(self):
        """ """
        return self.boundary

    def isClosed(self):
        """True if the shape is closed"""
        return self.getSingleWire().isClosed()

    def close_shape(self):
        """Close the shape with a LineSegment between shape's end and
            start point
        """
        if not self.isClosed():
            wire = self.getSingleWire()
            wls = Utils.wire_closure(wire)
            self.boundary.append(wls)

    @property
    def Placement(self):
        """ """
        if hasattr(self, "Placement"):
            return self._Placement
        else:
            return None

    @Placement.setter
    def Placement(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if isinstance(value, Base.Placement):
            self._Placement = value
            for s in self.allshapes:
                s.Placement = value
        else:
            raise ValueError("Placement must be a Base.Placement object")

    def getCenter(self):
        """Get the center of the shape bounding box"""
        return self.getSingleWire().BoundBox.Center

    def getCurves(self):
        """Convert the boundary shape in a set of Part.Curve"""
        output = []
        for o in self.boundary:
            if isinstance(o, Shape):
                out = o.getCurves()
                if out:
                    output += out
            elif isinstance(o, Part.Wire):
                out = Utils.convert_Wire_to_Curves(o)
                if out:
                    output += out
        return output

    def __getitem__(self, key):
        return self.boundary[key]

    def __setitem__(self, key, value):
        self.boundary[key] = value

    def __add__(self, other):
        # # Note: not sure if a deepcopy should be made
        # s = copy.deepcopy(self)
        # s += other
        s = None
        if isinstance(other, (Part.Wire, Shape)):
            s = Shape([self, other])
        else:
            raise ValueError("Only Part.Wire and Shape can be added.")
        return s

    def __iadd__(self, other):
        if isinstance(other, (Part.Wire, Shape)):
            self.boundary.append(other)
        else:
            raise ValueError("Only Part.Wire and Shape can be added.")
        return self

    def search(self, label: str):
        """Search for a shape with the specified label

        Parameters
        ----------
        label : str :
            shape label.            

        Returns
        -------
        output : list(Shape
            list of shapes that have the specified label.

        """
        output = []
        if self.label == label:
            output.append(self)
        for o in self.boundary:
            if isinstance(o, Shape):
                output += o.search(label)
        return output

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        new.append(" {}".format(self.label))
        new.append(" {}".format(self.boundary))
        new.append(" {}".format(self.lcar))
        new.append(")")
        return ", ".join(new)


class Shape2D():
    """This class is a container for geometrical 2D planar shapes."""

    def __init__(
            self,
            boundary: Shape,
            label: str = "",
            holes: Union[Shape, list[Shape]] = []
    ):
        """Constructor for Shape2D"""

        # initialize internal variable
        # necessary for the creation of the face
        self._boundary = []
        self._holes = []
        self.label = label
        # initalise face to None. It will be recalculated during the
        # assignment of boundary and holes
        self.face = None
        self.boundary = boundary
        self.holes = holes

    @property
    def boundary(self):
        """mirapy.geo.Shape: shape2D's outer boundary."""
        return self._boundary

    @boundary.setter
    def boundary(self, objs):
        """

        Parameters
        ----------
        objs :
            

        Returns
        -------

        """
        self._boundary = None
        # if objs is not a list, it is converted into a list.
        if not hasattr(objs, '__len__'):
            objs = [objs]
        if self.__checkBoundary(objs):
            self._boundary = objs
            self.face = self._createFace()
        else:
            raise ValueError("Only a closed Shape object can be used as "
                             "boundary.")

    def __checkBoundary(self, objs):
        if len(objs) == 0:
            return True
        elif all(isinstance(o, Shape) and o.isClosed() for o in objs):
            return True
        return False

    @property
    def holes(self):
        """mirapy.geo.Shape: shape2D's outer boundary."""
        return self._holes

    @holes.setter
    def holes(self, holes):
        """

        Parameters
        ----------
        holes :
            

        Returns
        -------

        """
        self._holes = []
        if not hasattr(holes, '__len__'):
            holes = [holes]
        if self.__checkHoles(holes):
            self._holes = holes
            self.face = self._createFace()
        else:
            raise ValueError("Only closed Shape objects can be used as holes.")

    def __checkHoles(self, holes):
        if len(holes) == 0:
            return True
        elif all(isinstance(o, Shape) and o.isClosed() for o in holes):
            print("Intersection between holes is not allowed. __checkHoles"
                  " does not check it, so pay attention."
                  " To be still implemented.")
            return True
        return False

    def _createFace(self):
        """ """
        s = self.boundary[0]
        face = Part.Face(s.getSingleWire())
        if len(self.holes) != 0:
            fholes = [Part.Face(h.getSingleWire()) for h in self.holes]
            # face.cutHoles([h.getSingleWire() for h in self.holes])
            # face.validate()
            s = face.cut(fholes)
            if len(s.Faces) == 1:
                face = s.Faces[0]
            else:
                raise ValueError("Any or more than one face have been"
                                 "created.")
        return face

    @property
    def Length(self):
        """float: total length of the shape perimiter."""
        return self.face.Length

    @property
    def Area(self):
        """float: total area of the shape."""
        return self.face.Area

    @property
    def Wires(self):
        """list(Part.Wire): list of wires of which the shape consists of."""
        return self.face.Wires

    @property
    def allshapes(self):
        """ """
        return self.boundary + self.holes

    @property
    def Placement(self):
        """ """
        if hasattr(self, "Placement"):
            return self._Placement
        else:
            return None

    @Placement.setter
    def Placement(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if isinstance(value, Base.Placement):
            self._Placement = value
            for s in self.allshapes:
                s.Placement = value
            self.face.Placement = value
        else:
            raise ValueError("Placement must be a Base.Placement object")

    def getSingleWires(self):
        """list(Part.Wire): list of wires of objs and holes."""
        wires = [o.getSingleWire() for o in self.allshapes]
        return wires

    def getCenter(self):
        """ """
        return self.face.BoundBox.Center

    def __getitem__(self, key):
        # wires = self.getSingleWires()
        return self.allshapes[key]

    def __setitem__(self, key, value):
        if key == 0:
            self.boundary[key] = value
        if key > 0:
            self.holes[key-1] = value

    def __add__(self, other):
        if isinstance(other, (Shape2D)):
            newface = self.face.fuse(other.face)
            faces = newface.removeSplitter().Faces
            if len(faces) == 1:
                newface = faces[0]
            else:
                raise ValueError("Any or more than one face have been"
                                 " created.")
            newShape = Shape2D.createFromFace(newface)
        else:
            raise ValueError("Only Shape2D objects can be added.")
        return newShape

    def __sub__(self, other):
        if isinstance(other, (Shape2D)):
            newface = self.face.cut(other.face)
            faces = newface.removeSplitter().Faces
            if len(faces) == 1:
                newface = faces[0]
            else:
                raise ValueError("Any or more than one face have been"
                                 " created.")
            newShape = Shape2D.createFromFace(newface)
        else:
            raise ValueError("Only Shape2D objects can be subtracted.")
        return newShape

    def search(self, label: str):
        """Search for a shape with the specified label

        Parameters
        ----------
        label : str
            shape label.            

        Returns
        -------
        output : list(Shape)
            list of shapes that have the specified label.

        """
        output = []
        if self.label == label:
            output.append(self)
        for o in self.allshapes:
            if isinstance(o, Shape):
                output += o.search(label)
        return output

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        new.append(" {}".format(self.label))
        new.append(" boundary: {}".format(self.boundary))
        new.append(" holes: {}".format(self.holes))
        new.append(")")
        return ", ".join(new)

    @staticmethod
    def createFromFace(face: Part.Face, label: str = ""):
        """

        Parameters
        ----------
        face: Part.Face
            
        label: str
             (Default value = "")

        Returns
        -------

        """
        if not isinstance(face, Part.Face):
            raise ValueError("First argument must be Part.Face.")
        boundary = Shape(face.Wires[0])
        holes = [Shape(w) for w in face.Wires[1:]]
        newShape = Shape2D(boundary, holes=holes, label=label)
        return newShape


class geoConstraint():
    """ """
    
    def __init__(self, point, angle = None, lscale = 1, label = None):
        self.line = None
        self.point = point
        self.__lscale = lscale
        self.angle = angle
        self.label = label
        
        self.poptions = {'s': 20, 'marker': 'o', 'c': 'red'}
        self.woptions = {'s': 20, 'facecolors': 'none', 'edgecolors': 'red'}
    
    @property
    def angle(self):
        """ """
        try:
            return self.__angle
        except:
            return None
    
    @angle.setter
    def angle(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if not value is None:
            self.__angle = value
            x = math.cos(numpy.deg2rad(self.angle)) * self.__lscale
            y = math.sin(numpy.deg2rad(self.angle)) * self.__lscale
            linesegment = Part.LineSegment(self.point,
                                           self.point + Base.Vector(x, y, 0))
            self.line = Part.Wire(Part.Shape([linesegment]).Edges)
        else:
            self.line = None
    
    @property
    def lscale(self):
        """ """
        try:
            return self.__lscale
        except:
            return None
    
    @angle.setter
    def lscale(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if value is not None:
            self.__lscale = value
            try:
                self.angle = self.__angle
            except:
                pass
        else:
            self.__lscale  = 1

    @property
    def Placement(self):
        """ """
        if hasattr(self, "Placement"):
            return self._Placement
        else:
            return None

    @Placement.setter
    def Placement(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        if isinstance(value, Base.Placement):
            self._Placement = value
            vertex = Part.Vertex(Base.Vector(self.point))
            vertex.Placement = value
            self.point = vertex.Point
            if hasattr(self, 'line') and self.line is not None:
                self.line.Placement = value
        else:
            raise ValueError("Placement must be a Base.Placement object")
    
    def __repr__(self):
        return (str(self.label) + ": ( point: " + str(self.point)
                + ", angle: " + str(self.angle) + " )")



class Utils():
    """ """
    
    @staticmethod    
    def _convert_Edge_to_Curve(edge):
        """Convert a Freecad Edge to the respective Part.Curve

        Parameters
        ----------
        edge :
            

        Returns
        -------

        """
        curve = edge.Curve
        first = edge.FirstParameter
        last = edge.LastParameter
        if edge.Orientation == "Reversed":
            first, last = last, first
        output = None

        if isinstance(curve, Part.Line):
            output = Part.LineSegment(curve.value(first), curve.value(last))
        elif isinstance(curve, Part.Ellipse):
            output = Part.ArcOfEllipse(curve, first, last)
            if edge.Orientation == "Reversed":
                output.Axis = -output.Axis
                p0 = curve.value(first)
                p1 = curve.value(last)
                output = Part.ArcOfEllipse(output.Ellipse,
                                           output.Ellipse.parameter(p0),
                                           output.Ellipse.parameter(p1),
                                           )
        elif isinstance(curve, Part.Circle):
            output = Part.ArcOfCircle(curve, first, last)
            if edge.Orientation == "Reversed":
                output.Axis = -output.Axis
                p0 = curve.value(first)
                p1 = curve.value(last)
                output = Part.ArcOfCircle(output.Circle,
                                          output.Circle.parameter(p0),
                                          output.Circle.parameter(p1),
                                          )
        elif isinstance(curve, Part.BezierCurve):
            output = Part.BezierCurve()
            poles = curve.getPoles()
            if edge.Orientation == "Reversed":
                poles.reverse()
            output.setPoles(poles)
            output.segment(first, last)
        elif isinstance(curve, Part.BSplineCurve):
            p = curve.discretize(100)
            if edge.Orientation == "Reversed":
                p.reverse()
            output = Part.BSplineCurve()
            output.interpolate(p)
        elif isinstance(curve, Part.OffsetCurve):
            c = curve.toNurbs()
            if isinstance(c, Part.BSplineCurve):
                if edge.Orientation == "Reversed":
                    c.reverse()
            output = Utils._convert_Edge_to_Curve(Part.Edge(c))
        else:
            print("Conversion of {} is still not supported!\
                  ".format(type(curve)))

        return output

    @staticmethod
    def convert_Wire_to_Curves(wire):
        """Convert a Part.Wire to the set of Part.Curve objects of which \
            it consists of.

        Parameters
        ----------
        wire :
            

        Returns
        -------

        """
        # print("Wire.Orientation: {}".format(wire.Orientation))
        output = []
        edges = wire.OrderedEdges
        for e in edges:
            output.append(Utils._convert_Edge_to_Curve(e))
        # if wire.Orientation == "Reversed":
        #     output.reverse()
        return output

    @staticmethod
    def wire_closure(wire):
        """

        Parameters
        ----------
        wire :
            

        Returns
        -------
        type
            Otherwire, None is returned.

        """
        if not wire.isClosed():
            ov = wire.OrderedVertexes
            ls = Part.LineSegment(ov[-1].Point, ov[0].Point)
            return Part.Wire(Part.Shape([ls]).Edges)
        return None

    @staticmethod    
    def close_wire(wire):
        """

        Parameters
        ----------
        wire :
            

        Returns
        -------

        """
        if not wire.isClosed():
            wls = Utils.wire_closure(wire)
            if wls is not None:
                return Part.Wire([wire, wls])
            else:
                raise ValueError("Wire not closed, but start and end point"
                                 "are coincident")
        return wire
    
#     @staticmethod    
#     def joinWires(wires, closed = False):
#         import copy
#         output = []
#         nw = copy.deepcopy(wires)
#         if closed:
#             nw.append(wires[0])    
        
#         for i in range(len(nw)-1):
#             w1 = nw[i]
#             w2 = nw[i+1]
#             output.append(w1)
#             if (w1.OrderedVertexes[-1].Point != w2.OrderedVertexes[0].Point):
#                 l = Part.LineSegment(w1.OrderedVertexes[-1].Point, w2.OrderedVertexes[0].Point)
#                 w = Part.Wire(Part.Shape([l]).Edges)
#                 output.append(w)
        
#         return output

#     @staticmethod    
#     def centroidnp(arr, w = None):
#         l = float(arr.shape[0])
#         b = arr
#         if not w is None:
#             b = (arr.T * w).T
#         sum_x = numpy.sum(b[:, 0])
#         sum_y = numpy.sum(b[:, 1])
#         return sum_x/l, sum_y/l
    
    @staticmethod
    def create_rectangle(*args):
        """Create a rectangle -> Part.Wire
            - P1 and P2
            - Center, w, and h

        Parameters
        ----------
        *args :
            

        Returns
        -------

        """
        if len(args) == 2:
            p1 = args[0]
            p2 = args[1]
            wire = Part.makePolygon([p1,
                                     Base.Vector(p2[0], p1[1], 0.),
                                     p2,
                                     Base.Vector(p1[0], p2[1], 0.),
                                     p1]
                                    )
            return wire

        if len(args) == 3:
            center = args[0]
            w = args[1]
            h = args[2]
            p1 = center - Base.Vector(w/2., h/2., 0.)
            p2 = center + Base.Vector(w/2., h/2., 0.)
            return Utils.create_rectangle(p1, p2)

        raise ValueError("input must be: (P1, P2) "
                         "or (Center, w, h)")

    @staticmethod
    def create_polygon(pointlist, closed=False):
        """

        Parameters
        ----------
        pointlist :
            
        closed :
             (Default value = False)

        Returns
        -------

        """
        if closed:
            pointlist = pointlist + pointlist[0:1]
        return Part.makePolygon(pointlist)

    @staticmethod
    def test_geo1():
        """ """
        pntslist = [Base.Vector(), Base.Vector(1., 0., 0.),
                    Base.Vector(1., 1., 0.), Base.Vector(0., 1., 0.)]
        wire = Utils.create_polygon(pntslist, True)
        shape = Shape(wire, label="polygon")
        return shape

    @staticmethod
    def test_geo2():
        """creation of a simple polygon shape"""
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.)]
        pntslist2 = [Base.Vector(2., 1., 0.), Base.Vector(0., 1., 0.),
                     Base.Vector()]
        wire1 = Utils.create_polygon(pntslist1)
        arc = Part.makeCircle(1., Base.Vector(1., 1., 0.),
                              Base.Vector(0., 0., 1.), 270., 360.)
        wire2 = Utils.create_polygon(pntslist2)
        shape = Shape(Part.Wire([wire1, arc, wire2]), label="witharc")
        return shape

    @staticmethod
    def test_geo3():
        """creation of a polygon with arc shape"""
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.)]
        pntslist2 = [Base.Vector(2., 1., 0.), Base.Vector(0., 1., 0.),
                     Base.Vector()]
        wire1 = Utils.create_polygon(pntslist1)
        arc = Part.makeCircle(1., Base.Vector(1., 1., 0.),
                              Base.Vector(0., 0., 1.), 270., 360.)
        wire2 = Utils.create_polygon(pntslist2)
        
        wire = Part.Wire([wire1, arc, wire2])
        wire.reverse()
        shape = Shape(wire, label="witharc")
        return shape

    @staticmethod
    def test_geo4():
        """polygon and arc geometry. Arc and a wire are reversed"""
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.)]
        pntslist2 = [Base.Vector(2., 1., 0.), Base.Vector(0., 1., 0.),
                     Base.Vector()]
        wire1 = Utils.create_polygon(pntslist1)
        
        arc = Part.makeCircle(1., Base.Vector(1., 1., 0.),
                              Base.Vector(0., 0., -1.), 0., 90.)
        arc.reverse()
        # reverse pntslist2
        pntslist2.reverse()

        wire2 = Utils.create_polygon(pntslist2)
        wire2.reverse()
        
        wire = Part.Wire([wire1, arc, wire2])
        wire.reverse()
        shape = Shape(wire, label="witharc")
        return shape

    @staticmethod
    def test_geo5():
        """shape of shapes"""
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.)]
        pntslist2 = [Base.Vector(2., 1., 0.), Base.Vector(0., 1., 0.),
                     Base.Vector()]
        wire1 = Utils.create_polygon(pntslist1)
        s1 = Shape(wire1, "wire1")
        
        arc = Part.makeCircle(1., Base.Vector(1., 1., 0.),
                              Base.Vector(0., 0., -1.), 0., 90.)
        arc.reverse()
        s2 = Shape(Part.Wire(arc), "reversed_arc")
        # reverse pntslist2
        pntslist2.reverse()

        wire2 = Utils.create_polygon(pntslist2)
        wire2.reverse()
        s3 = Shape(wire2, "wire2")
        
        shape = Shape([s1,s2,s3], label="witharc")
        return shape

    @staticmethod
    def test_geo6():
        """shapes with common parts"""
        # create shape1
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.)]
        pntslist2 = [Base.Vector(2., 1., 0.), Base.Vector(0., 1., 0.),
                     Base.Vector()]
        wire1 = Utils.create_polygon(pntslist1)
        s1 = Shape(wire1, "wire1")

        arc = Part.makeCircle(1., Base.Vector(1., 1., 0.),
                              Base.Vector(0., 0., -1.), 0., 90.)
        arc.reverse()
        warc = Part.Wire(arc)
        s2 = Shape(warc, "reversed_arc")
        # reverse pntslist2
        pntslist2.reverse()

        wire2 = Utils.create_polygon(pntslist2)
        wire2.reverse()
        s3 = Shape(wire2, "wire2")

        shape1 = Shape([s1, s2, s3], label="witharc")

        # create shape2
        pntslist3 = [Base.Vector(2., 1., 0.), Base.Vector(3., 1., 0.),
                     Base.Vector(3., -1., 0.), Base.Vector(0., -1., 0.),
                     Base.Vector()]
        wire3 = Utils.create_polygon(pntslist3)
        s3 = Shape(wire3, "wire3")
        shape2 = Shape([s1, s2, s3])
        return shape1, shape2

    def test_geo7():
        """ """
        pntslist1 = [Base.Vector(), Base.Vector(1., 0., 0.),
                     Base.Vector(1., 1., 0.), Base.Vector(0., 1., 0.)]
        wire1 = Utils.create_polygon(pntslist1, True)
        shape1 = Shape(wire1, label="polygon1")

        pntslist2 = [Base.Vector(), Base.Vector(0.5, 0., 0.),
                     Base.Vector(0.5, -1., 0.), Base.Vector(0., -1., 0.)]
        wire2 = Utils.create_polygon(pntslist2, True)
        shape2 = Shape(wire2, label="polygon2")
        return shape1, shape2



# if __name__ == "__main__":
#     # print("start")
#     # pntslist = [Base.Vector(1., 1., 0.), Base.Vector(0., 1., 0.),
#     #             Base.Vector(0., 0., 0.), Base.Vector(1., 0., 0.)]

#     # pntslist1 = [Base.Vector(1., 0., 0.), Base.Vector(2., 0., 0.),
#     #              Base.Vector(2., 1., 0.), Base.Vector(1., 1., 0.)]
    
#     # pntslist2 = [Base.Vector(1., 1., 0.), Base.Vector(1., 0., 0.)]

#     # w1 = Part.makePolygon(pntslist)
#     # w2 = Part.makePolygon(pntslist1)
#     # w3 = Part.makePolygon(pntslist2)
    
#     # w1.reverse()
    
#     # # print(w2.Orientation)
#     # # [print("{}({}): ({}, {})".format(e.Curve, e.Orientation,
#     # #                                  e.Curve.value(e.FirstParameter),
#     # #                                  e.Curve.value(e.LastParameter)))
#     # #  for e in w2.Edges]

#     # w2.reverse()

#     # print(w2.Orientation)
#     # [print("{}({}): ({}, {})".format(e.Curve, e.Orientation,
#     #                                   e.Curve.value(e.FirstParameter),
#     #                                   e.Curve.value(e.LastParameter)))
#     #   for e in w2.Edges]

#     # # print(Utils.convert_Wire_to_Curves(w1))
#     # # print(Utils.convert_Wire_to_Curves(w2))

#     # s1 = mirapy.core.Shape(w1)
#     # s2 = mirapy.core.Shape(w2)
#     # s2.physicalGroups = {1: "lpf"}

#     # s = s1 + s2

#     # # print(s.getSingleWire().isClosed())

#     # w = s.getSingleWire()
#     # # print(Utils.convert_Wire_to_Curves(w))

#     # axis = s1.plot2D()
#     # axis = s2.plot2D(axis, True)

#     # curves1 = Utils.convert_Wire_to_Curves(w1)
#     # # [print(c.StartPoint) for c in curves1]
    
#     # curves2 = Utils.convert_Wire_to_Curves(w2)
#     # # [print(c.StartPoint) for c in curves2]

#     # s0 = mirapy.core.Shape(Part.Wire(w.OrderedEdges))
#     # import mirapy
#     # c0 = mirapy.core.Component()
#     # c1 = mirapy.core.Component("C", s0, parent = c0)
#     # c2 = mirapy.core.Component("C2", s2, parent = c0)
#     # m = mirapy.core.Mesh()
#     # #m.meshfile = "Mesh.step"
#     # m(c0, clean=False)
    
#     # # doc = FreeCAD.newDocument()
#     # # shapeobj = doc.addObject("Part::Feature", "MyShape")
#     # # shapeobj.Shape = Part.Shape(w)
#     # # doc.recompute()
    
#     # # [print("{}({}): ({}, {})".format(e.Curve, e.Orientation,
#     # #                                   e.Curve.value(e.FirstParameter),
#     # #                                   e.Curve.value(e.LastParameter)))
#     # #   for e in w.OrderedEdges]    
    
#     # print("end")
    
#     print("Start test2")
#     pntslist = [Base.Vector(1., 1., 0.), Base.Vector(0., 1., 0.)]

#     pntslist1 = [Base.Vector(1., 0., 0.), Base.Vector(2., 0., 0.),
#                   Base.Vector(2., 1., 0.), Base.Vector(1., 1., 0.)]
    
#     pntslist2 = [Base.Vector(1., 1., 0.), Base.Vector(1., 0., 0.)]

#     w1_l = Part.makePolygon(pntslist)
#     w1_arc = Part.makeCircle(1., Base.Vector(1.,1.,0.), Base.Vector(0,0,-1), 
#                           90., 180.)

#     w1_arc.reverse()
    
#     w1 = Part.Wire([w1_l, w1_arc])
#     w1.reverse()
    
#     w2_l = Part.makePolygon(pntslist1[0:2])

#     curve = Part.BezierCurve()
#     curve.setPoles(pntslist1[1:4])
    
#     w2_b = Part.Wire(Part.Shape([curve]).Edges)

#     w2 = Part.Wire([w2_l, w2_b])
    
#     s1 = mirapy.geo.Shape(w1)
#     s2 = mirapy.geo.Shape(w2)
#     s2.physicalGroups = {1: "lpf"}

#     s = s1 + s2
#     print(s.getSingleWire().isClosed())
    
#     # w = s.getSingleWire()
#     # w.reverse()
#     # s = mirapy.core.Shape(w)
    
#     wx_l = Part.makePolygon([Base.Vector(1., 1., 0.), Base.Vector(-1., 2., 0.),
#                               Base.Vector(0., 0., 0.), Base.Vector(1., 0., 0.)])

#     sx = mirapy.geo.Shape([w1, wx_l])
#     sx.physicalGroups = {1: "lpx"}
    
#     import mirapy
#     c0 = mirapy.core.Component("DEMO")
#     c1 = mirapy.core.Component("C", s, parent = c0)
#     c2 = mirapy.core.Component("C2", s2, parent = c0)
#     c3 = mirapy.core.Component("Cx", sx, parent = c0)
#     m = mirapy.meshing.Mesh()
#     #m.meshfile = "Mesh.step"
#     m(c0, clean=False)
    
#     c0.plot2D()
    
#     print("end test2")
    
#     print("start test3")
#     import BOPTools.SplitAPI
#     list_of_shapes = [w1, w2]
#     pieces, map = list_of_shapes[0].generalFuse(list_of_shapes[1:], 1e-8)
#     s3 = mirapy.core.Shape(pieces.Wires)
    
#     list_of_shapes = [s.getSingleWire(), sx.getSingleWire()]
#     pieces, map = list_of_shapes[0].generalFuse(list_of_shapes[1:], 1e-8)
#     s4 = mirapy.core.Shape(pieces.Wires)    
    
#     f = s.face.fuse([sx.face]) #Part.Face(s.getSingleWire())
#     doc = FreeCAD.newDocument()
#     shapeobj = doc.addObject("Part::Feature", "Face")
#     shapeobj.Shape = Part.Shape(f)
#     doc.recompute()
#     #FreeCAD.ActiveDocument.Objects[0].Shape.exportStep("o.step")
#     solid = f.extrude(Base.Vector(0.,0.,30.))
#     shapeobj = doc.addObject("Part::Feature", "Solid")
#     shapeobj.Shape = Part.Shape(solid)    
#     doc.saveAs("test.FCStd")
    
#     print("end test3")
