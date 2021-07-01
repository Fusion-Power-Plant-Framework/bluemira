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

import pytest
import numpy

import freecad  # noqa: F401
import Part
from FreeCAD import Base

import bluemira.geometry._freecadapi as freecadapi


class TestFreecadapi:
    @classmethod
    def setup_class(cls):
        cls.square_points = ([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)])

    def test_fail_vector_to_numpy(self):
        with pytest.raises(TypeError):
            arr = freecadapi.vector_to_numpy(self.square_points)

    def test_fail_points_to_numpy(self):
        with pytest.raises(TypeError):
            arr = freecadapi.point_to_numpy(self.square_points)

    def test_single_vector_to_numpy(self):
        input = numpy.array((1., 0.5, 2.))
        vector = Base.Vector(input)
        arr = freecadapi.vector_to_numpy(vector)
        comparison = arr == input
        assert comparison.all()

    def test_vector_to_numpy(self):
        vectors = [Base.Vector(v) for v in self.square_points]
        arr = freecadapi.vector_to_numpy(vectors)
        comparison = arr == numpy.array(self.square_points)
        assert comparison.all()

    def test_point_to_numpy(self):
        vectors = [Part.Point(Base.Vector(v)) for v in self.square_points]
        arr = freecadapi.point_to_numpy(vectors)
        comparison = arr == numpy.array(self.square_points)
        assert comparison.all()

    def test_vertex_to_numpy(self):
        vertexes = [Part.Vertex(Base.Vector(v)) for v in self.square_points]
        arr = freecadapi.vertex_to_numpy(vertexes)
        comparison = arr == numpy.array(self.square_points)
        assert comparison.all()

    def test_make_polygon(self):
        # open wire
        open_wire: Part.Wire = freecadapi.make_polygon(self.square_points)
        vertexes = open_wire.Vertexes
        assert len(vertexes) == 4
        assert len(open_wire.Edges) == 3
        arr = freecadapi.vertex_to_numpy(vertexes)
        comparison = arr == numpy.array(self.square_points)
        assert comparison.all()
        assert not open_wire.isClosed()
        # closed wire
        closed_wire: Part.Wire = freecadapi.make_polygon(self.square_points, closed=True)
        vertexes = closed_wire.Vertexes
        assert len(vertexes) == 4
        assert len(closed_wire.Edges) == 4
        arr = freecadapi.vertex_to_numpy(vertexes)
        comparison = arr == numpy.array(self.square_points)
        assert comparison.all()
        assert closed_wire.isClosed()

    def test_make_bezier(self):
        bezier: Part.Wire = freecadapi.make_bezier(self.square_points)
        curve = bezier.Edges[0].Curve
        assert type(curve) == Part.BezierCurve

    def test_make_bspline(self):
        bspline: Part.Wire = freecadapi.make_bspline(self.square_points)
        curve = bspline.Edges[0].Curve
        assert type(curve) == Part.BSplineCurve

    def test_length(self):
        open_wire: Part.Wire = freecadapi.make_polygon(self.square_points)
        assert freecadapi.length(open_wire) == open_wire.Length == 3.
        closed_wire: Part.Wire = freecadapi.make_polygon(self.square_points, True)
        assert freecadapi.length(closed_wire) == closed_wire.Length == 4.

    def test_area(self):
        wire: Part.Wire = freecadapi.make_polygon(self.square_points, True)
        assert freecadapi.area(wire) == wire.Area == 0.
        face: Part.Face = Part.Face(wire)
        assert freecadapi.area(face) == face.Area == 1.

    def test_center_of_mass(self):
        wire: Part.Wire = freecadapi.make_polygon(self.square_points, True)
        face: Part.Face = Part.Face(wire)
        comparison = freecadapi.center_of_mass(wire) == numpy.array((0.5, 0.5, 0.))
        assert comparison.all()

    def test_scale_shape(self):
        factor = 2.
        wire: Part.Wire = freecadapi.make_polygon(self.square_points, True)
        scaled_wire = freecadapi.scale_shape(wire, factor)
        face: Part.Face = Part.Face(scaled_wire)
        assert freecadapi.area(face) == 1. * factor**2
        assert freecadapi.length(face) == freecadapi.length(scaled_wire) == 4. * factor
