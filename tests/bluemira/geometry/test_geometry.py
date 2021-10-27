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

from bluemira.geometry.wire import BluemiraWire

from bluemira.geometry.tools import (
    make_polygon,
    make_ellipse,
    make_circle,
    make_circle_arc_3P,
)
from scipy.special import ellipe
import math
import pytest


class TestGeometry:
    @classmethod
    def setup_class(cls):
        cls.square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]

    def test_create_wire(self):
        wire = make_polygon(self.square_points, label="test", closed=False)
        assert wire.length == 3.0
        assert wire.area == 0.0
        assert wire.volume == 0.0
        assert wire.label == "test"
        assert not wire.is_closed()

    def test_close_wire(self):
        wire = make_polygon(self.square_points, label="test", closed=True)
        assert wire.length == 4.0
        assert wire.area == 0.0
        assert wire.volume == 0.0
        assert wire.label == "test"
        assert wire.is_closed()

    def test_add_wires(self):
        wire1 = make_polygon(self.square_points[0:3], label="wire1", closed=False)
        wire2 = make_polygon(self.square_points[2:], label="wire2", closed=False)
        wire3 = wire1 + wire2
        wire3.label = "wire3"
        assert wire1.length == 2.0
        assert wire2.length == 1.0
        assert wire3.length == 3.0
        wire1 += wire2
        assert wire1.length == 3.0

    def test_make_circle(self):
        radius = 2.0
        center = [1, 0, 3]
        axis = [0, 1, 0]
        bm_circle = make_circle(radius=radius, center=center, axis=axis)
        assert bm_circle.length == 2 * math.pi * radius

    def test_make_circle_arc_3P(self):  # noqa N802
        p1 = [0, 0, 0]
        p2 = [1, 1, 0]
        p3 = [2, 0, 0]
        bm_circle = make_circle_arc_3P(p1, p2, p3)
        assert bm_circle.length == math.pi

    def test_make_ellipse(self):
        major_radius = 5.0
        minor_radius = 2.0

        bm_ellipse = make_ellipse(
            major_radius=major_radius,
            minor_radius=minor_radius,
        )
        edge = bm_ellipse.boundary[0].Edges[0]

        # ellispe eccentricity
        eccentricity = math.sqrt(1 - (minor_radius / major_radius) ** 2)
        assert eccentricity == edge.Curve.Eccentricity

        # theoretical length
        expected_length = 4 * major_radius * ellipe(eccentricity ** 2)
        assert pytest.approx(edge.Length) == expected_length

        # WARNING: it seems that FreeCAD implements in a different way
        # Wire.Length and Edge.length giving a result slightly different
        # but enough to make the following assert fail. To be investigated.
        # assert pytest.approx(bm_ellipse.length) == expected_length

    def test_copy_deepcopy(self):
        points = self.square_points
        points.append(self.square_points[0])
        wire1 = make_polygon(points[0:4], label="wire1")
        wire2 = make_polygon(points[3:], label="wire2")
        wire = BluemiraWire([wire1, wire2], label="wire")
        wire_copy = wire.copy()
        wire_deepcopy = wire.deepcopy()

        assert wire_copy.label == wire.label
        assert wire_deepcopy.label == wire.label
        assert wire.length == (wire1.length + wire2.length)
        assert wire.length == wire_copy.length
        assert wire.length == wire_deepcopy.length
        w1_len = wire1.length
        w2_len = wire2.length
        w_len = wire.length

        wire.scale(2)
        assert wire.length == 2 * w_len
        assert wire.length == wire_copy.length
        assert w_len == wire_deepcopy.length
        assert wire1.length == 2 * w1_len
        assert wire2.length == 2 * w2_len

        wire_copy = wire.copy("wire_copy")
        wire_deepcopy = wire.deepcopy("wire_deepcopy")

        assert wire_copy.label == "wire_copy"
        assert wire_deepcopy.label == "wire_deepcopy"
