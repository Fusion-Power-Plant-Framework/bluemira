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

from bluemira.geometry.tools import make_polygon


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
