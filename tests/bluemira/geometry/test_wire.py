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
import numpy as np

from bluemira.geometry.tools import make_polygon


class TestWire:
    def test_start_point(self):
        n_coords = 10
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(0, 2, n_coords)
        coords[2, :] = np.linspace(-5, 0, n_coords)
        wire = make_polygon(coords)

        start_point = wire.start_point()

        np.testing.assert_equal(start_point, np.array([0, 0, -5]))

    def test_end_point(self):
        n_coords = 10
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(0, 2, n_coords)
        coords[2, :] = np.linspace(-5, 0, n_coords)
        wire = make_polygon(coords)

        end_point = wire.end_point()

        np.testing.assert_equal(end_point, np.array([2, 0, 0]))
