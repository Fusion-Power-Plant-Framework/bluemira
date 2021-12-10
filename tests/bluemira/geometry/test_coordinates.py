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
import pytest
from copy import deepcopy

from bluemira.geometry.error import CoordinatesError
from bluemira.geometry.coordinates import Coordinates


class TestCoordinates:
    def test_array_init(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        c1 = Coordinates(xyz)
        c2 = Coordinates(xyz.T)
        assert c1 == c2
        assert np.allclose(c1.x, xyz[0])
        assert np.allclose(c1.y, xyz[1])
        assert np.allclose(c1.z, xyz[2])
        assert np.allclose(c1[0], xyz[0])
        assert np.allclose(c1[1], xyz[1])
        assert np.allclose(c1[2], xyz[2])

    def test_bad_array_init(self):
        a = np.random.rand(4, 4)
        with pytest.raises(CoordinatesError):
            Coordinates(a)

        a = np.array([[1, 2, 3, 4, 5], [1, 2, 3], [0, 0, 0, 0]], dtype=object)
        with pytest.raises(CoordinatesError):
            Coordinates(a)

    def test_obj_but_good_lengths(self):
        xyz = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], dtype=object)
        c = Coordinates(xyz)
        assert np.allclose(c.x, [1, 2, 3, 4, 5])
        assert np.allclose(c.y, [1, 2, 3, 4, 5])
        assert np.allclose(c.z, [0, 0, 0, 0, 0])

    def test_dict_init(self):
        xyz = {"x": [0, 1, 2, 3], "y": 1, "z": 0}
        c = Coordinates(xyz)
        assert np.allclose(c.x, np.array([0, 1, 2, 3]))
        assert np.allclose(c.y, np.array([1, 1, 1, 1]))
        assert np.allclose(c.z, np.array([0, 0, 0, 0]))

        xyz = {"x": [0, 0, 0, 0], "y": [1, 2, 2, 1], "z": [-1, -1, 1, 1]}
        c = Coordinates(xyz)
        assert np.allclose(c.x, xyz["x"])
        assert np.allclose(c.y, xyz["y"])
        assert np.allclose(c.z, xyz["z"])

    def test_ragged_dict_init(self):
        xyz = {"x": [0, 0, 0, 0], "y": [1, 2, 1], "z": [-1, -1, 1, 1]}
        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

    def test_iterable_init(self):
        xyz = [[0, 0, 0], (1, 1, 1), [2, 2, 2], [3, 3, 3]]
        c = Coordinates(xyz)

        assert np.allclose(c.x, list(range(4)))
        assert np.allclose(c.y, list(range(4)))
        assert np.allclose(c.z, list(range(4)))

        xyz = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], (0, 1, 2, 3, 4))
        c = Coordinates(xyz)

        assert np.allclose(c.x, list(range(5)))
        assert np.allclose(c.y, list(range(5)))
        assert np.allclose(c.z, list(range(5)))

    def test_bad_iterable_init(self):
        xyz = [[0, 0, 0], (1, 1), [2, 2, 2], [3, 3, 3]]

        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

        xyz = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], (0, 1, 2, 3))
        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

    def test_indexing(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        c1 = Coordinates(xyz)
        assert np.alltrue(xyz[0] == c1[0])
        assert np.alltrue(xyz[1] == c1[1])
        assert np.alltrue(xyz[2] == c1[2])

        assert np.allclose([3, 0, 3], c1[:, -1])

    def test_unpacking(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        x1, y1, z1 = xyz
        c1 = Coordinates(xyz)
        x2, y2, z2 = c1
        assert np.alltrue(x1 == x2)
        assert np.alltrue(y1 == y2)
        assert np.alltrue(z1 == z2)

    def test_translate(self):
        a = np.random.rand(3, 123)
        v = np.random.rand(3)
        c = Coordinates(a)
        c.translate(v)

        assert np.allclose(a + v.reshape(3, 1), c)

    def test_rotate(self):
        a = np.random.rand(3, 123)
        v = np.random.rand(3)
        base = np.random.rand(3)
        direction = np.random.rand(3)
        degree = 360 * np.random.rand(1)
        c = Coordinates(a)
        c2 = deepcopy(c)
        c.rotate(base, direction, degree)

        assert np.isclose(c.length, c2.length)
