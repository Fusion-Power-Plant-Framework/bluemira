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


def trace_torus_orbit(r_1, r_2, n_r_2_turns, n_points):
    """
    Trace the orbit of a particle travelling around an a torus.

    Parameters
    ----------
    r_1: float
      The radius of the x-y circle
    r_2: float
      The radius of the particle orbit around the x-y circle
    n_r_2_turns: float
      The number of orbits around the centreline when making one full turn of the x-y circle
    n_points: int
      The number of points to produce

    Returns
    -------
    xyz_array: np.ndarray
      Array of shape (3, n_points)
    """
    phi = np.linspace(0, 2 * np.pi, num=n_points)  # Major angle
    theta = np.linspace(0, 2 * np.pi * n_r_2_turns, num=n_points)  # Minor angle
    x = (r_1 + r_2 * np.cos(theta)) * np.cos(phi)
    y = (r_1 + r_2 * np.cos(theta)) * np.sin(phi)
    z = r_2 * np.sin(theta)
    xyz_array = np.array([x, y, z])
    return xyz_array


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

    def test_rotate_90(self):
        c = Coordinates(
            np.array(
                [
                    [2, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0],
                    [1, 1, 2, 2, 1],
                ]
            )
        )
        v1 = c.normal_vector
        assert np.allclose(v1, [0, 1, 0])
        assert c.check_ccw()

        c.rotate(degree=90)
        v2 = c.normal_vector
        assert np.allclose(abs(v2), [1, 0, 0])
        assert c.check_ccw([-1, 0, 0])

        c.rotate(degree=90)
        v3 = c.normal_vector
        assert np.allclose(abs(v3), [0, 1, 0])
        assert c.check_ccw([0, -1, 0])

        c.rotate(degree=90)
        v4 = c.normal_vector
        assert np.allclose(abs(v4), [1, 0, 0])
        assert c.check_ccw([1, 0, 0])

        c.rotate(degree=90)
        v5 = c.normal_vector
        assert np.allclose(abs(v5), [0, 1, 0])
        assert c.check_ccw([0, 1, 0])

    def test_is_planar(self):
        c = Coordinates(
            np.array(
                [
                    [2, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0],
                    [1, 1, 2, 2, 1],
                ]
            )
        )
        v = np.random.rand(3)
        base = np.random.rand(3)
        direction = np.random.rand(3)
        degree = 360 * np.random.rand(1)
        c.rotate(base, direction, degree)
        assert c.is_planar

    def test_complicated(self):
        xyz = trace_torus_orbit(5, 1, 10, 1000)
        c = Coordinates(xyz)

        assert c.closed
        assert not c.is_planar
