# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

from bluemira.geometry.tools import make_polygon
from bluemira.magnetostatics.tools import process_xyz_array, reduce_coordinates


def test_xyz_decorator():
    class TestClassDecorator:
        def __init__(self):
            self.a = 4

        @process_xyz_array
        def func(self, x, y, z):
            return np.array([self.a + x + y, y + z, x - z])

    tester = TestClassDecorator()

    result = tester.func(4, 5, 6)
    assert result.shape == (3,)

    result = tester.func([4], [5], [6])
    assert result.shape == (3,)

    result = tester.func(np.array(4), np.array([4]), 5)
    assert result.shape == (3,)

    x = np.array([3, 4, 5, 6])
    result = tester.func(x, x, x)
    assert result.shape == (3, 4)

    rng = np.random.default_rng()
    x = rng.random((16, 16))
    result = tester.func(x, x, x)
    assert result.shape == (3, 16, 16)

    result2 = np.zeros((3, 16, 16))
    for i in range(16):
        for j in range(16):
            result2[:, i, j] = tester.func(x[i, j], x[i, j], x[i, j])

    assert np.allclose(result2, result)


def test_reduce_coordinates():
    square = make_polygon({"x": [0, 1, 1, 0], "z": [0, 0, 1, 1]})
    square_points = square.vertexes
    disc_points = reduce_coordinates(square.discretize(20, byedges=True))
    assert np.allclose(square_points, disc_points)
