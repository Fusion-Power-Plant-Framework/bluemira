# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import (
    make_circle,
    make_circle_arc_3P,
    make_ellipse,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.temp import get_coil_points, make_coils_from_wire
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
    disc_points = reduce_coordinates(square.discretise(20, byedges=True))
    assert np.allclose(square_points, disc_points)


class TestDiscretisation:
    @classmethod
    def setup_class(cls):
        cls.circle = make_circle(2, center=(4, 0, 0), axis=(0, 1, 0))
        cls.ellipse = make_ellipse(
            center=(4, 0, 0),
            major_radius=4,
            minor_radius=2,
            major_axis=(0, 0, 1),
            minor_axis=(1, 0, 0),
        )
        points = Coordinates({
            "x": [
                1.0,
                1.0,
                2 - 0.5 * np.sqrt(2),
                2.0,
                6.0,
                2.0,
                2 - 0.5 * np.sqrt(2),
                1.0,
            ],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [
                -4.0,
                4.0,
                4 + 0.5 * np.sqrt(2),
                5.0,
                0.0,
                -5.0,
                -4 - 0.5 * np.sqrt(2),
                -4.0,
            ],
        }).T

        wires = [
            make_polygon(points[:2, :]),
            make_circle_arc_3P(points[1, :], points[2, :], points[3, :]),
            make_circle_arc_3P(points[3, :], points[4, :], points[5, :]),
            make_circle_arc_3P(points[5, :], points[6, :], points[7, :]),
        ]
        cls.dshape = BluemiraWire(wires)

    def test_comparison_plot(self):
        simple_kwargs = {"edgecolor": "red", "facecolor": "white"}
        detailed_kwargs = {"edgecolor": "blue", "facecolor": "white"}
        coils_circle_simple = make_coils_from_wire(self.circle, 0.06)
        coils_circle = make_coils_from_wire(self.circle, 0.06, simple=False)
        coils_ellipse_simple = make_coils_from_wire(self.ellipse, 0.06)
        coils_ellipse = make_coils_from_wire(self.ellipse, 0.06, simple=False)
        coils_dshape_simple = make_coils_from_wire(self.dshape, 0.06)
        coils_dshape = make_coils_from_wire(self.dshape, 0.06, simple=False)
        f = plt.figure()
        ax1 = f.add_subplot(1, 3, 1)
        coils_circle_simple.plot(ax=ax1, **simple_kwargs)
        coils_circle.plot(ax=ax1, **detailed_kwargs)
        ax2 = f.add_subplot(1, 3, 2)
        coils_ellipse_simple.plot(ax=ax2)
        coils_ellipse.plot(ax=ax2)
        ax3 = f.add_subplot(1, 3, 3)
        coils_dshape_simple.plot(ax=ax3)
        coils_dshape.plot(ax=ax3)

    def test_coil_points(self):
        coils_circle = make_polygon(get_coil_points(self.circle, 0.06))
        np.testing.assert_allclose(coils_circle.length, self.circle.length, rtol=1e-2)
        coils_ellipse = make_polygon(get_coil_points(self.ellipse, 0.06))
        np.testing.assert_allclose(coils_ellipse.length, self.ellipse.length, rtol=1e-2)
        coils_dshape = make_polygon(get_coil_points(self.dshape, 0.06))
        np.testing.assert_allclose(coils_dshape.length, self.dshape.length, rtol=1e-2)
