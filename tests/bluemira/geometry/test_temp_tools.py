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

import tests
import pytest
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.base import Plane
from bluemira.geometry.temp_tools import (
    check_linesegment,
    bounding_box,
    on_polygon,
    in_polygon,
    loop_plane_intersect,
    polygon_in_polygon,
)
from bluemira.geometry.loop import Loop


class TestCheckLineSegment:
    def test_true(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [0.5, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True
        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.0005, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True

        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [1.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True
        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True

    def test_false(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [5.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is False

        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.005, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is False


class TestBoundingBox:
    def test_null(self):
        x, y, z = np.zeros(100), np.zeros(100), np.zeros(100)
        xb, yb, zb = bounding_box(x, y, z)
        assert np.all(xb == 0)
        assert np.all(yb == 0)
        assert np.all(zb == 0)

    def test_random(self):
        x, y, z = np.random.rand(100), np.random.rand(100), np.random.rand(100)
        args = np.random.randint(0, 100, 8)
        x[args] = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
        y[args] = np.array([-2, -2, 2, 2, 2, -2, -2, 2])
        z[args] = np.array([-2, 2, -2, 2, -2, 2, -2, 2])
        xb, yb, zb = bounding_box(x, y, z)

        assert np.allclose(xb, np.array([-2, -2, -2, -2, 2, 2, 2, 2]))
        assert np.allclose(yb, np.array([-2, -2, 2, 2, -2, -2, 2, 2]))
        assert np.allclose(zb, np.array([-2, 2, -2, 2, -2, 2, -2, 2]))


class TestOnPolygon:
    def test_simple(self):
        loop = Loop(x=[0, 1, 2, 2, 0, 0], z=[-1, -1, -1, 1, 1, -1])
        for p in loop.d2.T:
            assert on_polygon(p[0], p[1], loop.d2.T) is True

        fails = [[4, 4], [5, 5], [0.1, 0.1]]
        for fail in fails:
            assert on_polygon(*fail, loop.d2.T) is False


class TestLoopPlane:
    def test_simple(self):
        loop = Loop(x=[0, 1, 2, 2, 0, 0], z=[-1, -1, -1, 1, 1, -1])
        plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = loop_plane_intersect(loop, plane)
        e = np.array([[0, 0, 0], [2, 0, 0]])
        assert np.allclose(intersect, e)

    def test_complex(self):
        loop = Loop(
            x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
            z=[-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
        )
        plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 2
        f, ax = plt.subplots()
        loop.plot(ax)
        for i in intersect:
            assert on_polygon(i[0], i[2], loop.d2.T)
            ax.plot(i[0], i[2], marker="o", color="r")
        plane = Plane([0, 0, 2.7], [1, 0, 2.7], [0, 1, 2.7])  # x-y offset
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 4
        for i in intersect:
            assert on_polygon(i[0], i[2], loop.d2.T)
            ax.plot(i[0], i[2], marker="o", color="r")

        plane = Plane([0, 0, 4], [1, 0, 4], [0, 1, 4])  # x-y offset
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 1
        for i in intersect:
            assert on_polygon(i[0], i[2], loop.d2.T)
            ax.plot(i[0], i[2], marker="o", color="r")

        plane = Plane([0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005])  # x-y offset
        intersect = loop_plane_intersect(loop, plane)
        assert intersect is None

    def test_other_dims(self):
        loop = Loop(
            x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
            y=[-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
        )
        plane = Plane([0, 0, 0], [1, 0, 0], [0, 0, 1])  # x-y
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 2
        f, ax = plt.subplots()
        loop.plot(ax)
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")

        plane = Plane([0, 10, 0], [1, 10, 0], [0, 10, 1])  # x-y
        intersect = loop_plane_intersect(loop, plane)
        assert intersect is None

    def test_xyzplane(self):
        loop = Loop(
            x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
            y=[-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
        )
        loop.translate([-2, 0, 0])
        plane = Plane([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = loop_plane_intersect(loop, plane)
        f, ax = plt.subplots()
        loop.plot(ax)
        for i in intersect:
            assert on_polygon(i[0], i[2], loop.d2.T)
            ax.plot(i[0], i[2], marker="o", color="r")

    def test_flat_intersect(self):
        # test that a shared segment with plane only gives two intersects
        loop = Loop(x=[0, 2, 2, 0, 0], z=[-1, -1, 1, 1, -1])
        plane = Plane([0, 0, 1], [0, 1, 1], [1, 0, 1])
        inter = loop_plane_intersect(loop, plane)
        assert np.allclose(inter, np.array([[0, 0, 1], [2, 0, 1]]))


class TestInPolygon:
    def test_simple(self):
        loop = Loop(x=[-2, 2, 2, -2, -2, -2], z=[-2, -2, 2, 2, 1.5, -2])
        in_points = [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ]

        out_points = [
            [-3, -3],
            [-3, 0],
            [-3, 3],
            [0, -3],
            [3, 3],
            [3, -3],
            [2.00000009, 0],
            [-2.0000000001, -1.999999999999],
        ]

        on_points = [
            [-2, -2],
            [2, -2],
            [2, 2],
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
            [-2, 2],
        ]

        if tests.PLOTTING:
            plt.close("all")
            f, ax = plt.subplots()
            loop.plot(ax, edgecolor="k")
            for point in in_points:
                check = in_polygon(*point, loop.d2.T)
                c = "b" if check else "r"
                ax.plot(*point, marker="s", color=c)
            for point in on_points:
                check = in_polygon(*point, loop.d2.T)
                c = "b" if check else "r"
                ax.plot(*point, marker="o", color=c)
            for point in out_points:
                check = in_polygon(*point, loop.d2.T)
                c = "b" if check else "r"
                ax.plot(*point, marker="*", color=c)

            plt.show()

        # Test single and arrays
        for p in in_points:
            assert in_polygon(*p, loop.d2.T), p
        assert np.all(polygon_in_polygon(np.array(in_points), loop.d2.T))

        for p in on_points:
            assert in_polygon(*p, loop.d2.T, include_edges=True), p
        assert np.all(
            polygon_in_polygon(np.array(on_points), loop.d2.T, include_edges=True)
        )

        for p in on_points:
            assert not in_polygon(*p, loop.d2.T), p

        assert np.all(~polygon_in_polygon(np.array(on_points), loop.d2.T))

        for p in out_points:
            assert not in_polygon(*p, loop.d2.T), p
        assert np.all(~polygon_in_polygon(np.array(out_points), loop.d2.T))

    def test_big(self):
        filename = get_bluemira_path("geometry/test_data", subfolder="tests")
        filename += "/in_polygon_test.pkl"
        with open(filename, "rb") as file:
            data = pickle.load(file)  # noqa (S301)

        x = data["X"]
        z = data["Z"]

        if tests.PLOTTING:
            f, ax = plt.subplots()

        n, m = x.shape
        mask = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if in_polygon(x[i, j], z[i, j], data["LCFS"].d2.T):
                    mask[i, j] = 1
        if tests.PLOTTING:
            data["LCFS"].plot(ax, fill=False, edgecolor="k")
            ax.contourf(data["X"], data["Z"], mask, levels=[0, 0.5, 1])
            plt.show()

        hits = np.count_nonzero(mask)
        assert hits == 1171, hits  # Recursion test


if __name__ == "__main__":
    pytest.main([__file__])
