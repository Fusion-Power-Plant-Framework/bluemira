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
import os
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.base import Plane, GeometryError
from bluemira.geometry.temp_tools import (
    check_linesegment,
    bounding_box,
    on_polygon,
    in_polygon,
    loop_plane_intersect,
    polygon_in_polygon,
    get_normal_vector,
    get_perimeter,
    get_area,
    rotation_matrix,
    offset,
    get_intersect,
    join_intersect,
)
from bluemira.geometry.loop import Loop

TEST_PATH = get_bluemira_path("bluemira/geometry/test_data", subfolder="tests")


class TestPerimeter:
    def test_simple(self):
        # 2 x 2 square
        x = [0, 2, 2, 0, 0]
        y = [0, 0, 2, 2, 0]
        assert get_perimeter(x, y) == 8.0


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


class TestGetNormal:
    def test_simple(self):
        x = np.array([0, 2, 2, 0, 0])
        z = np.array([0, 0, 2, 2, 0])
        y = np.zeros(5)
        n_hat = get_normal_vector(x, y, z)
        assert np.allclose(np.abs(n_hat), np.array([0, 1, 0]))

    def test_edge(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        z = np.array([1, 2, 4])
        n_hat = get_normal_vector(x, y, z)
        assert np.allclose(n_hat, 0.5 * np.array([np.sqrt(2), -np.sqrt(2), 0]))

    def test_error(self):
        fails = [
            [[0, 1], [0, 1], [0, 1]],
            [[0, 1, 2], [0, 1, 2], [0, 1]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        ]
        for fail in fails:
            with pytest.raises(GeometryError):
                get_normal_vector(
                    np.array(fail[0]), np.array(fail[1]), np.array(fail[2])
                )


class TestArea:
    def test_area(self):
        """
        Checked with:
        https://www.analyzemath.com/Geometry_calculators/irregular_polygon_area.html
        """
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        assert get_area(x, y) == 29.5
        loop = Loop(x=x, y=y)
        loop.rotate(43, p1=[3, 2, 1], p2=[42, 2, 1])
        assert np.isclose(get_area(*loop.xyz), 29.5)

    def test_error(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        with pytest.raises(GeometryError):
            get_area(x, y[:-1])


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

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop.plot(ax)

        for i in intersect:
            if tests.PLOTTING:
                ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], loop.d2.T)

        plane = Plane([0, 0, 2.7], [1, 0, 2.7], [0, 1, 2.7])  # x-y offset
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 4

        for i in intersect:
            if tests.PLOTTING:
                ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], loop.d2.T)

        plane = Plane([0, 0, 4], [1, 0, 4], [0, 1, 4])  # x-y offset
        intersect = loop_plane_intersect(loop, plane)
        assert len(intersect) == 1
        for i in intersect:
            if tests.PLOTTING:
                ax.plot(i[0], i[2], marker="o", color="r")

            assert on_polygon(i[0], i[2], loop.d2.T)

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

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop.plot(ax)
            for i in intersect:
                ax.plot(i[0], i[2], marker="o", color="r")
            plt.show()

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

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop.plot(ax)
        for i in intersect:
            if tests.PLOTTING:
                ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], loop.d2.T)

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
        """
        Regression test on a closed LCFS and an equilibrium grid.
        """
        filename = os.sep.join([TEST_PATH, "in_polygon_test.json"])

        lcfs = Loop.from_file(filename)

        x = np.linspace(4.383870967741935, 13.736129032258066, 65)
        z = np.linspace(-7.94941935483871, 7.94941935483871, 65)
        x, z = np.meshgrid(x, z, indexing="ij")

        n, m = x.shape
        mask = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if in_polygon(x[i, j], z[i, j], lcfs.d2.T):
                    mask[i, j] = 1
        if tests.PLOTTING:
            f, ax = plt.subplots()
            lcfs.plot(ax, fill=False, edgecolor="k")
            ax.contourf(x, z, mask, levels=[0, 0.5, 1])
            plt.show()

        hits = np.count_nonzero(mask)
        assert hits == 1171, hits


class TestRotationMatrix:
    def test_axes(self):
        axes = ["x", "y", "z"]
        axes2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for a1, a2 in zip(axes, axes2):
            r_1 = rotation_matrix(np.pi / 6, a1)
            r_2 = rotation_matrix(np.pi / 6, a2)
            assert np.allclose(r_1, r_2), a1

        axes = ["fail", "somthing", "1"]
        for axis in axes:
            with pytest.raises(GeometryError):
                rotation_matrix(30, axis)

    def test_ccw(self):
        p1 = [9, 0, 0]

        r_matrix = rotation_matrix(np.pi / 2, axis="z")
        p2 = r_matrix @ p1

        assert np.isclose(p2[1], 9), p2


class TestOffset:
    plot = tests.PLOTTING

    @classmethod
    def setup_class(cls):
        pass

    def test_rectangle(self):
        # Rectangle - positive offset
        x = [1, 3, 3, 1, 1, 3]
        y = [1, 1, 3, 3, 1, 1]
        o = offset(x, y, 0.25)
        assert sum(o[0] - np.array([0.75, 3.25, 3.25, 0.75, 0.75])) == 0
        assert sum(o[1] - np.array([0.75, 0.75, 3.25, 3.25, 0.75])) == 0
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*o, "r", marker="o")
            ax.set_aspect("equal")

    def test_triangle(self):
        x = [1, 2, 1.5, 1, 2]
        y = [1, 1, 4, 1, 1]
        t = offset(x, y, -0.25)
        assert (
            abs(sum(t[0] - np.array([1.29511511, 1.70488489, 1.5, 1.29511511])) - 0)
            < 1e-3
        )
        assert abs(sum(t[1] - np.array([1.25, 1.25, 2.47930937, 1.25])) - 0) < 1e-3
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*t, "r", marker="o")
            ax.set_aspect("equal")

    def test_complex_open(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1]
        # fmt:on

        c = offset(x, y, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*c, "r", marker="o")
            ax.set_aspect("equal")

    def test_complex_closed(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -3]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 1, 0, 2]
        # fmt:on

        c = offset(x, y, 1)
        if self.plot:
            f, ax = plt.subplots()
            ax.plot(x, y, "k")
            ax.plot(*c, "r", marker="o")
            ax.set_aspect("equal")


class TestIntersections:
    def test_get_intersect(self):
        loop1 = Loop(x=[0, 0.5, 1, 2, 3, 4, 0], z=[1, 1, 1, 1, 2, 5, 5])
        loop2 = Loop(x=[1.5, 1.5, 2.5, 2.5, 2.5], z=[4, -4, -4, -4, 5])
        shouldbe = [[1.5, 1], [2.5, 1.5], [2.5, 5]]
        intersect = np.array(get_intersect(loop1, loop2))
        correct = np.array(shouldbe).T
        assert np.allclose(intersect, correct)

        loop1 = Loop(x=[0, 0.5, 1, 2, 3, 4, 0], y=[1, 1, 1, 1, 2, 5, 5])
        loop2 = Loop(x=[1.5, 1.5, 2.5, 2.5, 2.5], y=[4, -4, -4, -4, 5])
        shouldbe = [[1.5, 1], [2.5, 1.5], [2.5, 5]]
        intersect = np.array(get_intersect(loop1, loop2))
        correct = np.array(shouldbe).T
        assert np.allclose(intersect, correct)

        loop1 = Loop(z=[0, 0.5, 1, 2, 3, 4, 0], y=[1, 1, 1, 1, 2, 5, 5])
        loop2 = Loop(z=[1.5, 1.5, 2.5, 2.5, 2.5], y=[4, -4, -4, -4, 5])
        shouldbe = [[1.5, 1][::-1], [2.5, 1.5][::-1], [2.5, 5][::-1]]
        intersect = np.array(get_intersect(loop1, loop2))
        correct = np.array(shouldbe).T
        assert np.allclose(intersect, correct)

    def test_join_intersect(self):
        loop1 = Loop(x=[0, 0.5, 1, 2, 3, 5, 4.5, 4, 0], z=[1, 1, 1, 1, 2, 4, 4.5, 5, 5])
        loop2 = Loop(x=[1.5, 1.5, 2.5, 2.5, 2.5], z=[4, -4, -4, -4, 5])
        join_intersect(loop1, loop2)
        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop1.plot(ax, fill=False, edgecolor="k", points=True)
            loop2.plot(ax, fill=False, edgecolor="r", points=True)
            plt.show()
        assert np.allclose(loop1[3], [1.5, 0, 1])
        assert np.allclose(loop1[5], [2.5, 0, 1.5])
        assert np.allclose(loop1[10], [2.5, 0, 5])

        loop1 = Loop(x=[0, 0.5, 1, 2, 3, 4, 0], y=[1, 1, 1, 1, 2, 5, 5])
        loop2 = Loop(x=[1.5, 1.5, 2.5, 2.5, 2.5], y=[4, -4, -4, -4, 5])
        join_intersect(loop1, loop2)

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop1.plot(ax, fill=False, edgecolor="k", points=True)
            loop2.plot(ax, fill=False, edgecolor="r", points=True)
            plt.show()
        assert np.allclose(loop1[3], [1.5, 1, 0])
        assert np.allclose(loop1[5], [2.5, 1.5, 0])
        assert np.allclose(loop1[8], [2.5, 5, 0])

        loop1 = Loop(z=[0, 0.5, 1, 2, 3, 4, 0], y=[1, 1, 1, 1, 2, 5, 5])
        loop2 = Loop(z=[1.5, 1.5, 2.5, 2.5, 2.5], y=[4, -4, -4, -4, 5])
        join_intersect(loop1, loop2)

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop1.plot(ax, fill=False, edgecolor="k", points=True)
            loop2.plot(ax, fill=False, edgecolor="r", points=True)
            plt.show()
        assert np.allclose(loop1[1], [0, 5, 2.5])
        assert np.allclose(loop1[4], [0, 1.5, 2.5])
        assert np.allclose(loop1[6], [0, 1, 1.5])

    def test_join_intersect_arg1(self):
        tf = Loop.from_file(os.sep.join([TEST_PATH, "test_TF_intersect.json"]))
        lp = Loop.from_file(os.sep.join([TEST_PATH, "test_LP_intersect.json"]))
        eq = Loop.from_file(os.sep.join([TEST_PATH, "test_EQ_intersect.json"]))
        up = Loop.from_file(os.sep.join([TEST_PATH, "test_UP_intersect.json"]))
        if tests.PLOTTING:
            f, ax = plt.subplots()
            for loop in [tf, up, eq, lp]:
                loop.plot(ax, fill=False)

        args = []
        intx, intz = [], []
        for loop in [lp, eq, up]:
            i = get_intersect(tf, loop)
            a = join_intersect(tf, loop, get_arg=True)
            args.extend(a)
            intx.extend(i[0])
            intz.extend(i[1])
        if tests.PLOTTING:
            for loop in [tf, up, eq, lp]:
                loop.plot(ax, fill=False, points=True)
            ax.plot(*tf.d2.T[args].T, "s", marker="o", color="r")
            ax.plot(intx, intz, "s", marker="^", color="k")
        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args]))
        assert np.allclose(np.sort(intz), np.sort(tf.z[args]))

    def test_join_intersect_arg2(self):
        tf = Loop.from_file(os.sep.join([TEST_PATH, "test_TF_intersect2.json"]))
        lp = Loop.from_file(os.sep.join([TEST_PATH, "test_LP_intersect2.json"]))
        eq = Loop.from_file(os.sep.join([TEST_PATH, "test_EQ_intersect2.json"]))
        up = Loop.from_file(os.sep.join([TEST_PATH, "test_UP_intersect2.json"]))
        if tests.PLOTTING:
            f, ax = plt.subplots()
            for loop in [tf, up, eq, lp]:
                loop.plot(ax, fill=False)

        args = []
        intx, intz = [], []
        for loop in [lp, eq, up]:
            i = get_intersect(tf, loop)
            a = join_intersect(tf, loop, get_arg=True)
            args.extend(a)
            intx.extend(i[0])
            intz.extend(i[1])
        if tests.PLOTTING:
            ax.plot(*tf.d2.T[args].T, "s", marker="o", color="r")
            ax.plot(intx, intz, "s", marker="^", color="k")
        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args])), f"{intx} != {tf.x[args]}"
        assert np.allclose(np.sort(intz), np.sort(tf.z[args])), f"{intz} != {tf.z[args]}"


if __name__ == "__main__":
    pytest.main([__file__])
