# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle  # noqa (S403)
import pytest

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.error import GeometryError
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import (
    inloop,
    circle_line_intersect,
    loop_volume,
    distance_between_points,
    circle_seg,
    circle_arc,
    get_intersect,
    join_intersect,
    loop_plane_intersect,
    check_linesegment,
    on_polygon,
    in_polygon,
    polygon_in_polygon,
    rotate_matrix,
    project_point_axis,
    bounding_box,
    polyarea,
    loop_surface,
    lineq,
)
import tests


TEST = get_BP_path("geometry/test_data", subfolder="tests")


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

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_join_intersect(self):
        loop1 = Loop(x=[0, 0.5, 1, 2, 3, 5, 4.5, 4, 0], z=[1, 1, 1, 1, 2, 4, 4.5, 5, 5])
        loop2 = Loop(x=[1.5, 1.5, 2.5, 2.5, 2.5], z=[4, -4, -4, -4, 5])
        join_intersect(loop1, loop2)
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
        f, ax = plt.subplots()
        loop1.plot(ax, fill=False, edgecolor="k", points=True)
        loop2.plot(ax, fill=False, edgecolor="r", points=True)
        plt.show()
        assert np.allclose(loop1[1], [0, 5, 2.5])
        assert np.allclose(loop1[4], [0, 1.5, 2.5])
        assert np.allclose(loop1[6], [0, 1, 1.5])

    def test_join_intersect_arg1(self):
        tf = Loop.from_file(os.sep.join([TEST, "test_TF_intersect.json"]))
        lp = Loop.from_file(os.sep.join([TEST, "test_LP_intersect.json"]))
        eq = Loop.from_file(os.sep.join([TEST, "test_EQ_intersect.json"]))
        up = Loop.from_file(os.sep.join([TEST, "test_UP_intersect.json"]))
        if not tests.PLOTTING:
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
        if not tests.PLOTTING:
            for loop in [tf, up, eq, lp]:
                loop.plot(ax, fill=False, points=True)
            ax.plot(*tf.d2.T[args].T, "s", marker="o", color="r")
            ax.plot(intx, intz, "s", marker="^", color="k")
        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args]))
        assert np.allclose(np.sort(intz), np.sort(tf.z[args]))

    def test_join_intersect_arg2(self):
        tf = Loop.from_file(os.sep.join([TEST, "test_TF_intersect2.json"]))
        lp = Loop.from_file(os.sep.join([TEST, "test_LP_intersect2.json"]))
        eq = Loop.from_file(os.sep.join([TEST, "test_EQ_intersect2.json"]))
        up = Loop.from_file(os.sep.join([TEST, "test_UP_intersect2.json"]))
        if not tests.PLOTTING:
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
        if not tests.PLOTTING:
            ax.plot(*tf.d2.T[args].T, "s", marker="o", color="r")
            ax.plot(intx, intz, "s", marker="^", color="k")
        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args])), f"{intx} != {tf.x[args]}"
        assert np.allclose(np.sort(intz), np.sort(tf.z[args])), f"{intz} != {tf.z[args]}"

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_join_intersect_fail(self):
        tf = Loop.from_file(os.sep.join([TEST, "test_TF_intersect3.json"]))
        lp = Loop.from_file(os.sep.join([TEST, "test_UP_intersect3.json"]))
        join_intersect(tf, lp, get_arg=True)
        f, ax = plt.subplots()
        tf.plot(ax, points=True)
        plt.show()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plasma_div(self):
        if "R" in globals():
            f, ax = plt.subplots()
            reactor = globals()["R"]
            div = reactor.DIV.geom["2D profile"]
            separatrix = reactor.PL.get_sep()
            div.plot(ax, fill=True)
            separatrix.plot(ax, fill=False)

            x_inter, z_inter = get_intersect(separatrix, div)
            for x, z in zip(x_inter, z_inter):
                ax.plot(x, z, "s", marker="o", color="r")
            args = join_intersect(separatrix, div, get_arg=True)
            for arg in args:
                ax.plot(*separatrix.d2.T[arg], "s", marker="^", color="b")


# =============================================================================
#     def test_clip_loop(self):
#         S = Loop.from_file(os.sep.join([TEST, ]))
# =============================================================================


class TestInLoop:
    def test_square_open(self):

        x_l, z_l = [0, 2, 2, 0], [0, 0, 2, 2]
        x, z = 1, 1
        in_loop = inloop(x_l, z_l, x, z, side="in")
        assert in_loop
        in_loop = inloop(x_l, z_l, x, z, side="out")
        assert not in_loop

    def test_square_closed(self):
        x_l, z_l = [0, 2, 2, 0, 0], [0, 0, 2, 2, 0]
        x, z = 1, 1
        in_loop = inloop(x_l, z_l, x, z, side="in")
        assert in_loop
        in_loop = inloop(x_l, z_l, x, z, side="out")
        assert not in_loop


def assert_permutations(x, y):
    assert list(sorted(x)) == list(sorted(y))


class TestCircleLine:
    def test_vertline(self):
        x, z = circle_line_intersect(0, 0, 5, 0, 1, 0, -1)
        assert_permutations(x, [0, 0])
        assert_permutations(z, [-5, 5])

    def test_horzline(self):
        x, z = circle_line_intersect(0, 0, 5, 1, 0, -1, 0)
        assert_permutations(x, [-5, 5])
        assert_permutations(z, [0, 0])

    def test_horztangent(self):
        x, z = circle_line_intersect(0, 0, 5, 5, 5, -5, 5)
        assert_permutations(x, [0])
        assert_permutations(z, [5])

    def test_verttangent(self):
        x, z = circle_line_intersect(0, 0, 5, 5, -5, 5, 5)
        assert_permutations(x, [5])
        assert_permutations(z, [0])

    def test_tangent(self):
        r = np.sqrt(2)
        x, z = circle_line_intersect(0, 0, r, 0, 2, 2, 0)
        assert round(abs(x[0] - 1), 7) == 0
        assert round(abs(z[0] - 1), 7) == 0
        assert len(x) == 1
        f, ax = plt.subplots()
        c = circle_seg(r, (0, 0))
        ax.plot(*c)
        ax.plot(x, z, marker="o", color="r")
        ax.plot([0, 2], [2, 0], marker="o", color="b")
        ax.set_aspect("equal")

    def test_nothing(self):
        none = circle_line_intersect(0, 0, 1, 0, 5, 5, 5)
        assert none is None
        none = circle_line_intersect(0, 0, 1, 5, 5, 5, 5)
        assert none is None


class TestDistance:
    def test_2d(self):
        d = distance_between_points([0, 0], [1, 1])
        assert d == np.sqrt(2)

    def test_3d(self):
        d = distance_between_points([0, 0, 0], [1, 1, 1])
        assert d == np.sqrt(3)

    def test_fail(self):
        with pytest.raises(GeometryError):
            distance_between_points([0, 0], [1, 1, 1])
        with pytest.raises(GeometryError):
            distance_between_points([0, 0, 0], [1, 1])
        with pytest.raises(GeometryError):
            distance_between_points([0, 0, 0, 0], [1, 1, 1, 1])
        with pytest.raises(GeometryError):
            distance_between_points([0], [1, 1])
        with pytest.raises(GeometryError):
            distance_between_points([0, 0], [1])
        with pytest.raises(GeometryError):
            distance_between_points([0], [1])


class TestCheckLineSegment:
    def test_true(self):
        a = [0, 0]
        b = [1, 0]
        c = [0.5, 0]
        assert check_linesegment(a, b, c) is True
        a = [0, 0]
        b = [0.001, 0]
        c = [0.0005, 0]
        assert check_linesegment(a, b, c) is True

        a = [0, 0]
        b = [1, 0]
        c = [1, 0]
        assert check_linesegment(a, b, c) is True
        a = [0, 0]
        b = [0.001, 0]
        c = [0, 0]
        assert check_linesegment(a, b, c) is True

    def test_false(self):
        a = [0, 0]
        b = [1, 0]
        c = [5, 0]
        assert check_linesegment(a, b, c) is False

        a = [0, 0]
        b = [0.001, 0]
        c = [0.005, 0]
        assert check_linesegment(a, b, c) is False


class TestArea:
    def test_area(self):
        """
        Checked with:
        https://www.analyzemath.com/Geometry_calculators/irregular_polygon_area.html
        """
        x = [0, 1, 2, 3, 4, 5, 6, 4, 3, 2]
        y = [0, -5, -3, -5, -1, 0, 2, 6, 4, 1]
        assert polyarea(x, y) == 29.5
        loop = Loop(x=x, y=y)
        loop.rotate(43, p1=[3, 2, 1], p2=[42, 2, 1])
        assert np.isclose(polyarea(*loop.xyz), 29.5)

    def test_error(self):
        x = [0, 1, 2, 3, 4, 5, 6, 4, 3, 2]
        y = [0, -5, -3, -5, -1, 0, 2, 6, 4, 1]
        with pytest.raises(GeometryError):
            polyarea(x, y[:-1])


class TestVolume:
    def test_volume(self):
        x = np.linspace(1, 2, 100000)
        y = np.sqrt(x - 1)
        yy = (x - 1) ** 2
        x_f = np.append(x[::-1], x)
        z_f = np.append(y[::-1], yy)
        # Example 4 https://www.rit.edu/studentaffairs/asc/sites/rit.edu.studen
        # taffairs.asc/files/docs/services/resources/handouts/C8_VolumesbyInteg
        # ration_BP_9_22_14.pdf
        assert round(abs(loop_volume(x_f[::-1], z_f[::-1]) - np.pi * 29 / 30), 7) == 0


class TestCircleArcSeg:
    def test_circle_seg(self):
        x, y = circle_seg(5, [6, 6], angle=90, start=0)
        assert x[0] == 11
        assert y[0] == 6
        assert x[-1] == 6
        assert y[-1] == 11
        x, y = circle_seg(5, [6, 6], angle=90, start=180)
        assert x[0] == 1
        assert round(abs(y[0] - 6), 7) == 0
        assert round(abs(x[-1] - 6), 7) == 0
        assert round(abs(y[-1] - 1), 7) == 0
        x, y = circle_seg(3, [2, 2], angle=180)
        assert x[0] == 2
        assert x[-1] == 2
        assert y[0] == -1
        assert y[-1] == 5

    def test_circle_arc(self):
        x, y = circle_arc([5.5, -0.5], [0.5, 0], angle=90)
        assert x[0] == 5.5
        assert y[0] == -0.5
        d = distance_between_points([5.5, -0.5], [0.5, 0])
        assert round(abs(x[-1] - 1), 7) == 0
        assert round(abs(y[-1] - 5), 7) == 0
        x, y = circle_arc([-5.5, -0.5], [0.5, 0], angle=90)
        assert round(abs(x[0] - -5.5), 7) == 0
        assert round(abs(y[0] - -0.5), 7) == 0
        d = distance_between_points([5.5, -0.5], [0.5, 0])
        assert round(abs(x[-1] - 1), 7) == 0
        assert round(abs(y[-1] - -6), 7) == 0


class TestPointAxisProjection:
    def test_point_axis(self):
        result = project_point_axis([4, 4, 0], [0, 0, 1])
        assert np.allclose(result, [0, 0, 0])
        result = project_point_axis([4, 4, 1], [0, 0, 1])
        assert np.allclose(result, [0, 0, 1])

        result = project_point_axis([4, 4, 0], [1, 0, 0])
        assert np.allclose(result, [4, 0, 0])

        result = project_point_axis([4, 4, 0], [0, 1, 0])
        assert np.allclose(result, [0, 4, 0])


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
        filename = get_BP_path("geometry/test_data", subfolder="tests")
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


class TestRotationMatrix:
    def test_axes(self):
        axes = ["x", "y", "z"]
        axes2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for a1, a2 in zip(axes, axes2):
            r_1 = rotate_matrix(np.pi / 6, a1)
            r_2 = rotate_matrix(np.pi / 6, a2)
            assert np.allclose(r_1, r_2), a1

        axes = ["fail", "somthing", "1"]
        for axis in axes:
            with pytest.raises(GeometryError):
                rotate_matrix(30, axis)

    def test_ccw(self):
        p1 = [9, 0, 0]

        r_matrix = rotate_matrix(np.pi / 2, axis="z")
        p2 = r_matrix @ p1

        assert np.isclose(p2[1], 9), p2


class TestLineEq:
    def test_line(self):
        a = [0, 0]
        b = [0, 1]
        m, c = lineq(a, b)
        assert m == 0
        assert c == 1

        a = [0, 0]
        b = [1, 1]
        m, c = lineq(a, b)
        assert m == 1
        assert c == 0
        _, _ = lineq(a, b, show=True)


class TestSurfaceArea:
    def test_torus(self):
        major = 9
        minor = 3
        area = 4 * np.pi ** 2 * major * minor

        x, z = circle_seg(minor, h=(major, 0), npoints=500)

        s_area = loop_surface(x, z)
        assert np.isclose(s_area, area)


if __name__ == "__main__":
    pytest.main([__file__])
