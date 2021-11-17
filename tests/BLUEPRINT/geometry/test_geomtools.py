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
from matplotlib import pyplot as plt
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
    loop_plane_intersect,
    on_polygon,
    rotate_matrix,
    project_point_axis,
    bounding_box,
    polyarea,
    loop_surface,
    lineq,
    get_normal_vector,
    make_box_xz,
    index_of_point_on_loop,
)


TEST = get_BP_path("BLUEPRINT/geometry/test_data", subfolder="tests")


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


class TestGetNormal:
    def test_simple(self):
        x = [0, 2, 2, 0, 0]
        z = [0, 0, 2, 2, 0]
        y = np.zeros(5)
        n_hat = get_normal_vector(x, y, z)
        assert np.allclose(np.abs(n_hat), np.array([0, 1, 0]))

    def test_edge(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        z = [1, 2, 4]
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
                get_normal_vector(*fail)


def test_make_box():
    x_max = 4.0
    x_min = 2.0
    z_max = 5.0
    z_min = 1.0

    box = make_box_xz(x_min, x_max, z_min, z_max)
    area_box = box.area
    area_check = (z_max - z_min) * (x_max - x_min)
    assert area_box == area_check

    # Swap x
    bad_x_max, bad_x_min = x_min, x_max
    with pytest.raises(GeometryError):
        bad_box = make_box_xz(bad_x_min, bad_x_max, z_min, z_max)

    # Swap z
    bad_z_max, bad_z_min = z_min, z_max
    with pytest.raises(GeometryError):
        bad_box = make_box_xz(x_min, x_max, bad_z_min, bad_z_max)


cases = [
    {"x": 0.5, "z": 0, "before": True, "idx": 0},
    {"x": 0.5, "z": 0, "before": False, "idx": 1},
    {"x": 0.5, "z": 1, "before": True, "idx": 2},
    {"x": 0.5, "z": 1, "before": False, "idx": 3},
    {"x": 0.0, "z": 0, "before": True, "idx": 0},
    {"x": 0.0, "z": 0, "before": False, "idx": 1},
    {"x": 1.5, "z": 0, "before": True, "idx": None},
    {"x": 1.5, "z": 0, "before": False, "idx": None},
]


@pytest.mark.parametrize("inputs", cases)
def test_idx_pt_on_loop(inputs):

    # Create test loops
    box_closed = make_box_xz(0, 1, 0, 1)
    box_open = Loop(x=box_closed.x[:-1], z=box_closed.z[:-1])
    assert box_closed.closed
    assert not box_open.closed

    # Create a point to test
    point_check = [inputs["x"], inputs["z"]]

    # Return index of point before or after
    before = inputs["before"]

    # Open and closed loop cases
    for box in [box_closed, box_open]:
        # Not on loop case: test error
        index_expect = inputs["idx"]
        if not index_expect:
            with pytest.raises(GeometryError):
                index = index_of_point_on_loop(box, point_check, before)
        else:
            index = index_of_point_on_loop(box, point_check, before)
            assert index == index_expect
