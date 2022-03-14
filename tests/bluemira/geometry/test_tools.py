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

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.tools import (
    _signed_distance_2D,
    convex_hull_wires_2d,
    extrude_shape,
    find_point_along_wire_at_length,
    make_circle,
    make_polygon,
    offset_wire,
    point_inside_shape,
    revolve_shape,
    signed_distance,
    signed_distance_2D_polygon,
    slice_shape,
    wire_value_at,
)
from bluemira.geometry.wire import BluemiraWire

generic_wire = make_polygon(
    [
        [0.0, -1.0, 0.0],
        [1.0, -2.0, 0.0],
        [2.0, -3.0, 0.0],
        [3.0, -4.0, 0.0],
        [4.0, -5.0, 0.0],
        [5.0, -6.0, 0.0],
        [6.0, -7.0, 0.0],
        [7.0, -8.0, 0.0],
        [8.0, -4.0, 0.0],
        [9.0, -2.0, 0.0],
        [10.0, 3.0, 0.0],
        [8.0, 2.0, 0.0],
        [6.0, 4.0, 0.0],
        [4.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


class TestSignedDistanceFunctions:
    @classmethod
    def setup_class(cls):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        y = np.array([0.0, -1.0, -1.0, -3.0, -4.0, -2.0, 1.0, 2.5, 3.0, 1.0, 0.0])
        z = np.zeros(len(x))

        cls.subject_2D_array = np.array([x, y]).T
        cls.subject_wire = make_polygon(np.array([x, y, z]).T)

    def test_sdf_2d(self):

        p1 = np.array([0, 0])  # Corner point
        p2 = np.array([0.5, -0.5])  # Mid edge point
        p3 = np.array([3, 0])  # Inside point
        p4 = np.array([-0.1, 0])  # Just outside point
        d1 = _signed_distance_2D(p1, self.subject_2D_array)
        assert d1 == 0
        d2 = _signed_distance_2D(p2, self.subject_2D_array)
        assert d2 == 0
        d3 = _signed_distance_2D(p3, self.subject_2D_array)
        assert d3 > 0
        d4 = _signed_distance_2D(p4, self.subject_2D_array)
        assert d4 == -0.1
        d = np.array([d1, d2, d3, d4])

        d_array = signed_distance_2D_polygon(
            np.array([p1, p2, p3, p4]), self.subject_2D_array
        )

        assert np.allclose(d, d_array)

    def test_sdf(self):
        # Overlapping
        target = make_polygon(
            [[0, 0, 0], [4, 0, 0], [4, 2.5, 0], [0, 2.5, 0], [0, 0, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd > 0
        # Touching
        target = make_polygon(
            [[0, 0, 0], [-4, 0, 0], [-4, -2.5, 0], [0, -2.5, 0], [0, 0, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd == 0
        # Not overlapping
        target = make_polygon(
            [[-1, 3.5, 0], [-1, -5, 0], [6, -5, 0], [6, 3.5, 0], [-1, 3.5, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd < 0


class TestWirePlaneIntersect:
    def test_simple(self):
        loop = make_polygon(
            [[0, 0, -1], [1, 0, -1], [2, 0, -1], [2, 0, 1], [0, 0, 1], [0, 0, -1]]
        )

        xy_plane = BluemiraPlacement(axis=[0, 0, 1])
        intersect = slice_shape(loop, xy_plane)
        e = np.array([[0, 0, 0], [2, 0, 0]])
        e.sort(axis=0)
        intersect.sort(axis=0)
        assert np.allclose(intersect, e)

    def test_complex(self):
        wire = make_polygon(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -2.0],
                [2.0, 0.0, -3.0],
                [3.0, 0.0, -4.0],
                [4.0, 0.0, -5.0],
                [5.0, 0.0, -6.0],
                [6.0, 0.0, -7.0],
                [7.0, 0.0, -8.0],
                [8.0, 0.0, -4.0],
                [9.0, 0.0, -2.0],
                [10.0, 0.0, 3.0],
                [8.0, 0.0, 2.0],
                [6.0, 0.0, 4.0],
                [4.0, 0.0, 2.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        xy_plane = BluemiraPlacement(axis=[0, 0, 1])
        intersect = slice_shape(wire, xy_plane)
        assert intersect.shape[0] == 2

        xy_plane = BluemiraPlacement(base=[0, 0, 2.7], axis=[0, 0, 1])
        intersect = slice_shape(wire, xy_plane)
        print(intersect)
        assert intersect.shape[0] == 4

        plane = BluemiraPlacement.from_3_points(
            [0, 0, 4], [1, 0, 4], [0, 1, 4]
        )  # x-y offset
        intersect = slice_shape(wire, plane)
        assert intersect.shape[0] == 1

        plane = BluemiraPlacement.from_3_points(
            [0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005]
        )  # x-y offset
        intersect = slice_shape(wire, plane)
        assert intersect is None

    def test_other_dims(self):
        shift = 0
        for plane in [
            BluemiraPlacement.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
            BluemiraPlacement(axis=[0, 1, 0]),
        ]:
            intersect = slice_shape(generic_wire, plane)
            assert intersect.shape[0] == 2

        shift = 10
        for plane in [
            BluemiraPlacement.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
        ]:
            intersect = slice_shape(generic_wire, plane)
            assert intersect is None

    def test_xyzplane(self):
        wire = generic_wire.copy()
        wire.translate((-2, 0, 0))
        plane = BluemiraPlacement.from_3_points([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = slice_shape(wire, plane)
        assert intersect.shape[0] == 2

    def test_flat_intersect(self):
        # test that a shared segment with plane only gives two intersects
        wire = make_polygon(
            [
                [0.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

        plane = BluemiraPlacement.from_3_points([0, 0, 1], [0, 1, 1], [1, 0, 1])
        inter = slice_shape(wire, plane)
        true = np.array([[0, 0, 1], [2, 0, 1]])
        true.sort(axis=0)
        inter.sort(axis=0)
        assert np.allclose(inter, true)

    def test_weird_wire(self):
        # test a wire that moves in 3 dimensions
        wire = make_polygon(
            [
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 1.0],
                [2.0, -3.0, 2.0],
                [3.0, -4.0, 1.0],
                [4.0, -5.0, 0.0],
                [5.0, -6.0, -1.0],
                [6.0, -7.0, -2.0],
                [7.0, -8.0, -1.0],
                [8.0, -4.0, 0.0],
                [9.0, -2.0, 1.0],
                [10.0, 3.0, 2.0],
                [8.0, 2.0, 1.0],
                [6.0, 4.0, 0.0],
                [4.0, 2.0, -1.0],
                [2.0, 0.0, -2.0],
                [0.0, -1.0, 0.0],
            ]
        )

        plane = BluemiraPlacement.from_3_points([1, -2, -1], [6, 4, 0], [9, -2, 1])

        intersect = slice_shape(wire, plane)
        assert intersect.shape[0] == 4


class TestSolidFacePlaneIntersect:

    big = 10
    small = 5
    centre = 15
    twopi = 2 * np.pi
    offset = 1

    cyl_rect = 2 * big + 2 * offset
    twopir = twopi * small

    xz_plane = BluemiraPlacement(axis=[0, 1, 0])
    xy_plane = BluemiraPlacement(axis=[0, 0, 1])
    yz_plane = BluemiraPlacement(axis=[1, 0, 0])

    @pytest.mark.parametrize(
        "plane, length, hollow",
        [
            # hollow
            (xz_plane, offset, True),
            (yz_plane, offset, True),
            (xy_plane, twopir, True),
            (BluemiraPlacement(base=[0, 0, 0.5], axis=[0, 0, 1]), twopir, True),
            (BluemiraPlacement(base=[0, 0, offset], axis=[0, 0, 1]), twopir, True),
            # solid
            (xz_plane, cyl_rect, False),
            (yz_plane, cyl_rect, False),
            # tangent intersecting plane doesnt work at solid base??
            pytest.param(xy_plane, twopir, False, marks=[pytest.mark.xfail]),
            (BluemiraPlacement(base=[0, 0, 0.5], axis=[0, 0, 1]), twopir, False),
            (BluemiraPlacement(base=[0, 0, offset], axis=[0, 0, 1]), twopir, False),
        ],
    )
    def test_cylinder(self, plane, length, hollow):
        circ = make_circle(self.small)
        if not hollow:
            circ = BluemiraFace(circ)
        cylinder = extrude_shape(circ, (0, 0, self.offset))
        _slice = slice_shape(cylinder, plane)
        assert _slice is not None
        assert all([np.isclose(sl.length, length) for sl in _slice]), [
            f"{sl.length}, {length}" for sl in _slice
        ]

    def test_solid_nested_donut(self):

        circ = make_circle(self.small, [0, 0, self.centre], axis=[0, 1, 0])
        circ2 = make_circle(self.big, [0, 0, self.centre], axis=[0, 1, 0])

        face = BluemiraFace([circ2, circ])

        # cant join a face to itself atm 20/12/21
        donut = revolve_shape(face, direction=[1, 0, 0], degree=359)

        _slice = slice_shape(donut, self.xz_plane)

        no_big = 0
        no_small = 0
        for sl in _slice:
            try:
                assert np.isclose(sl.length / self.twopi, self.big)
                no_big += 1
            except AssertionError:
                assert np.isclose(sl.length / self.twopi, self.small)
                no_small += 1
        assert no_big == 2
        assert no_small == 2

    def test_primitive_cut(self):

        path = PrincetonD({"x2": {"value": self.big}}).create_shape()
        p2 = offset_wire(path, self.offset)
        face = BluemiraFace([p2, path])
        extruded = extrude_shape(face, (0, 1, 0))

        _slice_xy = slice_shape(extruded, self.xy_plane)
        _slice_xz = slice_shape(
            extruded, BluemiraPlacement(base=[0, 1, 0], axis=[0, 1, 0])
        )

        assert len(_slice_xy) == 2
        assert len(_slice_xz) == 2

    def test_polygon_cut(self):

        face = BluemiraFace(generic_wire)
        _slice_face = slice_shape(face, BluemiraPlacement())
        assert generic_wire.length == _slice_face[0].length

        solid = extrude_shape(face, (1, 2, 3))
        _slice_solid = slice_shape(solid, BluemiraPlacement(axis=[3, 2, 1]))
        assert len(_slice_solid) == 1


class TestPointInside:
    def test_simple(self):
        polygon = BluemiraFace(
            make_polygon({"x": [-2, 2, 2, -2, -2, -2], "z": [-2, -2, 2, 2, 1.5, -2]})
        )
        in_points = [
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
        ]
        for point in in_points:
            assert point_inside_shape(point, polygon)

        out_points = [
            [-3, 0, -3],
            [-3, 0, 0],
            [-3, 0, 3],
            [0, 0, -3],
            [3, 0, 3],
            [3, 0, -3],
            [2.005, 0, 0],
            [2.001, 0, -1.9999],
            # TODO: This is not very good FreeCAD..
            # [2.00000009, 0, 0],
            # [-2.0000000001, 0, -1.999999999999],
        ]

        for point in out_points:
            assert not point_inside_shape(point, polygon)


class TestPointAlongWire:
    def test_point_along_wire_at_length_2d(self):
        # Line in 2d: z = 3x - 4
        coords = np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [-1, 2, 5, 8, 11]])
        wire = make_polygon(coords)
        desired_len = np.sqrt(2.5)

        point = wire_value_at(wire, distance=desired_len)

        np.testing.assert_almost_equal(point, [1.5, 0, 0.5], decimal=2)

    def test_point_along_wire_at_length_3d(self):
        circle = make_circle(radius=1, axis=[1, 1, 1])
        semi_circle = make_circle(radius=1, axis=[1, 1, 1], end_angle=180)

        point = wire_value_at(circle, distance=np.pi)

        np.testing.assert_almost_equal(point, semi_circle.end_point().T[0], decimal=2)

    @pytest.mark.parametrize("len_offset", [0.1, 1, 10, 200])
    def test_ValueError_given_length_gt_wire_length(self, len_offset):
        coords = np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [-1, 2, 5, 8, 11]])
        wire = make_polygon(coords)
        wire_length = wire.length

        with pytest.raises(ValueError):
            find_point_along_wire_at_length(wire, wire_length + len_offset)


class TestConvexHullWires2d:
    def test_ValueError_given_wires_empty(self):
        with pytest.raises(ValueError):
            convex_hull_wires_2d([], 10)

    def test_hull_around_two_circles_xz_plane(self):
        circle_1 = make_circle(radius=1, center=[-0.5, 0, 0.5], axis=(0, 1, 0))
        circle_2 = make_circle(radius=1, center=[0.5, 0, -0.5], axis=(0, 1, 0))

        hull = convex_hull_wires_2d([circle_1, circle_2], ndiscr=200)

        assert hull.is_closed
        assert np.allclose(hull.center_of_mass, [0, 0, 0])
        bounding_box = hull.bounding_box
        assert bounding_box.z_min == -1.5
        assert bounding_box.z_max == 1.5
        assert bounding_box.x_min == -1.5
        assert bounding_box.x_max == 1.5
        assert bounding_box.y_min == bounding_box.y_max == 0

    def test_hull_around_two_circles_xy_plane(self):
        circle_1 = make_circle(radius=1, center=[-0.5, 1, 0.5], axis=(1, 1, 1))
        circle_2 = make_circle(radius=1, center=[0.5, -2, -0.5], axis=(1, 1, 1))

        hull = convex_hull_wires_2d([circle_1, circle_2], ndiscr=1000, plane="xy")

        assert hull.is_closed
        assert np.allclose(hull.center_of_mass, [0, -0.5, 0])
        bounding_box = hull.bounding_box
        assert bounding_box.z_min == bounding_box.z_max == 0

    @pytest.mark.parametrize("bad_plane", ["ab", "", None, ["x", "y"]])
    def test_ValueError_if_invalid_plane(self, bad_plane):
        circle = make_circle(radius=1)

        with pytest.raises(ValueError):
            convex_hull_wires_2d([circle], 10, plane=bad_plane)
