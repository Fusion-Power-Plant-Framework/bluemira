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

from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import BluemiraPlane


class TestPlane:
    def test_instantiation_xy(self):
        xy_plane = BluemiraPlane([0, 0, 0], [0, 0, 1], 0.0)
        xy_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [1, -1, 0], [1, 1, 0])
        assert xy_plane.axis == xy_plane_2.axis
        assert xy_plane.base == xy_plane_2.base
        assert xy_plane.angle == xy_plane_2.angle

    def test_instantiation_xz(self):
        xz_plane = BluemiraPlane([0, 0, 0], [0, -1, 0], 0.0)
        xz_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, -1], [1, 0, 1])
        assert xz_plane.axis == xz_plane_2.axis
        assert xz_plane.base == xz_plane_2.base
        assert xz_plane.angle == xz_plane_2.angle

    def test_instantiation_yz(self):
        yz_plane = BluemiraPlane([0, 0, 0], [1, 0, 0], 0.0)
        yz_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [0, 1, -1], [0, 1, 1])
        assert yz_plane.axis == yz_plane_2.axis
        assert yz_plane.base == yz_plane_2.base
        assert yz_plane.angle == yz_plane_2.angle

    def test_instantiation_xyz(self):
        xyz_plane = BluemiraPlane([0, 0, 0], [1, 1, 1], 0.0)
        xyz_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [1, -1, 0], [1, 0, -1])
        assert np.allclose(xyz_plane.axis, xyz_plane_2.axis)
        assert xyz_plane.base == xyz_plane_2.base
        assert xyz_plane.angle == xyz_plane_2.angle

    def test_colinear(self):
        bad_sets = [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # co-linear
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # all the same zero
            [[0, 0, 1], [0, 0, 0], [0, 0, -1]],  # co-linear bad order
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # two the same #1
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # two the same #2
            [
                [
                    1,
                    1,
                    1,
                ],
                [1, 1, 1],
                [1, 1, 1],
            ],  # all the same non-zero
        ]
        for bad in bad_sets:
            with pytest.raises(GeometryError):
                BluemiraPlane.from_3_points(*bad)

    def test_instantiation_matrix(self):
        xy_plane = BluemiraPlane([0, 0, 0], [0, 0, 1], 0.0)
        matrix = xy_plane.to_matrix()
        xy2_plane = BluemiraPlane.from_matrix(matrix)
        self._check_planes_equal(xy_plane, xy2_plane)

        for _ in range(10):
            plane = BluemiraPlane(np.random.rand(3), np.random.rand(3), np.random.rand())
            matrix = plane.to_matrix()
            plane2 = BluemiraPlane.from_matrix(matrix)
            self._check_planes_equal(plane, plane2)
            assert np.allclose(plane.axis, plane2.axis)

    @staticmethod
    def _check_planes_equal(plane, plane2):
        assert np.allclose(plane.to_matrix(), plane2.to_matrix())
        assert np.isclose(plane.angle, plane2.angle)
        assert np.allclose(plane.base, plane2.base)
        assert np.allclose(plane.axis, plane2.axis)
