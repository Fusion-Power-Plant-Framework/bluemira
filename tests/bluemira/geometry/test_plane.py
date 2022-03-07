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

from bluemira.geometry.plane import BluemiraPlane


class TestPlacement:
    def test_instantiation(self):
        base = np.random.random((1, 3))[0]
        # create a random axis. A constant value has been added to avoid [0,0,0]
        axis = np.random.random((1, 3))[0] + 0.01
        plane = BluemiraPlane(base, axis)
        np.testing.assert_equal(plane.base, base)
        np.testing.assert_equal(plane.axis * np.linalg.norm(axis), axis)

    def test_instantiation_xy(self):
        xy_plane = BluemiraPlane([0, 0, 0], [0, 0, 1])
        xy_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
        np.testing.assert_equal(xy_plane.axis, xy_plane_2.axis)
        np.testing.assert_equal(xy_plane.base, xy_plane_2.base)

    def test_instantiation_yz(self):
        yz_plane = BluemiraPlane([0, 0, 0], [1, 0, 0])
        yz_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [0, 1, 0], [0, 0, 1])
        np.testing.assert_equal(yz_plane.axis, yz_plane_2.axis)
        np.testing.assert_equal(yz_plane.base, yz_plane_2.base)

    def test_instantiation_xz(self):
        xz_plane = BluemiraPlane([0, 0, 0], [0, -1, 0])
        xz_plane_2 = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 0, 1])
        np.testing.assert_equal(xz_plane.axis, xz_plane_2.axis)
        np.testing.assert_equal(xz_plane.base, xz_plane_2.base)

    def test_move_plane(self):
        v = np.array([np.random.random(), np.random.random(), np.random.random()])
        plane = BluemiraPlane([0, 0, 0], [0, 0, 1])
        plane.move(v)
        np.testing.assert_equal(plane.base, v)

    def test_create_face(self):
        base = np.random.random((1, 3))[0]
        # create a random axis. A constant value has been added to avoid [0,0,0]
        axis = np.random.random((1, 3))[0] + 0.01
        plane = BluemiraPlane(base, axis)
        lx = 20
        ly = 10
        bmface = plane.to_face(lx, ly)
        np.testing.assert_almost_equal(bmface.center_of_mass, plane.base)
        np.testing.assert_almost_equal(
            np.array([bmface.length, bmface.area]), np.array([2 * (lx + ly), lx * ly])
        )
