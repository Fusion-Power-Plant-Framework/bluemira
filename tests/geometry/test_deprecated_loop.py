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


import os
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.error import GeometryError
from bluemira.utilities.plot_tools import Plot3D

TEST = get_bluemira_path("geometry/test_data", subfolder="tests")


class TestLoop:
    @classmethod
    def setup_class(cls):
        # Ccw square A = 1
        cls.a = Loop(x=[0, 1, 1, 0, 0], y=None, z=np.array([0, 0, 1, 1, 0]))
        # Ccw triangle 3-4 corner
        cls.tt = Loop(x=[0, 2, 0], z=[0, 0, 1])
        # Ccw bucket
        cls.b = Loop(x=[0, 1, 2, 3], z=[0, -1, -1, -0.5])
        # Open Clockwise hexagon
        cls.h = Loop(x=[4, 2, 1.5, 3, 4.5], y=[1, 1, 3, 4.5, 3])
        # Clockwise triangle A = 8
        cls.t = Loop(x=[4, 3, 2, 0, 2, 4, 4], z=[0, 0, 0, 0, 2, 4, 0], y=np.pi)
        #  yz  ccw square A = 4
        cls.s = Loop(x=10, y=[0, 2, 2, 0, 0], z=np.array([0, 0, 2, 2, 0]))
        cls.aloop = Loop.from_array(
            np.array([[0, 1, -1, 1, -1], [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
        )

    def test_arrayfail(self):
        with pytest.raises(GeometryError):
            aloop = Loop.from_array(
                np.array(
                    [
                        [0, 1, -1, 1, -1],
                        [1, 1, 1, 1, 1],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                    ]
                )
            )

    def test_ccw(self):
        assert self.a.ccw
        assert self.b.ccw
        assert self.h.ccw
        assert self.t.ccw

    def test_length(self):
        square = self.a.copy()
        assert square.length == 4.0
        square.rotate(2 * np.random.rand(1), p1=np.random.rand(3), p2=np.random.rand(3))
        assert np.isclose(square.length, 4.0)

    def test_area(self):
        assert self.t.area == 8
        assert self.s.area == 4
        assert self.h.area == 7.25

    def test_open(self):
        assert self.h.closed is False
        self.h.close()
        assert self.h.closed is True
        assert self.t.closed is True
        self.t.open()
        assert self.t.closed is False

    def test_closed(self):
        # Open Clockwise hexagon
        h = Loop(x=[4, 2, 1.5, 3, 4.5], y=[1, 1, 3, 4.5, 3])
        assert not h._check_closed()
        assert len(h) == 5
        h.close()  # Lightning strikes twice
        assert h._check_closed()
        assert len(h) == 6
        h.close()
        assert h._check_closed()
        assert len(h) == 6

    def test_reorder(self):
        # Ccw closed square A = 1 with 0 at minmin
        a = Loop(x=[0, 1, 1, 0, 0], y=None, z=np.array([0, 0, 1, 1, 0]))
        assert len(a) == 5
        a.reorder(2)
        assert len(a) == 5

        b = Loop(x=[0, 1, 1, 0], y=[0, 0, 1, 1])
        with pytest.raises(GeometryError):
            b.reorder(2)

    def test_planes(self):
        assert self.h._check_plane([3, 2, 0])
        assert not self.h._check_plane([3, 2, 10])
        assert self.a._check_plane([3, 0, 10])
        assert self.b._check_plane([3, 0, 10])
        assert not self.b._check_plane([3, 2, 45.5])
        assert self.tt._check_plane([3, 0, 10])
        assert not self.tt._check_plane([3, 2, 45.5])

    def test_rotate(self):
        self.h.close()
        self.h.rotate(
            theta=uniform(0, 360),
            p1=[uniform(0, 1), uniform(0, 1), uniform(0, 1)],
            p2=[uniform(0, 1), uniform(0, 1), uniform(0, 1)],
        )
        # Delta 1e-4 still fails occasionally
        assert abs(self.h.area - 7.25) < 1e-7

    def test_ndim(self):
        assert self.t.ndim == 2
        self.t.rotate(theta=uniform(0, 360), p1=[0.2, 0.3, 0.9], p2=[0.2, 0.9, 0.05])
        assert self.t.ndim == 3

    def test_n_hat(self):
        loop = Loop(
            y=[
                -0.05,
                0.05,
                0.05,
                0.025,
                0.025,
                0.05,
                0.05,
                -0.05,
                -0.05,
                -0.025,
                -0.025,
                -0.05,
                -0.05,
            ],
            z=[
                -0.05,
                -0.05,
                -0.025,
                -0.025,
                0.025,
                0.025,
                0.05,
                0.05,
                0.025,
                0.025,
                -0.025,
                -0.025,
                -0.05,
            ],
        )
        assert np.allclose(np.abs(loop.n_hat), np.array([1, 0, 0]))

        dcm = np.array(
            [
                [-0.70710678, 0.70710678, 0.0],
                [-0.70710678, -0.70710678, 0.0],
                [0.0, -0.0, 1.0],
            ]
        )
        loop2 = loop.rotate_dcm(dcm.T, update=False)
        assert np.allclose(
            np.abs(loop2.n_hat), np.array([0.5 * np.sqrt(2), 0.5 * np.sqrt(2), 0])
        )

    def test_offset(self):
        b = self.a.offset(1)
        assert np.array_equal(b.x, np.array([-1, 2, 2, -1, -1]))
        assert np.array_equal(b.z, np.array([-1, -1, 2, 2, -1]))
        loop1 = Loop.from_file(os.sep.join([TEST, "edge_case_offset.json"]))
        # Edge case malparido
        offset_loop = loop1.offset(0.03)  # Esse fantasma foi finalmente capturado...
        assert offset_loop.closed
        assert offset_loop.ccw

    def test_checkalreadyin(self):
        a = Loop(x=[0, 4, 4, 0, 0], y=[0, 0, 2, 2, 0])
        assert a._check_already_in([4, 0])
        assert not a._check_already_in([4, 4])
        a = Loop(x=[0, 4, 4, 0, 0], z=[0, 0, 2, 2, 0])
        assert a._check_already_in([4, 0])
        assert not a._check_already_in([4, 4])
        a = Loop(y=[0, 4, 4, 0, 0], z=[0, 0, 2, 2, 0])
        assert a._check_already_in([4, 0])
        assert not a._check_already_in([4, 4])


@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class Test3Dplotting:
    def test_looks_good(self):
        loop = Loop(x=[0.4, 4, 6, 7, 8, 4, 0.4], y=[1, 1, 2, 2, 3, 3, 1], z=0)
        ax = Plot3D()
        loop.rotate(30, p1=[0, 0, 0], p2=[0, 1, 0])  # Killer edge case
        loop.plot(ax)
        loop.rotate(40, p1=[0, 0, 0], p2=[0, 0, 1])
        loop.plot(ax)
        loop.rotate(30, p1=[0, 0, 0], p2=[0, 1, 2])  # Killer edge case
        loop.plot(ax)
        loop.translate([10, 0.0, -5])
        loop.plot(ax)
        loop.rotate(-70, p1=[0, 1, 0], p2=[-9, -8, 4])
        loop.plot(ax)
        plt.show()

    def test_edges_again(self):
        loop = Loop(x=[0, 2, 2, 0, 0], y=[0, 0, 2, 2, 0])
        loop.translate([10, 10, 10])
        ax = Plot3D()
        loop.plot(ax)
        loop.rotate(30, p1=[0, 0, 0], p2=[0, 0, 1])
        loop.plot(ax)
        loop.rotate(-30, p1=[0, 0, 0], p2=[0, 0, 1])
        loop.rotate(30, p1=[0, 0, 0], p2=[0, 1, 0])  # this is the killer!
        loop.plot(ax)
        u_loop = Loop([0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [2, 0, 0, 2, 2])
        u_loop.translate([3, 0, 0])
        u_loop.plot(ax)
        plt.show()
