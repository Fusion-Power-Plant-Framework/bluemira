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
from itertools import cycle
from random import uniform

import numpy as np
import pytest
from matplotlib import pyplot as plt

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_tools import get_intersect
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.utilities.plottools import Plot3D

TEST = get_bluemira_path("BLUEPRINT/geometry/test_data", subfolder="tests")


class TestLoop:
    plot = tests.PLOTTING

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

        # Star shaped loop
        cls.star_shape = Loop(
            x=np.array([6.5, 9.0, 14, 10.5, 12.0, 6.5, 2.0, 3.5, 0.0, 5.0]),
            z=-np.array([0.0, 5.0, 5.5, 9.0, 14.0, 11.5, 14.0, 9.0, 5.5, 5]),
        )
        cls.star_shape.close()

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

    def test_area(self):
        assert self.t.area == 8
        assert self.s.area == 4
        assert self.h.area == 7.25

    def test_open(self):
        assert self.h.closed is False
        self.h.close()
        assert self.h.closed is True
        assert self.t.closed is True
        self.t.open_()
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
        offset_loop = loop1.offset(0.03)
        assert offset_loop.closed
        assert offset_loop.ccw

    def test_offset_clipper(self):
        offset_star = self.star_shape.offset_clipper(1)

        # Check if the base and the offested loop intersects
        x_int, z_int = get_intersect(self.star_shape, offset_star)
        assert len(x_int) == 0 and len(z_int) == 0

        # Test the area of the intersected loop
        ref_offset_area = 137.24360110736922
        tested_offset_area = offset_star.area
        assert np.isclose(ref_offset_area, tested_offset_area, rtol=1.0e-1)

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

    def test_join(self):
        """
        Runs the lightning never strikes twice protection protocol
        """
        a = Loop(x=[0, 4, 4, 0, 0], y=[0, 0, 2, 2, 0])
        for _ in range(4):
            i = a.receive_projection([2, -2], 90, get_arg=True)
            assert len(a) == 6
            assert i == 1, f"i = {i}"
        if self.plot:
            a.plot(points=True)
        for _ in range(4):
            i = a.receive_projection([3, 100], -90, get_arg=True)
            assert len(a) == 7
            assert i == 4
        for _ in range(4):
            i = a.receive_projection([2, 100], -90, get_arg=True)
            assert len(a) == 8
            assert i == 5
        for _ in range(4):
            i = a.receive_projection([-2, 1], 0, get_arg=True)
            assert len(a) == 9
            assert i == 7
        if self.plot:
            a.plot(points=True)

    def test_receiveprojection(self):
        bb = Loop.from_file(os.sep.join([TEST, "bbprojfail.json"]))
        cut = [9.1727, 20]
        a = bb.receive_projection(cut, -90)
        assert not np.allclose(a, np.array([9.1727, -6.16878442]))

    def test_intersect(self):
        a = Loop(x=[0, 4, 4, 0, 0], y=[0, 0, 2, 2, 0])
        i = a.intersect([2, 0], [2, 10])
        assert np.allclose(i, np.array([[2.0, 0.0], [2.0, 2.0]]))

    def test_chopbyline(self, plot=False):
        loop = Loop.from_file(os.sep.join([TEST, "testchopbyline.json"]))
        ib, ob = loop.chop_by_line([8, 8], -90)
        # find matching points
        mp = np.array(
            np.all((ob.xyz.T[:, None, :] == ib.xyz.T[None, :, :]), axis=-1).nonzero()
        ).T.tolist()
        # take the first two because it is closed
        assert len(mp) - 2 == 2
        if plot:
            for p, m in zip(mp, cycle(["o", "^", "*"])):
                plt.plot(*ob.d2.T[p[0]], color="r", marker=m)
                plt.plot(*ib.d2.T[p[1]], color="b", marker=m)

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_clip(self):
        cts = Loop.from_file(os.sep.join([TEST, "ctstest.json"]))
        pts = MultiLoop.from_file(os.sep.join([TEST, "tstest.json"]))
        a = pts.clip(cts)
        f, ax = plt.subplots()
        a.plot(ax=ax)
        cts.plot(ax=ax)

    def test_section(self):
        loop = Loop(x=[0, 2, 2, 0, 0], z=[-1, -1, 1, 1, -1])
        plane = Plane([0, 0, 0], [0, 1, 0], [1, 0, 0])
        inter = loop.section(plane)
        assert np.allclose(inter[0], np.array([0, 0, 0])), inter[0]
        assert np.allclose(inter[1], np.array([2, 0, 0])), inter[1]

    def test_point_inside(self):
        loop = Loop(x=[-2, 2, 2, -2, -2], z=[-2, -2, 2, 2, -2])
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
        for p in in_points:
            assert loop.point_inside(p), p
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
        for p in out_points:
            assert not loop.point_inside(p), p
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
        for p in on_points:
            assert not loop.point_inside(p), p

    def test_cross_section(self):
        """
        Test that the CrossSection mesh nodes for a Loop match reference values
        """
        expected_nodes = np.array(
            [
                [0.0, 0.0],
                [0.95, 0.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 0.6],
                [1.0, 1.0],
                [0.64644661, 0.64644661],
                [1.0, 0.25],
                [1.0, 0.125],
                [1.0, 0.0625],
                [0.475, 0.0],
                [0.82272759, 0.55],
                [0.7125, 0.0],
                [0.83125, 0.0],
                [0.896875, 0.09375],
                [0.771875, 0.11140625],
                [0.8530552, 0.23482732],
                [0.3232233, 0.3232233],
                [0.91283749, 0.375],
                [0.69917639, 0.24763242],
                [0.62277167, 0.1193469],
                [0.48483496, 0.48483496],
                [0.8232233, 0.8232233],
                [0.50891309, 0.27405462],
                [0.66763584, 0.05967345],
                [0.7421875, 0.05570313],
                [0.69732334, 0.11537658],
                [0.890625, 0.0],
                [0.9234375, 0.046875],
                [0.8640625, 0.046875],
                [0.975, 0.0],
                [1.0, 0.03125],
                [0.975, 0.03125],
                [1.0, 0.55],
                [0.91136379, 0.575],
                [0.91136379, 0.525],
                [1.0, 0.8],
                [0.91161165, 0.91161165],
                [0.91161165, 0.71161165],
                [0.834375, 0.10257813],
                [0.8015625, 0.05570313],
                [0.9265276, 0.24241366],
                [0.9265276, 0.17991366],
                [1.0, 0.1875],
                [0.9484375, 0.078125],
                [0.73552569, 0.17951934],
                [0.66097403, 0.18348966],
                [0.73483496, 0.73483496],
                [0.7345871, 0.5982233],
                [0.82297545, 0.68661165],
                [0.60404474, 0.26084352],
                [0.56584238, 0.19670076],
                [0.88294634, 0.30491366],
                [0.95641875, 0.3125],
                [1.0, 0.09375],
                [0.9484375, 0.109375],
                [0.771875, 0.0],
                [0.8124651, 0.17311678],
                [0.8749651, 0.16428866],
                [0.49195655, 0.13702731],
                [0.4160682, 0.29863896],
                [0.39911165, 0.16161165],
                [1.0, 0.375],
                [0.95641875, 0.4375],
                [0.65378127, 0.51741748],
                [0.59200567, 0.36623369],
                [0.76095199, 0.39881621],
                [0.77611579, 0.24122987],
                [0.2375, 0.0],
                [0.16161165, 0.16161165],
                [0.86778254, 0.4625],
                [0.56564078, 0.56564078],
                [0.80600694, 0.31131621],
                [0.59375, 0.0],
                [0.54888584, 0.05967345],
                [0.49687402, 0.37944479],
                [0.40402913, 0.40402913],
            ]
        )

        loop = Loop(
            x=[0.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            y=[0.0, 0.0, 0.0, 0.5, 0.55, 0.6, 1.0, 0.0],
        )
        cross_section, _ = loop.generate_cross_section([0.1], 0.1, 30)

        assert np.allclose(cross_section.mesh_nodes, expected_nodes)

    def test_cross_section_geom_details(self):
        """
        Test that the geometry details for a Loop match reference values
        """
        expected_points = [
            [0.0, 0.0],
            [0.95, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.6],
            [1.0, 1.0],
        ]

        expected_facets = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]

        expected_control_point = [2.0 / 3.0, 1.0 / 3.0]

        expected_hole = []

        loop = Loop(
            x=[0.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            y=[0.0, 0.0, 0.0, 0.5, 0.55, 0.6, 1.0, 0.0],
        )
        _, clean_loop = loop.generate_cross_section([0.1], 0.1, 30)

        assert np.allclose(clean_loop.get_points(), expected_points)
        assert np.allclose(clean_loop.get_closed_facets(), expected_facets)
        assert np.allclose(clean_loop.get_control_point(), expected_control_point)
        assert np.allclose(clean_loop.get_hole(), expected_hole)


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
