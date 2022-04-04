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

import filecmp
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.geomtools import circle_seg, rotate_matrix
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell


class TestShell:
    loop = Loop(x=[6, 10, 10, 6], y=4, z=[-1, -5, 5, 1])
    loop.close()
    shell = Shell.from_offset(loop, 0.3)

    def test_translate(self):
        self.shell.translate([0, 1, 0])
        assert self.shell.inner.y[0] == 5
        assert self.shell.outer.y[0] == 5
        self.shell.translate([1, 1, 0])
        assert self.shell.inner.y[0] == 6
        assert self.shell.outer.y[0] == 6
        assert min(self.shell.inner.x) == 7

    def test_rotate(self):
        angle = 30  # degrees
        dcm = rotate_matrix(np.deg2rad(angle), axis="z")
        new_rotate = self.shell.rotate(angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False)
        new_dcm_rot = self.shell.rotate_dcm(dcm, update=False)
        self.shell.rotate_dcm(dcm)

        assert self.shell == new_rotate == new_dcm_rot

    def test_plotting(self):
        plot_defaults(force=True)

        inner = Loop(
            x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            z=[0, -5, -7, -5, -8, -2, 0, 2, 4, 5, 6, 7, 6, 4, 3, 2, 0],
        )
        outer = inner.offset(1)
        shell = Shell(inner, outer)
        shell.rotate(30, p1=[4, 3, 1], p2=[6, 4, 2])
        shell.plot()

        path = get_bluemira_path("BLUEPRINT/geometry/test_data", subfolder="tests")
        name_new = os.sep.join([path, "test_3d_shell_fig_new.png"])
        plt.savefig(name_new)
        name_old = os.sep.join([path, "test_3d_shell_fig.png"])

        assert filecmp.cmp(name_new, name_old, shallow=False)

    def test_cross_section(self):
        """
        Test that the CrossSection mesh nodes for a Shell match reference values
        """
        expected_nodes = [
            [0.2, 0.1],
            [0.85, 0.1],
            [0.9, 0.1],
            [0.9, 0.5],
            [0.9, 0.6],
            [0.9, 0.8],
            [0.6, 0.4],
            [0.0, 0.0],
            [0.95, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.6],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.475, 0.0],
            [1.0, 0.25],
            [1.0, 0.8],
            [0.75, 0.75],
            [0.25, 0.25],
            [0.525, 0.1],
            [0.7125, 0.0],
            [0.2375, 0.0],
            [0.125, 0.125],
            [0.9, 0.3],
            [0.875, 0.875],
            [0.75, 0.6],
            [0.4, 0.25],
            [0.83125, 0.0],
            [1.0, 0.125],
            [0.875, 0.04355469],
            [1.0, 0.0625],
            [0.8125, 0.8125],
            [0.125, 0.0],
            [0.1875, 0.1875],
            [0.3, 0.175],
            [0.9, 0.7],
            [1.0, 0.7],
            [1.0, 0.375],
            [0.9, 0.4],
            [0.825, 0.7],
            [0.3625, 0.1],
            [0.59375, 0.0],
            [0.6875, 0.1],
            [0.375, 0.375],
            [0.625, 0.625],
            [0.28125, 0.1],
            [0.21875, 0.05],
            [0.3, 0.05],
            [0.853125, 0.02177734],
            [0.8625, 0.07177734],
            [0.840625, 0.05],
            [0.95, 0.75],
            [0.95, 0.8],
            [0.9, 0.75],
            [0.1625, 0.05],
            [0.18125, 0.0],
            [0.35625, 0.0],
            [0.41875, 0.05],
            [0.325, 0.25],
            [0.3875, 0.3125],
            [0.3125, 0.3125],
            [0.4875, 0.3875],
            [0.55, 0.45],
            [0.4375, 0.4375],
            [0.95, 0.175],
            [0.95, 0.275],
            [0.9, 0.2],
            [0.95, 0.1125],
            [1.0, 0.1875],
            [0.78125, 0.78125],
            [0.7875, 0.725],
            [0.81875, 0.75625],
            [0.8875, 0.07177734],
            [0.875, 0.1],
            [0.640625, 0.05],
            [0.653125, 0.0],
            [0.7, 0.05],
            [0.95, 0.3375],
            [0.95, 0.3875],
            [0.9, 0.35],
            [0.9375, 0.8375],
            [0.8875, 0.8375],
            [0.95, 0.45],
            [0.95, 0.5],
            [0.9, 0.45],
            [0.95, 0.55],
            [0.9, 0.55],
            [0.8625, 0.75],
            [0.85625, 0.80625],
            [0.9, 0.65],
            [0.95, 0.6],
            [0.95, 0.65],
            [0.9375, 0.9375],
            [1.0, 0.9],
            [0.1625, 0.1125],
            [0.125, 0.0625],
            [1.0, 0.55],
            [0.675, 0.5],
            [0.6875, 0.6125],
            [0.6125, 0.5125],
            [0.19375, 0.14375],
            [0.15625, 0.15625],
            [1.0, 0.09375],
            [0.95, 0.08125],
            [0.5625, 0.5625],
            [0.60625, 0.1],
            [0.559375, 0.05],
            [0.275, 0.2125],
            [0.35, 0.2125],
            [0.890625, 0.0],
            [0.9125, 0.02177734],
            [0.78125, 0.05],
            [0.771875, 0.0],
            [0.925, 0.05],
            [0.975, 0.03125],
            [0.84375, 0.84375],
            [0.975, 0.0],
            [1.0, 0.03125],
            [0.24375, 0.18125],
            [0.25, 0.1375],
            [0.0625, 0.0625],
            [0.0625, 0.0],
            [0.21875, 0.21875],
            [1.0, 0.65],
            [0.95, 0.7],
            [1.0, 0.75],
            [1.0, 0.4375],
            [1.0, 0.3125],
            [0.75, 0.675],
            [0.7875, 0.65],
            [0.5, 0.05],
            [0.44375, 0.1],
            [0.76875, 0.1],
            [0.534375, 0.0],
            [0.5, 0.325],
            [0.6875, 0.6875],
        ]

        outer_loop = Loop(
            x=[0.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
            y=[0.0, 0.0, 0.0, 0.5, 0.55, 0.6, 1.0, 0.5, 0.0],
        )

        inner_loop = Loop(
            x=[0.2, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.2],
            y=[0.1, 0.1, 0.1, 0.5, 0.55, 0.6, 0.8, 0.4, 0.1],
        )

        test_shell = Shell(inner_loop, outer_loop)
        cross_section, _ = test_shell.generate_cross_section([0.1], 0.1, 30)

        assert np.allclose(cross_section.mesh_nodes, expected_nodes)

    def test_cross_section_geom_details(self):
        """
        Test that the geometry details for a Shell match reference values
        """
        expected_points = [
            [0.2, 0.1],
            [0.85, 0.1],
            [0.9, 0.1],
            [0.9, 0.5],
            [0.9, 0.6],
            [0.9, 0.8],
            [0.6, 0.4],
            [0.0, 0.0],
            [0.95, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.6],
            [1.0, 1.0],
            [0.5, 0.5],
        ]

        expected_facets = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 0],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 7],
        ]

        expected_hole = [0.683333333333334, 0.31666666666666704]

        outer_loop = Loop(
            x=[0.0, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
            y=[0.0, 0.0, 0.0, 0.5, 0.55, 0.6, 1.0, 0.5, 0.0],
        )

        inner_loop = Loop(
            x=[0.2, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.2],
            y=[0.1, 0.1, 0.1, 0.5, 0.55, 0.6, 0.8, 0.4, 0.1],
        )

        test_shell = Shell(inner_loop, outer_loop)
        _, clean_shell = test_shell.generate_cross_section([0.1], 0.1, 30)

        assert np.allclose(clean_shell.get_points(), expected_points)
        assert np.allclose(clean_shell.get_closed_facets(), expected_facets)
        assert np.allclose(clean_shell.get_hole(), expected_hole)

        # Note that control points are generated stochastically for Shells.
        # So just check that the control point is in the polygon.
        assert test_shell.point_inside(clean_shell.get_control_point())


def test_fail_if_intersect():
    # Create a circle of radius 1
    x_arr, z_arr = circle_seg(1.0)
    loop_1 = Loop(x=x_arr, z=z_arr)
    # Translate the loop
    loop_2 = loop_1.translate([0.5, 0.0, 0.0], update=False)
    with pytest.raises(GeometryError) as err:
        # Try to create a shell - should raise and error
        shell = Shell(loop_1, loop_2)
