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
"""
Created on Fri Aug  2 12:33:20 2019

@author: matti
"""
import numpy as np

from BLUEPRINT.geometry.geombase import Plane, point_dict_to_array
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.shell import Shell


def test_read_write_geombase(tmpdir):
    # Make a star loop
    outer_len = 2.0
    inner_len = 4.0 / 3.0
    n_points = 5
    x = []
    y = []
    angle_diff = 2.0 * np.pi / n_points
    for i in range(0, n_points):
        angle = angle_diff * i
        x_tip = outer_len * np.sin(angle)
        y_tip = outer_len * np.cos(angle)
        x.append(x_tip)
        y.append(y_tip)

        angle_dip = angle + angle_diff / 2.0
        x_dip = inner_len * np.sin(angle_dip)
        y_dip = inner_len * np.cos(angle_dip)
        x.append(x_dip)
        y.append(y_dip)
    h = Loop(x, y)
    h.close()
    g = h.offset(1)
    j = g.offset(2)
    shell = Shell.from_offset(h, 1)
    multi = MultiLoop([h, g, j], stitch=False)

    # Temporary path
    path = tmpdir.mkdir("geometry")
    # Test no extension
    filename = str(path.join("testh-Loop"))
    h.to_json(filename)
    h2 = Loop.from_file(filename)
    assert np.all(h.xyz - h2.xyz == 0)
    filename = str(path.join("testh-Loop.json"))
    h.to_json(filename)
    h2 = Loop.from_file(filename)
    assert np.all(h.xyz - h2.xyz == 0)

    filename = str(path.join("testG-Shell.json"))
    shell.to_json(filename)
    shell2 = Shell.from_file(filename)
    assert np.all(shell.inner.xyz - shell2.inner.xyz == 0)
    assert np.all(shell.outer.xyz - shell2.outer.xyz == 0)

    filename = str(path.join("testJ-MultiLoop.json"))
    multi.to_json(filename)
    multi2 = MultiLoop.from_file(filename)
    for i in range(len(multi)):
        assert np.all(multi.loops[i].xyz - multi2.loops[i].xyz == 0)


class TestPointDictArray:
    def test_point_dict_array(self):
        d = {"x": 1, "y": 2, "z": 3}
        assert np.all(point_dict_to_array(d) == np.array([1, 2, 3]))
        d = {"y": 2, "z": 3, "x": 1}
        assert np.all(point_dict_to_array(d) == np.array([1, 2, 3]))
        d = {"z": 3, "y": 2, "x": 1}
        assert np.all(point_dict_to_array(d) == np.array([1, 2, 3]))


class TestPlane:
    def test_plan_dims(self):
        p_xy = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        p_yz = Plane([0, 0, 0], [0, 1, 0], [0, 0, 1])
        p_xz = Plane([0, 0, 0], [1, 0, 0], [0, 0, 1])

        assert p_xy.plan_dims == ["x", "y"]
        assert p_yz.plan_dims == ["y", "z"]
        assert p_xz.plan_dims == ["x", "z"]

        p_xy = Plane([0, 0, 0], [1, 0, 0], [0, -1, 0])
        p_yz = Plane([0, 0, 0], [0, 1, 0], [0, 0, -1])
        p_xz = Plane([0, 0, 0], [1, 0, 0], [0, 0, -1])

        assert p_xy.plan_dims == ["x", "y"]
        assert p_yz.plan_dims == ["y", "z"]
        assert p_xz.plan_dims == ["x", "z"]

        p_xy = Plane([0, 0, 0], [-1, 0, 0], [0, -1, 0])
        p_yz = Plane([0, 0, 0], [0, -1, 0], [0, 0, -1])
        p_xz = Plane([0, 0, 0], [-1, 0, 0], [0, 0, -1])

        assert p_xy.plan_dims == ["x", "y"]
        assert p_yz.plan_dims == ["y", "z"]
        assert p_xz.plan_dims == ["x", "z"]

    def test_intersect(self):
        p_xy = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        p_yz = Plane([0, 0, 0], [0, 1, 0], [0, 0, 1])
        p_xz = Plane([0, 0, 0], [1, 0, 0], [0, 0, 1])

        o, x = p_xy.intersect(p_xz)
        assert np.allclose(o, np.array([0, 0, 0]))
        assert np.allclose(np.abs(x), np.array([1, 0, 0]))

        o, y = p_xy.intersect(p_yz)
        assert np.allclose(o, np.array([0, 0, 0]))
        assert np.allclose(np.abs(y), np.array([0, 1, 0]))

        o, z = p_yz.intersect(p_xz)
        assert np.allclose(o, np.array([0, 0, 0]))
        assert np.allclose(np.abs(z), np.array([0, 0, 1]))

        fail1, fail2 = p_xy.intersect(p_xy)
        assert fail1 is None
        assert fail2 is None

        p_xy4 = Plane([0, 0, 4], [1, 0, 4], [0, 1, 4])
        p_yz4 = Plane([4, 0, 0], [4, 1, 0], [4, 0, 1])
        p_xz4 = Plane([0, 4, 0], [1, 4, 0], [0, 4, 1])

        o, x = p_xz4.intersect(p_xy4)
        assert np.allclose(o, np.array([0, 4, 4]))
        assert np.allclose(np.abs(x), np.array([1, 0, 0]))

        o, y = p_xy4.intersect(p_yz4)
        assert np.allclose(o, np.array([-4, 0, -4]))
        assert np.allclose(np.abs(y), np.array([0, 1, 0]))

        o, z = p_xz4.intersect(p_yz4)
        assert np.allclose(o, np.array([4, 4, 0]))
        assert np.allclose(np.abs(z), np.array([0, 0, 1]))
