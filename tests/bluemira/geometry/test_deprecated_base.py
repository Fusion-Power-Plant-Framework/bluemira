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

from bluemira.geometry._deprecated_base import Plane
from bluemira.geometry._deprecated_loop import Loop


def test_read_write_geombase(tmpdir):
    # Make some geometry objects
    x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
    y = [-4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 1]
    h = Loop(x, y)
    g = h.offset(1)
    j = g.offset(2)  # Note this breaks offset :'(

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
