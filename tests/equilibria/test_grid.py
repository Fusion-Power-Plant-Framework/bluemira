# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid, integrate_dx_dz, volume_integral


class TestGrid:
    def test_init(self):
        g = Grid(0, 20, -10, 10, 100, 100)
        assert g.x_min != 0
        g = Grid(10, 5, -10, 10, 100, 100)
        assert g.x_min == 5
        assert g.x_max == 10
        g = Grid(5, 10, 10, -10, 100, 100)
        assert g.z_min == -10
        assert g.z_max == 10

        bad_dims = [
            [0, 20, -10, -10, 100, 100],
            [0, 0, -10, 10, 100, 100],
            [0, 0, 0, 0, 100, 100],
        ]
        for bad in bad_dims:
            with pytest.raises(EquilibriaError):
                _ = Grid(*bad)

    def test_eqdsk(self):
        eqdict = {
            "xgrid1": 0.4,
            "xdim": 16,
            "zmid": 0.5,
            "zdim": 10,
            "nx": 100,
            "nz": 100,
        }

        g = Grid.from_eqdict(eqdict)

        assert g.nx == 100
        assert g.nz == 100
        assert np.isclose(g.x_size, 16.0)
        assert np.isclose(g.z_size, 10.0)

    def test_dicretisation(self):
        g = Grid(0, 20, -10, 10, 9, 100)
        assert g.nx == 10
        g = Grid(0, 20, -10, 10, 100, 9)
        assert g.nz == 10

    def test_point_inside(self):
        g = Grid(4, 10, -10, 10, 65, 65)
        points = [[5, 0], [5, 5], [5, 1], [5, -1], [5, -5]]
        for p in points:
            assert g.point_inside(*p)

        x, z = g.bounds
        points = list(zip(x, z))
        for p in points:
            assert g.point_inside(*p)

        fails = [[-1, 0], [-10, 0], [-10, -10], [5, -12], [5, 12], [100, 0]]
        for f in fails:
            assert not g.point_inside(*f)


def test_integrate_dx_dz():
    nx, nz = 100, 100
    f = np.ones((nx, nz))

    integral = integrate_dx_dz(f, 1, 1)
    # NOTE: The integration is at the centre points of each grid cell
    assert np.isclose(integral, (nx - 1) * (nz - 1))


def test_volume_integral():
    nx, nz = 100, 100
    x = np.linspace(0, 100, nx)
    f = np.ones((nx, nz))

    integral = volume_integral(f, x, 1, 1)

    x_centres = 0.5 * (x + np.roll(x, 1))[1:]
    # NOTE: The integration is at the centre points of each grid cell
    true = 2 * np.pi * (nx - 1) * np.sum(x_centres)
    assert np.isclose(integral, true)
