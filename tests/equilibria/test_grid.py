# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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
        points = list(zip(x, z, strict=False))
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
