# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.plasma import NoPlasmaCoil, PlasmaCoil


class TestPlasmaCoil:
    points = (
        [6, 0],
        [6.0, 0.0],
        [np.array([6, 7]), np.array([0, 0])],
        [np.array([6.0, 7.0]), np.array([0.0, 0.0])],
        [[16, 17], [16, 17]],
        [np.array([[16, 17], [5, 6]]), np.array([[16, 17], [0, 0]])],
    )

    @classmethod
    def setup_class(cls):
        grid = Grid(4, 10, -5, 5, 50, 50)
        plasma_psi = np.zeros((50, 50))
        j_tor = np.zeros((50, 50))
        plasma_psi[25, 25] = 1.0
        j_tor[25, 25] = 1.0
        cls.plasma_coil = PlasmaCoil(plasma_psi, j_tor, grid=grid)

    @pytest.mark.parametrize(("x", "z"), points)
    def test_psi(self, x, z):
        result = self.plasma_coil.psi(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize(("x", "z"), points)
    def test_Bx(self, x, z):
        result = self.plasma_coil.Bx(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize(("x", "z"), points)
    def test_Bz(self, x, z):
        result = self.plasma_coil.Bx(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize(("x", "z"), points)
    def test_Bp(self, x, z):
        result = self.plasma_coil.Bp(x, z)
        assert result.shape == np.array(x).shape

    def test_none_call(self):
        psi = self.plasma_coil.psi()
        Bx = self.plasma_coil.Bx()
        Bz = self.plasma_coil.Bz()
        Bp = self.plasma_coil.Bp()
        assert psi.shape == (50, 50)
        assert Bx.shape == (50, 50)
        assert Bz.shape == (50, 50)
        assert Bp.shape == (50, 50)

    def test_one_None_one_array(self):
        with pytest.raises(EquilibriaError):
            self.plasma_coil.psi(None, 1.0)
        with pytest.raises(EquilibriaError):
            self.plasma_coil.psi(np.array([1.0]), None)

    def test_bad_input_shapes(self):
        with pytest.raises(EquilibriaError):
            self.plasma_coil.psi(np.zeros(3), np.zeros(4))


class TestNoPlasmaCoil:
    def test_call(self):
        grid = Grid(4, 10, -5, 5, 100, 100)
        nocoil = NoPlasmaCoil(grid)

        assert nocoil.psi().shape == (100, 100)
        assert nocoil.Bx().shape == (100, 100)
        assert nocoil.Bz().shape == (100, 100)
        assert nocoil.Bp().shape == (100, 100)

        x = np.array([1, 2, 3, 4])
        z = np.array([1, 2, 3, 4])
        assert nocoil.psi(x, z).shape == (4,)
        assert nocoil.Bx(x, z).shape == (4,)
        assert nocoil.Bz(x, z).shape == (4,)
        assert nocoil.Bp(x, z).shape == (4,)
