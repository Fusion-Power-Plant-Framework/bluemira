# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.constants import EPS, to_kelvin
from bluemira.utilities.tools import is_num
from tests.materials.materials_helpers import MATERIAL_CACHE


class TestProperty:
    tungsten = MATERIAL_CACHE.get_material("Tungsten")
    SS_316 = MATERIAL_CACHE.get_material("SS316-LN")
    copper = MATERIAL_CACHE.get_material("Pure Cu")
    cucrzr = MATERIAL_CACHE.get_material("CuCrZr")

    def test_array_limits(self):
        a = self.tungsten.k(to_kelvin([20, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([20, 30, 400, 1000, 1001]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([19, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([19, 30, 400, 1000, 1001]))
        del a

    def test_interp(self):
        self.SS_316.rho([300])
        self.SS_316.rho(300)
        assert self.SS_316.rho(to_kelvin(20)) == 7930
        assert self.SS_316.rho(to_kelvin(300)) == 7815
        a = self.SS_316.rho([300, 400, 500])
        b = self.SS_316.rho(np.array([300, 400, 500]))
        assert (a - b).all() == 0

    def test_cu_rho(self):
        assert self.copper.rho(to_kelvin(20)) == 8940  # Eq check

    def test_cucrzr_rho(self):
        assert self.cucrzr.rho(to_kelvin(20)) == 8900  # Eq check


class TestMaterials:
    beryllium = MATERIAL_CACHE.get_material("Beryllium")
    SS_316 = MATERIAL_CACHE.get_material("SS316-LN")
    nb_3_sn = MATERIAL_CACHE.get_material("Nb3Sn - WST")
    nb_3_sn_2 = MATERIAL_CACHE.get_material("Nb3Sn - EUTF4")
    nbti = MATERIAL_CACHE.get_material("NbTi")

    def test_density_load(self):
        self.beryllium.temperature = 300
        assert hasattr(self.beryllium, "density")
        assert is_num(self.beryllium.density())
        assert type(self.beryllium.density()) is float

    def test_material_card(self):
        pytest.importorskip("openmc")
        openmc_mat = self.beryllium.to_openmc_material()
        assert openmc_mat.density == pytest.approx(self.beryllium.density())

    def test_t_tmp(self):
        """
        Tests Doppler broadening material card for serpent II
        """
        pytest.importorskip("openmc")
        openmc_mat = self.SS_316.to_openmc_material()
        assert openmc_mat.temperature == pytest.approx(293.15)
        self.SS_316.temperature = 400
        openmc_mat = self.SS_316.to_openmc_material()
        assert openmc_mat.temperature == pytest.approx(400)

    def test_superconductor_plot(self):
        b_min, b_max = 3, 16
        t_min, t_max = 2, 6
        eps = -0.66
        self.nb_3_sn.plot(b_min, b_max, t_min, t_max, eps)
        self.nb_3_sn_2.plot(b_min, b_max, t_min, t_max, eps)
        self.nbti.plot(b_min, b_max, t_min, t_max)


class TestLiquids:
    water = MATERIAL_CACHE.get_material("H2O")

    def test_temp_pressure(self):
        assert self.water.temperature == pytest.approx(293.15, rel=0, abs=EPS)
        assert self.water.pressure == pytest.approx(101325, rel=0, abs=EPS)
        assert self.water.density() == pytest.approx(998.207815375)

    def test_material_card(self):
        pytest.importorskip("openmc")
        self.water.temperature, self.water.pressure = 500, 200000
        openmc_mat = self.water.to_openmc_material()
        assert openmc_mat.density == pytest.approx(self.water.density())
