# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
"""
Created on Tue Sep 17 11:18:37 2019

@author: matti
"""

import numpy as np
import pytest
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    from neutronics_material_maker.utils import make_serpent_material

from BLUEPRINT.utilities.tools import is_num, kgm3togcm3, tokelvin

import tests
from tests.materials.setup_methods import TEST_MATERIALS_CACHE


class TestProperty:
    def test_property(self):
        beryllium = TEST_MATERIALS_CACHE.get_material("Beryllium")
        with pytest.raises(ValueError):
            beryllium.Cp(-200)
        with pytest.raises(ValueError):
            beryllium.Cp([-200])
        with pytest.raises(ValueError):
            beryllium.Cp(20000)
        with pytest.raises(ValueError):
            beryllium.Cp([20000])
        beryllium.Cp(400)
        beryllium.Cp([400])

    def test_array(self):
        beryllium = TEST_MATERIALS_CACHE.get_material("Beryllium")
        t_array = beryllium.Cp(np.array([400, 500, 550]))
        t_list = beryllium.Cp([400, 500, 550])
        assert (t_array - t_list).all() == 0

    def test_array_limits(self):
        tungsten = TEST_MATERIALS_CACHE.get_material("Tungsten")
        a = tungsten.k(tokelvin([20, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):
            a = tungsten.k(tokelvin([20, 30, 400, 1000, 1001]))
        with pytest.raises(ValueError):
            a = tungsten.k(tokelvin([19, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):
            a = tungsten.k(tokelvin([19, 30, 400, 1000, 1001]))
        del a

    def test_interp(self):
        ss_316 = TEST_MATERIALS_CACHE.get_material("SS316-LN")
        ss_316.rho([300])
        ss_316.rho(300)
        assert ss_316.rho(tokelvin(20)) == 7930
        assert ss_316.rho(tokelvin(300)) == 7815
        a = ss_316.rho([300, 400, 500])
        b = ss_316.rho(np.array([300, 400, 500]))
        assert (a - b).all() == 0

    def test_curho(self):
        copper = TEST_MATERIALS_CACHE.get_material("Pure Cu")
        assert copper.rho(tokelvin(20)) == 8940  # Eq check

    def test_cucrzrrho(self):
        cucrzr = TEST_MATERIALS_CACHE.get_material("CuCrZr")
        assert cucrzr.rho(tokelvin(20)) == 8900  # Eq check


class TestMaterials:
    def test_density_load(self):
        beryllium = TEST_MATERIALS_CACHE.get_material("Beryllium")
        beryllium.temperature = 300
        assert hasattr(beryllium, "density")
        assert is_num(beryllium.density)
        assert type(beryllium.density) == float

    def test_material_card(self):
        beryllium = TEST_MATERIALS_CACHE.get_material("Beryllium")
        pytest.importorskip("openmc")
        s = make_serpent_material(beryllium)
        s = s.splitlines()[0]
        # Check serpent header updated with correct density
        assert float(s.split(" ")[2]) == pytest.approx(kgm3togcm3(beryllium.density))

    def test_t_tmp(self):
        """
        Tests Doppler broadening material card for serpent II
        """
        ss_316 = TEST_MATERIALS_CACHE.get_material("SS316-LN")
        pytest.importorskip("openmc")
        s = make_serpent_material(ss_316)
        assert " tmp 293.15 " in s.splitlines()[0]
        ss_316.temperature_in_K = 400
        s = make_serpent_material(ss_316)
        assert " tmp 400 " in s.splitlines()[0]

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_superconductor_plot(self):
        nb_3_sn = TEST_MATERIALS_CACHE.get_material("Nb3Sn - WST")
        nb_3_sn_2 = TEST_MATERIALS_CACHE.get_material("Nb3Sn - EUTF4")
        nbti = TEST_MATERIALS_CACHE.get_material("NbTi")
        bmin, bmax = 3, 16
        tmin, tmax = 2, 6
        eps = -0.66
        nb_3_sn.plot(bmin, bmax, tmin, tmax, eps)
        nb_3_sn_2.plot(bmin, bmax, tmin, tmax, eps)
        nbti.plot(bmin, bmax, tmin, tmax)


class TestLiquids:
    def test_temp_pressure(self):
        water = TEST_MATERIALS_CACHE.get_material("H2O")
        assert water.temperature == 293.15
        assert water.pressure == 101325
        assert water.density == pytest.approx(998.207815375)

    def test_material_card(self):
        pytest.importorskip("openmc")
        water = TEST_MATERIALS_CACHE.get_material("H2O")
        water.temperature, water.pressure = 500, 200000
        s = make_serpent_material(water)
        s = s.splitlines()[0]
        assert float(s.split(" ")[2]) == pytest.approx(kgm3togcm3(water.density))


if __name__ == "__main__":
    pytest.main([__file__])
