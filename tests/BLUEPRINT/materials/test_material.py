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
from BLUEPRINT.materials import materials_cache

import tests


class TestProperty:
    tungsten = materials_cache.get_material("Tungsten")
    SS_316 = materials_cache.get_material("SS316-LN")
    copper = materials_cache.get_material("Pure Cu")
    cucrzr = materials_cache.get_material("CuCrZr")

    def test_array_limits(self):
        a = self.tungsten.k(tokelvin([20, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):
            a = self.tungsten.k(tokelvin([20, 30, 400, 1000, 1001]))
        with pytest.raises(ValueError):
            a = self.tungsten.k(tokelvin([19, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):
            a = self.tungsten.k(tokelvin([19, 30, 400, 1000, 1001]))
        del a

    def test_interp(self):
        self.SS_316.rho([300])
        self.SS_316.rho(300)
        assert self.SS_316.rho(tokelvin(20)) == 7930
        assert self.SS_316.rho(tokelvin(300)) == 7815
        a = self.SS_316.rho([300, 400, 500])
        b = self.SS_316.rho(np.array([300, 400, 500]))
        assert (a - b).all() == 0

    def test_curho(self):
        assert self.copper.rho(tokelvin(20)) == 8940  # Eq check

    def test_cucrzrrho(self):
        assert self.cucrzr.rho(tokelvin(20)) == 8900  # Eq check


class TestMaterials:
    beryllium = materials_cache.get_material("Beryllium")
    SS_316 = materials_cache.get_material("SS316-LN")
    nb_3_sn = materials_cache.get_material("Nb3Sn - WST")
    nb_3_sn_2 = materials_cache.get_material("Nb3Sn - EUTF4")
    nbti = materials_cache.get_material("NbTi")
    plot = tests.PLOTTING

    def test_density_load(self):
        self.beryllium.temperature = 300
        assert hasattr(self.beryllium, "density")
        assert is_num(self.beryllium.density)
        assert type(self.beryllium.density) == float

    def test_material_card(self):
        pytest.importorskip("openmc")
        s = make_serpent_material(self.beryllium)
        s = s.splitlines()[0]
        # Check serpent header updated with correct density
        assert float(s.split(" ")[2]) == pytest.approx(
            kgm3togcm3(self.beryllium.density)
        )

    def test_t_tmp(self):
        """
        Tests Doppler broadening material card for serpent II
        """
        pytest.importorskip("openmc")
        s = make_serpent_material(self.SS_316)
        assert " tmp 293.15 " in s.splitlines()[0]
        self.SS_316.temperature_in_K = 400
        s = make_serpent_material(self.SS_316)
        assert " tmp 400 " in s.splitlines()[0]

    def test_superconductor_plot(self):
        if self.plot:
            bmin, bmax = 3, 16
            tmin, tmax = 2, 6
            eps = -0.66
            self.nb_3_sn.plot(bmin, bmax, tmin, tmax, eps)
            self.nb_3_sn_2.plot(bmin, bmax, tmin, tmax, eps)
            self.nbti.plot(bmin, bmax, tmin, tmax)


class TestLiquids:
    water = materials_cache.get_material("H2O")

    def test_temp_pressure(self):
        assert self.water.temperature == 293.15
        assert self.water.pressure == 101325
        assert self.water.density == pytest.approx(998.207815375)

    def test_material_card(self):
        pytest.importorskip("openmc")
        self.water.temperature, self.water.pressure = 500, 200000
        s = make_serpent_material(self.water)
        s = s.splitlines()[0]
        assert float(s.split(" ")[2]) == pytest.approx(kgm3togcm3(self.water.density))


if __name__ == "__main__":
    pytest.main([__file__])
