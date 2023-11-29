# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from bluemira.base.constants import (
    E_I,
    E_IJ,
    E_IJK,
    gas_flow_uc,
    raw_uc,
    to_celsius,
    to_kelvin,
)
from bluemira.utilities.tools import levi_civita_tensor


def test_lct_constants():
    for i, lct in enumerate([E_I, E_IJ, E_IJK], start=1):
        np.testing.assert_equal(lct, levi_civita_tensor(dim=i))


class TestTemperatureConverters:
    params = ("unit", ["celsius", "kelvin", "rankine", "reaumur", "fahrenheit"])

    @pytest.mark.parametrize(*params)
    def test_to_kelvin_rasies_error(self, unit):
        with pytest.raises(ValueError):  # noqa: PT011
            to_kelvin(-1000, unit)

    @pytest.mark.parametrize(*params)
    def test_to_celsius_raises_error(self, unit):
        with pytest.raises(ValueError):  # noqa: PT011
            to_celsius(-1000, unit)


class TestConverter:
    def test_percentage_conversion(self):
        assert raw_uc(1, "percent", "%") == 1
        assert raw_uc(1, "count", "%") == 100

    def test_raw_flow_conversion(self):
        assert np.isclose(raw_uc(1, "mol", "Pa m^3"), 2271.0954641485578)
        assert np.isclose(raw_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)
        assert np.isclose(raw_uc(2271.0954641485578, "Pa m^3", "mol"), 1)
        assert np.isclose(raw_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

    def test_gas_flow_conversion(self):
        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)
        assert np.isclose(gas_flow_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s", 298.15), 2478.95)
        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)

        assert np.isclose(gas_flow_uc(2478.95, "Pa m^3/s", "mol/s", 298.15), 1)
        assert np.isclose(gas_flow_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

    def test_energy_temperature_conversion(self):
        assert np.isclose(raw_uc(1, "eV", "K"), 11604.518121550082)
        assert np.isclose(raw_uc(1, "eV/s", "K/s"), 11604.518121550082)
        assert np.isclose(raw_uc(11604.518121550082, "K", "eV"), 1)
        assert np.isclose(raw_uc(11604.518121550082, "K/s", "eV/s"), 1)

    def test_units_with_scales(self):
        # .....I know.....
        assert np.isclose(raw_uc(1, "10^19/m^3", "um^-3"), 10)
        assert np.isclose(raw_uc(1, "10^19/m^3", "10um^-3"), 1)
