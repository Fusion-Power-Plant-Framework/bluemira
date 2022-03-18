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
import pytest

from bluemira.base.constants import E_I, E_IJ, E_IJK, to_celsius, to_kelvin
from bluemira.utilities.tools import levi_civita_tensor


def test_lct_constants():
    for i, lct in enumerate([E_I, E_IJ, E_IJK], start=1):
        np.testing.assert_equal(lct, levi_civita_tensor(dim=i))


class TestTemperatureConverters:
    params = ("unit", ["celsius", "kelvin", "rankine", "reaumur", "fahrenheit"])

    @pytest.mark.parametrize(*params)
    def test_to_kelvin_rasies_error(self, unit):
        with pytest.raises(ValueError):
            to_kelvin(-1000, unit)

    @pytest.mark.parametrize(*params)
    def test_to_celsius_raises_error(self, unit):
        with pytest.raises(ValueError):
            to_celsius(-1000, unit)


if __name__ == "__main__":
    pytest.main([__file__])
