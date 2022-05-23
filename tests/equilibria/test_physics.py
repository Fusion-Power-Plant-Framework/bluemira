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
import pandas as pd

from bluemira.equilibria.physics import calc_psib, coulomb_logarithm


class TestPhysics:
    def test_psi_bd(self):
        psib = calc_psib(320, 9, 19.6e6, 0.8)

        assert round(abs(psib - 143), 0) == 0  # 142.66 ~ 143 V.s
        # CREATE DEMO 2015 test case


class TestCoulombLogarithm:
    @classmethod
    def setup_class(cls):
        df = pd.DataFrame(
            columns=["Case", "n [1/m^3]", "T [eV]", "ln Lambda (Goldston)", "ln_lambda"]
        )

        df.loc[0] = ["Solar wind", 10.0**7, 10.0, 26]
        df.loc[1] = ["Van Allen belts", 10.0**9, 10.0**2, 26]
        df.loc[2] = ["Earth's ionosphere", 10.0**11, 10.0**-1, 14]
        df.loc[3] = ["Solar corona", 10.0**13, 10.0**2, 21]
        df.loc[4] = ["Gas discharge", 10.0**16, 10.0**0, 12]
        df.loc[5] = ["Process plasma", 10.0**18, 10.0**2, 15]
        df.loc[6] = ["Fusion experiment", 10.0**19, 10.0**3, 17]
        df.loc[7] = ["Fusion reactor", 10.0**20, 10.0**4, 18]
        df.loc[8] = ["Hartmut's case", 10.0**20, 1000.0, 16.5]
        cls.data = df

    def test_coulomb_logarithm_values(self):
        for i in range(9):
            T = self.data.loc[i, "T [eV]"]
            n = self.data.loc[i, "n [1/m^3]"]
            value = round(coulomb_logarithm(T, n), 1)
            reference_value = self.data.loc[i, "ln Lambda (Goldston)"]
            np.testing.assert_allclose(value, reference_value, rtol=0.1)
