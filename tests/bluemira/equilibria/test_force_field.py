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

import pytest

import numpy as np
from scipy.special import ellipe, ellipk

from bluemira.base.constants import MU_0

from bluemira.equilibria.force_field import ForceField
from bluemira.equilibria.coils import Coil, CoilSet


class TestForceField:
    @classmethod
    def setup_class(cls):
        coils = []

        x = [5, 5]
        z = [5, -5]
        for i, (xi, zi) in enumerate(zip(x, z)):
            c = Coil(xi, zi, current=0, ctype="PF", name=f"PF_{i+1}", dx=0, dz=0)

            coils.append(c)
        cls.coilset = CoilSet(coils)
        dummy = Coil(5, 0, current=0, ctype="Plasma", dx=0, dz=0)
        cls.ff = ForceField(cls.coilset, dummy)

    def test_Fz(self):  # noqa (N802)
        """
        Check the vertical forces between a Helmholtz pair.
        Verbose: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6472319
        """
        forces, _ = self.ff.calc_force(10e6 * np.ones(2))
        x, z = self.coilset.coils["PF_1"].x, self.coilset.coils["PF_1"].z
        xc, zc = self.coilset.coils["PF_2"].x, self.coilset.coils["PF_2"].z
        i1 = i2 = 10e6
        a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
        k = 4 * x * xc / a ** 2
        fz = (
            (MU_0 * i1 * i2 * (z - zc) * np.sqrt(k))
            / (4 * np.sqrt(x * xc))
            * (((2 - k) / (1 - k)) * ellipe(k) - 2 * ellipk(k))
        )
        assert np.isclose(fz / 1e6, np.abs(forces[0, 1]), 6)
        assert np.isclose(np.abs(forces[0, 1]), np.abs(forces[1, 1]))


if __name__ == "__main__":
    pytest.main([__file__])
