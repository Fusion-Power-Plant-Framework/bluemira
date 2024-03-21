# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
from scipy.special import ellipe, ellipk

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.profiles import CustomProfile


class TestForceField:
    @classmethod
    def setup_class(cls):
        coils = []

        x = [5, 5]
        z = [5, -5]
        for i, (xi, zi) in enumerate(zip(x, z, strict=False)):
            c = Coil(xi, zi, current=10e6, ctype="PF", name=f"PF_{i + 1}", dx=0, dz=0)

            coils.append(c)
        cls.coilset = CoilSet(*coils)
        cls.eq = Equilibrium(
            cls.coilset,
            Grid(0.1, 10, -10, 10, 10, 10),
            CustomProfile(np.linspace(0, 1, 10), np.linspace(1, 0, 10), 9, 6, I_p=0.0),
        )

    def test_Fz(self):
        """
        Check the vertical forces between a Helmholtz pair.
        Verbose: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6472319
        """
        forces = self.eq.get_coil_forces()
        (x, xc), (z, zc) = self.coilset.x, self.coilset.z
        i1 = i2 = 10e6
        a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
        k = 4 * x * xc / a**2
        fz = (
            (MU_0 * i1 * i2 * (z - zc) * np.sqrt(k))
            / (4 * np.sqrt(x * xc))
            * (((2 - k) / (1 - k)) * ellipe(k) - 2 * ellipk(k))
        )
        assert np.isclose(fz / 1e6, np.abs(forces[0, 1]), 6)
        assert np.isclose(np.abs(forces[0, 1]), np.abs(forces[1, 1]))
