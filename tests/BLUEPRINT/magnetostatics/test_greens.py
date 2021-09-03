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
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.magnetostatics.greens import (
    greens_psi,
    greens_all,
    greens_Bx,
    greens_Bz,
)

import tests

if tests.PLOTTING:
    plot_defaults()


class TestGreens:
    def setup_method(self):
        nx, nz = 65, 65
        self.x_1d = np.linspace(0.01, 30, nx)
        self.z_1d = np.linspace(-15, 15, nz)
        self.x, self.z = np.meshgrid(self.x_1d, self.z_1d, indexing="ij")

    def test_fast_greens(self):
        """
        Test for equivalence between greens function implementations
        """
        xc, zc = [4, 4, 0.5, 20, 20], [4, -4, 0, -10, 10]
        for x, z in zip(xc, zc):
            psi = greens_psi(x, z, self.x, self.z)
            Bx = greens_Bx(x, z, self.x, self.z)
            Bz = greens_Bz(x, z, self.x, self.z)
            psif, bxf, bzf = greens_all(x, z, self.x, self.z)
            assert np.amax(np.abs(psi - psif)) < 1e-7, np.amax(np.abs(psi - psif))
            assert np.allclose(psi, psif)
            assert np.amax(np.abs(Bx - bxf)) < 1e-7, np.amax(np.abs(Bx - bxf))
            assert np.allclose(Bx, bxf)
            assert np.amax(np.abs(Bz - bzf)) < 1e-7, np.amax(np.abs(Bz - bzf))
            assert np.allclose(Bz, bzf)

    def test_psi_axis(self):
        for z in [-10, -5, 0, 5, 10]:
            assert greens_psi(4, 4, 0, z) == 0


if __name__ == "__main__":
    pytest.main([__file__])
