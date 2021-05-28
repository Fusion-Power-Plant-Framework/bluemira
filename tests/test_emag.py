# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Tests for mirapy.emag

@author: ivan
"""
import pytest

import mirapy.emag as emag
import numpy as np


class TestEmagMethods:
    def test_Utils_Bteo_coil(self):
        """
        Test B calculation of a coil along the axis
        """
        rc = 1.0
        zc = 0.0
        pr = 0.0
        pz = 0.0
        Ic = 1.0e6

        B = emag.Utils.Bteo_coil(rc, zc, pr, pz, Ic)
        B == pytest.approx(0.6283185307179)

    def test_Green_calculatePsi(self):
        """
        Test Green.calculatePsi method
        """
        Rc = np.array([1, 2])
        Zc = np.array([0, 0])
        R = np.array([0, 1, 2])
        Z = np.array([0, 1, 0])

        psi = emag.Greens.calculatePsi(Rc, Zc, R, Z)
        assert len(psi) == 3


if __name__ == "__main__":
    pytest.main([__file__])
