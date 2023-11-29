# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.plasma_physics.rules_of_thumb import (
    calc_cyl_safety_factor,
    calc_qstar_freidberg,
    calc_qstar_uckan,
    estimate_Le,
    estimate_M,
)


class TestHirshmanInductanceRules:
    # Table I of
    # https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma  # noqa: W505,E501
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    a = np.array([2.975, 2.285, 1.848, 1.507, 1.217, 0.957, 0.716, 0.487, 0.272])
    b = np.array([0.228, 0.325, 0.403, 0.465, 0.512, 0.542, 0.553, 0.538, 0.508])
    c = np.array([1.008, 1.038, 1.093, 1.177, 1.301, 1.486, 1.769, 2.223, 2.864])
    d = np.array([0.022, 0.056, 0.087, 0.113, 0.134, 0.148, 0.155, 0.152, 0.133])

    @pytest.mark.parametrize("kappa", [1.0, 2.0])
    def test_le(self, kappa):
        le_table = self.a * (1 - self.eps) / (1 - self.eps + self.b * kappa)
        le = estimate_Le(1 / self.eps, kappa=kappa)
        np.testing.assert_allclose(le, le_table, rtol=3.1e-2)

    @pytest.mark.parametrize("kappa", [1.0, 2.0])
    def test_M(self, kappa):
        m_table = (1 - self.eps) ** 2 / (
            (1 - self.eps) ** 2 * self.c + self.d * np.sqrt(kappa)
        )
        m = estimate_M(1 / self.eps, kappa=kappa)
        np.testing.assert_allclose(m_table, m, rtol=1.1e-2)


class TestSafetyFactors:
    def test_uckan_is_cylindrical(self):
        R_0, A, B_0, I_p = 9, 3, 6, 20e6
        qstar_uckan = calc_qstar_uckan(R_0, A, B_0, I_p, 1.0, 0.0)
        qstar_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
        np.testing.assert_almost_equal(qstar_uckan, qstar_cyl)

    def test_freidberg_is_cylindrical(self):
        R_0, A, B_0, I_p = 9, 3, 6, 20e6
        qstar_uckan = calc_qstar_freidberg(R_0, A, B_0, I_p, 1.0)
        qstar_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
        np.testing.assert_almost_equal(qstar_uckan, qstar_cyl)
