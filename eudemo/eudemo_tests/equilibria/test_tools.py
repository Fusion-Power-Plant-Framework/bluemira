# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from eudemo.equilibria.tools import estimate_kappa95


class TestKappaLaw:
    """
    As per the conclusions of the CREATE report 2L4NMJ
    """

    @pytest.mark.parametrize(
        ("A", "m_s", "expected"),
        [
            (3.6, 0.3, 1.58),
            (3.1, 0.3, 1.68),
            (2.6, 0.3, 1.73),
            (3.6, 0, 1.66),
            (3.1, 0, 1.77),
            (2.6, 0, 1.80),
        ],
    )
    def test_kappa(self, A, m_s, expected):
        k95 = estimate_kappa95(A, m_s)
        np.testing.assert_allclose(k95, expected, rtol=5e-3)
