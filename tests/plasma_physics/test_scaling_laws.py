# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.plasma_physics.scaling_laws import P_LH, IPB98y2, lambda_q


class TestLambdaQScaling:
    def test_ITER(self):
        """
        As per https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.215001
        """
        lambda_q_value, min_v, max_v = lambda_q(5.3, 2.42, 120e6, 6.2, error=True)
        # In the paper above, the conclusions section give 0.94 mm for the extrapolation,
        # which I do not find (~ 0.96 mm) but there are additional terms which are not
        # listed in the regression (elongation, Z_eff, I_p)
        np.testing.assert_almost_equal(lambda_q_value, 0.94e-3, decimal=4)

        assert min_v < lambda_q_value < max_v


class TestPLH:
    def test_ITER(self):
        """
        As per https://infoscience.epfl.ch/record/135655/files/1742-6596_123_1_012033.pdf
        """
        p_lh_threshold, min_v, max_v = P_LH(1e20, 5.3, 3.1, 6.2, error=True)
        # No value provided for this scaling (3), but ballpark is right
        np.testing.assert_approx_equal(p_lh_threshold, 90e6, significant=1)
        assert min_v < p_lh_threshold < max_v


class TestIPB98y2:
    def test_ITER(self):
        """
        As per I dunno
        """

        value = IPB98y2(15e6, 5.3, 100e6, 1e20, 2.5, 6.2, 3.1, 1.59)

        assert round(value, 0) == pytest.approx(3.0, rel=0, abs=EPS)
