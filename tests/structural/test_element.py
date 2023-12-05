# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

from bluemira.structural.element import _k_array


class TestK:
    def test_shape(self):
        k = _k_array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert k.shape == (12, 12)
        k2 = k.T
        for i in range(12):
            for j in range(12):
                assert k[i, j] == k2[j, i], f"{i}, {j}"

        # Check array is symmetric
        assert np.allclose(k, k.T, rtol=1e-9)
