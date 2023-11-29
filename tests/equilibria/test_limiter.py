# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numpy as np

from bluemira.equilibria.limiter import Limiter


def test_limiter():
    x = [1, 2, 3, 4]
    z = [0, -2, 0, 2]
    limiter = Limiter(x, z)

    assert len(limiter) == 4

    lims = [[1, 0], [2, -2], [3, 0], [4, 2]]
    for i, lim in enumerate(limiter):
        assert (lim == np.array(lims[i])).all()
