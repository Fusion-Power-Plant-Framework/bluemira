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


import pytest
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


if __name__ == "__main__":
    pytest.main([__file__])
