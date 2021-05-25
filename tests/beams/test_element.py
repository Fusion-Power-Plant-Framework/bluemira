# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
import pytest
import numpy as np
from BLUEPRINT.beams.element import _k_array


class Testk:
    def test_shape(self):
        k = _k_array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        assert k.shape == (12, 12)
        k2 = k.T
        for i in range(12):
            for j in range(12):
                assert k[i, j] == k2[j, i], f"{i}, {j}"

        # Check array is symmetric
        assert np.allclose(k, k.T, rtol=1e-9)
        # from matplotlib import pyplot as plt
        # f, ax = plt.subplots()
        # ax.matshow(k)


if __name__ == "__main__":
    pytest.main([__file__])
