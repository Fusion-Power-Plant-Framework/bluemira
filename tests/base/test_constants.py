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

from BLUEPRINT.base.constants import E_IJK, E_IJ, E_I
from BLUEPRINT.utilities.tools import levi_civita_tensor


def test_lct_constants():
    for i, lct in enumerate([E_I, E_IJ, E_IJK], start=1):
        np.testing.assert_equal(lct, levi_civita_tensor(dim=i))


if __name__ == "__main__":
    pytest.main([__file__])
