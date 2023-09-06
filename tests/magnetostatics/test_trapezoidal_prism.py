# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import numpy as np
import pytest

from bluemira.base.constants import raw_uc
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def test_paper_example():
    """
    Verification test.

    Babic and Aykel example

    https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594?saml_referrer=
    """
    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([2 * 2.154700538379251, 0, 0]),  # This gives b=1
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        1,
        1,
        np.pi / 3,
        np.pi / 6,
        4e5,
    )
    field = source.field(2, 2, 2)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    # As per Babic and Aykel paper
    # Assume truncated last digit and not rounded...
    field_7decimals = np.trunc(abs_field * 10**7) / 10**7
    field_7true = 15.5533805
    assert field_7decimals == field_7true

    # Test singularity treatments:
    field = source.field(1, 1, 1)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    # Assume truncated last digit and not rounded...
    field_9decimals = np.trunc(abs_field * 10**9) / 10**9
    field_9true = 53.581000397
    assert field_9decimals == field_9true


class TestTrapezoidalPrismCurrentSource:
    def test_error_on_self_intersect(self):
        with pytest.raises(MagnetostaticsError):
            source = TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                np.pi / 2,
                np.pi / 2,
                current=1.0,
            )
