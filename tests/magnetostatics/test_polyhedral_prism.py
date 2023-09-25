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
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def test_benchmark():
    """
    Verification test.

    Benchmarked against cube using trapezoidal current source with 45 deg offset.
    """
    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([0, 0, 4]),
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        0.5,
        0.5,
        0,
        0,
        1e6,
    )
    source.rotate(45, "z")
    source2 = PolyhedralPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([0, 4, 0]),
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        4,
        4,
        np.sqrt(0.5 * 1.0**2),
        0,
        0,
        1e6,
        100,
    )
    field = source.field(2, 0, 4)
    field2 = source2.field(2, 0, 4)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    abs_field2 = raw_uc(np.sqrt(sum(field2**2)), "T", "mT")  # Field in mT
    # Assume truncated last digit and not rounded...
    field_7decimals = np.trunc(abs_field * 10**6) / 10**6
    field_7decimals2 = np.trunc(abs_field2 * 10**6) / 10**6
    assert field_7decimals == field_7decimals2

    # Test singularity treatments:
    field = source.field(np.sqrt(0.5 * 1.0**2), 0, 2)
    field2 = source2.field(np.sqrt(0.5 * 1.0**2), 0, 2)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    abs_field2 = raw_uc(np.sqrt(sum(field2**2)), "T", "mT")  # Field in mT
    # Assume truncated last digit and not rounded...
    field_6decimals = np.trunc(abs_field * 10**6) / 10**6
    field_6decimals2 = np.trunc(abs_field2 * 10**6) / 10**6
    assert field_6decimals == field_6decimals2


class TestPolyhedralPrismCurrentSource:
    @pytest.mark.parametrize("angle", [-54, -45.0001])
    def test_error_on_self_intersect(self, angle):
        with pytest.raises(MagnetostaticsError):
            PolyhedralPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                4,
                2,
                np.sqrt(0.5 * 1.0**2),
                angle,
                angle,
                1e6,
                5,
            )

    def test_no_error_on_triangle(self):
        PolyhedralPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            4,
            2,
            np.sqrt(0.5 * 1.0**2),
            -45,
            -45,
            1e6,
            5,
        )

    @pytest.mark.parametrize("angle", [90, 180, 270, 360])
    def test_error_on_angle_limits(self, angle):
        with pytest.raises(MagnetostaticsError):
            PolyhedralPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                4,
                2,
                np.sqrt(0.5 * 1.0**2),
                angle,
                0,
                1e6,
                5,
            )

    @pytest.mark.parametrize("angle1,angle2", [[10, -10], [-20, 30]])
    def test_error_on_mixed_sign_angles(self, angle1, angle2):
        with pytest.raises(MagnetostaticsError):
            PolyhedralPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([1, 1, 0]),
                4,
                2,
                np.sqrt(0.5 * 1.0**2),
                angle1,
                angle2,
                1e6,
                5,
            )

    @pytest.mark.parametrize("angle1,angle2", [[1, 1], [2, 3], [0, 2], [-2, 0]])
    def test_no_error_on_double_sign_angles(self, angle1, angle2):
        PolyhedralPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            4,
            2,
            np.sqrt(0.5 * 1.0**2),
            angle1,
            angle2,
            1e6,
            5,
        )
