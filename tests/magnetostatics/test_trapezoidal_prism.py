# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.constants import EPS, raw_uc
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


class TestTrapezoidalPrismBabicAykel:
    """
    Verification test.

    Babic and Aykel example

    https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594
    """

    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([2 * 2.154700538379251, 0, 0]),  # This gives b=1
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        1,
        1,
        60.0,
        30.0,
        4e5,
    )

    @pytest.mark.parametrize(
        ("point", "value", "precision"),
        [((2, 2, 2), 15.5533805, 7), ((1, 1, 1), 53.581000397, 9)],
    )
    def test_paper_singularity_values(self, point, value, precision):
        field = self.source.field(*point)
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**precision) / 10**precision
        assert field_ndecimals == pytest.approx(value, rel=0, abs=EPS)

    def test_paper_inside_conductor(self):
        field = self.source.field(0.5, 0.5, 0.5)
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**7) / 10**7
        assert field_ndecimals == pytest.approx(34.9969156, rel=0, abs=EPS)


class TestTrapezoidalPrismCurrentSource:
    @pytest.mark.parametrize("angle", [54, 45.0001])
    def test_error_on_self_intersect(self, angle):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle,
                angle,
                current=1.0,
            )

    def test_no_error_on_triangle(self):
        TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.5,
            0.1,
            45,
            45,
            current=1.0,
        )

    @pytest.mark.parametrize("angle", [90, 180, 270, 360])
    def test_error_on_angle_limits(self, angle):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle,
                0.25 * np.pi,
                current=1.0,
            )

    @pytest.mark.parametrize(("angle1", "angle2"), [(10, -10), (-20, 30)])
    def test_error_on_mixed_sign_angles(self, angle1, angle2):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle1,
                angle2,
                current=1.0,
            )

    @pytest.mark.parametrize(("angle1", "angle2"), [(1, 1), (2, 3), (0, 2), (-2, 0)])
    def test_no_error_on_double_sign_angles(self, angle1, angle2):
        TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.5,
            0.1,
            angle1,
            angle2,
            current=1.0,
        )
