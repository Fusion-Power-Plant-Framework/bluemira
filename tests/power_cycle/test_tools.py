# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from bluemira.power_cycle.tools import match_domains, unique_domain


@pytest.mark.parametrize("epslon", [1e-10, 1e-7, 1e-5])
def test_unique_domain(epslon):
    x = np.array([1, 1, 2, 2, 3, 3])
    y = np.array([1, 2, 3, 4, 5, 6])

    new_x, new_y = unique_domain(x, y, epslon=epslon, mode="careful")
    slow_x = np.array([
        1,
        1 + epslon,
        2,
        2 + epslon,
        3,
        3 + epslon,
    ])
    assert np.array_equal(new_x, slow_x)
    assert np.array_equal(new_y, y)

    new_x, new_y = unique_domain(x, y, epslon=epslon, mode="fast")
    fast_x = np.array([
        1 + 0 * epslon,
        1 + 1 * epslon,
        2 + 2 * epslon,
        2 + 3 * epslon,
        3 + 4 * epslon,
        3 + 5 * epslon,
    ])
    assert np.array_equal(new_x, fast_x)
    assert np.array_equal(new_y, y)

    with pytest.raises(ValueError, match="Invalid argument"):
        unique_domain(x, y, mode="invalid_mode")


def test_match_domains():
    x_set = [np.array([1, 2, 3]), np.array([2, 3, 4])]
    y_set = [np.array([1, 2, 3]), np.array([4, 6, 8])]
    out_of_bounds_value = 0

    expected_x = np.array([1, 2, 3, 4])
    expected_y_set = [
        np.array([1, 2, 3, out_of_bounds_value]),
        np.array([out_of_bounds_value, 4, 6, 8]),
    ]
    matched_x, matched_y_set = match_domains(x_set, y_set)
    assert np.array_equal(matched_x, expected_x)
    assert np.array_equal(matched_y_set[0], expected_y_set[0])
    assert np.array_equal(matched_y_set[1], expected_y_set[1])
