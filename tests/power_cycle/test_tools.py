# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import timeit as ti
from functools import partial

import numpy as np
import pytest

from bluemira.power_cycle.tools import match_domains, unique_domain


@pytest.mark.parametrize("epsilon", [1e-10, 1e-7, 1e-5])
def test_unique_domain(epsilon):
    x = np.array([1, 1, 2, 2, 3, 3])
    new_x = unique_domain(x, epsilon=epsilon)
    expected_x = np.array([
        1,
        1 + epsilon,
        2,
        2 + epsilon,
        3,
        3 + epsilon,
    ])
    assert np.array_equal(new_x, expected_x)


@pytest.mark.parametrize("epsilon", [1e-10, 1e-7, 1e-5])
def test_timing_unique_domain(epsilon):
    n_repeat = 1000
    times = ti.Timer(partial(test_unique_domain, epsilon)).repeat(
        repeat=10, number=n_repeat
    )
    avg_time = min(times) / n_repeat
    print(f"For epsilon={epsilon}, avg.'unique_domain' runtime: {avg_time} s")
    assert avg_time < 1


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
