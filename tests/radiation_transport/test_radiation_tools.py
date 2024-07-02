# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.radiation_tools import (
    calculate_line_radiation_loss,
    calculate_z_species,
    exponential_decay,
    filtering_in_or_out,
    gaussian_decay,
    interpolated_field_values,
    linear_interpolator,
)


def test_gaussian_decay():
    decayed_val = gaussian_decay(10, 1, 50)
    gap_1 = decayed_val[0] - decayed_val[1]
    gap_2 = decayed_val[1] - decayed_val[2]
    gap_3 = decayed_val[-2] - decayed_val[-1]
    assert gap_1 < gap_2 < gap_3


def test_exponential_decay():
    decayed_val = exponential_decay(10, 1, 50, decay=True)
    gap_1 = decayed_val[0] - decayed_val[1]
    gap_2 = decayed_val[1] - decayed_val[2]
    gap_3 = decayed_val[-2] - decayed_val[-1]
    assert gap_1 > gap_2 > gap_3


def test_calculate_z_species():
    t_ref = np.array([0, 10])
    z_ref = np.array([10, 20])
    frac = 0.1
    t_test = 5
    z = calculate_z_species(t_ref, z_ref, frac, t_test)
    assert z == pytest.approx(22.5)


def test_calculate_line_radiation_loss():
    ne = 1e20
    p_loss = 1e-31
    frac = 0.01
    rad = calculate_line_radiation_loss(ne, p_loss, frac)
    assert rad == pytest.approx(0.796, abs=1e-3)


def test_interpolated_field_values():
    x = np.array([0, 1, 1, 0])
    z = np.array([0, 0, 1, 1])
    field = np.array([0, 1, 2, 3])

    interpolator = linear_interpolator(x, z, field)
    x_new = np.array([0.5, 1.5])
    z_new = np.array([0.5, 1.5])

    field_grid = interpolated_field_values(x_new, z_new, interpolator)

    assert field_grid.shape == (2, 2)
    assert np.isclose(field_grid[0, 0], interpolator(0.5, 0.5))
    assert np.isclose(field_grid[1, 1], interpolator(1.5, 1.5))


def test_filtering_in_or_out():
    test_domain_x = [1.0, 2.0, 3.0]
    test_domain_z = [4.0, 5.0, 6.0]

    include_func = filtering_in_or_out(test_domain_x, test_domain_z, include_points=True)
    assert include_func([2.0, 5.0])
    assert not include_func([0.0, 0.0])

    exclude_func = filtering_in_or_out(
        test_domain_x, test_domain_z, include_points=False
    )
    assert not exclude_func([2.0, 5.0])
    assert exclude_func([0.0, 0.0])
