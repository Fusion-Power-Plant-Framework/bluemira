# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.radiation_transport.radiation_tools import (
    calculate_line_radiation_loss,
    calculate_total_radiated_power,
    calculate_zeff,
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


def test_calculate_line_radiation_loss():
    ne = 1e20
    p_loss = 1e-31
    frac = 0.01
    rad = calculate_line_radiation_loss(ne, p_loss, frac)
    assert rad == pytest.approx(10, abs=1e-3)


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


def test_calculate_zeff():
    """
    Test the calculate_zeff function with known inputs and expected outputs.
    """
    # Inputs
    impurities_content = np.array([0.01])
    imp_data_z_ref = [np.array([1, 2, 3])]
    imp_data_t_ref = [np.array([0, 5, 10])]
    impurity_symbols = np.array(["C"])
    te = [np.array([0, 5, 10])]

    # Calling function
    zeff, avg_zeff, total_fraction, intermediate_values = calculate_zeff(
        impurities_content, imp_data_z_ref, imp_data_t_ref, impurity_symbols, te
    )

    # Expected outputs
    expected_zeff = np.array([1.0, 2.0, 3.0])
    expected_avg_zeff = 2.0
    expected_total_fraction = 0.01
    expected_species_fractions = [0.01]
    expected_species_zi = [2.0]
    expected_symbols = ["C"]

    # Assertions
    assert np.allclose(zeff, expected_zeff)
    assert np.isclose(avg_zeff, expected_avg_zeff)
    assert np.isclose(total_fraction, expected_total_fraction)
    assert intermediate_values["species_fractions"] == expected_species_fractions
    assert np.allclose(intermediate_values["species_zi"], expected_species_zi)
    assert intermediate_values["symbols"] == expected_symbols


def test_calculate_total_radiated_power():
    """
    Test the calculate_total_radiated_power function
    """
    # Define x and z coordinates forming a rectangle
    x = np.array([1.0, 1.0, 2.0, 2.0])
    z = np.array([1.0, 2.0, 1.0, 2.0])

    # Define p_rad as a constant value over the rectangle
    p_rad = np.array([1.0, 1.0, 1.0, 1.0])

    # Calling the function
    p_total = calculate_total_radiated_power(x, z, p_rad)

    # Expected total radiated power
    expected_volume = np.pi * (2.0**2 - 1.0**2) * (2.0 - 1.0)
    expected_p_total = expected_volume * 1.0

    # Assertion
    assert p_total == pytest.approx(expected_p_total, rel=1e-6)
