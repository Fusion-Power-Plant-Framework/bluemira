# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numba as nb
import numpy as np
import pytest

from bluemira.base.constants import EPS, MU_0_2PI, MU_0_4PI
from bluemira.magnetostatics.greens import (
    GREENS_ZERO,
    clip_nb,
    ellipe_nb,
    ellipk_nb,
    greens_Bx,
    greens_Bz,
    greens_all,
    greens_dbz_dx,
    greens_dpsi_dx,
    greens_dpsi_dz,
    greens_psi,
)

# Regression testing


@nb.jit(nopython=True)
def greens_Bx_old(xc, zc, x, z, d_xc=0, d_zc=0):  # noqa: ARG001
    """
    Calculate radial magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc: float or 1-D array
        Coil x coordinates [m]
    zc: float or 1-D array
        Coil z coordinates [m]
    x: float or np.array(N, M)
        Calculation x locations
    z: float or np.array(N, M)
        Calculation z
    d_xc: float
        The coil half-width (overload argument)
    d_zc: float
        The coil half-height (overload argument)

    Returns
    -------
    Bx: float or np.array(N, M)
        Radial magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0
        if x <= 0
    """
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    i1 = ellipk_nb(k2) / a
    i2 = ellipe_nb(k2) / (a**3 * (1 - k2))
    return MU_0_2PI * ((z - zc) * (-i1 + i2 * ((z - zc) ** 2 + x**2 + xc**2)) / x)


@nb.jit(nopython=True)
def greens_Bz_old(xc, zc, x, z, d_xc=0, d_zc=0):  # noqa: ARG001
    """
    Calculate vertical magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc: float or 1-D array
        Coil x coordinates [m]
    zc: float or 1-D array
        Coil z coordinates [m]
    x: float or np.array(N, M)
        Calculation x locations
    z: float or np.array(N, M)
        Calculation z locations
    d_xc: float
        The coil half-width (overload argument)
    d_zc: float
        The coil half-height (overload argument)

    Returns
    -------
    Bz: float or np.array(N, M)
        Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0
        if x <= 0
    """
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    e, k = ellipe_nb(k2), ellipk_nb(k2)
    i1 = 4 * k / a
    i2 = 4 * e / (a**3 * (1 - k2))
    part_a = (z - zc) ** 2 + x**2 + xc**2
    part_b = -2 * x * xc
    return MU_0_4PI * xc * ((xc + x * part_a / part_b) * i2 - i1 * x / part_b)


def test_greens_vs_greens_all():
    nx, nz = 100, 100
    x_coil, z_coil = 4, 0

    x_1d = np.linspace(0.1, 10, nx)
    z_1d = np.linspace(-5, 5, nz)
    x_2d, z_2d = np.meshgrid(x_1d, z_1d, indexing="ij")

    # Analytical field values
    psi, Bx, Bz = greens_all(x_coil, z_coil, x_2d, z_2d)
    Bx2 = greens_Bx(x_coil, z_coil, x_2d, z_2d)
    Bz2 = greens_Bz(x_coil, z_coil, x_2d, z_2d)
    psi2 = greens_psi(x_coil, z_coil, x_2d, z_2d)

    np.testing.assert_allclose(Bx, Bx2)
    np.testing.assert_allclose(Bz, Bz2)
    np.testing.assert_allclose(psi, psi2)


class TestGreenFieldsRegression:
    rng = np.random.default_rng(846023420)
    fixtures = []  # noqa: RUF012
    for _ in range(5):  # Tested with 8000, no failures
        fixtures.append(  # noqa: PERF401
            [
                10 * np.clip(rng.random(), 0.01, None),
                10 - 5 * rng.random(),
                10 * np.clip(rng.random((100, 100)), 0.01, None),
                10 - 5 * rng.random((100, 100)),
            ]
        )

    @pytest.mark.parametrize(("xc", "zc", "x", "z"), fixtures)
    def test_greens_Bx(self, xc, zc, x, z):
        self.runner(greens_Bx, greens_Bx_old, xc, zc, x, z)

    @pytest.mark.parametrize(("xc", "zc", "x", "z"), fixtures)
    def test_greens_Bz(self, xc, zc, x, z):
        self.runner(greens_Bz, greens_Bz_old, xc, zc, x, z)

    def runner(self, new_greens_func, old_greens_func, xc, zc, x, z):
        new = new_greens_func(xc, zc, x, z)
        old = old_greens_func(xc, zc, x, z)
        np.testing.assert_allclose(1e7 * new, 1e7 * old, atol=EPS)


class TestGreensEdgeCases:
    @pytest.mark.parametrize(
        "func",
        [
            greens_psi,
            greens_dpsi_dz,
            greens_dpsi_dx,
            greens_Bx,
            greens_Bz,
            greens_dbz_dx,
        ],
    )
    @pytest.mark.parametrize("fail_point", [[0, 0, 0, 0]])
    def test_greens_on_axis(self, func, fail_point):
        with pytest.raises(ZeroDivisionError):
            func(*fail_point)

    @pytest.mark.parametrize("func", [greens_Bx, greens_Bz, greens_dbz_dx])
    @pytest.mark.parametrize("fail_point", [[1, 1, 0, 10], [-1, -1, 0, 10]])
    def test_greens_on_axis_field(self, func, fail_point):
        with pytest.raises(ZeroDivisionError):
            func(*fail_point)

    @pytest.mark.parametrize("func", [greens_Bx, greens_dpsi_dz, greens_dbz_dx])
    @pytest.mark.parametrize("zero_point", [[1, 1, 1, 1], [-1, -1, -1, -1]])
    def test_greens_at_same_point(self, func, zero_point):
        np.testing.assert_allclose(0.0, func(*zero_point), atol=1e-6)

    @pytest.mark.parametrize(
        "small_point", [[1, 1, 1 + 1e-4, 1 + 1e-4], [-1, -1, -(1 + 1e-4), -(1 + 1e-4)]]
    )
    def test_greens_at_small_distance_from_point(self, small_point):
        print(greens_dbz_dx(*small_point))
        np.testing.assert_allclose(0.00025, greens_dbz_dx(*small_point), atol=1e-6)
