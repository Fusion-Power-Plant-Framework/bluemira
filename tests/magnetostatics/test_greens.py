# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numba as nb
import numpy as np
import pytest
from scipy.special import ellipe as scipy_ellipe
from scipy.special import ellipk as scipy_ellipk

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
from tests.magnetostatics.greens_old import greens_Bx as old_greens_Bx
from tests.magnetostatics.greens_old import greens_Bz as old_greens_Bz
from tests.magnetostatics.greens_old import greens_dbz_dx as old_greens_dbz_dx
from tests.magnetostatics.greens_old import greens_dpsi_dx as old_greens_dpsi_dx
from tests.magnetostatics.greens_old import greens_dpsi_dz as old_greens_dpsi_dz
from tests.magnetostatics.greens_old import greens_psi as old_greens_psi

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


class TestEllipticalFunctionsRegression:
    rng = np.random.default_rng(846023420)
    fixtures = [  # noqa: RUF012
        # nans
        np.nan,
        [np.nan, np.nan],
        # in valid range [0, 1)
        0.314,
        np.linspace(0.0, np.nextafter(1.0, 0.0, dtype=np.float64), 201),
        np.full((4, 3, 2), 0.42),
        # bottom of valid range
        0.0,
        np.zeros((3,)),
        np.zeros((1, 2)),
        # top of valid range
        1.0,
        np.ones((3,)),
        np.ones((1, 2)),
        # positive and out of valid range
        1.5,
        np.full((3,), 1.5),
        np.full((2, 3), 2.0),
        # in valid negative range
        np.linspace(-0.0, np.nextafter(-1, 0, dtype=np.float64), 201),
        np.full((4, 3, 2), -0.42),
        # large negative
        -100.0,
        np.full((3,), -10.0),
        # approaching range bounds
        np.array([1e-9, 1e-22]),
        np.array([1.0 - 1e-9, 1 - 1e-16]),
        np.array([-1e-9, -1e-22]),
        np.array([-1.0 + 1e-9, -1 + 1e-16]),
        # non-C-contiguous
        np.asfortranarray(np.full((4, 3), 0.5)),
        # Near limits
        np.linspace(-EPS, EPS, 201),
        np.linspace(1.0 - EPS, 1.0 + EPS, 201),
        np.linspace(2.0 - EPS, 2.0 + EPS, 201),
        # Odd balls
        np.inf,
        -np.inf,
        [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi],
    ]
    for _ in range(5):  # Tested with 80000, no failures
        fixtures.extend([
            rng.uniform(-2, 2, 10),
        ])

    @pytest.mark.parametrize("m", fixtures)
    def test_ellipe_nb(self, m):
        self.runner(ellipe_nb, scipy_ellipe, m)

    @pytest.mark.parametrize("m", fixtures)
    def test_ellipk_nb(self, m):
        self.runner(ellipk_nb, scipy_ellipk, m)

    def runner(self, new_ellip_func, old_ellip_func, m):
        new = new_ellip_func(m)
        old = old_ellip_func(m)
        np.testing.assert_allclose(new, old, rtol=0.0, atol=EPS)


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


class TestGreenFieldsRegression2:
    rng = np.random.default_rng(84602342)
    fixtures = []  # noqa: RUF012
    for i in range(5):  # Tested with 8000, 8 failures, see below
        fixtures.append(  # noqa: PERF401
            (
                i,
                10 * np.clip(rng.random(), 0.01, None),
                10 - 5 * rng.random(),
                10 * np.clip(rng.random(), 0.01, None),
                10 - 5 * rng.random(),
            )
        )

    @pytest.mark.parametrize(
        ("case", "xc", "zc", "x", "z"),
        fixtures,
    )
    @pytest.mark.parametrize(
        ("new_greens_func", "old_greens_func"),
        [
            (greens_psi, old_greens_psi),
            (greens_Bx, old_greens_Bx),
            (greens_Bz, old_greens_Bz),
            (greens_dbz_dx, old_greens_dbz_dx),
            (greens_dpsi_dx, old_greens_dpsi_dx),
            (greens_dpsi_dz, old_greens_dpsi_dz),
        ],
    )
    def test_greens_func(
        self,
        case,
        xc,
        zc,
        x,
        z,
        new_greens_func,
        old_greens_func,
    ):
        self.runner(new_greens_func, old_greens_func, case, xc, zc, x, z)

    @pytest.mark.xfail(
        reason="Approaching logarithmic singularity - differences in"
        " numerical implementation."
    )
    @pytest.mark.parametrize(
        ("new_greens_func", "old_greens_func", "case", "xc", "zc", "x", "z"),
        [
            (
                greens_psi,
                old_greens_psi,
                0,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_dpsi_dx,
                old_greens_dpsi_dx,
                1,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_dpsi_dz,
                old_greens_dpsi_dz,
                2,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_Bx,
                old_greens_Bx,
                3,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_Bz,
                old_greens_Bz,
                4,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_dbz_dx,
                old_greens_dbz_dx,
                5,
                9.6286576367563210,
                8.1240150987258204,
                9.6257638186005128,
                8.1325033392959813,
            ),
            (
                greens_dpsi_dx,
                old_greens_dpsi_dx,
                6,
                9.5983221038397453,
                6.1333202772184414,
                9.64806483606313318,
                6.1604391248090558,
            ),
            (
                greens_dpsi_dz,
                old_greens_dpsi_dz,
                8,
                9.5983221038397453,
                6.1333202772184414,
                9.64806483606313318,
                6.1604391248090558,
            ),
        ],
    )
    def test_regression_failures(
        self, new_greens_func, old_greens_func, case, xc, zc, x, z
    ):
        """
        This is to document some regression failures, which all occur xhen xc, zc -> x, z
        The maximum relative and absolute errors are still very low
        This is due to the alternative treatment of some calculations, which are more
        robust in the new implementation
        """
        self.runner(new_greens_func, old_greens_func, case, xc, zc, x, z)

    def runner(self, new_greens_func, old_greens_func, case, xc, zc, x, z):
        new = new_greens_func(xc, zc, x, z)
        old = old_greens_func(xc, zc, x, z)

        np.testing.assert_allclose(
            new,
            old,
            rtol=0.0,
            atol=EPS,
            equal_nan=True,
            err_msg=(
                f"{new_greens_func.__name__} failed on case {case}\n"
                f"xc  = {xc:.16e}\n"
                f"zc  = {zc:.16e}\n"
                f"x   = {x:.16e}\n"
                f"z   = {z:.16e}\n"
                f"new = {new:.16e}\n"
                f"old = {old:.16e}"
            ),
        )


class TestGreensEdgeCases:
    @pytest.mark.parametrize(
        "func",
        [
            greens_psi,
            greens_dpsi_dz,
            greens_dpsi_dx,
            greens_Bx,
            greens_dbz_dx,
        ],
    )
    @pytest.mark.parametrize("axis_point", [[1, 1, 0, 0]])
    def test_greens_on_axis(self, func, axis_point):
        assert func(*axis_point) == 0.0  # noqa: RUF069

    def test_greens_Bz_axis(self):
        assert greens_Bz(1, 1, 0, 0) > 0.0

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
    @pytest.mark.parametrize("origin_point", [[0, 0, 0, 0]])
    def test_greens_origin_singularity(self, func, origin_point):
        with pytest.raises(ZeroDivisionError):
            func(*origin_point)

    @pytest.mark.parametrize("func", [greens_Bx, greens_dpsi_dz, greens_dbz_dx])
    @pytest.mark.parametrize("zero_point", [[1, 1, 1, 1], [-1, -1, -1, -1]])
    def test_greens_at_same_point(self, func, zero_point):
        np.testing.assert_allclose(0.0, func(*zero_point), atol=1e-6)

    @pytest.mark.parametrize(
        "small_point", [[1, 1, 1 + 1e-4, 1 + 1e-4], [1, -1, 1 + 1e-4, -(1 + 1e-4)]]
    )
    def test_greens_at_small_distance_from_point(self, small_point):
        print(greens_dbz_dx(*small_point))
        np.testing.assert_allclose(0.00025, greens_dbz_dx(*small_point), atol=1e-6)
