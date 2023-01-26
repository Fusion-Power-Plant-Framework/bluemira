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

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0_2PI, MU_0_4PI
from bluemira.magnetostatics.greens import (
    GREENS_ZERO,
    clip_nb,
    ellipe_nb,
    ellipk_nb,
    greens_Bx,
    greens_Bz,
    greens_dpsi_dx,
    greens_dpsi_dz,
    greens_psi,
)

# Regression testing


@nb.jit(nopython=True)
def greens_Bx_old(xc, zc, x, z, d_xc=0, d_zc=0):  # noqa :N802
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
def greens_Bz_old(xc, zc, x, z, d_xc=0, d_zc=0):
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
