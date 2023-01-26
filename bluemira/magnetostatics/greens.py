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

"""
Green's functions mappings for psi, Bx, and Bz
"""
import numba as nb
import numpy as np
from scipy.special import ellipe, ellipk

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI

__all__ = [
    "greens_psi",
    "greens_Bx",
    "greens_Bz",
    "greens_dpsi_dx",
    "greens_dpsi_dz",
    "greens_all",
]

# Offset from 0<x<1
#     Used in calculating Green's functions to avoid np.nan
GREENS_ZERO = 1e-8


@nb.vectorize(nopython=True, cache=True)
def clip_nb(val, val_min, val_max):
    """
    Clips (limits) val between val_min and val_max. Vectorised for speed and
    compatibility with numba.

    Parameters
    ----------
    val: float
        The value to be clipped.
    val_min: float
        The minimum value.
    val_max: float
        The maximum value.

    Returns
    -------
    clipped_val: float
        The clipped values.
    """
    if val < val_min:
        return val_min
    elif val > val_max:
        return val_max
    return val


@nb.vectorize(nopython=True)
def ellipe_nb(k):
    """
    Vectorised scipy ellipe
    """
    return ellipe(k)


ellipe_nb.__doc__ += ellipe.__doc__


@nb.vectorize(nopython=True)
def ellipk_nb(k):
    """
    Vectorised scipy ellipk
    """
    return ellipk(k)


ellipk_nb.__doc__ += ellipk.__doc__


def circular_coil_inductance_elliptic(radius, rc):
    """
    Calculate the inductance of a circular coil by elliptic integrals.

    Parameters
    ----------
    radius: float
        The radius of the circular coil
    rc: float
        The radius of the coil cross-section

    Returns
    -------
    inductance: float
        The self-inductance of the circular coil [H]
    """
    k = 4 * radius * (radius - rc) / (2 * radius - rc) ** 2
    args = (
        np.array(arg, dtype=np.float64) for arg in (k, GREENS_ZERO, 1.0 - GREENS_ZERO)
    )
    k = clip_nb(*args)
    return MU_0 * (2 * radius - rc) * ((1 - k**2 / 2) * ellipk(k) - ellipe(k))


def circular_coil_inductance_kirchhoff(radius, rc):
    """
    Calculate the inductance of a circular coil by Kirchhoff's approximation.

    radius: float
        The radius of the circular coil
    rc: float
        The radius of the coil cross-section

    Returns
    -------
    inductance: float
        The self-inductance of the circular coil [H]
    """
    return MU_0 * radius * (np.log(8 * radius / rc) - 2 + 0.25)


@nb.jit(nopython=True)
def greens_psi(xc, zc, x, z, d_xc=0, d_zc=0):
    """
    Calculate poloidal flux at (x, z) due to a unit current at (xc, zc)
    using Greens function.

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
    psi: float or np.array(N, M)
        Poloidal magnetic flux per radian response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0

    Notes
    -----
    \t:math:`{\\psi}(x, z)=\\int\\int G(x, z; x_{c}, z_{c})J_{{\\phi}}`
    \t:math:`(x_{c}, z_{c})dx_{c}dz_{c}`\n
    Where:
    \t:math:`G(x, z; x_{c}, z_{c})=\\dfrac{{\\mu}_{0}}{2{\\pi}}`
    \t:math:`\\dfrac{\\sqrt{xx_{c}}}{k}`
    \t:math:`[(2-\\mathbf{K}(k)-2\\mathbf{E}(k)]`\n
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`J_{\\phi}(x_{c}, z_{c}) = 1`
    """
    k2 = 4 * x * xc / ((x + xc) ** 2 + (z - zc) ** 2)
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    # K, E in scipy are set as K(k^2), E(k^2)
    return (
        MU_0_2PI
        * np.sqrt(x * xc)
        * ((2 - k2) * ellipk_nb(k2) - 2 * ellipe_nb(k2))
        / np.sqrt(k2)
    )


@nb.jit(nopython=True)
def greens_dpsi_dx(xc, zc, x, z, d_xc=0, d_zc=0):
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    i1 = ellipk_nb(k2) / a
    i2 = ellipe_nb(k2) / (a**3 * (1 - k2))
    return MU_0_2PI * x * ((xc**2 - (z - zc) ** 2 - x**2) * i2 + i1)

    h2 = (z - zc) ** 2
    u2 = (x + xc) ** 2 + h2
    k2 = 4 * x * xc / u2
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    d2 = (xc - x) ** 2 + h2
    w2 = xc**2 - x**2 - h2
    return MU_0_2PI * (x / np.sqrt(u2)) * (w2 / d2 * ellipe_nb(k2) + ellipk_nb(k2))


@nb.jit(nopython=True)
def greens_dpsi_dz(xc, zc, x, z, d_xc=0, d_zc=0):
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    i1 = ellipk_nb(k2) / a
    i2 = ellipe_nb(k2) / (a**3 * (1 - k2))
    return MU_0_2PI * ((z - zc) * (i1 - i2 * ((z - zc) ** 2 + x**2 + xc**2)))
    h = z - zc
    h2 = h**2
    u2 = (x + xc) ** 2 + h2
    k2 = 4 * x * xc / u2
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    v2 = x**2 + xc**2 + h2
    d2 = (xc - x) ** 2 + h2
    return MU_0_2PI * (h / np.sqrt(u2)) * (ellipk_nb(k2) - v2 / d2 * ellipe_nb(k2))


@nb.jit(nopython=True)
def greens_Bx(xc, zc, x, z, d_xc=0, d_zc=0):  # noqa :N802
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
    return -1 / x * greens_dpsi_dz(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_Bz(xc, zc, x, z, d_xc=0, d_zc=0):  # noqa :N802
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
    return 1 / x * greens_dpsi_dx(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_all(xc, zc, x, z):
    """
    Speed optimisation of Green's functions for psi, Bx, and Bz

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

    Returns
    -------
    psi: float or np.array(N, M)
        Poloidal magnetic flux per radian response at (x, z)
    Bx: float or np.array(N, M)
        Radial magnetic field response at (x, z)
    Bz: float or np.array(N, M)
        Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0
        if x <= 0
    """
    a = np.hypot((x + xc), (z - zc))
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    e, k = ellipe_nb(k2), ellipk_nb(k2)
    i_1 = 4 * k / a
    i_2 = 4 * e / (a**3 * (1 - k2))
    a_part = (z - zc) ** 2 + x**2 + xc**2
    b_part = -2 * x * xc
    g_bx = MU_0_4PI * xc * (z - zc) * (i_1 - i_2 * a_part) / b_part
    g_bz = MU_0_4PI * xc * ((xc + x * a_part / b_part) * i_2 - i_1 * x / b_part)
    g_psi = MU_0_4PI * a * ((2 - k2) * k - 2 * e)
    return g_psi, g_bx, g_bz
