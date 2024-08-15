# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Green's functions mappings for psi, Bx, and Bz
"""

import numba as nb
import numpy as np
from scipy.special import ellipe, ellipk

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI

__all__ = [
    "greens_Bx",
    "greens_Bz",
    "greens_all",
    "greens_dpsi_dx",
    "greens_dpsi_dz",
    "greens_psi",
]

# Offset from 0<x<1
#     Used in calculating Green's functions to avoid np.nan
GREENS_ZERO = 1e-8


@nb.vectorize(nopython=True, cache=True)
def clip_nb(
    val: float | np.ndarray, val_min: float, val_max: float
) -> float | np.ndarray:
    """
    Clips (limits) val between val_min and val_max. Vectorised for speed and
    compatibility with numba.

    Parameters
    ----------
    val:
        The value to be clipped.
    val_min:
        The minimum value.
    val_max:
        The maximum value.

    Returns
    -------
    The clipped values.
    """
    if val < val_min:
        return val_min
    if val > val_max:
        return val_max
    return val


@nb.vectorize(nopython=True)
def ellipe_nb(k: float | np.ndarray) -> float | np.ndarray:
    """
    Vectorised scipy ellipe

    Notes
    -----
    K, E in scipy are set as K(k^2), E(k^2)
    """
    return ellipe(k)


ellipe_nb.__doc__ += ellipe.__doc__


@nb.vectorize(nopython=True)
def ellipk_nb(k: float | np.ndarray) -> float | np.ndarray:
    """
    Vectorised scipy ellipk

    Notes
    -----
    K, E in scipy are set as K(k^2), E(k^2)
    """
    return ellipk(k)


ellipk_nb.__doc__ += ellipk.__doc__


@nb.jit(nopython=True)
def circular_coil_inductance_elliptic(radius: float, rc: float) -> float:
    """
    Calculate the inductance of a circular coil by elliptic integrals.

    Parameters
    ----------
    radius:
        The radius of the circular coil
    rc:
        The radius of the coil cross-section

    Returns
    -------
    The self-inductance of the circular coil [H]

    Notes
    -----
    The inductance is given by

    .. math::
        L = \\mu_{0} (2 r - r_c) \\Biggl((1 - k^2 / 2)~
        \\int_0^{\\frac{\\pi}{2}} \\frac{d\\theta}{\\sqrt{1 - k~
        \\sin (\\theta)^2}} - \\int_0^{\\frac{\\pi}{2}}~
        \\sqrt{1 - k \\sin (\\theta)^2} \\, d\\theta\\Biggr)

    where :math:`r` is the radius, :math:`\\mu_{0}` is the vacuum
    permeability, and

    .. math::
        k = \\max\\left(10^{-8}, \\min~
        \\left(\\frac{4r(r - r_c)}{(2r - r_c)^2}~
        , 1.0 - 10^{-8}\\right)\\right)
    """
    k = 4 * radius * (radius - rc) / (2 * radius - rc) ** 2
    k = clip_nb(k, GREENS_ZERO, 1.0 - GREENS_ZERO)
    return MU_0 * (2 * radius - rc) * ((1 - k**2 / 2) * ellipk_nb(k) - ellipe_nb(k))


def circular_coil_inductance_kirchhoff(radius: float, rc: float) -> float:
    """
    Calculate the inductance of a circular coil by Kirchhoff's approximation.

    radius:
        The radius of the circular coil
    rc:
        The radius of the coil cross-section

    Returns
    -------
    The self-inductance of the circular coil [H]

    Notes
    -----

    .. math::
        Inductance = \\mu_{0} * radius * (log(8 * radius / rc) - 2 + 0.25)

    where :math:`\\mu_{0}` is the vacuum permeability
    """
    return MU_0 * radius * (np.log(8 * radius / rc) - 2 + 0.25)


@nb.jit(nopython=True)
def greens_psi(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> float | np.ndarray:
    """
    Calculate poloidal flux at (x, z) due to a unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Poloidal magnetic flux per radian response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0

    Notes
    -----
    \t:math:`G_{\\psi}(x_{c}, z_{c}; x, z) = \\dfrac{{\\mu}_{0}}{2{\\pi}}`
    \t:math:`\\dfrac{\\sqrt{xx_{c}}}{k}`
    \t:math:`[(2-\\mathbf{K}(k^2)-2\\mathbf{E}(k^2)]`\n
    Where:
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind
    """
    _, k2 = calc_a_k2(xc, zc, x, z)
    e, k = calc_e_k(k2)
    return MU_0_2PI * np.sqrt(x * xc) * ((2 - k2) * k - 2 * e) / np.sqrt(k2)


@nb.jit(nopython=True)
def greens_dpsi_dx(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> float | np.ndarray:
    """
    Calculate the radial derivative of the poloidal flux at (x, z)
    due to a unit current at (xc, zc) using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Radial derivative of the poloidal flux response at (x, z)

    Notes
    -----
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial x}}(x_{c}, z_{c}; x, z) =`
    \t:math:`\\dfrac{\\mu_0}{2\\pi}`
    \t:math:`\\dfrac{1}{u}`
    \t:math:`[\\dfrac{w^2}{d^2}\\mathbf{E}(k^2)+\\mathbf{K}(k^2)]`\n
    Where:
    \t:math:`h^{2}\\equiv z_{c}-z`\n
    \t:math:`u^2\\equiv(x+x_{c})^2+h^2`\n
    \t:math:`d^{2}\\equiv (x - x_{c})^2 + h^2`\n
    \t:math:`w^{2}\\equiv x^2 -x_{c}^2 - h^2`\n
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind

    The implementation used here refactors the above to avoid some zero divisions.
    """
    a, k2 = calc_a_k2(xc, zc, x, z)
    i1, i2 = calc_i1_i2(a, k2)
    return MU_0_2PI * x * ((xc**2 - (z - zc) ** 2 - x**2) * i2 + i1)


@nb.jit(nopython=True)
def greens_dpsi_dz(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> float | np.ndarray:
    """
    Calculate the vertical derivative of the poloidal flux at (x, z)
    due to a unit current at (xc, zc) using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Vertical derivative of the poloidal flux response at (x, z)

    Notes
    -----
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial z}}(x_{c}, z_{c}; x, z) =`
    \t:math:`\\dfrac{\\mu_0}{2\\pi}`
    \t:math:`\\dfrac{h}{u}`
    \t:math:`[\\mathbf{K}(k^2) - \\dfrac{v^2}{d^2}\\mathbf{E}(k^2)]`\n
    Where:
    \t:math:`h^{2}\\equiv z_{c}-z`\n
    \t:math:`u^2\\equiv(x+x_{c})^2+h^2`\n
    \t:math:`d^{2}\\equiv (x - x_{c})^2 + h^2`\n
    \t:math:`v^{2}\\equiv x^2 +x_{c}^2 + h^2`\n
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind

    The implementation used here refactors the above to avoid some zero divisions.
    """
    a, k2 = calc_a_k2(xc, zc, x, z)
    i1, i2 = calc_i1_i2(a, k2)
    return MU_0_2PI * ((z - zc) * (i1 - i2 * ((z - zc) ** 2 + x**2 + xc**2)))


@nb.jit(nopython=True)
def calc_a_k2(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
):
    a = np.hypot((x + xc), (z - zc))
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    return a, k2


@nb.jit(nopython=True)
def calc_e_k(
    k2: float | np.ndarray,
):
    return ellipe_nb(k2), ellipk_nb(k2)


@nb.jit(nopython=True)
def calc_i1_i2(
    a: float | np.ndarray,
    k2: float | np.ndarray,
    e: float | np.ndarray | None = None,
    k: float | np.ndarray | None = None,
):
    if (e is None) or (k is None):
        e, k = calc_e_k(k2)
    i1 = 4 * k / a
    i2 = 4 * e / (a**3 * (1 - k2))
    return i1, i2


@nb.jit(nopython=True)
def greens_Bx(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,
    d_zc: float = 0,
) -> float | np.ndarray:
    """
    Calculate radial magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Radial magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if x == 0

    Notes
    -----
    \t:math:`G_{B_{x}}(x_{c}, z_{c}; x, z) = -\\dfrac{1}{x}`
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial z}}`
    \t:math:`(x_{c}, z_{c}; x, z)`
    """
    return -1 / x * greens_dpsi_dz(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_Bz(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,
    d_zc: float = 0,
) -> float | np.ndarray:
    """
    Calculate vertical magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if x == 0

    Notes
    -----
    \t:math:`G_{B_{z}}(x_{c}, z_{c}; x, z) = \\dfrac{1}{x}`
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial x}}`
    \t:math:`(x_{c}, z_{c}; x, z)`
    """
    return 1 / x * greens_dpsi_dx(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_all(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Speed optimisation of Green's functions for psi, Bx, and Bz

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations

    Returns
    -------
    psi:
        Poloidal magnetic flux per radian response at (x, z)
    Bx:
        Radial magnetic field response at (x, z)
    Bz:
        Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0
        if x <= 0
    """
    a, k2 = calc_a_k2(xc, zc, x, z)
    e, k = calc_e_k(k2)
    i1, i2 = calc_i1_i2(a, k2, e, k)
    i1 *= 4
    i2 *= 4
    a_part = (z - zc) ** 2 + x**2 + xc**2
    b_part = -2 * x * xc
    g_bx = MU_0_4PI * xc * (z - zc) * (i1 - i2 * a_part) / b_part
    g_bz = MU_0_4PI * xc * ((xc + x * a_part / b_part) * i2 - i1 * x / b_part)
    g_psi = MU_0_4PI * a * ((2 - k2) * k - 2 * e)
    return g_psi, g_bx, g_bz
