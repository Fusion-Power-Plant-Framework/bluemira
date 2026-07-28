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

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnetostatics._ellipe import ellipe_nb
from bluemira.magnetostatics._ellipk import ellipk_nb

__all__ = [
    "circular_coil_inductance_elliptic",
    "circular_coil_inductance_kirchhoff",
    "greens_Bx",
    "greens_Bz",
    "greens_all",
    "greens_dbz_dx",
    "greens_dpsi_dx",
    "greens_dpsi_dz",
    "greens_psi",
    "square_coil_inductance_kirchhoff",
]


GREENS_ZERO = 1e-8


@nb.vectorize(cache=True)
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
    :
        The clipped values.
    """
    if val < val_min:
        return val_min
    if val > val_max:
        return val_max
    return val


@nb.njit(cache=True, inline="always")
def _common_geometry(xc, zc, x, z):
    r"""
    Calculate the geometry shared by the Green's functions.

    .. math::

        h = z-z_c,

    .. math::

        u^2 = (x+x_c)^2+h^2,

    .. math::

        k^2 = \frac{4xx_c}{u^2}.
    """
    h = z - zc
    xp = x + xc

    u2 = xp * xp + h * h
    u = np.sqrt(u2)
    k2 = clip_nb(4.0 * x * xc / u2, GREENS_ZERO, 1.0 - GREENS_ZERO)

    return h, u, u2, k2


@nb.njit(cache=True, inline="always")
def _elliptic_integrals(k2):
    """Evaluate E(k²) and K(k²)."""
    return ellipe_nb(k2), ellipk_nb(k2)


@nb.njit(cache=True, inline="always")
def _i1_i2(u, u2, k2, e, k):
    r"""
    Calculate

    .. math::

        I_1 = \frac{K(k^2)}{u},

    .. math::

        I_2 = \frac{E(k^2)}{u^3(1-k^2)}.
    """
    inv_u = 1.0 / u

    i1 = k * inv_u
    i2 = e * inv_u / (u2 * (1.0 - k2))

    return i1, i2


@nb.njit(cache=True, inline="always")
def _radial_term(xc, x, h, i1, i2):
    r"""
    Calculate

    .. math::

        R = I_1+w^2I_2,

    where

    .. math::

        w^2=x_c^2-x^2-h^2.
    """
    w2 = xc * xc - x * x - h * h
    return i1 + w2 * i2


@nb.njit(cache=True, inline="always")
def _vertical_term(xc, x, h, i1, i2):
    r"""
    Calculate

    .. math::

        Z = I_1-v^2I_2,

    where

    .. math::

        v^2=x^2+x_c^2+h^2.
    """
    v2 = x * x + xc * xc + h * h
    return i1 - v2 * i2


@nb.njit(cache=True, inline="always")
def _radial_response(xc, zc, x, z):
    """Calculate the term shared by dpsi/dx and Bz."""
    h, u, u2, k2 = _common_geometry(xc, zc, x, z)
    e, k = _elliptic_integrals(k2)
    i1, i2 = _i1_i2(u, u2, k2, e, k)

    return _radial_term(xc, x, h, i1, i2)


@nb.njit(cache=True, inline="always")
def _vertical_response(xc, zc, x, z):
    """Calculate the quantities shared by dpsi/dz and Bx."""
    h, u, u2, k2 = _common_geometry(xc, zc, x, z)
    e, k = _elliptic_integrals(k2)
    i1, i2 = _i1_i2(u, u2, k2, e, k)

    return h, _vertical_term(xc, x, h, i1, i2)


@nb.njit(cache=True, inline="always")
def _axis_bz(xc, zc, z):
    r"""
    Calculate the analytical vertical field on the symmetry axis.

    .. math::

        B_z(0,z)=
        \frac{\mu_0x_c^2}
        {2\left[x_c^2+(z-z_c)^2\right]^{3/2}}.
    """
    h = z - zc
    u2 = xc * xc + h * h

    return 0.5 * MU_0 * xc * xc / (u2 * np.sqrt(u2))


@nb.njit(cache=True, inline="always")
def _elliptic_derivatives(e, k, k2):
    r"""
    Calculate elliptic-integral derivatives with respect to ``k²``.

    .. math::

        \frac{dE}{d(k^2)}=\frac{E-K}{2k^2},\qquad
        \frac{dK}{d(k^2)}
        =\frac{E-(1-k^2)K}{2k^2(1-k^2)}.
    """
    one_minus_k2 = 1.0 - k2
    inv_2k2 = 0.5 / k2

    dE_dk2 = (e - k) * inv_2k2  # noqa: N806
    dK_dk2 = (e - one_minus_k2 * k) * inv_2k2 / one_minus_k2  # noqa: N806

    return dK_dk2, dE_dk2


@nb.njit(cache=True)
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
    :
        Poloidal magnetic flux per radian response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc == x == 0

    Notes
    -----
    .. math::
        G_\\psi=    \frac{\\mu_0}{4\\pi}[(2-k^2)K(k^2)-2E(k^2)]

    The analytical axis value is 0.0. This is checked for as x <= 0.

    Negative x or xc values are outside the intended domain.
    """
    axis = x <= 0.0

    _, u, _, k2 = _common_geometry(xc, zc, x, z)
    e, k = _elliptic_integrals(k2)

    result = MU_0_4PI * u * ((2.0 - k2) * k - 2.0 * e)

    return np.where(axis, 0.0, result)


@nb.njit(cache=True)
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
    :
        Radial derivative of the poloidal flux response at (x, z)

    Notes
    -----
    .. math::

        \frac{\\partial G_\\psi}{\\partial x}
        =\frac{\\mu_0}{2\\pi}x
        \\left(I_1+w^2I_2\right).

    The implementation used here refactors the above to avoid some zero divisions.

    The analytical axis value is 0.0. This is checked for as x <= 0.

    Negative x or xc values are outside the intended domain.
    """
    radial_term = _radial_response(xc, zc, x, z)

    # The explicit factor of x supplies the exact axis value for mixed arrays.
    return MU_0_2PI * x * radial_term


@nb.njit(cache=True)
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
    :
        Vertical derivative of the poloidal flux response at (x, z)

    Notes
    -----
    .. math::

        \frac{\\partial G_\\psi}{\\partial z}
        =\frac{\\mu_0}{2\\pi}h
        \\left(I_1-v^2I_2\right).

    The implementation used here refactors the above to avoid some zero divisions.

    The analytical axis value is 0.0. This is checked for as x <= 0.

    Negative x or xc values are outside the intended domain.
    """
    axis = x <= 0.0

    h, vertical_term = _vertical_response(xc, zc, x, z)
    result = MU_0_2PI * h * vertical_term

    return np.where(axis, 0.0, result)


@nb.njit(cache=True)
def greens_Bx(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> float | np.ndarray:
    """
    Calculate the radial magnetic-field response.

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
    :
        Radial magnetic field response at (x, z)

    Notes
    -----
    .. math::

        G_{B_x}
        =-\frac{\\mu_0}{2\\pi}\frac{h}{x}
        \\left(I_1-v^2I_2\right).

    The analytical axis value is 0.0. This is checked for as x <= 0.

    Negative x or xc values are outside the intended domain.
    """
    axis = x <= 0.0
    x_safe = np.where(axis, 1.0, x)

    h, vertical_term = _vertical_response(xc, zc, x, z)
    result = -MU_0_2PI * h * vertical_term / x_safe

    return np.where(axis, 0.0, result)


@nb.njit(cache=True)
def greens_Bz(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> float | np.ndarray:
    """
    Calculate the vertical magnetic-field response.

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
    :
        Vertical magnetic field response at (x, z)

    Notes
    -----
    .. math::

        G_{B_z}
        =\frac{\\mu_0}{2\\pi}
        \\left(I_1+w^2I_2\right).

    On the symmetry axis,

    .. math::

        G_{B_z}(0,z)=
        \frac{\\mu_0x_c^2}
        {2\\left[x_c^2+(z-z_c)^2\right]^{3/2}}.

    The axis value is calculated analytically.

    Negative x or xc values are outside the intended domain.
    """
    h, u, u2, k2 = _common_geometry(xc, zc, x, z)
    e, k = _elliptic_integrals(k2)
    i1, i2 = _i1_i2(u, u2, k2, e, k)

    xc2 = xc * xc
    w2 = xc2 - x * x - h * h

    bz = MU_0_2PI * (i1 + w2 * i2)
    bz_axis = 0.5 * MU_0 * xc2 / (u2 * u)

    return np.where(x <= 0.0, bz_axis, bz)


@nb.njit(cache=True)
def greens_dbz_dx(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
) -> float | np.ndarray:
    r"""
    Calculate the radial derivative of the vertical magnetic field.

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
    :
        The gradient of the vertical magnetic field

    Notes
    -----
    .. math::

        \frac{\partial B_z}{\partial x}
        =\frac{\partial B_x}{\partial z}
        =\frac{\mu_0}{2\pi}
        \left[
            \frac{1}{u}\frac{\partial F}{\partial x}
            -\frac{x+x_c}{u^3}F
        \right],

    where :math:`F=K(k^2)+E(k^2)w^2/d^2` and
    :math:`d^2=(x-x_c)^2+h^2`.

    The analytical axis value is 0.0. This is checked for as x <= 0.

    Negative x or xc values are outside the intended domain.
    """
    axis = x <= 0.0
    x_safe = np.where(axis, 1.0, x)

    h, u, u2, k2 = _common_geometry(xc, zc, x, z)

    xm = x - xc
    h2 = h * h

    # Calculate d² directly to avoid cancellation near the filament.
    d2 = xm * xm + h2
    w2 = xc * xc - x * x - h2

    e, k = _elliptic_integrals(k2)
    dK_dk2, dE_dk2 = _elliptic_derivatives(  # noqa: N806
        e,
        k,
        k2,
    )

    d2_is_zero = np.isclose(d2, 0.0)
    u2_is_zero = np.isclose(u2, 0.0)

    singular = np.logical_or(d2_is_zero, u2_is_zero)
    invalid = np.logical_or(axis, singular)

    d2_safe = np.where(d2_is_zero, GREENS_ZERO, d2)
    u2_safe = np.where(u2_is_zero, GREENS_ZERO, u2)
    u_safe = np.where(u2_is_zero, np.sqrt(u2_safe), u)

    inv_d2 = 1.0 / d2_safe
    inv_d2_2 = inv_d2 * inv_d2
    inv_u = 1.0 / u_safe
    inv_u3 = inv_u / u2_safe

    w2_over_d2 = w2 * inv_d2
    elliptic_term = k + e * w2_over_d2

    # Differentiate the underlying geometric parameter. The elliptic
    # functions themselves continue to use the clipped parameter.
    k2_geometric = 4.0 * x * xc / u2_safe

    dk2_dx = k2_geometric * (1.0 / x_safe - 2.0 * (x + xc) / u2_safe)

    d_elliptic_term_dx = (
        dK_dk2 * dk2_dx
        + dE_dk2 * dk2_dx * w2_over_d2
        - 2.0 * e * x * inv_d2
        - 2.0 * e * w2 * xm * inv_d2_2
    )

    result = MU_0_2PI * (d_elliptic_term_dx * inv_u - elliptic_term * (x + xc) * inv_u3)

    return np.where(invalid, 0.0, result)


@nb.njit(cache=True)
def greens_all(
    xc: float | np.ndarray,
    zc: float | np.ndarray,
    x: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """
    Calculate poloidal flux, radial field, and vertical field together.

    Geometry, masks, and elliptic integrals are each evaluated once.

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
        Poloidal flux response
    Bx:
        Radial magnetic field response
    Bz:
        Vertical magnetic field response
    """
    axis = x <= 0.0
    x_safe = np.where(axis, 1.0, x)

    h, u, u2, k2 = _common_geometry(xc, zc, x, z)
    e, k = _elliptic_integrals(k2)
    i1, i2 = _i1_i2(u, u2, k2, e, k)

    radial_term = _radial_term(xc, x, h, i1, i2)
    vertical_term = _vertical_term(xc, x, h, i1, i2)

    psi = MU_0_4PI * u * ((2.0 - k2) * k - 2.0 * e)

    bx = -MU_0_2PI * h * vertical_term / x_safe

    bz = MU_0_2PI * radial_term

    axis_bz = _axis_bz(xc, zc, z)

    psi = np.where(axis, 0.0, psi)
    bx = np.where(axis, 0.0, bx)
    bz = np.where(axis, axis_bz, bz)

    return psi, bx, bz


@nb.njit(cache=True)
def circular_coil_inductance_elliptic(
    radius: float | np.ndarray, rc: float | np.ndarray
) -> float | np.ndarray:
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
    :
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


def circular_coil_inductance_kirchhoff(
    radius: float | np.ndarray, rc: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the inductance of a circular coil by Kirchhoff's approximation.

    radius:
        The radius of the circular coil
    rc:
        The radius of the coil cross-section

    Returns
    -------
    :
        The self-inductance of the circular coil [H]

    Notes
    -----

    .. math::
        Inductance = \\mu_{0} * radius * (log(8 * radius / rc) - 2 + 0.25)

    where :math:`\\mu_{0}` is the vacuum permeability
    """
    return MU_0 * radius * (np.log(8 * radius / rc) - 2 + 0.25)


def square_coil_inductance_kirchhoff(
    radius: float | np.ndarray, width: float | np.ndarray, height: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the inductance of a square coil by Kirchhoff's approximation.

    radius:
        The radius of the square coil
    width:
        The width of the coil cross-section
    height
        The height of the coil cross-section

    Returns
    -------
    The self-inductance of the square coil [H]

    Notes
    -----
    .. math::

        Inductance = \\mu_0 radius (ln(8\\frac{radius}{width + height}) - 0.5)

    where :math:`\\mu_{0}` is the vacuum permeability
    """
    return MU_0 * radius * (np.log(8 * radius / (width + height)) - 0.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.magnetostatics.greens_old import greens_Bz as old_greens_Bz

    x = np.linspace(0, 0.1, 500)
    z = np.zeros_like(x)
    coil_x, coil_z = 4, 4

    psi = greens_psi(coil_x, coil_z, x, z)
    dpsi_dx = greens_dpsi_dx(coil_x, coil_z, x, z)
    dpsi_dz = greens_dpsi_dz(coil_x, coil_x, x, z)
    bx = greens_Bx(coil_x, coil_z, x, z)
    bz = greens_Bz(coil_x, coil_z, x, z)
    old_bz = old_greens_Bz(coil_x, coil_z, x, z)

    for r in [psi, dpsi_dx, dpsi_dz, bx]:
        f, ax = plt.subplots()
        ax.plot(x, 1e6 * r)
        plt.show()

    f, ax = plt.subplots()
    ax.plot(x, bz, label="Bz")
    ax.plot(x, old_bz, ls="--", label="old Bz")
    ax.legend()
    plt.show()
