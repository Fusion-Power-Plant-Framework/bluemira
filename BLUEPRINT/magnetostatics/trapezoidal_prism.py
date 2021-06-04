# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Analytical expressions for the field inside an arbitrarily shaped winding pack
of rectangular cross-section, following equations as described in:

https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594?saml_referrer=
including corrections from:
https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.675
"""
import numpy as np
import numba as nb

__all__ = ["Bx_analytical_prism", "Bz_analytical_prism"]


@nb.jit(cache=True)
def primitive_sxn_bound(cos_theta, sin_theta, r, q, t):
    """
    Function primitive of Bx evaluated at a bound.

    Parameters
    ----------
    cos_theta: float
        The cosine of theta in radians
    sin_theta: float
        The sine of theta in radians
    r: float
        The r local coordinate value
    q: float
        The q local coordinate value
    t: float
        The t local coordinate value

    Returns
    -------
    sxn_t: float
        The value of the primitive function at integral bound t

    Notes
    -----
    Uses corrected formulae available at:
    https://onlinelibrary.wiley.com/doi/abs/10.1002/jnm.675

    Singularities all resolve to: lim(ln(1)) --> 0
    """
    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t ** 2 + q ** 2)
    divisor_2 = cos_theta * np.sqrt(r ** 2 * cos_theta ** 2 + q ** 2)
    divisor_3 = q * np.sqrt(
        t ** 2 + 2 * r * t * sin_theta * cos_theta + (r ** 2 + q ** 2) * cos_theta ** 2
    )

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += t * np.arcsinh((t * sin_theta + r * cos_theta) / divisor_1)
    if divisor_2 != 0:
        result += r * cos_theta * np.arcsinh((t + r * cos_theta * sin_theta) / divisor_2)
    if divisor_3 != 0:
        result += q * np.arctan((q ** 2 * sin_theta - t * r * cos_theta) / divisor_3)

    return result


@nb.jit(cache=True)
def primitive_sxn(theta, r, q, l1, l2):
    """
    Analytical integral of Bx function primitive.

    Parameters
    ----------
    theta: float
        The angle in radians
    r: float
        The r local coordinate value
    q: float
        The q local coordinate value
    l1: float
        The first local coordinate bound
    l2: float
        The second local coordinate bound

    Returns
    -------
    sxn_l2_l1: float
        The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_sxn_bound(cos_theta, sin_theta, r, q, l2) - primitive_sxn_bound(
        cos_theta, sin_theta, r, q, l1
    )


@nb.jit(cache=True)
def primitive_szn_bound(cos_theta, sin_theta, r, ll, t):
    """
    Function primitive of Bz evaluated at a bound.

    Parameters
    ----------
    cos_theta: float
        The cosine of theta
    sin_theta: float
        The sine of theta
    r: float
        The r local coordinate value
    ll: float
        The l local coordinate value
    t: float
        The t local coordinate value

    Returns
    -------
    szn_t: float
        The value of the primitive function at integral bound t

    Notes
    -----
    Singularities all resolve to: lim(ln(1)) --> 0
    """
    sqrt_term = np.sqrt(
        t ** 2 * cos_theta ** 2
        + ll ** 2
        + 2 * r * ll * sin_theta * cos_theta
        + r ** 2 * cos_theta ** 2
    )

    # First compute the divisors of each of the terms to determine if there
    # are singularities
    divisor_1 = cos_theta * np.sqrt(t ** 2 + r ** 2 * cos_theta ** 2)
    divisor_2 = cos_theta * np.sqrt(t ** 2 + ll ** 2)
    divisor_3 = np.sqrt(
        ll ** 2 + 2 * ll * r * sin_theta * cos_theta + r ** 2 * cos_theta ** 2
    )
    divisor_4 = r * cos_theta * sqrt_term
    divisor_5 = ll * sqrt_term

    # All singularities resolve to 0?
    result = 0
    if divisor_1 != 0:
        result += (
            t * sin_theta * np.arcsinh((ll + r * sin_theta * cos_theta) / divisor_1)
        )
    if divisor_2 != 0:
        result -= t * np.arcsinh((ll * sin_theta + r * cos_theta) / divisor_2)
    if divisor_3 != 0:
        result -= r * cos_theta ** 2 * np.arcsinh((t * cos_theta) / divisor_3)
    if divisor_4 != 0:
        result -= (
            r
            * cos_theta
            * sin_theta
            * np.arctan((t * (ll + r * sin_theta * cos_theta)) / divisor_4)
        )
    if divisor_5 != 0:
        result += ll * np.arctan((t * (ll * sin_theta + r * cos_theta)) / divisor_5)

    return result


@nb.jit(cache=True)
def primitive_szn(theta, r, ll, q1, q2):
    """
    Analytical integral of Bz function primitive.

    Parameters
    ----------
    theta: float
        The angle in radians
    r: float
        The r local coordinate value
    ll: float
        The l local coordinate value
    q1: float
        The first local coordinate bound
    q2: float
        The second local coordinate bound

    Returns
    -------
    sxn_q2_q1: float
        The value of the integral of the Bx function primitive
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return primitive_szn_bound(cos_theta, sin_theta, r, ll, q2) - primitive_szn_bound(
        cos_theta, sin_theta, r, ll, q1
    )


@nb.jit(cache=True)
def Bx_analytical_prism(alpha, beta, l1, l2, q1, q2, r1, r2):
    """
    Calculate magnetic field in the local x coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha: float
        The first trapezoidal angle [rad]
    beta: float
        The second trapezoidal angle [rad]
    l1: float
        The local l1 coordinate [m]
    l2: float
        The local l2 coordinate [m]
    q1: float
        The local q1 coordinate [m]
    q2: float
        The local q2 coordinate [m]
    r1: float
        The local r1 coordinate [m]
    r2: float
        The local r2 coordinate [m]

    Returns
    -------
    Bx: float
        The magnetic field response in the x coordinate direction
    """
    return (
        primitive_sxn(alpha, r1, q2, l1, l2)
        - primitive_sxn(alpha, r1, q1, l1, l2)
        + primitive_sxn(beta, r2, q2, l1, l2)
        - primitive_sxn(beta, r2, q1, l1, l2)
    )


@nb.jit(cache=True)
def Bz_analytical_prism(alpha, beta, l1, l2, q1, q2, r1, r2):
    """
    Calculate magnetic field in the local z coordinate direction due to a
    trapezoidal prism current source.

    Parameters
    ----------
    alpha: float
        The first trapezoidal angle [rad]
    beta: float
        The second trapezoidal angle [rad]
    l1: float
        The local l1 coordinate [m]
    l2: float
        The local l2 coordinate [m]
    q1: float
        The local q1 coordinate [m]
    q2: float
        The local q2 coordinate [m]
    r1: float
        The local r1 coordinate [m]
    r2: float
        The local r2 coordinate [m]

    Returns
    -------
    Bz: float
        The magnetic field response in the z coordinate direction
    """
    return (
        primitive_szn(alpha, r1, l2, q1, q2)
        - primitive_szn(alpha, r1, l1, q1, q2)
        + primitive_szn(beta, r2, l2, q1, q2)
        - primitive_szn(beta, r2, l1, q1, q2)
    )
