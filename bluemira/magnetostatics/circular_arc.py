# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Analytical expressions for the field due to a circular current arc of
rectangular cross-section, following equations as described in:

.. doi:: 10.1109/TMAG.1985.1064259
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from bluemira.base.constants import EPS, MU_0_4PI
from bluemira.geometry._private_tools import make_circle_arc
from bluemira.magnetostatics.baseclass import CrossSectionCurrentSource
from bluemira.magnetostatics.tools import (
    integrate,
    jit_llc3,
    jit_llc4,
    process_xyz_array,
)

__all__ = ["CircularArcCurrentSource"]


LOG_EPS = np.finfo(float).tiny
TWO_PI = 2.0 * np.pi

# Full integrands free of singularities


@jit_llc4
def brc_integrand_full(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Brc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    log_arg = r_j - r_pc * cos_psi + sqrt_term
    if log_arg <= 0:
        log_arg = LOG_EPS
    return cos_psi * sqrt_term + r_pc * cos_psi**2 * np.log(log_arg)


@jit_llc4
def bzc_integrand_full_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Bzc integrand without singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    term_1 = 0.0
    if z_k != 0:
        log_arg_1 = r_j - r_pc * cos_psi + sqrt_term
        if log_arg_1 <= 0:
            log_arg_1 = LOG_EPS

        term_1 = -z_k * np.log(log_arg_1)

    log_arg_2 = -z_k + sqrt_term
    if log_arg_2 <= 0:
        log_arg_2 = LOG_EPS
    return term_1 - r_pc * cos_psi * np.log(log_arg_2)


# Integrands to treat singularities


@jit_llc4
def brc_integrand_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the first part of the Brc integrand (no singularities)

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return cos_psi * sqrt_term


@jit_llc4
def bf1_r_pccos2_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    log_arg = r_j - r_pc * cos_psi + sqrt_term
    if log_arg <= 0:
        log_arg = LOG_EPS

    return r_pc * cos_psi**2 * np.log(log_arg)


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p1(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 1

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi**2
        * np.log(
            (r_pc * cos_psi - r_j)
            + np.sqrt((r_pc * cos_psi - r_j) ** 2 + r_pc**2 * np.sin(psi) ** 2)
        )
    )


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p2(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 2

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi**2
        * np.log(
            (r_j - r_pc * cos_psi)
            + np.sqrt((r_j - r_pc * cos_psi) ** 2 + r_pc**2 * np.sin(psi) ** 2)
        )
    )


@jit_llc4
def bf1_zk_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF1(-z_k) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    if z_k == 0:
        return 0.0

    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    log_arg = r_j - r_pc * cos_psi + sqrt_term
    if log_arg <= 0:
        log_arg = LOG_EPS

    return -z_k * np.log(log_arg)


@jit_llc4
def bf2_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand.

    For r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    log_arg = -z_k + sqrt_term
    if log_arg <= 0:
        log_arg = LOG_EPS

    return -r_pc * cos_psi * np.log(log_arg)


@jit_llc3
def bf2_0_pi_integrand(psi: float, r_pc: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    log_arg = z_k + np.sqrt(2 * r_pc**2 * (1 - cos_psi) + z_k**2)
    if log_arg <= 0:
        log_arg = LOG_EPS

    return r_pc * cos_psi * np.log(log_arg)


@jit_llc4
def bf3_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF3 integrand

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    psi:
        Angle [rad]

    Returns
    -------
    The result of the integrand at a single point

    Notes
    -----
    Treats the sin(psi) = 0 singularity
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if sin_psi != 0:
        sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
        return (
            r_pc
            * sin_psi
            * np.arctan((z_k * (r_j - r_pc * cos_psi)) / (r_pc * sin_psi * sqrt_term))
        )
    return 0


# More singularity treatments...


def bf1_r_pccos2_zk0_0_pi(r_pc: float, r_j: float) -> float:
    """If log_arg <= 0:
        log_arg = LOG_EPS

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]

    Returns
    -------
    The result of the integral at a single point for 0 to pi integral
    """
    if r_pc == r_j:
        return r_pc * (0.5 * np.pi * np.log(r_pc) + 0.2910733)
    # r_j < r_pc
    result = 0.5 * np.pi * r_pc * (np.log(r_pc) - np.log(2) - 0.5)
    result -= integrate(bf1_r_pccos2_0_pi_integrand_p1, (r_pc, r_j), 0, 0.5 * np.pi)
    result += integrate(bf1_r_pccos2_0_pi_integrand_p2, (r_pc, r_j), 0.5 * np.pi, np.pi)
    return result


def bf2_rj_rpc_0_pi(r_pc: float, z_k: float) -> float:
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    The result of the integral for 0 to pi
    """
    return np.pi * r_pc + integrate(bf2_0_pi_integrand, (r_pc, z_k), 0, np.pi)


# Primitive functions
# BTC integrands


@jit_llc4
def btc_integrand_full(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the Btc integrand without singularities.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return sin_psi * sqrt_term + r_pc * sin_psi * cos_psi * np.log(
        r_j - r_pc * cos_psi + sqrt_term
    )


@jit_llc4
def btc_integrand_p1(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the first part of the Btc integrand, without logarithmic
    singularities.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return sin_psi * sqrt_term


@jit_llc4
def bf1_r_pcsincos_integrand(psi: float, r_pc: float, r_j: float, z_k: float) -> float:
    """
    Calculate the BF1(r_pc * sin(psi) * cos(psi)) integrand.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]

    Returns
    -------
    :
        The result of the integrand at a single point
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc**2 - 2 * r_pc * r_j * cos_psi + r_j**2 + z_k**2)
    return r_pc * sin_psi * cos_psi * np.log(r_j - r_pc * cos_psi + sqrt_term)


@jit_llc3
def bf1_r_pcsincos_0_pi_integrand_p1(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the transformed BF1(r_pc * sin(psi) * cos(psi)) integrand
    on 0 to pi / 2 for the singular z_k == 0 case.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]

    Returns
    -------
    :
        The result of the transformed integrand.
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    return (
        r_pc
        * sin_psi
        * cos_psi
        * np.log(
            (r_pc * cos_psi - r_j)
            + np.sqrt((r_pc * cos_psi - r_j) ** 2 + r_pc**2 * sin_psi**2)
        )
    )


@jit_llc3
def bf1_r_pcsincos_0_pi_integrand_p2(psi: float, r_pc: float, r_j: float) -> float:
    """
    Calculate the regular BF1(r_pc * sin(psi) * cos(psi)) integrand
    on pi / 2 to pi for the singular z_k == 0 case.

    Parameters
    ----------
    psi:
        Angle [rad]
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]

    Returns
    -------
    :
        The result of the regular integrand.
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    return (
        r_pc
        * sin_psi
        * cos_psi
        * np.log(
            (r_j - r_pc * cos_psi)
            + np.sqrt((r_j - r_pc * cos_psi) ** 2 + r_pc**2 * sin_psi**2)
        )
    )


def bf1_r_pcsincos_zk0_0_pi(r_pc: float, r_j: float) -> float:
    """
    Calculate the BF1(r_pc * sin(psi) * cos(psi)) integral for 0 to pi.

    This treats the z_k == 0 singular case following Feng's treatment for
    the circular arc conductor, where f1(psi) = r_pc * sin(psi) * cos(psi).

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]

    Returns
    -------
    :
        The result of the integral from 0 to pi.
    """
    if r_pc == r_j:
        return -(2.0 / 3.0) * r_pc

    result = r_pc * (np.log(r_pc) - 0.5)
    result -= integrate(
        bf1_r_pcsincos_0_pi_integrand_p1,
        (r_pc, r_j),
        0,
        0.5 * np.pi,
    )
    result += integrate(
        bf1_r_pcsincos_0_pi_integrand_p2,
        (r_pc, r_j),
        0.5 * np.pi,
        np.pi,
    )
    return result


def primitive_brc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Brc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Brc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    # Sub-set of paper singularities that don't perform well in integration
    is_singular_corner = z_k == 0 and r_j == r_pc and 0 <= phi_pc <= theta

    if not is_singular_corner:
        # No singularities
        return integrate(brc_integrand_full, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    # Singularity free treatment of first term
    result = integrate(brc_integrand_p1, args, -phi_pc, theta - phi_pc)

    # Dodge singularities in second term
    if phi_pc == 0:
        if theta == np.pi:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)

        if theta == TWO_PI:
            # cos(-psi)^2 == cos(psi)^2
            result += 2 * bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        else:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
            result -= integrate(bf1_r_pccos2_integrand, args, -theta, np.pi - theta)

    elif phi_pc == theta:
        if phi_pc == np.pi:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        else:
            # cos(-psi)^2 == cos(psi)^2
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
            result -= integrate(bf1_r_pccos2_integrand, args, np.pi, np.pi - theta)
    else:
        # cos(-psi)^2 == cos(psi)^2
        result += 2 * bf1_r_pccos2_zk0_0_pi(r_pc, r_j)
        result -= integrate(bf1_r_pccos2_integrand, args, phi_pc - theta, TWO_PI - theta)

    return result


def primitive_btc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Btc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius, inner or outer, of the coil [m]
    z_k:
        The z coordinate, upper or lower, of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc [rad]

    Returns
    -------
    :
        The result of the Btc primitive.
    """
    args = (r_pc, r_j, z_k)
    # Sub-set of paper singularities that don't perform well in integration
    is_singular_corner = z_k == 0 and r_j == r_pc and 0 <= phi_pc <= theta

    if not is_singular_corner:
        return integrate(btc_integrand_full, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    result = integrate(btc_integrand_p1, args, -phi_pc, theta - phi_pc)

    # Dodge singularities in the BF1 term.
    #
    # For f(psi) = r_pc * sin(psi) * cos(psi):
    # f(-psi) = -f(psi), so the 0 to 2*pi integral cancels.
    if phi_pc == 0:
        if theta == np.pi:
            result += bf1_r_pcsincos_zk0_0_pi(r_pc, r_j)

        elif theta == TWO_PI:
            result += 0.0

        else:
            result += bf1_r_pcsincos_zk0_0_pi(r_pc, r_j)
            result -= integrate(
                bf1_r_pcsincos_integrand,
                args,
                -theta,
                np.pi - theta,
            )

    elif phi_pc == theta:
        if phi_pc == np.pi:
            # Integral from -pi to 0 equals the integral from pi to 0 after
            # using the odd symmetry treatment in Feng's notation.
            result += -bf1_r_pcsincos_zk0_0_pi(r_pc, r_j)

        else:
            result += -bf1_r_pcsincos_zk0_0_pi(r_pc, r_j)
            result -= integrate(
                bf1_r_pcsincos_integrand,
                args,
                np.pi,
                np.pi - theta,
            )

    else:
        # The full 0 to 2*pi contribution of this odd integrand is zero.
        result -= integrate(
            bf1_r_pcsincos_integrand,
            args,
            phi_pc - theta,
            TWO_PI - theta,
        )

    return result


def primitive_bzc(
    r_pc: float, r_j: float, z_k: float, phi_pc: float, theta: float
) -> float:
    """
    Calculate the Bzc primitives and treat singularities.

    Parameters
    ----------
    r_pc:
        The radius of the point at which to evaluate field [m]
    r_j:
        The radius (inner or outer) of the coil [m]
    z_k:
        The z coordinate (upper or lower) of the coil [m]
    phi_pc:
        Angle of the point at which to evaluate field [rad]
    theta:
        Azimuthal angle of the circular arc

    Returns
    -------
    The result of the Bzc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    bf1_singularities = (z_k == 0) and (r_j <= r_pc) and (0 <= phi_pc <= theta)
    bf2_singularities = (r_j == r_pc) and (z_k >= 0) and (0 <= phi_pc <= theta)
    bf3_singularities = r_pc == 0
    if not bf1_singularities and not bf2_singularities and not bf3_singularities:
        # No singularities (almost)
        return integrate(
            bzc_integrand_full_p1, args, -phi_pc, theta - phi_pc
        ) + integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    result = 0
    if bf1_singularities:
        # Treat BF1(-z_k)
        if phi_pc == 0 and theta not in {np.pi, TWO_PI}:
            # At pi and 2 * pi the BF1 integral is 0
            # Elsewhere:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, -theta, np.pi - theta)

        if phi_pc == theta:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, np.pi, np.pi - theta)

        else:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, phi_pc - theta, TWO_PI - theta)

    else:
        # BF1 is normal
        result += integrate(bf1_zk_integrand, args, -phi_pc, theta - phi_pc)

    if bf2_singularities:
        # Treat BF2
        if phi_pc == 0:
            result += bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, -theta, np.pi - theta)

        if phi_pc == theta:
            result += bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, np.pi, np.pi - theta)

        else:
            result += 2 * bf2_rj_rpc_0_pi(r_pc, z_k)
            result -= integrate(bf2_integrand, args, phi_pc - theta, TWO_PI - theta)

    else:
        # BF2 is normal
        result += integrate(bf2_integrand, args, -phi_pc, theta - phi_pc)

    if r_pc != 0:
        # r_pc = 0, BF3 evaluates to 0
        result += integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)

    return result


# Full field calculations in working coordinates


def Bx_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local x coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the x coordinate direction
    """
    return (
        primitive_brc(r_p, r1, z1, theta_p, theta)
        - primitive_brc(r_p, r1, z2, theta_p, theta)
        - primitive_brc(r_p, r2, z1, theta_p, theta)
        + primitive_brc(r_p, r2, z2, theta_p, theta)
    )


def Bt_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local tangential coordinate direction due
    to a circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    :
        The magnetic field response in the local tangential direction.
    """
    return (
        primitive_btc(r_p, r1, z1, theta_p, theta)
        - primitive_btc(r_p, r1, z2, theta_p, theta)
        - primitive_btc(r_p, r2, z1, theta_p, theta)
        + primitive_btc(r_p, r2, z2, theta_p, theta)
    )


def Bz_analytical_circular(
    r1: float, r2: float, z1: float, z2: float, theta: float, r_p: float, theta_p: float
) -> float:
    """
    Calculate magnetic field in the local z coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1:
        Inner coil radius [m]
    r2:
        Outer coil radius [m]
    z1:
        The first modified z coordinate [m]
    z2:
        The second modified z coordinate [m]
    theta:
        Azimuthal angle of the circular arc [rad]
    r_p:
        The radius of the point at which to evaluate the field [m]
    theta_p:
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    The magnetic field response in the z coordinate direction
    """
    return (
        primitive_bzc(r_p, r1, z1, theta_p, theta)
        - primitive_bzc(r_p, r1, z2, theta_p, theta)
        - primitive_bzc(r_p, r2, z1, theta_p, theta)
        + primitive_bzc(r_p, r2, z2, theta_p, theta)
    )


class CircularArcCurrentSource(CrossSectionCurrentSource):
    """
    3-D circular arc prism current source with a rectangular cross-section and
    uniform current distribution.

    Parameters
    ----------
    origin:
        The origin of the current source in global coordinates [m]
    ds:
        The direction vector of the current source in global coordinates [m]
    normal:
        The normalised normal vector of the current source in global coordinates [m]
    t_vec:
        The normalised tangent vector of the current source in global coordinates [m]
    breadth:
        The breadth of the current source (half-width) [m]
    depth:
        The depth of the current source (half-height) [m]
    radius:
        The radius of the circular arc from the origin [m]
    dtheta:
        The azimuthal width of the arc [°]
    current:
        The current flowing through the source [A]

    Notes
    -----
    The origin is at the centre of the circular arc, with the ds vector pointing
    towards the start of the circular arc.

    Cylindrical coordinates are used for calculations under the hood.
    """

    def __init__(
        self,
        origin: npt.NDArray[np.float64],
        ds: npt.NDArray[np.float64],
        normal: npt.NDArray[np.float64],
        t_vec: npt.NDArray[np.float64],
        breadth: float,
        depth: float,
        radius: float,
        dtheta: float,
        current: float,
    ):
        self._origin = origin
        self._breadth = breadth
        self._depth = depth
        self._length = 0.5 * (breadth + depth)  # For plotting only
        self._radius = radius
        self._update_r1r2()

        self._dtheta = np.deg2rad(dtheta)
        self._rho = current / (4 * breadth * depth)
        self._dcm = np.array([ds, normal, t_vec], dtype=float)
        self._points = self._calculate_points()

    @property
    def radius(self) -> float:
        """
        The radius of the CircularArcCurrentSource
        """
        return self._radius

    @radius.setter
    def radius(self, radius: float):
        """
        Set the radius.

        Parameters
        ----------
        radius:
            The radius of the CircularArcCurrentSource
        """
        self._radius = radius
        self._update_r1r2()

    @property
    def breadth(self) -> float:
        """
        The breadth of the CircularArcCurrentSource
        """
        return self._breadth

    @breadth.setter
    def breadth(self, breadth: float):
        """
        Set the breadth of the CircularArcCurrentSource.

        Parameters
        ----------
        breadth:
            The breadth of the CircularArcCurrentSource
        """
        self._breadth = breadth
        self._update_r1r2()

    def _update_r1r2(self):
        """
        Update
        """
        self._r1 = self.radius - self.breadth
        self._r2 = self.radius + self.breadth

    @staticmethod
    def _local_to_cylindrical(point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert from local to cylindrical coordinates.

        Returns
        -------
        :
            Cylindrical coordinates of point.
        """
        x, y, z = point
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])

    def _cylindrical_to_working(self, zp: float) -> tuple[float, float, float, float]:
        """
        Convert from local cylindrical coordinates to working coordinates.

        Returns
        -------
        :
            r +- breadth and z +- depth.
        """
        z1 = zp + self._depth
        z2 = zp - self._depth
        return self._r1, self._r2, z1, z2

    def _BxByBz(self, rp: float, tp: float, zp: float) -> npt.NDArray[np.float64]:
        """
        Calculate the field at a point in local coordinates.

        Returns
        -------
        :
            (Bx, By, Bz) at a point
        """
        r1, r2, z1, z2 = self._cylindrical_to_working(zp)

        br = Bx_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)
        bz = Bz_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)
        if np.isclose(abs(self._dtheta), TWO_PI, rtol=0.0, atol=EPS):
            bt = 0.0
        else:
            bt = Bt_analytical_circular(r1, r2, z1, z2, self._dtheta, rp, tp)

        bx = br * np.cos(tp) - bt * np.sin(tp)
        by = br * np.sin(tp) + bt * np.cos(tp)

        return np.array([bx, by, bz])

    @process_xyz_array
    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        :
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        # Convert to local cylindrical coordinates
        point = self._global_to_local([point])[0]
        rp, tp, zp = self._local_to_cylindrical(point)
        # Calculate field in local coordinates
        b_local = MU_0_4PI * self._rho * self._BxByBz(rp, tp, zp)
        # Convert field to global coordinates
        return self._dcm.T @ b_local

    def _calculate_points(self) -> npt.NDArray[np.float64]:
        """
        Calculate extrema points of the current source for plotting and debugging.

        Returns
        -------
        :
            extrema points
        """
        r = self.radius
        a = self.breadth
        b = self._depth

        # Circle arcs
        n = 200
        theta = self._dtheta
        ones = np.ones(n)
        arc_1x, arc_1y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_2x, arc_2y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_3x, arc_3y = make_circle_arc(r + a, 0, 0, angle=theta, n_points=n)
        arc_4x, arc_4y = make_circle_arc(r - a, 0, 0, angle=theta, n_points=n)
        arc_1 = np.array([arc_1x, arc_1y, -b * ones]).T
        arc_2 = np.array([arc_2x, arc_2y, -b * ones]).T
        arc_3 = np.array([arc_3x, arc_3y, b * ones]).T
        arc_4 = np.array([arc_4x, arc_4y, b * ones]).T

        n_slices = int(2 + self._dtheta // (0.25 * np.pi))
        slices = np.linspace(0, n - 1, n_slices, endpoint=True, dtype=int)
        points = [arc_1, arc_2, arc_3, arc_4]

        # Rectangles
        points.extend([
            np.vstack([arc_1[s], arc_2[s], arc_3[s], arc_4[s], arc_1[s]]) for s in slices
        ])

        return np.array([self._local_to_global(p) for p in points], dtype=object)

    def plot(self, ax: plt.Axes | None = None, *, show_coord_sys: bool = False):
        """
        Plot the CircularArcCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        super().plot(ax=ax, show_coord_sys=show_coord_sys)
        ax = plt.gca()
        theta = self._dtheta
        x, y = make_circle_arc(
            self.radius, 0, 0, angle=theta / 2, start_angle=theta / 4, n_points=200
        )
        centre_arc = np.array([x, y, np.zeros(200)]).T
        points = self._local_to_global(centre_arc)
        ax.plot(*points.T, color="r")
        ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]], marker="^", color="r")
