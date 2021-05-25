# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Analytical expressions for the field inside a circular current arc of
rectangular cross-section, following equations as described in:

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1064259
"""
import numpy as np
from BLUEPRINT.magnetostatics.utilities import jit_llc3, jit_llc4, integrate

__all__ = ["Bx_analytical_circular", "Bz_analytical_circular"]


# Full integrands free of singularities


@jit_llc4
def brc_integrand_full(psi, r_pc, r_j, z_k):
    """
    Calculate the Brc integrand without singularities.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    return cos_psi * sqrt_term + r_pc * cos_psi ** 2 * np.log(
        r_j - r_pc * cos_psi + sqrt_term
    )


@jit_llc4
def bzc_integrand_full(psi, r_pc, r_j, z_k):
    """
    Calculate the Bzc integrand without singularities.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    result = -z_k * np.log(r_j - r_pc * cos_psi + sqrt_term) - r_pc * cos_psi * np.log(
        -z_k + sqrt_term
    )
    if sin_psi != 0 and r_pc != 0:
        result += (
            r_pc
            * sin_psi
            * np.arctan((z_k * (r_j - r_pc * cos_psi)) / (r_pc * sin_psi * sqrt_term))
        )
    return result


# Integrands to treat singularities


@jit_llc4
def brc_integrand_p1(psi, r_pc, r_j, z_k):
    """
    Calculate the first part of the Brc integrand (no singularities)

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    return cos_psi * sqrt_term


@jit_llc4
def bf1_r_pccos2_integrand(psi, r_pc, r_j, z_k):
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    return r_pc * cos_psi ** 2 * np.log(r_j - r_pc * cos_psi + sqrt_term)


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p1(psi, r_pc, r_j):
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 1

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi ** 2
        * np.log(
            (r_pc * cos_psi - r_j)
            + np.sqrt((r_pc * cos_psi - r_j) ** 2 + r_pc ** 2 * np.sin(psi) ** 2)
        )
    )


@jit_llc3
def bf1_r_pccos2_0_pi_integrand_p2(psi, r_pc, r_j):
    """
    Calculate the BF1(r_pc*cos(psi)^2) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0. Part 2

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc
        * cos_psi ** 2
        * np.log(
            (r_j - r_pc * cos_psi)
            + np.sqrt((r_j - r_pc * cos_psi) ** 2 + r_pc ** 2 * np.sin(psi) ** 2)
        )
    )


@jit_llc4
def bf1_zk_integrand(psi, r_pc, r_j, z_k):
    """
    Calculate the BF1(-z_k) integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    return -z_k * np.log(r_j - r_pc * cos_psi + sqrt_term)


@jit_llc4
def bf2_integrand(psi, r_pc, r_j, z_k):
    """
    Calculate the BF2 integrand.

    For r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point
    """
    cos_psi = np.cos(psi)
    sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
    return -r_pc * cos_psi * np.log(-z_k + sqrt_term)


@jit_llc3
def bf2_0_pi_integrand(psi, r_pc, z_k):
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j == r_pc and z_k >= 0.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point for 0 to pi integral
    """
    cos_psi = np.cos(psi)
    return (
        r_pc * cos_psi * np.log(z_k + np.sqrt(2 * r_pc ** 2 * (1 - cos_psi) + z_k ** 2))
    )


@jit_llc4
def bf3_integrand(psi, r_pc, r_j, z_k):
    """
    Calculate the BF3 integrand

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    psi: float
        Angle [rad]

    Returns
    -------
    result: float
        The result of the integrand at a single point

    Notes
    -----
    Treats the sin(psi) = 0 singularity
    """
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if sin_psi != 0:
        sqrt_term = np.sqrt(r_pc ** 2 - 2 * r_pc * r_j * cos_psi + r_j ** 2 + z_k ** 2)
        return (
            r_pc
            * sin_psi
            * np.arctan((z_k * (r_j - r_pc * cos_psi)) / (r_pc * sin_psi * sqrt_term))
        )
    return 0


# More singularity treatments...


def bf1_r_pccos2_zk0_0_pi(r_pc, r_j):
    """
    Calculate the BF1(r_pc*cos(psi)^2) integral for 0 to pi.

    From 0 to pi for r_j <= r_pc and z_k == 0

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]

    Returns
    -------
    BF1(r_pc*cos(psi)^2): float
        The result of the integral at a single point for 0 to pi integral
    """
    if r_pc == r_j:
        return r_pc * (0.5 * np.pi * np.log(r_pc) + 0.2910733)
    # r_j < r_pc
    result = 0.5 * np.pi * r_pc * (np.log(r_pc) - np.log(2) - 0.5)
    result -= integrate(bf1_r_pccos2_0_pi_integrand_p1, (r_pc, r_j), 0, 0.5 * np.pi)
    result += integrate(bf1_r_pccos2_0_pi_integrand_p2, (r_pc, r_j), 0.5 * np.pi, np.pi)
    return result


def bf2_rj_rpc_0_pi(r_pc, z_k):
    """
    Calculate the BF2 integrand for a 0 to pi integral

    From 0 to pi for r_j < r_pc and z_k == 0.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]

    Returns
    -------
    result: float
        The result of the integral for 0 to pi
    """
    return np.pi * r_pc + integrate(bf2_0_pi_integrand, (r_pc, z_k), 0, np.pi)


# Primitive functions


def primitive_brc(r_pc, r_j, z_k, phi_pc, theta):
    """
    Calculate the Brc primitives and treat singularities.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    phi_pc: float
        Angle of the point at which to evaluate field [rad]
    theta: float
        Azimuthal angle of the circular arc

    Returns
    -------
    result: float
        The result of the Brc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    singularities = (z_k == 0) and (r_j <= r_pc) and (0 <= phi_pc <= theta)
    if not singularities:
        # No singularities
        return integrate(brc_integrand_full, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    # Singularity free treatment of first term
    result = integrate(brc_integrand_p1, args, -phi_pc, theta - phi_pc)

    # Dodge singularities in second term
    if phi_pc == 0:
        if theta == np.pi:
            result += bf1_r_pccos2_zk0_0_pi(r_pc, r_j)

        if theta == 2 * np.pi:
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
        result -= integrate(
            bf1_r_pccos2_integrand, args, phi_pc - theta, 2 * np.pi - theta
        )

    return result


def primitive_bzc(r_pc, r_j, z_k, phi_pc, theta):
    """
    Calculate the Bzc primitives and treat singularities.

    Parameters
    ----------
    r_pc: float
        The radius of the point at which to evaluate field [m]
    r_j: float
        The radius (inner or outer) of the coil [m]
    z_k: float
        The z coordinate (upper or lower) of the coil [m]
    phi_pc: float
        Angle of the point at which to evaluate field [rad]
    theta: float
        Azimuthal angle of the circular arc

    Returns
    -------
    result: float
        The result of the Bzc primitive
    """
    args = (r_pc, r_j, z_k)  # The function arguments for integration
    bf1_singularities = (z_k == 0) and (r_j <= r_pc) and (0 <= phi_pc <= theta)
    bf2_singularities = (r_j == r_pc) and (z_k >= 0) and (0 <= phi_pc <= theta)
    bf3_singularities = r_pc == 0
    if not bf1_singularities and not bf2_singularities and not bf3_singularities:
        # No singularities
        return integrate(bzc_integrand_full, args, -phi_pc, theta - phi_pc)

    # Treat singularities
    result = 0
    if bf1_singularities:
        # Treat BF1(-z_k)
        if phi_pc == 0:
            if theta != np.pi and theta != 2 * np.pi:
                # At pi and 2 * pi the BF1 integral is 0
                # Elsewhere:
                # result += 0 (the first part of BF1 is 0)
                result -= integrate(bf1_zk_integrand, args, -theta, np.pi - theta)

        if phi_pc == theta:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(bf1_zk_integrand, args, np.pi, np.pi - theta)

        else:
            # result += 0 (the first part of BF1 is 0)
            result -= integrate(
                bf1_zk_integrand, args, phi_pc - theta, 2 * np.pi - theta
            )

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
            result -= integrate(bf2_integrand, args, phi_pc - theta, 2 * np.pi - theta)

    else:
        # BF2 is normal
        result += integrate(bf2_integrand, args, -phi_pc, theta - phi_pc)

    if r_pc != 0:
        # r_pc = 0, BF3 evaluates to 0
        result += integrate(bf3_integrand, args, -phi_pc, theta - phi_pc)

    return result


# Full field calculations in working coordinates


def Bx_analytical_circular(r1, r2, z1, z2, theta, r_p, theta_p):
    """
    Calculate magnetic field in the local x coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1: float
        Inner coil radius [m]
    r2: float
        Outer coil radius [m]
    z1: float
        The first modified z coordinate [m]
    z2: float
        The second modified z coordinate [m]
    theta: float
        Azimuthal angle of the circular arc [rad]
    r_p: float
        The radius of the point at which to evaluate the field [m]
    theta_p: float
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    Bx: float
        The magnetic field response in the x coordinate direction
    """
    return (
        primitive_brc(r_p, r1, z1, theta_p, theta)
        - primitive_brc(r_p, r1, z2, theta_p, theta)
        - primitive_brc(r_p, r2, z1, theta_p, theta)
        + primitive_brc(r_p, r2, z2, theta_p, theta)
    )


def Bz_analytical_circular(r1, r2, z1, z2, theta, r_p, theta_p):
    """
    Calculate magnetic field in the local z coordinate direction due to a
    circular arc current source.

    Parameters
    ----------
    r1: float
        Inner coil radius [m]
    r2: float
        Outer coil radius [m]
    z1: float
        The first modified z coordinate [m]
    z2: float
        The second modified z coordinate [m]
    theta: float
        Azimuthal angle of the circular arc [rad]
    r_p: float
        The radius of the point at which to evaluate the field [m]
    theta_p: float
        The angle of the point at which to evaluate the field [rad]

    Returns
    -------
    Bz: float
        The magnetic field response in the z coordinate direction
    """
    return (
        primitive_bzc(r_p, r1, z1, theta_p, theta)
        - primitive_bzc(r_p, r1, z2, theta_p, theta)
        - primitive_bzc(r_p, r2, z1, theta_p, theta)
        + primitive_bzc(r_p, r2, z2, theta_p, theta)
    )
