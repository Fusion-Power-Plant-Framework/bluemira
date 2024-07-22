# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tokamak plasma collision formulae.
"""

import numpy as np

from bluemira.base.constants import (
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    EPS_0,
    H_PLANCK,
    PROTON_MASS,
    raw_uc,
)


def debye_length(temperature: float, density: float) -> float:
    """
    Debye length

    Parameters
    ----------
    temperature:
        Temperature [K]
    density:
        Density [m^-3]

    Returns
    -------
    Debye length [m]

    Notes
    -----
    Debye length is given by the formula:

    .. math::
        \\lambda_D = \\sqrt{\\frac{\\varepsilon_0 k_B T}{n e^2}}

    where:

    - :math:`\\varepsilon_0` is the vacuum permittivity,

    - :math:`k_B` is the Boltzmann constant,

    - :math:`T` is the temperature in Kelvin,

    - :math:`n` is the number density of particles,

    - :math:`e` is the elementary charge.

    The conversion of the temperature from Kelvin to Joules
    implicitly includes the Boltzmann constant (:math:`k_B`).

    """
    return np.sqrt(
        EPS_0 * raw_uc(temperature, "K", "J") / (ELEMENTARY_CHARGE**2 * density)
    )


def reduced_mass(mass_1: float, mass_2: float) -> float:
    """
    Calculate the reduced mass of a two-particle system

    Parameters
    ----------
    mass_1:
        Mass of the first particle
    mass_2:
        Mass of the second particle

    Returns
    -------
    Reduced mass

    Notes
    -----
    Reduced mass of a two-particle system :

    .. math::

        \\mu_{AB} = \\frac{m_A m_B}{m_A + m_B}

    where :math:`m_A` and :math:`m_B` are the masses
    of the particles.
    """
    return (mass_1 * mass_2) / (mass_1 + mass_2)


def thermal_velocity(temperature: float, mass: float) -> float:
    """
    Parameters
    ----------
    temperature:
        Temperature [K]
    mass:
        Mass of the particle [kg]

    Returns
    -------
    Thermal velocity [m/s]

    Notes
    -----
    The thremal velocity is calculated as

     .. math::

        \\sqrt{\\frac{2 k_B T}{m}}

    The conversion of the temperature from Kelvin to Joules
    implicitly includes the Boltzmann constant (  :math:`k_B` ).

    The  :math:`\\sqrt 2` term is for a 3-dimensional system and the
    most probable velocity in the particle velocity distribution.

    """
    return np.sqrt(2) * np.sqrt(
        raw_uc(temperature, "K", "J")  # = Joule = kg*m^2/s^2
        / mass
    )  # sqrt(m^2/s^2) = m/s


def de_broglie_length(velocity: float, mu_12: float) -> float:
    """
    Calculate the de Broglie wavelength

    Parameters
    ----------
    velocity:
        Velocity [m/s]
    mu_12:
        Reduced mass [kg]

    Returns
    -------
    De Broglie wavelength [m]

    Notes
    -----
    The de Broglie length is given by

    .. math::

        \\lambda_{th} = \\frac{h}{2 \\cdot \\mu_{12} \\cdot velocity}

    where  :math:`h` is the Planck Constant.

    """
    return H_PLANCK / (2 * mu_12 * velocity)


def impact_parameter_perp(velocity: float, mu_12: float) -> float:
    """
    Calculate the perpendicular impact parameter, a.k.a. b90

    Parameters
    ----------
    velocity:
        Velocity [m/s]
    mu_12:
        Reduced mass [kg]

    Returns
    -------
    Perpendicular impact parameter [m]

    Notes
    -----

    .. math::

        b_{90} = \\frac{e^2}{4 \\pi \\epsilon_0 \\mu_{12} v^2}

    where:

    - :math:`e` is the elementary charge (absolute charge of an electron)

    - :math:`\\epsilon_0` is the vacuum permittivity,

    - :math:`\\mu_{12}` is the reduced mass,

    - :math:`v` is the relative velocity.
    """
    return ELEMENTARY_CHARGE**2 / (4 * np.pi * EPS_0 * mu_12 * velocity**2)


def coulomb_logarithm(temperature: float, density: float) -> float:
    """
    Calculate the value of the Coulomb logarithm for an electron hitting a proton.

    Parameters
    ----------
    temperature:
        Temperature [K]
    density:
        Density [1/m^3]

    Returns
    -------
    Coulomb logarithm value

    Notes
    -----
    The Coulomb logarithm is calculated using the formula:

    .. math::

        \\ln{\\Lambda} = \\ln{\\left(1 + \\left(\\frac{\\lambda_{Debye}}
        {b_{min}}\\right)^2\\right)^{1/2}}

    where:

    :math:`\\lambda_{Debye}` is the Debye length,

    :math:`b_{min}` is the minimum impact parameter, which
    is defined as the maximum value between :math:`b_{90}`
    (the perpendicular impact parameter) and the de Broglie
    wavelegth :math:`\\lambda_{th}`

    """
    lambda_debye = debye_length(temperature, density)
    mu_12 = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    v = thermal_velocity(temperature, ELECTRON_MASS)
    lambda_de_broglie = de_broglie_length(v, mu_12)
    b_perp = impact_parameter_perp(v, mu_12)
    b_min = max(lambda_de_broglie, b_perp)
    return np.log(np.sqrt(1 + (lambda_debye / b_min) ** 2))


def spitzer_conductivity(Z_eff: float, T_e: float, ln_lambda: float) -> float:
    """
    Formula for electrical conductivity in a plasma as per L. Spitzer.

    Parameters
    ----------
    Z_eff:
        Effective charge [dimensionless]
    T_e:
        Electron temperature on axis [keV]
        The equation takes in temperature as [eV], so an in-line conversion is used here.
    ln_lambda:
        Coulomb logarithm value

    Returns
    -------
    Plasma resistivity [1/Ohm/m]

    Notes
    -----
    Spitzer and Haerm, 1953

    \t:math:`\\sigma = 1.92e4 (2-Z_{eff}^{-1/3}) \\dfrac{T_{e}^{3/2}}{Z_{eff}ln\\Lambda}`
    """
    return (
        1.92e4
        * (2 - Z_eff ** (-1 / 3))
        * raw_uc(T_e, "keV", "eV") ** 1.5
        / (Z_eff * ln_lambda)
    )
