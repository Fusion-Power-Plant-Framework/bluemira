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
Tokamak plasma collision formulae.
"""

import numpy as np

from bluemira.base.constants import (
    ELECTRON_MASS,
    EPS_0,
    EV_TO_J,
    H_PLANCK,
    K_BOLTZMANN,
    PROTON_MASS,
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
    """
    return np.sqrt(EPS_0 * K_BOLTZMANN * temperature / (EV_TO_J**2 * density))


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
    The sqrt(2) term is for a 3-dimensional system and the most probable velocity in
    the particle velocity distribution.
    """
    return np.sqrt(2) * np.sqrt(K_BOLTZMANN * temperature / mass)


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
    """
    return H_PLANCK / (2 * mu_12 * velocity)


def impact_parameter_perp(velocity: float, mu_12: float):
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
    """
    return EV_TO_J**2 / (4 * np.pi * EPS_0 * mu_12 * velocity**2)


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
        Effective charge [a.m.u.]
    T_e:
        Electron temperature on axis [eV]
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
    return 1.92e4 * (2 - Z_eff ** (-1 / 3)) * T_e**1.5 / (Z_eff * ln_lambda)
