# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

import numpy as np

from bluemira.base.constants import (
    ELECTRON_MASS,
    EPS_0,
    EV_TO_J,
    H_PLANCK,
    K_BOLTZMANN,
    MU_0,
    PROTON_MASS,
)


def debye_length(temperature, density):
    """
    Debye length

    Parameters
    ----------
    temperature: float
        Temperature [K]
    density: float
        Density [m^-3]

    Returns
    -------
    debye_length: float
        Debye length [m]
    """
    return np.sqrt(EPS_0 * K_BOLTZMANN * temperature / (EV_TO_J**2 * density))


def reduced_mass(mass_1, mass_2):
    """
    Calculate the reduced mass of a two-particle system

    Parameters
    ----------
    mass_1: float
        Mass of the first particle
    mass_2: float
        Mass of the second particle

    Returns
    -------
    mu_12: float
        Reduced mass
    """
    return (mass_1 * mass_2) / (mass_1 + mass_2)


def thermal_velocity(temperature, mass):
    """
    Parameters
    ----------
    temperature: float
        Temperature [K]
    mass: float
        Mass of the particle [kg]

    Notes
    -----
    The sqrt(2) term is for a 3-dimensional system and the most probable velocity in
    the particle velocity distribution.
    """
    return np.sqrt(2) * np.sqrt(K_BOLTZMANN * temperature / mass)


def de_broglie_length(velocity, mu_12):
    """
    Calculate the de Broglie wavelength

    Parameters
    ----------
    velocity: float
        Velocity [m/s]
    mu_12: float
        Reduced mass [kg]

    Returns
    -------
    lambda_de_broglie: float
        De Broglie wavelength [m]
    """
    return H_PLANCK / (2 * mu_12 * velocity)


def impact_parameter_perp(velocity, mu_12):
    """
    Calculate the perpendicular impact parameter

    Parameters
    ----------
    velocity: float
        Velocity [m/s]
    mu_12: float
        Reduced mass [kg]

    Returns
    -------
    b90: float
        Perpendicular impact parameter [m]
    """
    return EV_TO_J**2 / (4 * np.pi * EPS_0 * mu_12 * velocity**2)


def coulomb_logarithm(temperature, density):
    """
    Calculate the value of the Coulomb logarithm for an electron hitting a proton.

    Parameters
    ----------
    temperature: float
        Temperature [K]
    density: float
        Density [1/m^3]

    Returns
    -------
    ln_lambda: float
        Coulomb logarithm value
    """
    lambda_debye = debye_length(temperature, density)
    mu_12 = reduced_mass(ELECTRON_MASS, PROTON_MASS)
    v = thermal_velocity(temperature, ELECTRON_MASS)
    lambda_de_broglie = de_broglie_length(v, mu_12)
    b_perp = impact_parameter_perp(v, mu_12)
    b_min = max(lambda_de_broglie, b_perp)
    return np.log(np.sqrt(1 + (lambda_debye / b_min) ** 2))


def spitzer_conductivity(Z_eff, T_e, ln_lambda):
    """
    Formula for electrical conductivity in a plasma as per L. Spitzer.

    Parameters
    ----------
    Z_eff: float
        Effective charge [a.m.u.]
    T_e: float
        Electron temperature on axis [eV]
    ln_lambda: float
        Coulomb logarithm value

    Returns
    -------
    sigma: float
        Plasma resistivity [1/Ohm/m]

    Notes
    -----
    Spitzer and Haerm, 1953

    \t:math:`\\sigma = 1.92e4 (2-Z_{eff}^{-1/3}) \\dfrac{T_{e}^{3/2}}{Z_{eff}ln\\Lambda}`
    """
    return 1.92e4 * (2 - Z_eff ** (-1 / 3)) * T_e**1.5 / (Z_eff * ln_lambda)


def estimate_loop_voltage(R_0, B_t, Z_eff, T_e, n_e, q_0):
    """
    A 0-D estimate of the loop voltage during burn

    Parameters
    ----------
    R_0: float
        Major radius [m]
    B_t: float
        Toroidal field on axis [T]
    Z_eff: float
        Effective charge [a.m.u.]
    T_e: float
        Electron temperature on axis [eV]
    n_e: float
        Electron density [1/m^3]
    q_0: float
        Safety factor on axis

    Returns
    -------
    v_loop: float
        Loop voltage during burn [V]

    Notes
    -----
    H. Zohm, W. Morris (2022)

    \t:math:`v_{loop}=2\\pi R_{0}\\dfrac{2\\pi B_{t}}{\\mu_{0}q_{0}\\sigma_{0}R_{0}}`

    where :math:`\\sigma_{0}` is the Spitzer conductivity on axis:
    \t:math:`\\sigma_{0} = 1.92e4 (2-Z_{eff}^{-1/3}) \\dfrac{T_{e}^{3/2}}{Z_{eff}ln\\Lambda}`

    Assumes no non-inductive current on axis

    Assumes a circular cross-section on axis

    There is no neo-classical resistivity on axis because there are no trapped particles
    """  # noqa: W505
    ln_lambda = coulomb_logarithm(T_e * EV_TO_J / K_BOLTZMANN, n_e)
    sigma = spitzer_conductivity(Z_eff, T_e, ln_lambda)

    # Current density on axis
    j_0 = 2 * B_t / (MU_0 * q_0 * R_0)
    v_loop = 2 * np.pi * R_0 * j_0 / sigma
    return v_loop
