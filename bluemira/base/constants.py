# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
A collection of generic physical constants, conversions, and miscellaneous constants.
"""

import numpy as np
import seaborn as sns


# =============================================================================
# Physical constants
# =============================================================================

# Speed of light
C_LIGHT = 299792458  # [m/s]

# Vacuum permeability
MU_0 = 4 * np.pi * 1e-7  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
MU_0_4PI = 1e-7  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
ONE_4PI = 1 / (4 * np.pi)

# Gravitational constant
GRAVITY = 9.81  # [m/s^2]  # nO ESCAPING

# Avogadro's number
N_AVOGADRO = 6.02214e23  # [1/mol] Number of particles in a mol

# Stefan-Boltzmann constant: black-body radiation constant of proportionality
SIGMA_BOLTZMANN = 5.670367e-8  # [W/m^2.K^4]

# Boltzmann constant kB = R/N_a
K_BOLTZMANN = 1.38064852e-23  # [J/K]

# Tritium half-life
T_HALFLIFE = 12.32  # [yr]

# Tritium decay constant
T_LAMBDA = np.log(2) / T_HALFLIFE  # [1/yr]

# Tritium molar mass
T_MOLAR_MASS = 3.016050  # [u] or [g/mol]

# Deuterium molar mass
D_MOLAR_MASS = 2.014102  # [u] or [g/mol]

# Helium molar mass
HE_MOLAR_MASS = 4.002603  # [u] or [g/mol]

# Helium-3 molar mass
HE3_MOLAR_MASS = 3.0160293  # [u] or [g/mol]

# neutron molar mass
NEUTRON_MOLAR_MASS = 1.008665  # [u] or [g/mol]

# proton molar mass
PROTON_MOLAR_MASS = 1.007276466879  # [u] or [g/mol]

# electron molar mass
ELECTRON_MOLAR_MASS = 5.48579909070e-4  # [u] or [g/mol]

# Absolute zero in Kelvin
ABS_ZERO_K = 0  # [K]

# Absolute zero in Celsius
ABS_ZERO_C = -273.15  # [°C]

# =============================================================================
# Conversions
# =============================================================================

# Electron-volts to Joules
EV_TO_J = 1.602176565e-19

# Joules to Electron-volts
J_TO_EV = 1 / EV_TO_J

# Atomic mass units to kilograms
AMU_TO_KG = 1.660539040e-27

# Years to seconds
YR_TO_S = 60 * 60 * 24 * 365

# Seconds to years
S_TO_YR = 1 / YR_TO_S

# =============================================================================
# Working constants
# =============================================================================

# Numpy's default float precision limit
EPS = np.finfo(np.float).eps

# Levi Civita Tensors
E_IJK = np.zeros((3, 3, 3))
E_IJK[0, 1, 2] = E_IJK[1, 2, 0] = E_IJK[2, 0, 1] = 1
E_IJK[0, 2, 1] = E_IJK[2, 1, 0] = E_IJK[1, 0, 2] = -1

E_IJ = np.array([[0, 1], [-1, 0]])

E_I = np.array([1])

# =============================================================================
# Alphabets
# =============================================================================

GREEK_ALPHABET = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]

GREEK_ALPHABET_CAPS = [s.capitalize() for s in GREEK_ALPHABET]

# =============================================================================
# Colors
# =============================================================================

EXIT_COLOR = "\x1b[0m"

ANSI_COLOR = {
    "white": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "orange": "\x1b[38;5;208m",
    "blue": "\x1b[38;5;27m",
    "purple": "\x1b[35m",
    "cyan": "\x1b[36m",
    "lightgrey": "\x1b[37m",
    "darkgrey": "\x1b[90m",
    "lightred": "\x1b[91m",
    "lightgreen": "\x1b[92m",
    "yellow": "\x1b[93m",
    "lightblue": "\x1b[94m",
    "pink": "\x1b[95m",
    "lightcyan": "\x1b[96m",
    "darkred": "\x1b[38;5;124m",
}


# This is specifically NOT the MATLAB color palette.
BLUEMIRA_PAL_MAP = {
    "blue": "#0072c2",
    "orange": "#d85319",
    "yellow": "#f0b120",
    "purple": "#7d2f8e",
    "green": "#75ac30",
    "cyan": "#4cbdf0",
    "red": "#a21430",
    "pink": "#f77ec7",
    "grey": "#a8a495",
}


BLUEMIRA_PALETTE = sns.color_palette(list(BLUEMIRA_PAL_MAP.values()))
