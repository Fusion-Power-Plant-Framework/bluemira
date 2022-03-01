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

from functools import lru_cache
from typing import List, Union

import numpy as np
from periodictable import elements
from pint import UnitRegistry, set_application_registry

LOCALE = "en_GB"
ureg = UnitRegistry(
    fmt_locale=LOCALE, preprocessors=[lambda x: x.replace("%", " percent ")]
)
ureg.default_format = "~P"
set_application_registry(ureg)

SECOND = ureg.second
METRE = ureg.metre
KILOGRAM = ureg.kilogram
AMP = ureg.ampere
CELSIUS = ureg.celsius
MOL = ureg.mol
DEGREE = ureg.degree
DENSITY = KILOGRAM / METRE**3
PART_DENSITY = METRE**-3
FLUX_DENSITY = METRE**-2 / SECOND

ureg.define("displacements_per_atom  = count = dpa")
ureg.define("full_power_year = year = fpy")
ureg.define("percent = 0.01 count = %")

# =============================================================================
# Physical constants
# =============================================================================

# Speed of light
C_LIGHT = ureg.Quantity("c").to_base_units().magnitude  # [m/s]

# Vacuum permeability
MU_0 = ureg.Quantity("mu_0").to_base_units().magnitude  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
MU_0_4PI = 1e-7  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
ONE_4PI = 1 / (4 * np.pi)

# Gravitational constant
GRAVITY = ureg.Quantity("gravity").to_base_units().magnitude  # [m/s^2]  # nO ESCAPING

# Avogadro's number, [1/mol] Number of particles in a mol
N_AVOGADRO = ureg.Quantity("avogadro_number").to_base_units().magnitude

# Stefan-Boltzmann constant: black-body radiation constant of proportionality
SIGMA_BOLTZMANN = ureg.Quantity("sigma").to_base_units().magnitude  # [W/m^2.K^4]

# Boltzmann constant kB = R/N_a
K_BOLTZMANN = ureg.Quantity("k_B").to_base_units().magnitude  # [J/K]

# neutron molar mass, [u] or [g/mol]
NEUTRON_MOLAR_MASS = (
    ureg.Quantity("m_n").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude

# proton molar mass, [u] or [g/mol]
PROTON_MOLAR_MASS = (
    ureg.Quantity("m_p").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude

# electron molar mass, [u] or [g/mol]
ELECTRON_MOLAR_MASS = (
    ureg.Quantity("m_e").to("g") * ureg.Quantity("avogadro_constant").to_base_units()
).magnitude

# Tritium half-life
# https://www.nist.gov/pml/radiation-physics/radioactivity/radionuclide-half-life-measurements
# http://www.lnhb.fr/nuclear-data/nuclear-data-table/
# http://www.lnhb.fr/nuclides/H-3_tables.pdf
T_HALFLIFE = 12.312  # [yr]

# Tritium decay constant
T_LAMBDA = np.log(2) / T_HALFLIFE  # [1/yr]

# Tritium molar mass,  [u] or [g/mol]
T_MOLAR_MASS = elements.isotope("T").mass

# Deuterium molar mass, [u] or [g/mol]
D_MOLAR_MASS = elements.isotope("D").mass

# Helium molar mass, [u] or [g/mol]
HE_MOLAR_MASS = elements.isotope("He").mass

# Helium-3 molar mass, [u] or [g/mol]
HE3_MOLAR_MASS = elements.isotope("3-He").mass

# Absolute zero in Kelvin
ABS_ZERO_K = 0  # [K]

# Absolute zero in Celsius
ABS_ZERO_C = ureg.Quantity(0, ureg.kelvin).to("celsius").magnitude  # [°C]

# =============================================================================
# Conversions
# =============================================================================

# Electron-volts to Joules
EV_TO_J = ureg.Quantity(1, ureg.eV).to("joule").magnitude

# Joules to Electron-volts
J_TO_EV = ureg.Quantity(1, ureg.joule).to("eV").magnitude

# Atomic mass units to kilograms
AMU_TO_KG = ureg.Quantity(1, ureg.amu).to("kg").magnitude

# Years to seconds
YR_TO_S = ureg.Quantity(1, ureg.year).to("s").magnitude

# Seconds to years
S_TO_YR = ureg.Quantity(1, ureg.second).to("year").magnitude


def to_celsius(kelvin: Union[float, np.array, List[float]]) -> Union[float, np.array]:
    """
    Convert a temperature in Kelvin to Celsius.

    Parameters
    ----------
    kelvin: Union[float, np.array, List[float]]
        The temperature to convert [K]

    Returns
    -------
    temp_in_celsius: Union[float, np.array]
        The temperature [°C]
    """
    if np.any(np.less(kelvin, ABS_ZERO_K)):
        raise ValueError("Negative temperature in K specified.")
    return ureg.Quantity(kelvin, ureg.kelvin).to("celsius").magnitude


def to_kelvin(celsius: Union[float, np.array, List[float]]) -> Union[float, np.array]:
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp_in_celsius: Union[float, np.array, List[float]]
        The temperature to convert [°C]

    Returns
    -------
    temp_in_kelvin: Union[float, np.array]
        The temperature [K]
    """
    if np.any(np.less(celsius, ABS_ZERO_C)):
        raise ValueError("Negative temperature in K specified.")
    return ureg.Quantity(celsius, ureg.celsius).to("kelvin").magnitude


def from_keV(
    value: Union[float, np.array, List[float]],
    to: str = "celsius",
    *,
    _from: str = "keV"
):
    """
    Convert a temperature in keV to Celsius.

    Parameters
    ----------
    value: Union[float, np.array, List[float]]
        The temperature to convert [keV]
    to: str
        Unit to convert to eg celsius or kelvin
    from: str
        allows modification of keV prefix to eg eV or MeV etc

    Returns
    -------
    temp: Union[float, np.array]
        The temperature
    """
    return (ureg.Quantity(value, _from) / ureg.Quantity("k_B")).to(to).magnitude


def kgm3_to_gcm3(density: Union[float, np.array, List[float]]) -> Union[float, np.array]:
    """
    Convert a density in kg/m3 to g/cm3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [kg/m3]

    Returns
    -------
    density_gcm3 : Union[float, np.array]
        The density [g/cm3]
    """
    return density / 1000.0


def gcm3_to_kgm3(density: Union[float, np.array, List[float]]) -> Union[float, np.array]:
    """
    Convert a density in g/cm3 to kg/m3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [g/cm3]

    Returns
    -------
    density_kgm3 : Union[float, np.array]
        The density [kg/m3]
    """
    return density * 1000.0


def pam3s_to_mols(flow_in_pam3_s):
    """
    Convert a flow in Pa.m^3/s to a flow in mols.

    Parameters
    ----------
    flow_in_pam3_s: Union[float, np.array]
        The flow in Pa.m^3/s to convert

    Returns
    -------
    flow_in_mols: Union[float, np.array]
        The flow in mol/s

    Notes
    -----
    At 273.15 K for a diatomic gas
    """
    return (
        (ureg.Quantity(flow_in_pam3_s, "Pa m^3") / _pam3_mol_const())
        .to_base_units()
        .magnitude
    )


def mols_to_pam3s(flow_in_mols):  # noqa :N802
    """
    Convert a flow in mols to a flow in Pa.m^3/s.

    Parameters
    ----------
    flow_in_mols: Union[float, np.array]
        The flow in mol/s to convert

    Returns
    -------
    flow_in_pam3_s: Union[float, np.array]
        The flow in Pa.m^3/s

    Notes
    -----
    At 273.15 K for a diatomic gas
    """
    return (
        (ureg.Quantity(flow_in_mols, "mole") * _pam3_mol_const())
        .to_base_units()
        .magnitude
    )


@lru_cache(1)
def _pam3_mol_const():
    return ureg.Quantity("molar_gas_constant") * ureg.Quantity(0, ureg.celsius).to(
        "kelvin"
    )


# =============================================================================
# Working constants
# =============================================================================

# Numpy's default float precision limit
EPS = np.finfo(float).eps

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
