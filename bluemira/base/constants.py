# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of generic physical constants, conversions, and miscellaneous constants.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from periodictable import elements
from pint import Context, Quantity, Unit, UnitRegistry, set_application_registry
from pint.util import UnitsContainer

if TYPE_CHECKING:
    from collections.abc import Callable


class CoilType(Enum):
    """
    CoilType Enum
    """

    PF = auto()
    CS = auto()
    DUM = auto()
    NONE = auto()

    @classmethod
    def _missing_(cls, value: str | CoilType) -> CoilType:
        if not isinstance(value, str):
            raise TypeError("Input must be a string.")
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}."
                f" Select from {(*cls._member_names_,)}"
            ) from None


class BMUnitRegistry(UnitRegistry):
    """
    Bluemira UnitRegistry

    Extra conversions:

    eV <-> Kelvin
    Pa m^3 <-> mol

    Extra units:

    displacements_per_atom (dpa)
    full_power_year (fpy)
    atomic_parts_per_million (appm)
    USD ($)

    """

    def __init__(self):
        # Preprocessor replacements have spaces so
        # the units dont become prefixes or get prefixed
        # space before on % so that M% is not a thing
        # M$ makes sense if a bit non-standard
        super().__init__(
            fmt_locale="en_GB",
            preprocessors=[
                lambda x: x.replace("$", "USD "),
            ],
        )

        self._gas_flow_temperature = None
        self._contexts_added = False

    def _add_contexts(self, contexts: list[Context] | None = None):
        """
        Add new contexts to registry
        """
        if not self._contexts_added:
            self.contexts = [
                self._energy_temperature_context(),
                self._mass_energy_context(),
                self._flow_context(),
            ]

            for c in self.contexts:
                self.add_context(c)

            self._contexts_added = True

        if contexts:
            for c in contexts:
                self.add_context(c)

    def enable_contexts(self, *contexts: Context, **kwargs):
        """
        Enable contexts
        """
        self._add_contexts(contexts)

        super().enable_contexts(*[*self.contexts, *contexts], **kwargs)
        # Extra units
        self.define("displacements_per_atom  = count = dpa")
        self.define("full_power_year = year = fpy")
        self.define("@alias ppm = atomic_parts_per_million = appm")
        # Other currencies need to be set up in a new context
        self.define("USD = [currency]")

    def _energy_temperature_context(self):
        """
        Converter between energy and temperature

        temperature = energy / k_B

        Returns
        -------
        pint context

        """
        e_to_t = Context("Energy_to_Temperature")

        t_units = "[temperature]"
        ev_units = "[energy]"

        conversion = self.Quantity("k_B")

        return self._transform(
            e_to_t,
            t_units,
            ev_units,
            lambda _, x: x * conversion,
            lambda _, x: x / conversion,
        )

    def _mass_energy_context(self):
        """
        Converter between mass and energy

        energy = mass * speed-of-light^2

        Returns
        -------
        pint context
        """
        m_to_e = Context("Mass_to_Energy")

        m_units = "[mass]"
        e_units = "[energy]"

        conversion = self.Quantity("c^2")

        return self._transform(
            m_to_e,
            m_units,
            e_units,
            lambda _, x: x * conversion,
            lambda _, x: x / conversion,
        )

    @property
    def flow_conversion(self):
        """Gas flow conversion factor R * T"""
        return self.Quantity("molar_gas_constant") * self.gas_flow_temperature

    @property
    def gas_flow_temperature(self):
        """
        Gas flow temperature in kelvin

        If Quantity provided to setter it will convert units (naïvely)
        """
        if self._gas_flow_temperature is None:
            self._gas_flow_temperature = self.Quantity(0, "celsius").to("kelvin")
        return self._gas_flow_temperature

    @gas_flow_temperature.setter
    def gas_flow_temperature(self, value: float | Quantity | None):
        self._gas_flow_temperature = (
            value.to("kelvin")
            if isinstance(value, Quantity)
            else value
            if value is None
            else self.Quantity(value, "kelvin")
        )

    def _flow_context(self):
        """
        Convert between flow in mol and Pa m^3

        Pa m^3 = R * temperature * mol

        https://en.wikipedia.org/wiki/Standard_temperature_and_pressure#Molar_volume_of_a_gas

        Returns
        -------
        pint context

        """
        mols_to_pam3 = Context("Mol to Pa.m^3 for a gas")

        mol_units = "[substance]"
        pam3_units = "[energy]"

        return self._transform(
            mols_to_pam3,
            mol_units,
            pam3_units,
            lambda ureg, x: x * ureg.flow_conversion,
            lambda ureg, x: x / ureg.flow_conversion,
        )

    @staticmethod
    def _transform(
        context: Context,
        units_from: str,
        units_to: str,
        forward_transform: Callable[[UnitRegistry, complex | Quantity], float],
        reverse_transform: Callable[[UnitRegistry, complex | Quantity], float],
    ) -> Context:
        formatters = ["{}", "{} / [time]"]

        for form in formatters:
            context.add_transformation(
                form.format(units_from), form.format(units_to), forward_transform
            )
            context.add_transformation(
                form.format(units_to), form.format(units_from), reverse_transform
            )

        return context


ureg = BMUnitRegistry()
ureg.enable_contexts()
set_application_registry(ureg)

# For reference
TIME = ureg.second
LENGTH = ureg.metre
MASS = ureg.kilogram
CURRENT = ureg.ampere
TEMP = ureg.kelvin
QUANTITY = ureg.mol
ANGLE = ureg.degree
DENSITY = MASS / LENGTH**3
PART_DENSITY = LENGTH**-3
FLUX_DENSITY = LENGTH**-2 / TIME

base_unit_defaults = {
    "[time]": TIME,
    "[length]": LENGTH,
    "[mass]": MASS,
    "[current]": CURRENT,
    "[temperature]": TEMP,
    "[substance]": QUANTITY,
    "[luminosity]": "candela",
    "[angle]": ANGLE,
}

combined_unit_defaults = {
    "[energy]": "joules",
    "[pressure]": "pascal",
    "[magnetic_field]": "tesla",
    "[electric_potential]": "volt",
    "[power]": "watt",
    "[force]": "newton",
    "[resistance]": "ohm",
}

combined_unit_dimensions = {
    "[energy]": {"[length]": 2, "[mass]": 1, "[time]": -2},
    "[pressure]": {"[length]": -1, "[mass]": 1, "[time]": -2},
    "[magnetic_field]": {"[current]": -1, "[mass]": 1, "[time]": -2},
    "[electric_potential]": {"[length]": 2, "[mass]": 1, "[time]": -2},
    "[power]": {"[length]": 2, "[mass]": 1, "[time]": -3},
    "[force]": {"[length]": 2, "[mass]": 1, "[time]": -3},
    "[resistance]": {"[current]": -2, "[length]": 2, "[mass]": 1, "[time]": -3},
}

ANGLE_UNITS = [
    "radian",
    "turn",
    "degree",
    "arcminute",
    "arcsecond",
    "milliarcsecond",
    "grade",
    # "mil",  # this break milli conversion with current implementation
    "steradian",
    "square_degree",
]

# =============================================================================
# Physical constants
# =============================================================================

# Speed of light
C_LIGHT = ureg.Quantity("c").to_base_units().magnitude  # [m/s]

# Vacuum permeability
MU_0 = ureg.Quantity("mu_0").to_base_units().magnitude  # [T.m/A] or [V.s/(A.m)]

# Vacuum permittivity
EPS_0 = ureg.Quantity("eps_0").to_base_units().magnitude  # [A^2.s^4/kg/m^3]

# absolute charge of an electron
ELEMENTARY_CHARGE = ureg.Quantity("e").to_base_units().magnitude  # [e]

# Commonly used..
MU_0_2PI = 2e-7  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
MU_0_4PI = 1e-7  # [T.m/A] or [V.s/(A.m)]

# Commonly used..
ONE_4PI = 1 / (4 * np.pi)

# Gravitational constant
GRAVITY = ureg.Quantity("gravity").to_base_units().magnitude  # [m/s^2]  # nO ESCAPING

# Avogadro's number, [1/mol] Number of particles in a mol
N_AVOGADRO = ureg.Quantity("avogadro_number").to_base_units().magnitude

# Stefan-Boltzmann constant: black-body radiation constant of proportionality
SIGMA_BOLTZMANN = ureg.Quantity("sigma").to_base_units().magnitude  # [W/(m^2.K^4)]

# Boltzmann constant kB = R/N_a
K_BOLTZMANN = ureg.Quantity("k_B").to_base_units().magnitude  # [J/K]

# Plank constant
H_PLANCK = ureg.Quantity("hbar").to_base_units().magnitude

# Electron charge, [C]
E_CHARGE = ureg.Quantity("e").to_base_units().magnitude

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


# electron mass [kg]
ELECTRON_MASS = ureg.Quantity("m_e").to_base_units().magnitude

# proton mass [kg]
PROTON_MASS = ureg.Quantity("m_p").to_base_units().magnitude

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
ABS_ZERO_C = ureg.Quantity(0, ureg.kelvin).to(ureg.celsius).magnitude  # [°C]

ABS_ZERO = {ureg.kelvin: ABS_ZERO_K, ureg.celsius: ABS_ZERO_C}

# =============================================================================
# Conversions
# =============================================================================

# Years to seconds
YR_TO_S = ureg.Quantity(1, ureg.year).to(ureg.second).magnitude

# Seconds to years
S_TO_YR = ureg.Quantity(1, ureg.second).to(ureg.year).magnitude


def units_compatible(unit_1: str, unit_2: str) -> bool:
    """
    Test if units are compatible.

    Parameters
    ----------
    unit_1:
        unit 1 string
    unit_2:
        unit 2 string

    Returns
    -------
    :
        True if compatible, False otherwise
    """
    try:
        raw_uc(1, unit_1, unit_2)
    except ValueError:
        return False
    else:
        return True


def raw_uc(
    value: npt.ArrayLike,
    unit_from: str | ureg.Unit,
    unit_to: str | ureg.Unit,
) -> float | np.ndarray:
    """
    Raw unit converter

    Converts a value from one unit to another

    Parameters
    ----------
    value:
        value to convert
    unit_from:
        unit to convert from
    unit_to:
        unit to convert to

    Returns
    -------
    converted value
    """
    try:
        return (
            ureg.Quantity(value, ureg.Unit(unit_from)).to(ureg.Unit(unit_to)).magnitude
        )
    except ValueError:
        # Catch scales on units eg the ridculousness of this unit: 10^19/m^3
        unit_from_q = ureg.Quantity(unit_from)
        unit_to_q = ureg.Quantity(unit_to)
        return (
            ureg.Quantity(value * unit_from_q).to(unit_to_q.units).magnitude
            / unit_to_q.magnitude
        )


def gas_flow_uc(
    value: npt.ArrayLike,
    unit_from: str | ureg.Unit,
    unit_to: str | ureg.Unit,
    gas_flow_temperature: float | Quantity | None = None,
) -> int | float | np.ndarray:
    """
    Converts around Standard temperature and pressure for gas unit conversion.
    Accurate for Ideal gases.

    https://en.wikipedia.org/wiki/Standard_temperature_and_pressure#Molar_volume_of_a_gas

    Parameters
    ----------
    value:
        value to convert
    unit_from:
        unit to convert from
    unit_to:
        unit to convert to
    gas_flow_temperature:
        Gas flow temperature if not provided is 273.15 K,
        if not a `Quantity` the units are assumed to be kelvin

    Returns
    -------
    converted value
    """
    if gas_flow_temperature is not None:
        ureg.gas_flow_temperature = gas_flow_temperature
    try:
        return raw_uc(value, unit_from, unit_to)
    finally:
        ureg.gas_flow_temperature = None


def to_celsius(
    temp: npt.ArrayLike, unit: str | Unit = ureg.kelvin
) -> float | np.ndarray:
    """
    Convert a temperature in Kelvin to Celsius.

    Parameters
    ----------
    temp:
        The temperature to convert, default [K]
    unit:
        change the unit of the incoming value

    Returns
    -------
    The temperature [°C]
    """
    converted_val = raw_uc(temp, unit, ureg.celsius)
    _temp_check(ureg.celsius, converted_val)
    return converted_val


def to_kelvin(
    temp: npt.ArrayLike, unit: str | Unit = ureg.celsius
) -> float | np.ndarray:
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp:
        The temperature to convert, default [°C]
    unit:
        change the unit of the incoming value


    Returns
    -------
    The temperature [K]
    """
    converted_val = raw_uc(temp, unit, ureg.kelvin)
    _temp_check(ureg.kelvin, converted_val)
    return converted_val


def _temp_check(unit: Unit, val: complex | Quantity):
    """
    Check temperature is above absolute zero

    Parameters
    ----------
    unit:
        pint Unit
    val:
        value to check

    Raises
    ------
    ValueError
        if below absolute zero
    """
    if unit.dimensionality == UnitsContainer({"[temperature]": 1}) and np.any(
        np.less(
            val,
            ABS_ZERO.get(unit, ureg.Quantity(0, ureg.kelvin).to(unit).magnitude),
        )
    ):
        raise ValueError("Negative temperature in K specified.")


def kgm3_to_gcm3(
    density: npt.ArrayLike,
) -> float | np.ndarray:
    """
    Convert a density in kg/m3 to g/cm3

    Parameters
    ----------
    density:
        The density [kg/m3]

    Returns
    -------
    The density [g/cm3]
    """
    return raw_uc(density, "kg.m^-3", "g.cm^-3")


def gcm3_to_kgm3(
    density: npt.ArrayLike,
) -> float | np.ndarray:
    """
    Convert a density in g/cm3 to kg/m3

    Parameters
    ----------
    density:
        The density [g/cm3]

    Returns
    -------
    The density [kg/m3]
    """
    return raw_uc(density, "g.cm^-3", "kg.m^-3")


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


class RNGSeeds(Enum):
    """
    Random Seeds for necessary use cases
    """

    equilibria_harmonics = 2944412338698111642
    timeline_tools_lognorm = 6613659347120864846
    timeline_tools_truncnorm = 9523110846560405221
    timeline_tools_expo = 15335509124046896388
    timeline_outages = 5876826953682921855
