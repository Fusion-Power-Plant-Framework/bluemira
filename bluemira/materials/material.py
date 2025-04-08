# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
The home of base material objects. Use classes in here to make new materials.
"""

from __future__ import annotations

import abc
from dataclasses import KW_ONLY, asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import asteval
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

from bluemira.base.constants import to_celsius, to_kelvin
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.materials.constants import P_DEFAULT, T_DEFAULT
from bluemira.materials.error import MaterialsError
from bluemira.materials.tools import _try_calc_property, matproperty, to_openmc_material
from bluemira.utilities.tools import array_or_num, is_num

if TYPE_CHECKING:
    from collections.abc import Callable

    import openmc
    from numpy.typing import ArrayLike

# Set any custom symbols for use in asteval
asteval_user_symbols = {
    "PropsSI": PropsSI,
    "to_celsius": to_celsius,
    "to_kelvin": to_kelvin,
    "polynomial": np.polynomial,
}


class Material(Protocol):
    """Material Typing"""

    name: str
    density_unit: str
    percent_type: str
    temperature: float

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """Convert bluemira material to openmc material"""
        ...


@dataclass
class MaterialProperty:
    """
    Defines a property of a material within a valid temperature range.

    Parameters
    ----------
    value:
        If supplied as a string then this will define a temperature-, pressure-, and/or
        eps_vol-dependent calculation to be evaluated when the property is retrieved,
        otherwise it will define a constant value.
    temp_max:
        The maximum temperature [K] at which the property is valid. If not provided
        and no temp_max_celsius then all temperatures above 0K are valid.
    temp_min:
        The maximum temperature [K] at which the property is valid. If not
        provided and no temp_min_celsius then properties will be valid down to 0K.
    reference:
        The optional reference e.g. paper/database/website for the property.
    """

    value: float | str
    temp_max: float | None = None
    temp_min: float | None = None
    reference: str = ""
    obj: Any = field(default=None, repr=False)

    def __call__(
        self,
        temperature: float | None = None,
        pressure: float | None = None,
        B: float | None = None,
        eps_vol: float = 0.0,
    ) -> float | ArrayLike:
        """
        Evaluates the property at a given temperature, pressure, and/or eps_vol.

        Parameters
        ----------
        temperature:
            The temperature [K].
        pressure:
            The optional pressure [Pa].
        B:
            The magnetic flux density [T].
        esp_vol:
            The optional cell volume [m^3].

        Returns
        -------
        The property evaluated at the given temperature, pressure, and/or eps_vol.
        """
        if isinstance(self.value, str):
            aeval = asteval.Interpreter(usersyms=asteval_user_symbols)
            aeval.symtable["eps_vol"] = eps_vol

            if "temperature" in self.value:
                if temperature is None:
                    temperature = self.obj.temperature
                temperature = self._validate_temperature(temperature)
                aeval.symtable["temperature"] = temperature

            if "pressure" in self.value:
                if pressure is None and hasattr(self.obj, "pressure"):
                    pressure = self.obj.pressure
                pressure = self._validate_pressure(pressure)
                aeval.symtable["pressure"] = pressure

            if "magfield" in self.value:
                if B is None and hasattr(self.obj, "B"):
                    B = self.obj.B
                B = self._validate_B(B)
                aeval.symtable["B"] = B
            prop_val = array_or_num(aeval.eval(self.value))

            if len(aeval.error) > 0:
                raise aeval.error[0].exc(aeval.error[0].msg)

            return prop_val

        return self.value

    def to_dict(self):
        """
        Serialise the material property to a dictionary.

        Returns
        -------
        :
            Dictionary representation of the material property
        """
        return asdict(self)

    @staticmethod
    def _validate_B(B: float | ArrayLike):  # noqa: N802
        if B is None:
            raise ValueError("Magnetic Field is not set")
        return B

    @staticmethod
    def _validate_pressure(pressure: float | ArrayLike):
        if pressure is None:
            raise ValueError("Pressure is not set")
        return pressure

    def _validate_temperature(self, temperature: float | ArrayLike) -> float | ArrayLike:
        """
        Check that the property is valid for the requested temperature range.

        Parameters
        ----------
        temperature:
            The temperatures requested to value the property at.

        Returns
        -------
        :
            The validated temperatures

        Raises
        ------
        ValueError
            If any of the requested temperatures are outside of the valid range
        """
        if temperature is None:
            raise ValueError("Temperature is not set")
        temperatures = np.atleast_1d(temperature)
        if self.temp_min is not None and (temperatures < self.temp_min).any():
            bluemira_debug(f"obj: {self.obj}")
            bluemira_debug(f"{self.temp_min}, {self.temp_max}, {temperatures}")
            raise ValueError(
                "Material property not valid outside of temperature range: "
                f"{temperatures} < T_min = {self.temp_min}"
            )
        if self.temp_max is not None and (temperatures > self.temp_max).any():
            bluemira_debug(f"obj: {self.obj}")
            bluemira_debug(f"{self.temp_min}, {self.temp_max}, {temperatures}")
            raise ValueError(
                "Material property not valid outside of temperature range: "
                f"{temperature} > T_max = {self.temp_max}"
            )
        return temperatures


class MaterialPropertyDescriptor:
    """Material Property descriptor

    Converts various value inputs to a MaterialProperty
    """

    def __init__(self, _default=None):
        self._default = self._mutate_value(_default)

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> Callable[[], MaterialProperty] | MaterialProperty:
        """
        Returns
        -------
        :
            The MaterialProperty of the dataclass entry
        """
        if obj is None:
            return lambda: self._default
        if self._default.obj is None:
            self._default.obj = obj

        return getattr(obj, self._name, self._default)

    def _mutate_value(
        self,
        value: dict[str, float | str | None] | float | str | MaterialProperty | None,
        obj=None,
    ) -> MaterialProperty:
        if isinstance(value, dict):
            # Convert temp_*_celsius to temp_*
            if change := {"temp_min_celsius", "temp_max_celsius"}.intersection(value):
                if {"temp_min", "temp_max"}.difference(value):
                    for key in change:
                        value[key.rsplit("_", 1)[0]] = to_kelvin(value[key])
                for key in change:
                    del value[key]
            # empty dictionary
            value = MaterialProperty(**value, obj=obj) if value else self._default

        elif isinstance(value, float | int | str | type(None)):
            value = MaterialProperty(value=value, obj=obj)
        elif not isinstance(value, MaterialProperty):
            raise TypeError("Can't convert value to MaterialProperty")
        return value

    def __set__(
        self,
        obj: Any,
        value: dict[str, float | str | None]
        | float
        | str
        | Callable[[], MaterialProperty]
        | MaterialProperty
        | None,
    ):
        """
        Set the MaterialProperty of the dataclass entry
        """
        if callable(value) and not isinstance(value, MaterialProperty):
            value = value()
        setattr(obj, self._name, self._mutate_value(value, obj))


@dataclass
class Void:
    """
    Void material class.
    """

    name: str
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor(1)
    density_unit: str = "atom/cm3"
    temperature: float = T_DEFAULT
    percent_type: str = "ao"
    elements: dict[str, float] = field(default_factory=lambda: {"H": 1})
    zaid_suffix: str | None = None
    material_id: int | None = None

    @staticmethod
    def E(temperature: float | None = None) -> float:  # noqa: N802, ARG004
        """
        Returns
        -------
        :
            Young's modulus of the material at the given temperature.
        """
        return 0.0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Returns
        -------
        :
            Poisson's ratio at a given temperature.
        """
        return 0.0

    @staticmethod
    def rho(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Returns
        -------
        :
            The density at a given temperature.
        """
        return 0.0

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Returns
        -------
        :
            The openmc material
        """
        temperature = self.temperature if temperature is None else temperature
        return to_openmc_material(
            name=self.name,
            material_id=self.material_id,
            density=self.density(temperature),
            density_unit=self.density_unit,
            percent_type=self.percent_type,
            isotopes=None,
            elements=self.elements,
            zaid_suffix=self.zaid_suffix,
        )


@dataclass
class MassFractionMaterial:
    """
    Mass fraction material

    Parameters
    ----------
    name:
        The material's name.
    elements:
        The constituent elements and their corresponding mass fractions.
    nuclides:
        The constituent nuclides and their corresponding mass fractions.
    density:
        The density. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    density_unit:
        The unit that the density is supplied in, may be any one of kg/m3, g/cm3, g/cc,
        by default kg/m3.
    temperature:
        The temperature [K].
    zaid_suffix:
        The nuclear library to apply to the zaid, for example ".31c", this is used in
        MCNP and Serpent material cards.
    material_id:
        The id number or mat number used in the MCNP and OpenMC material cards.
    poissons_ratio:
        Poisson's ratio. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    thermal_conductivity:
        Thermal conductivity [W.m/K]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    youngs_modulus:
        Young's modulus [GPa]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    specific_heat:
        Specific heat [J/kg/K]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    coefficient_thermal_expansion:
        CTE [10^-6/T]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    electrical_resistivity:
        Electrical resistivity [(10^-8)Ohm.m]. If supplied as a string then this will
        define a temperature-dependent calculation, otherwise it will define a constant
        value.
    magnetic_saturation:
        Magnetic saturation [Am^2/kg]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    viscous_remanent_magnetisation:
        Viscous remanent magnetisation [Am^2/kg]. If supplied as a string then this will
        define a temperature-dependent calculation, otherwise it will define a constant
        value.
    coercive_field:
        Coercive field [A/m]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    minimum_yield_stress:
        Minimum yield stress [MPa]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    average_yield_stress:
        Average yield stress [MPa]. If supplied as a string then this will define a
        temperature-dependent calculation, otherwise it will define a constant value.
    minimum_ultimate_tensile_stress:
        Minimum ultimate tensile stress [MPa]. If supplied as a string then this will
        define a temperature-dependent calculation, otherwise it will define a constant
        value.
    average_ultimate_tensile_stress:
        Average ultimate tensile stress [MPa]. If supplied as a string then this will
        define a temperature-dependent calculation, otherwise it will define a constant
        value.
    """

    # Properties to interface with neutronics material maker
    name: str
    elements: dict[str, float] | None = None
    nuclides: dict[str, float] | None = None
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    density_unit: str = "kg/m3"
    temperature: float = T_DEFAULT
    zaid_suffix: str | None = None
    material_id: int | None = None
    percent_type: str = "wo"
    enrichment: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    enrichment_target: str | None = None
    enrichment_type: str | None = None

    # Engineering properties
    poissons_ratio: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    thermal_conductivity: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    youngs_modulus: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    specific_heat: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    coefficient_thermal_expansion: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )
    electrical_resistivity: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    magnetic_saturation: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    viscous_remanent_magnetisation: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )
    coercive_field: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    minimum_yield_stress: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    average_yield_stress: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    minimum_ultimate_tensile_stress: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )
    average_ultimate_tensile_stress: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )

    def __post_init__(
        self,
    ):
        """Value checking for required args marked as optional

        Raises
        ------
        MaterialsError
            Neither elements or nuclides provided
        """
        if not (self.elements or self.nuclides):
            raise MaterialsError("No elements or nuclides specified.")

    def __str__(self) -> str:
        """
        Returns
        -------
        :
            A string representation of the MfMaterial.
        """
        return self.name

    def __eq__(self, other: object) -> bool:
        """Equality Check"""  # noqa: DOC201
        if not isinstance(other, type(self)):
            # TODO @je-cook: a mixutre with only one component could be the same
            # 3654
            return False
        raise NotImplementedError("Material equality not implemented")

    def __hash__(self):
        """Hash of class"""  # noqa: DOC201
        return hash(self.elements)

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

        Returns
        -------
        :
            The openmc material

        """
        temperature = self.temperature if temperature is None else temperature
        return to_openmc_material(
            name=self.name,
            material_id=self.material_id,
            density=self.rho(temperature),
            density_unit=self.density_unit,
            percent_type=self.percent_type,
            isotopes=self.nuclides,
            elements=self.elements,
            enrichment=self.enrichment(temperature),
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
            temperature=temperature,
        )

    def mu(self, temperature: float) -> float:
        """
        Returns
        -------
        :
            Poisson's ratio
        """
        return _try_calc_property(self, "poissons_ratio", temperature)

    def k(self, temperature: float) -> float:
        """
        Returns
        -------
        :
            Thermal conductivity in W.m/K
        """
        return _try_calc_property(self, "thermal_conductivity", temperature)

    def E(self, temperature: float, **kwargs) -> float:  # noqa: N802, ARG002
        """
        Returns
        -------
        :
            Young's modulus in GPa
        """
        return _try_calc_property(self, "youngs_modulus", temperature)

    def Cp(self, temperature: float, **kwargs) -> float:  # noqa: N802, ARG002
        """
        Returns
        -------
        :
            Specific heat in J/kg/K
        """
        return _try_calc_property(self, "specific_heat", temperature)

    def CTE(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Mean coefficient of thermal expansion in 10**-6/T
        """
        return _try_calc_property(self, "coefficient_thermal_expansion", temperature)

    def rho(self, temperature: float, **kwargs) -> float:  # noqa: ARG002
        """
        Returns
        -------
        :
            Mass density in kg/m**3
        """
        return _try_calc_property(self, "density", temperature)

    def erho(self, temperature: float, **kwargs) -> float:  # noqa: ARG002
        """
        Returns
        -------
        :
            Electrical resistivity in Ohm.m
        """
        return _try_calc_property(self, "electrical_resistivity", temperature)

    def Ms(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Magnetic saturation in Am^2/kg
        """
        return _try_calc_property(self, "magnetic_saturation", temperature)

    def Mt(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Viscous remanent magnetisation in Am^2/kg
        """
        return _try_calc_property(self, "viscous_remanent_magnetisation", temperature)

    def Hc(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Coercive field in A/m
        """
        return _try_calc_property(self, "coercive_field", temperature)

    def Sy(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Minimum yield stress in MPa
        """
        return _try_calc_property(self, "minimum_yield_stress", temperature)

    def Syavg(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Average yield stress in MPa
        """
        return _try_calc_property(self, "average_yield_stress", temperature)

    def Su(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Minimum ultimate tensile stress in MPa
        """
        return _try_calc_property(self, "minimum_ultimate_tensile_stress", temperature)

    def Suavg(self, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Average ultimate tensile stress in MPa
        """
        return _try_calc_property(self, "average_ultimate_tensile_stress", temperature)


class Superconductor(abc.ABC):
    """
    Presently gratuitous use of multiple inheritance to convey plot function
    and avoid repetition. In future perhaps also a useful thing.
    """

    def plot(
        self,
        b_min: float,
        b_max: float,
        t_min: float,
        t_max: float,
        eps: float | None = None,
        n: int = 101,
        m: int = 100,
    ):
        """
        Plots superconducting surface parameterisation
        strain `eps` only used for Nb3Sn
        """
        jc = np.zeros([m, n])
        fields = np.linspace(b_min, b_max, n)
        temperatures = np.linspace(t_min, t_max, m)
        for j, b in enumerate(fields):
            for i, t in enumerate(temperatures):
                args = (b, t, eps) if eps else (b, t)
                jc[i, j] = self.Jc(*args)
        fig = plt.figure()
        fields, temperatures = np.meshgrid(fields, temperatures)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(self.name)
        ax.set_xlabel("B [T]")
        ax.set_ylabel("T [K]")
        ax.set_zlabel("$j_{c}$ [A/mm^2]")
        ax.plot_surface(fields, temperatures, jc, cmap=plt.cm.viridis)
        ax.view_init(30, 45)

    @abc.abstractmethod
    def Jc(self):  # noqa: N802
        """Jc"""
        ...


@dataclass
class NbTiSuperconductor(MassFractionMaterial, Superconductor):
    """
    Niobium-Titanium superconductor class.
    """

    _: KW_ONLY
    c_0: float
    bc_20: float
    tc_0: float
    alpha: float
    beta: float
    gamma: float

    def Bc2(self, temperature: float) -> float:
        """
        Returns
        -------
        :
            Critical field

        Notes
        -----
        :math:`B_{C2}^{*}(T) = B_{C20}(1-(\\frac{T}{T_{C0}})^{1.7})`
        """
        return self.bc_20 * (1 - (temperature / self.tc_0) ** 1.7)

    def Jc(self, B: float, temperature: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Critical current

        Notes
        -----
        :math:`j_{c}(B, T) = \\frac{C_{0}}{B}(1-(\\frac{T}{T_{C0}})^{1.7})
        ^{\\gamma}(\\frac{B}{B_{C2}(T)})^{\\alpha}(1-(\\frac{B}{B_{C2}(T)}))
        ^{\\beta}`
        """
        a = self.c_0 / B * (1 - (temperature / self.tc_0) ** 1.7) ** self.gamma
        ii = B / self.Bc2(temperature)
        b = ii**self.alpha
        # The below is an "elegant" dodge of numpy RuntimeWarnings encountered
        # when raising a negative number to a fractional power, which in this
        # parameterisation only occurs if a non-physical (<0) current density
        # is returned.
        c = (1 - ii) ** self.beta if 1 - ii > 0 else 0
        return a * b * c

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

        Returns
        -------
        :
            The openmc material
        """
        temperature = self.temperature if temperature is None else temperature

        return to_openmc_material(
            name=self.name,
            temperature=temperature,
            zaid_suffix=self.zaid_suffix,
            material_id=self.material_id,
            density=self.rho(temperature),
            density_unit=self.density_unit,
            percent_type=self.percent_type,
            isotopes=self.nuclides,
            elements=self.elements,
            enrichment=self.enrichment(temperature),
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
        )


@dataclass
class NbSnSuperconductor(MassFractionMaterial, Superconductor):
    """
    Niobium-Tin Superconductor class.
    """

    _: KW_ONLY
    c_a1: float
    c_a2: float
    eps_0a: float
    eps_m: float
    b_c20m: float
    t_c0max: float
    c: float
    p: float
    q: float

    @property
    def eps_sh(self):
        """
        EPS_sh, shear strain

        .. math::
            \\dfrac{C_{a2} \\epsilon_{0,a}}{\\sqrt{C_{a1}**2 - C_{a2}**2}}

        """
        return self.c_a2 * self.eps_0a / np.sqrt(self.c_a1**2 - self.c_a2**2)

    def Tc_star(self, B: float, eps: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Critical temperature

        Notes
        -----
        :math:`T_{C}^{*}(B, {\\epsilon}) = T_{C0max}^{*}s({\\epsilon})^{1/3}
        (1-b_{0})^{1/1.52}`
        """
        if B == 0:
            return self.t_c0max * self.s(eps) ** (1 / 3)
        b = (1 - self.Bc2_star(0, eps)) ** (1 / 1.52j).real
        return self.t_c0max * self.s(eps) ** (1 / 3) * b

    def Bc2_star(self, temperature: float, eps: float) -> float:
        """
        Returns
        -------
        :
            Critical field

        Notes
        -----
        :math:`B_{C}^{*}(T, {\\epsilon}) = B_{C20max}^{*}s({\\epsilon})
        (1-t^{1.52})`
        """
        if temperature == 0:
            return self.b_c20m * self.s(eps)
        return self.b_c20m * self.s(eps) * (1 - (self._t152(temperature, eps)))

    def Jc(self, B: float, temperature: float, eps: float) -> float:  # noqa: N802
        """
        Returns
        -------
        :
            Critical current

        Notes
        -----
        :math:`j_{c} = \\frac{C}{B}s({\\epsilon})(1-t^{1.52})(1-t^{2})b^{p}
        (1-b)^{q}`
        """
        b = self.b(B, temperature, eps)
        t = self.reduced_t(temperature, eps)
        # Ensure physical current density with max (j, 0)
        # Limits of parameterisation likely to be encountered sooner
        return max(
            (
                self.c
                / B
                * self.s(eps)
                * (1 - self._t152(temperature, eps))
                * (1 - t**2)
                * b**self.p
            )
            * (1 - b**self.q),
            0,
        )

    def _t152(self, temperature: float, eps: float) -> float:
        # 1.52 = 30000/19736
        t = self.reduced_t(temperature, eps) ** 1.52j
        return t.real

    def reduced_t(self, temperature: float, eps: float) -> float:
        """
        Returns
        -------
        :
            Reduced temperature

        Notes
        -----
        :math:`t = \\frac{T}{T_{C}^{*}(0, {\\epsilon})}`
        """
        return temperature / self.Tc_star(0, eps)

    def b(self, field: float, temperature: float, eps: float) -> float:
        """
        Returns
        -------
        :
            Reduced magnetic field

        Notes
        -----
        :math:`b = \\frac{B}{B_{C2}^{*}(0,{\\epsilon})}`

        """
        return field / self.Bc2_star(temperature, eps)

    def s(self, eps: float) -> float:
        """
        Strain function \n
        :math:`s({\\epsilon}) = 1+ \\frac{1}{1-C_{a1}{\\epsilon}_{0,a}}[C_{a1}
        (\\sqrt{{\\epsilon}_{sk}^{2}+{\\epsilon}_{0,a}^{2}}-\\sqrt{({\\epsilon}-
        {\\epsilon}_{sk})^{2}+{\\epsilon}_{0,a}^{2}})-C_{a2}{\\epsilon}]`

        Returns
        -------
        :
            The material strain
        """
        return 1 + 1 / (1 - self.c_a1 * self.eps_0a) * (
            self.c_a1
            * (
                np.sqrt(self.eps_sh**2 + self.eps_0a**2)
                - np.sqrt((eps - self.eps_sh) ** 2 + self.eps_0a**2)
            )
            - self.c_a2 * eps
        )

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

        Returns
        -------
        :
            The openmc Material
        """
        temperature = self.temperature if temperature is None else temperature
        return to_openmc_material(
            name=self.name,
            temperature=temperature,
            zaid_suffix=self.zaid_suffix,
            material_id=self.material_id,
            density=self.rho(temperature),
            density_unit=self.density_unit,
            percent_type=self.percent_type,
            isotopes=self.nuclides,
            elements=self.elements,
            enrichment=self.enrichment(temperature),
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
        )


@dataclass
class NbSnSuperconductorMag(MassFractionMaterial, Superconductor):
    """
    Nb3Sn Superconductor class.
    """

    _: KW_ONLY
    # cp300: Specific heat known data-point at 300 K [J/K/kg]
    cp300: float
    # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
    gamma: float
    # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
    beta: float
    c_a1: float
    c_a2: float
    eps_0a: float
    eps_m: float
    b_c20m: float
    t_c0max: float
    c: float
    p: float
    q: float
    c_: float = 1.0

    def Cp(self, temperature: float, **kwargs):  # noqa: N802, ARG002
        """
        Nb3Sn specific heat

        Ref
        ---
        ITER DRG1 Annex, Superconducting Material Database, Article 5, N 11 FDR
        42 01-07-05 R 0.1, Thermal, Electrical and Mechanical Properties of Materials
        at Cryogenic Temperatures

        Note
        ----
            Most specific heat data are listed in J/K/kg units. To represent the data
            in the J/K/m3 units a density of 8040 kg/m3 was assumed.

        Parameters
        ----------
        temperature:
            Operating temperature [K]

        Returns
        -------
            float [J/K/m³]
        """
        cp_low_nc = (self.beta * (temperature**3)) + (self.gamma * temperature)
        # cp_Nb3Sn = 1 / ((1 / self.cp300) + (1 / cp_low_nc))
        return 1 / ((1 / self.cp300) + (1 / cp_low_nc))

    def erho(self, temperature: float, **kwargs):  # noqa: PLR6301, ARG002
        """
        Electrical Resistivity

        Ref
        ---
        Data collected by L. Rossi, M. Sorbi, “MATPRO: A Computer Library of Material
        Property at Cryogenic Temperature”, INFN/TC-06/02, CARE-Note-2005-018-HHH

        Parameters
        ----------
        T:
            Operating temperature [K]

        Returns
        -------
        ------_
            float  [Ohm m]
        """
        return (-1e-4 * temperature**2 + 0.0938 * temperature + 22.601) * 1e-8

    def Jc(self, B: float, temperature: float, eps: float = 0.55, **kwargs) -> float:  # noqa: N802, ARG002
        """
        Returns the strand critical current density.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        temperature:
            Operating temperature in Kelvins.
        eps:
            Total applied measured strain in percentage. Default is 0.55.

        Returns
        -------
        Strand current density in Amperes per square meter.
        """
        T_ = temperature  # noqa: N806
        int_eps = -eps / 100
        eps_sh = self.c_a2 * self.eps_0a / (np.sqrt(self.c_a1**2 - self.c_a2**2))
        s_eps = 1 + (
            self.c_a1
            * (
                np.sqrt(eps_sh**2 + self.eps_0a**2)
                - np.sqrt((int_eps - eps_sh) ** 2 + self.eps_0a**2)
            )
            - self.c_a2 * int_eps
        ) / (1 - self.c_a1 * self.eps_0a)
        Bc0_eps = self.b_c20m * s_eps
        Tc0_eps = self.t_c0max * (s_eps) ** (1 / 3)  # noqa: N806
        t = T_ / Tc0_eps
        BcT_eps = Bc0_eps * (1 - t ** (1.52))
        b = B / BcT_eps
        hT = (1 - t ** (1.52)) * (1 - t**2)  # noqa: N806
        fPb = (b**self.p) * (1 - b) ** self.q  # noqa: N806
        return self.c_ * (self.c / B) * s_eps * fPb * hT


@dataclass
class CopperRRR(MassFractionMaterial):
    """
    Copper (OFHC, high purity RRR~100, annealed)

    Ref
    ---
    EFDA Material Data Compilation for Superconductor Simulation
    P. Bauer, H. Rajainmaki, E. Salpietro
    EFDA CSU, Garching, 04/18/07

    """

    _: KW_ONLY
    # RRR: Residual-resistance ratio (ratio of the resistivity of a material at
    # room temperature and at 0 K)
    RRR: int
    # cp300: Specific heat known data-point at 300 K [J/K/kg]
    cp300: float  # = field(default=3.454e6)
    # gamma: Debye fit parameter for Cp calculation [J/K**2/kg]
    gamma: float  # = field(default=0.011)
    # beta: Grueneisen fit parameter for Cp calculation [J/K**4/kg]
    beta: float  # = field(default=0.0011)

    # parameters for the calculation of the electrical resistivity
    a_rho1: float
    b_rho1: float
    c_rho1: float
    d_rho1: float
    e_rho1: float
    f_rho1: float

    a_rho2: float
    b_rho2: float
    c_rho2: float
    d_rho2: float

    a_A: float  # noqa: N815

    a_a: float
    b_a: float
    c_a: float
    d_a: float
    e_a: float

    def erho(self, temperature: float, B: float) -> float:
        """
        Electrical Resistivity in [Ohm.m]

        Ref
        ---
        NIST MONOGRAPH 177 J.Simon, E.S.Drexler and R.P.Reed, "Properties of
        Copper and Copper Alloys at Cryogenic Temperatures", 850 pages, February 1992,
        U.S. Government Printing Office, Washington, DC 20402-9325

        Parameters
        ----------
        temperature:
            Operating temperature [K]
        B:
            Operating magnetic field [T]

        Returns
        -------
            float  [Ohm.m]
        """
        rho1 = (self.a_rho1 * (temperature**self.b_rho1)) / (
            1
            + (
                self.c_rho1
                * (temperature**self.d_rho1)
                * (np.exp(-((self.e_rho1 / temperature) ** self.f_rho1)))
            )
        )

        rho2 = (
            (self.a_rho2 / self.RRR)
            + rho1
            + self.b_rho2 * ((self.c_rho2 * rho1) / (self.RRR * rho1 + self.d_rho2))
        )

        A = np.log10(self.a_A * B / rho2)

        a = (
            self.a_a
            + (self.b_a * A)
            + (self.c_a * (A**2))
            - (self.d_a * (A**3))
            + (self.e_a * (A**4))
        )

        return rho2 * (1 + (10**a))

    def Cp(self, temperature: float, **kwargs):  # noqa: N802, ARG002
        """
        Specific heat [J/kg/K]

        Ref
        ---
        L. Dresner, “Stability of Superconductors”, Plenum Press, NY, 1995

        Note
        ----
        The specific heat over the whole temperature range is obtained through
        fitting the function to the known low temperature and high temperature data.

        Parameters
        ----------
        temperature:
            Operating temperature [K]

        Returns
        -------
            float
        """
        c_plow = (self.beta * (temperature**3)) + (self.gamma * temperature)
        return 1 / (1 / self.cp300 + 1 / c_plow)


@dataclass
class Liquid:
    """
    Liquid material base class.
    """

    name: str
    symbol: str
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    density_unit: str = "kg/m3"
    temperature: float = T_DEFAULT
    pressure: float = P_DEFAULT
    zaid_suffix: str | None = None
    material_id: int | None = None
    percent_type: str = "ao"

    def __str__(self) -> str:
        """
        Returns
        -------
        :
            Get a string representation of the Liquid.
        """
        return self.name

    def rho(self, temperature: float, pressure: float | None = None) -> float:
        """
        Returns
        -------
        :
            Mass density
        """
        return _try_calc_property(
            self,
            "density",
            temperature,
            self.pressure if pressure is None else pressure,
        )

    @staticmethod
    def E(temperature: float | None = None) -> float:  # noqa: N802, ARG004
        """
        Returns
        -------
        :
            Youngs modulus (0 for all liquids)
        """
        return 0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Hmm... not sure about this one

        Returns
        -------
        :
            Poisson ratio
        """
        return 0

    def to_openmc_material(self, temperature: float | None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

        Returns
        -------
        :
            The openmc Material
        """
        temperature = self.temperature if temperature is None else temperature

        return to_openmc_material(
            name=self.symbol,
            chemical_equation=self.symbol,
            density=self.rho(temperature),
            density_unit=self.density_unit,
            percent_type=self.percent_type,
            temperature=temperature,
            pressure=self.pressure,
            zaid_suffix=self.zaid_suffix,
            material_id=self.material_id,
        )


@dataclass
class UnitCellCompound:
    """
    Unit cell compound
    """

    # Properties to interface with neutronics material maker
    name: str
    symbol: str
    volume_of_unit_cell_cm3: float
    atoms_per_unit_cell: int
    packing_fraction: float = 1.0
    temperature: float = T_DEFAULT
    zaid_suffix: str | None = None
    material_id: int | None = None
    percent_type: str = "ao"
    density_unit: str = "g/cm3"
    enrichment: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    enrichment_target: str = "Li6"
    enrichment_type: str = "ao"

    # Engineering properties
    specific_heat: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    coefficient_thermal_expansion: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )

    def __str__(self):
        """
        Returns
        -------
        :
            A string representation of the UcCompound.
        """
        return self.name

    def Cp(self, temperature: float, **kwargs) -> float:  # noqa: N802, ARG002
        """
        Returns
        -------
        :
            Specific heat in J/kg/K
        """
        return _try_calc_property(self, "specific_heat", temperature)

    def CTE(  # noqa: N802
        self, temperature: float, eps_vol: float | None = None
    ) -> float:
        """
        Returns
        -------
        :
            Mean coefficient of thermal expansion in 10**-6/T
        """
        return _try_calc_property(
            self, "coefficient_thermal_expansion", temperature, eps_vol
        )

    def to_openmc_material(self, temperature: None = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Returns
        -------
        :
            The openmc Material
        """
        return to_openmc_material(
            name=self.name,
            chemical_equation=self.symbol,
            volume_of_unit_cell_cm3=self.volume_of_unit_cell_cm3,
            atoms_per_unit_cell=self.atoms_per_unit_cell,
            percent_type=self.percent_type,
            temperature=self.temperature,
            packing_fraction=self.packing_fraction,
            enrichment=self.enrichment(temperature),
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
            density_unit=self.density_unit,
            zaid_suffix=self.zaid_suffix,
            material_id=self.material_id,
        )


class BePebbleBed(UnitCellCompound):
    """
    Beryllium Pebble Bed.
    """

    @matproperty(t_min=to_kelvin(25), t_max=to_kelvin(800))
    @staticmethod
    def CTE(temperature: float, eps_vol: float = 0) -> float:  # noqa: N802
        """
        .. doi:: 10.1016/S0920-3796(02)00165-5

        Returns
        -------
        :
            Mean coefficient of thermal expansion
        """
        # NOTE: Effect of inelastic volumetric strains [%] not negligible
        # esp_vol calculated roughly as f(T), as per 2M2BH9
        temperature = to_celsius(temperature)
        if eps_vol == 0:

            def calc_eps_vol(temp):
                """
                Returns
                -------
                :
                    Inelastic volumetric strains [%] based on T (C)
                """
                if temp >= 600:  # noqa: PLR2004
                    return 0.5
                if temp >= 500:  # noqa: PLR2004
                    return 0.3
                if temp < 500:  # noqa: PLR2004
                    return 0.2
                return None

            eps_vol = np.vectorize(calc_eps_vol)(temperature)
        if is_num(eps_vol):
            eps_vol *= np.ones_like(temperature)
        return (
            1.81
            + 0.0012 * temperature
            - 5e-7 * temperature**2
            + eps_vol
            * (
                9.03
                - 1.386e-3 * temperature
                - 7.6e-6 * temperature**2
                + 2.1e-9 * temperature**3
            )
        )


@dataclass
class Plasma:
    """
    A generic plasma material.
    """

    name: str
    isotopes: dict[str, float]
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor(1e-6)
    density_unit: str = "g/cm3"
    temperature: float = T_DEFAULT
    zaid_suffix: str | None = None
    material_id: int | None = None
    percent_type: str = "ao"

    def __str__(self) -> str:
        """
        Returns
        -------
        :
            A string representation of the plasma.
        """
        return self.name

    @staticmethod
    def E(temperature: float | None = None) -> float:  # noqa: N802, ARG004
        """
        Returns
        -------
        :
            Young's modulus.
        """
        return 0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Returns
        -------
        :
            Poisson's ratio.
        """
        return 0

    def to_openmc_material(self, temperature: None = None) -> openmc.Material:  # noqa: ARG002
        """
        Convert the material to an OpenMC material.

        Returns
        -------
        :
            The openmc Material
        """
        return to_openmc_material(
            name=self.name,
            density=self.density(self.temperature),
            density_unit=self.density_unit,
            isotopes=self.isotopes,
            percent_type=self.percent_type,
            temperature=self.temperature,
            material_id=self.material_id,
            zaid_suffix=self.zaid_suffix,
        )
