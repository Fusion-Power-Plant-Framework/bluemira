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
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Union

import asteval
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

from bluemira.base.constants import to_celsius, to_kelvin
from bluemira.materials.constants import P_DEFAULT, T_DEFAULT
from bluemira.materials.error import MaterialsError
from bluemira.materials.tools import _try_calc_property, matproperty, to_openmc_material
from bluemira.utilities.tools import array_or_num, is_num

if TYPE_CHECKING:
    import openmc
    from numpy.typing import ArrayLike

# Set any custom symbols for use in asteval
asteval_user_symbols = {
    "PropsSI": PropsSI,
    "to_celsius": to_celsius,
    "to_kelvin": to_kelvin,
}


class Material(Protocol):
    """Material Typing"""

    name: str
    density_unit: str
    percent_type: str

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
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

    value: Union[float, str]
    temp_max: Optional[float] = None
    temp_min: Optional[float] = None
    reference: str = ""
    obj: Any = field(default=None, repr=False)

    def __call__(
        self,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        eps_vol: float = 0.0,
    ) -> Union[float, ArrayLike]:
        """
        Evaluates the property at a given temperature, pressure, and/or eps_vol.

        Parameters
        ----------
        temperature:
            The temperature [K].
        pressure:
            The optional pressure [Pa].
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

            prop_val = array_or_num(aeval.eval(self.value))

            if len(aeval.error) > 0:
                raise aeval.error[0].exc(aeval.error[0].msg)

            return prop_val

        return self.value

    def to_dict(self):
        """
        Serialise the material property to a dictionary.
        """
        return asdict(self)

    @staticmethod
    def _validate_pressure(pressure: Union[float, ArrayLike]):
        if pressure is None:
            raise ValueError("Pressure is not set")
        return pressure

    def _validate_temperature(
        self, temperature: Union[float, ArrayLike]
    ) -> Union[float, ArrayLike]:
        """
        Check that the property is valid for the requested temperature range.

        Parameters
        ----------
        temperature:
            The temperatures requested to value the property at.

        Raises
        ------
        ValueError
            If any of the requested temperatures are outside of the valid range
        """
        if temperature is None:
            raise ValueError("Temperature is not set")
        temperatures = np.atleast_1d(temperature)
        if self.temp_min is not None and (temperatures < self.temp_min).any():
            raise ValueError(
                "Material property not valid outside of temperature range: "
                f"{temperatures} < T_min = {self.temp_min}"
            )
        if self.temp_max is not None and (temperatures > self.temp_max).any():
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

    def __get__(self, obj: Any, _) -> MaterialProperty:
        """Get the hex colour"""
        if obj is None:
            return self._default
        if self._default.obj is None:
            self._default.obj = obj

        return getattr(obj, self._name, self._default)

    def _mutate_value(
        self,
        value: Union[
            Union[Dict[str, Union[float, str, None]], float, str, None], MaterialProperty
        ],
        obj=None,
    ) -> MaterialProperty:
        if isinstance(value, dict):
            if "temp_min" not in value:
                # muck with keys
                # TODO(je-cook) should deprecate ?
                if "temp_min_kelvin" in value:
                    value["temp_max"] = value["temp_max_kelvin"]
                    value["temp_min"] = value["temp_min_kelvin"]
                    for k in (
                        "temp_min_kelvin",
                        "temp_max_kelvin",
                        "temp_min_celsius",
                        "temp_max_celsius",
                    ):
                        del value[k]

                if "temp_min_celsius" in value:
                    value["temp_max"] = to_kelvin(value["temp_max_celsius"])
                    value["temp_min"] = to_kelvin(value["temp_min_celsius"])
                    for k in ("temp_min_celsius", "temp_max_celsius"):
                        del value[k]
            # empty dictionary
            value = MaterialProperty(**value, obj=obj) if value else self._default

        elif isinstance(value, (float, int, str, type(None))):
            value = MaterialProperty(value=value, obj=obj)
        elif not isinstance(value, MaterialProperty):
            raise TypeError("Can't convert value to MaterialProperty")
        return value

    def __set__(
        self,
        obj: Any,
        value: Union[
            Union[Dict[str, Union[float, str, None]], float, str, None], MaterialProperty
        ],
    ):
        """
        Set the colour

        Notes
        -----
        The value can be anything accepted by matplotlib.colors.to_hex
        """
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
    elements: Dict[str, float] = field(default_factory=lambda: {"H": 1})
    zaid_suffix: Optional[str] = None
    material_id: Optional[int] = None

    @staticmethod
    def E(temperature: float | None = None) -> float:  # noqa: N802, ARG004
        """
        Young's modulus of the material at the given temperature.
        """
        return 0.0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Poisson's ratio at a given temperature.
        """
        return 0.0

    @staticmethod
    def rho(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        The density at a given temperature.
        """
        return 0.0

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.
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
    elements: Optional[Dict[str, float]] = None
    nuclides: Optional[Dict[str, float]] = None
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    density_unit: str = "kg/m3"
    temperature: float = T_DEFAULT
    zaid_suffix: Optional[str] = None
    material_id: Optional[int] = None
    percent_type: str = "wo"
    enrichment: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    enrichment_target: Optional[str] = None
    enrichment_type: Optional[str] = None

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

        dataclass limitation py3.8
        """
        if not (self.elements or self.nuclides):
            raise MaterialsError("No elements or nuclides specified.")

    def __str__(self) -> str:
        """
        Get a string representation of the MfMaterial.
        """
        return self.name

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

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
            enrichment=_try_calc_property(self, "enrichment", temperature)
            if self.enrichment
            else None,
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
            temperature=temperature,
        )

    def mu(self, temperature: float) -> float:
        """
        Poisson's ratio
        """
        return _try_calc_property(self, "poissons_ratio", temperature)

    def k(self, temperature: float) -> float:
        """
        Thermal conductivity in W.m/K
        """
        return _try_calc_property(self, "thermal_conductivity", temperature)

    def E(self, temperature: float) -> float:  # noqa: N802
        """
        Young's modulus in GPa
        """
        return _try_calc_property(self, "youngs_modulus", temperature)

    def Cp(self, temperature: float) -> float:  # noqa: N802
        """
        Specific heat in J/kg/K
        """
        return _try_calc_property(self, "specific_heat", temperature)

    def CTE(self, temperature: float) -> float:  # noqa: N802
        """
        Mean coefficient of thermal expansion in 10**-6/T
        """
        return _try_calc_property(self, "coefficient_thermal_expansion", temperature)

    def rho(self, temperature: float) -> float:
        """
        Mass density in kg/m**3
        """
        return _try_calc_property(self, "density", temperature)

    def erho(self, temperature: float) -> float:
        """
        Electrical resistivity in 10^(-8)Ohm.m
        """
        return _try_calc_property(self, "electrical_resistivity", temperature)

    def Ms(self, temperature: float) -> float:  # noqa: N802
        """
        Magnetic saturation in Am^2/kg
        """
        return _try_calc_property(self, "magnetic_saturation", temperature)

    def Mt(self, temperature: float) -> float:  # noqa: N802
        """
        Viscous remanent magnetisation in Am^2/kg
        """
        return _try_calc_property(self, "viscous_remanent_magnetisation", temperature)

    def Hc(self, temperature: float) -> float:  # noqa: N802
        """
        Coercive field in A/m
        """
        return _try_calc_property(self, "coercive_field", temperature)

    def Sy(self, temperature: float) -> float:  # noqa: N802
        """
        Minimum yield stress in MPa
        """
        return _try_calc_property(self, "minimum_yield_stress", temperature)

    def Syavg(self, temperature: float) -> float:  # noqa: N802
        """
        Average yield stress in MPa
        """
        return _try_calc_property(self, "average_yield_stress", temperature)

    def Su(self, temperature: float) -> float:  # noqa: N802
        """
        Minimum ultimate tensile stress in MPa
        """
        return _try_calc_property(self, "minimum_ultimate_tensile_stress", temperature)

    def Suavg(self, temperature: float) -> float:  # noqa: N802
        """
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
    def Jc():  # noqa: N802, D102
        ...


@dataclass
class NbTiSuperconductor(MassFractionMaterial, Superconductor):
    """
    Niobium-Titanium superconductor class.
    """

    c_0: Optional[float] = None
    bc_20: Optional[float] = None
    tc_0: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None

    def __post_init__(self):
        """Value checking for required args marked as optional

        dataclass limitation py3.8
        """
        if None in {
            self.c_0,
            self.bc_20,
            self.tc_0,
            self.alpha,
            self.beta,
            self.gamma,
        }:
            raise ValueError(
                "Not all required values set TODO(je-cook) kwargs in dataclasses"
            )

    def Bc2(self, temperature: float) -> float:
        """
        Critical field \n
        :math:`B_{C2}^{*}(T) = B_{C20}(1-(\\frac{T}{T_{C0}})^{1.7})`
        """
        return self.bc_20 * (1 - (temperature / self.tc_0) ** 1.7)

    def Jc(self, B: float, temperature: float) -> float:  # noqa: N802
        """
        Critical current \n
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
        # TODO: Check the above..
        c = (1 - ii) ** self.beta if 1 - ii > 0 else 0
        return a * b * c

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

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
            enrichment=_try_calc_property(self, "enrichment", temperature)
            if self.enrichment
            else None,
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
        )


@dataclass
class NbSnSuperconductor(MassFractionMaterial, Superconductor):
    """
    Niobium-Tin Superconductor class.
    """

    c_a1: Optional[float] = None
    c_a2: Optional[float] = None
    eps_0a: Optional[float] = None
    eps_m: Optional[float] = None
    b_c20m: Optional[float] = None
    t_c0max: Optional[float] = None
    c: Optional[float] = None
    p: Optional[float] = None
    q: Optional[float] = None

    def __post_init__(self):
        """Value checking for required args marked as optional

        dataclass limitation py3.8
        """
        if None in {
            self.c_a1,
            self.c_a2,
            self.eps_0a,
            self.eps_m,
            self.b_c20m,
            self.t_c0max,
            self.c,
            self.p,
            self.q,
        }:
            raise ValueError(
                "Not all required values set TODO(je-cook) kwargs in dataclasses"
            )
        self.eps_sh = self.c_a2 * self.eps_0a / np.sqrt(self.c_a1**2 - self.c_a2**2)

    def Tc_star(self, B: float, eps: float) -> float:  # noqa: N802
        """
        Critical temperature

        :math:`T_{C}^{*}(B, {\\epsilon}) = T_{C0max}^{*}s({\\epsilon})^{1/3}
        (1-b_{0})^{1/1.52}`
        """
        if B == 0:
            return self.t_c0max * self.s(eps) ** (1 / 3)
        b = (1 - self.Bc2_star(0, eps)) ** (1 / 1.52j).real
        return self.t_c0max * self.s(eps) ** (1 / 3) * b

    def Bc2_star(self, temperature: float, eps: float) -> float:
        """
        Critical field

        :math:`B_{C}^{*}(T, {\\epsilon}) = B_{C20max}^{*}s({\\epsilon})
        (1-t^{1.52})`
        """
        if temperature == 0:
            return self.b_c20m * self.s(eps)
        return self.b_c20m * self.s(eps) * (1 - (self._t152(temperature, eps)))

    def Jc(self, B: float, temperature: float, eps: float) -> float:  # noqa: N802
        """
        Critical current

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
        Reduced temperature \n
        :math:`t = \\frac{T}{T_{C}^{*}(0, {\\epsilon})}`
        """
        return temperature / self.Tc_star(0, eps)

    def b(self, field: float, temperature: float, eps: float) -> float:
        """
        Reduced magnetic field \n
        :math:`b = \\frac{B}{B_{C2}^{*}(0,{\\epsilon})}`
        """
        return field / self.Bc2_star(temperature, eps)

    def s(self, eps: float) -> float:
        """
        Strain function \n
        :math:`s({\\epsilon}) = 1+ \\frac{1}{1-C_{a1}{\\epsilon}_{0,a}}[C_{a1}
        (\\sqrt{{\\epsilon}_{sk}^{2}+{\\epsilon}_{0,a}^{2}}-\\sqrt{({\\epsilon}-
        {\\epsilon}_{sk})^{2}+{\\epsilon}_{0,a}^{2}})-C_{a2}{\\epsilon}]`
        """
        return 1 + 1 / (1 - self.c_a1 * self.eps_0a) * (
            self.c_a1
            * (
                np.sqrt(self.eps_sh**2 + self.eps_0a**2)
                - np.sqrt((eps - self.eps_sh) ** 2 + self.eps_0a**2)
            )
            - self.c_a2 * eps
        )

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

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
            enrichment=_try_calc_property(self, "enrichment", temperature)
            if self.enrichment
            else None,
            enrichment_target=self.enrichment_target,
            enrichment_type=self.enrichment_type,
        )


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
    zaid_suffix: Optional[str] = None
    material_id: Optional[int] = None
    percent_type: str = "ao"

    def __str__(self) -> str:
        """
        Get a string representation of the Liquid.
        """
        return self.name

    def rho(self, temperature: float, pressure: float | None = None) -> float:
        """
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
        Youngs modulus (0 for all liquids)
        """
        return 0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Hmm... not sure about this one
        """
        return 0

    def to_openmc_material(self, temperature: Optional[float] = None) -> openmc.Material:
        """
        Convert the material to an OpenMC material.

        Parameters
        ----------
        temperature:
            The temperature [K].

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
    enrichment: Optional[float] = None
    temperature: float = T_DEFAULT
    zaid_suffix: Optional[str] = None
    material_id: Optional[int] = None
    percent_type: str = "ao"
    density_unit: str = "g/cm3"
    enrichment_target: str = "Li6"
    enrichment_type: str = "ao"

    # Engineering properties
    specific_heat: MaterialPropertyDescriptor = MaterialPropertyDescriptor()
    coefficient_thermal_expansion: MaterialPropertyDescriptor = (
        MaterialPropertyDescriptor()
    )

    def __str__(self):
        """
        Get a string representation of the UcCompound.
        """
        return self.name

    def Cp(self, temperature: float) -> float:  # noqa: N802
        """
        Specific heat in J/kg/K
        """
        return _try_calc_property(self, "specific_heat", temperature)

    def CTE(  # noqa: N802
        self, temperature: float, eps_vol: float | None = None
    ) -> float:
        """
        Mean coefficient of thermal expansion in 10**-6/T
        """
        return _try_calc_property(
            self, "coefficient_thermal_expansion", temperature, eps_vol
        )

    def to_openmc_material(self, temperature: None = None) -> openmc.Material:  # noqa: ARG002
        """
        Convert the material to an OpenMC material.
        """
        return to_openmc_material(
            name=self.name,
            chemical_equation=self.symbol,
            volume_of_unit_cell_cm3=self.volume_of_unit_cell_cm3,
            atoms_per_unit_cell=self.atoms_per_unit_cell,
            percent_type=self.percent_type,
            temperature=self.temperature,
            packing_fraction=self.packing_fraction,
            enrichment=self.enrichment,
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
        https://www.sciencedirect.com/science/article/pii/S0920379602001655
        """
        # NOTE: Effect of inelastic volumetric strains [%] not negligible
        # esp_vol calculated roughly as f(T), as per 2M2BH9
        temperature = to_celsius(temperature)
        if eps_vol == 0:

            def calc_eps_vol(temp):
                """
                Calculates inelastic volumetric strains [%] based on T (C)
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
            eps_vol = eps_vol * np.ones_like(temperature)
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
    isotopes: Dict[str, float]
    density: MaterialPropertyDescriptor = MaterialPropertyDescriptor(1e-6)
    density_unit: str = "g/cm3"
    temperature: float = T_DEFAULT
    zaid_suffix: Optional[str] = None
    material_id: Optional[int] = None
    percent_type: str = "ao"

    def __str__(self) -> str:
        """
        Get a string representation of the plasma.
        """
        return self.name

    @staticmethod
    def E(temperature: float | None = None) -> float:  # noqa: N802, ARG004
        """
        Young's modulus.
        """
        return 0

    @staticmethod
    def mu(temperature: float | None = None) -> float:  # noqa: ARG004
        """
        Poisson's ratio.
        """
        return 0

    def to_openmc_material(self, temperature: None = None) -> openmc.Material:  # noqa: ARG002
        """
        Convert the material to an OpenMC material.
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
