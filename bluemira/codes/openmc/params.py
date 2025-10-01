# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from bluemira.base.parameter_frame import Parameter, ParameterFrame

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class OpenMCNeutronicsSolverParams(ParameterFrame):
    """

    Parameters
    ----------
    major_radius:
        Major radius of the machine
    profile_rho_ped:
        Pedestal location in normalized (minor) radius
    reactor_power:
        total reactor fusion power when operating at 100%
    n_profile_alpha:
        Electron density profile alpha exponent
    n_e_core:
        Core electron density
    n_e_ped:
        Pedestal electron density
    n_e_sep:
        Separatrix electron density
    T_profile_alpha:
        Electron temperature profile alpha exponent
    T_profile_beta:
        Electron temperature profile beta exponent
    T_e_core:
        Core electron temperature
    T_e_ped:
        Pedestal electron temperature
    T_e_sep:
        Separatrix electron temperature
    T_ie_ratio:
        Ion to electron temperature ratio (volume-averaged)
    n_i_fuel:
        Volume-averaged fuel ion density
    n_e:
        Volume-averaged electron density
    shaf_shift:
        Radial Shafranov shift
    """

    """Major Radius"""
    R_0: Parameter[float]

    """Pedestal location in normalized (minor) radius"""
    profile_rho_ped: Parameter[float]

    """Reactor power"""
    reactor_power: Parameter[float]
    """electronn density profile descriptors"""
    n_profile_alpha: Parameter[float]
    n_e_core: Parameter[float]
    n_e_ped: Parameter[float]
    n_e_sep: Parameter[float]
    """temperature profile descriptors"""
    T_profile_alpha: Parameter[float]
    T_profile_beta: Parameter[float]
    T_e_core: Parameter[float]
    T_e_ped: Parameter[float]
    T_e_sep: Parameter[float]

    T_ie_ratio: Parameter[float]
    """Ion to electron temperature ratio (volume-averaged)."""
    n_i_fuel: Parameter[float]
    """Volume-averaged fuel ion density [1/metre **3]."""
    n_e: Parameter[float]
    """Volumed-averaged plasma electron density [1/metre ** 3]."""

    """Shafranov shift"""
    shaf_shift: Parameter[float]


@dataclass(frozen=True)
class PlasmaSourceParameters:
    """
    Parameters describing the plasma source,
    i.e. where the plasma is positioned (and therefore where the power is concentrated),
    and what temperature the plasma is at.

    Parameters
    ----------
    reactor_power:
        total reactor (thermal) power when operating at 100%
    """

    rho_pedestal: float  # [dimensionless]
    reactor_power: float  # [W]

    electron_density_alpha: float  # [dimensionless]
    electron_density_core: float  # [1/m^3]
    electron_density_ped: float  # [1/m^3]
    electron_density_sep: float  # [1/m^3]

    electron_temperature_alpha: float  # [dimensionless]
    electron_temperature_beta: float  # [dimensionless]
    electron_temperature_core: float  # [keV]
    electron_temperature_ped: float  # [keV]
    electron_temperature_sep: float  # [keV]

    ie_temperature_ratio: Parameter[float]
    """Ion to electron temperature ratio (volume-averaged)."""
    va_fuel_ion_density: Parameter[float]
    """Volume-averaged fuel ion density [1/metre **3]."""
    va_electron_density: Parameter[float]
    """Volumed-averaged plasma electron density [1/metre ** 3]."""

    # mapping from parameter names in params.json (extracted by
    # OpenMCNeutronicsSolverParams) to the fields in this dataclass.
    _unit: ClassVar[Mapping[str, str]] = MappingProxyType({
        "rho_pedestal": "1",
        "reactor_power": "W",
        "electron_density_alpha": "1",
        "electron_density_core": "1/m^3",
        "electron_density_ped": "1/m^3",
        "electron_density_sep": "1/m^3",
        "electron_temperature_alpha": "1",
        "electron_temperature_beta": "1",
        "electron_temperature_core": "keV",
        "electron_temperature_ped": "keV",
        "electron_temperature_sep": "keV",
        "ie_temperature_ratio": "1",
        "va_electron_density": "1/m^3",
        "va_fuel_ion_density": "1/m^3",
    })
    _mapping: ClassVar[Mapping[str, str]] = MappingProxyType({
        "rho_pedestal": "profile_rho_ped",
        "electron_density_alpha": "n_profile_alpha",
        "electron_density_core": "n_e_core",
        "electron_density_ped": "n_e_ped",
        "electron_density_sep": "n_e_sep",
        "electron_temperature_alpha": "T_profile_alpha",
        "electron_temperature_beta": "T_profile_beta",
        "electron_temperature_core": "T_e_core",
        "electron_temperature_ped": "T_e_ped",
        "electron_temperature_sep": "T_e_sep",
        "ie_temperature_ratio": "T_ie_ratio",
        "va_electron_density": "n_e",
        "va_fuel_ion_density": "n_i_fuel",
    })

    @classmethod
    def from_parameterframe(cls, params: ParameterFrame):
        """Create an object of this class (PlasmaSourceParameters) from a ParameterFrame
        (specifically, an object of the class OpenMCNeutronicsSolverParams), with the
        appropriate units.

        Returns
        -------
        self
            A PlasmaSourceParameters dataclass.
        """
        param_dict = {}
        for k in fields(cls):
            param = getattr(params, cls._mapping.get(k.name, k.name))
            numerical_value = param.value_as(cls._unit[k.name])
            param_dict[k.name] = numerical_value

        return cls(**param_dict)
