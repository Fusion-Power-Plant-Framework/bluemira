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

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.error import GeometryError

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class OpenMCNeutronicsSolverParams(ParameterFrame):
    """

    Parameters
    ----------
    major_radius:
        Major radius of the machine
    aspect_ratio:
        aspect ratio of the machine
    elongation:
        elongation of the plasma
    triangularity:
        triangularity of the plasma
    reactor_power:
        total reactor (thermal) power when operating at 100%
    peaking_factor:
        (max. heat flux on fw)/(avg. heat flux on fw)
    temperature:
        plasma temperature (assumed to be uniform throughout the plasma)
    shaf_shift:
        Shafranov shift
        shift of the centre of flux surfaces, i.e.
        mean(min radius, max radius) of the LCFS,
        towards the outboard radial direction.
    """

    """Major Radius"""
    R_0: Parameter[float]
    """Aspect ratio"""
    A: Parameter[float]
    """Pedestal location in normalized (minor) radius"""
    profile_rho_ped: Parameter[float]

    """Plasma elongation"""
    kappa: Parameter[float]
    """Plasma triangularity"""
    delta: Parameter[float]
    """Reactor power"""
    reactor_power: Parameter[float]
    """ion density profile descriptors"""
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

    shaf_shift:
        Shafranov shift shift of the centre of flux surfaces, i.e.
        mean(min radius, max radius) of the LCFS, towards the outboard radial direction.
    """

    rho_pedestal: float  # [dimensionless]
    reactor_power: float  # [W]

    ion_density_alpha: float  # [dimensionless]
    ion_density_core: float  # [1/m^3]
    ion_density_ped: float  # [1/m^3]
    ion_density_sep: float  # [1/m^3]

    ion_temperature_alpha: float  # [dimensionless]
    ion_temperature_beta: float  # [dimensionless]
    ion_temperature_core: float  # [eV]
    ion_temperature_ped: float  # [eV]
    ion_temperature_sep: float  # [eV]

    # mapping from parameter names in params.json (extracted by
    # OpenMCNeutronicsSolverParams) to the fields in this dataclass.
    _unit: ClassVar[Mapping[str, str]] = MappingProxyType({
        "rho_pedestal": "1",
        "reactor_power": "W",
        "ion_density_alpha": "1",
        "ion_density_core": "1/m^3",
        "ion_density_ped": "1/m^3",
        "ion_density_sep": "1/m^3",
        "ion_temperature_alpha": "1",
        "ion_temperature_beta": "1",
        "ion_temperature_core": "keV",
        "ion_temperature_ped": "keV",
        "ion_temperature_sep": "keV",
    })
    _mapping: ClassVar[Mapping[str, str]] = MappingProxyType({
        "rho_pedestal": "profile_rho_ped",
        "ion_density_alpha": "n_profile_alpha",
        "ion_density_core": "n_e_core",
        "ion_density_ped": "n_e_ped",
        "ion_density_sep": "n_e_sep",
        "ion_temperature_alpha": "T_profile_alpha",
        "ion_temperature_beta": "T_profile_beta",
        "ion_temperature_core": "T_e_core",
        "ion_temperature_ped": "T_e_ped",
        "ion_temperature_sep": "T_e_sep",
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
