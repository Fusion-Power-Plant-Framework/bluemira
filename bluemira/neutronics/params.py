# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.error import GeometryError


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
    vertical_shift:
        how far (upwards) in the z direction is the centre of the plasma
        shifted compared to the geometric center of the poloidal cross-section.
    """

    R_0: Parameter[float]
    """Major Radius"""
    A: Parameter[float]
    """Aspect ratio"""
    kappa: Parameter[float]
    """Plasma elongation"""
    delta: Parameter[float]
    """Plasma triangularity"""
    reactor_power: Parameter[float]  # [W]
    peaking_factor: Parameter[float]  # [dimensionless]
    T_e: Parameter[float]
    """Average plasma electron temperature [J]"""
    shaf_shift: Parameter[float]
    """Shafranov shift"""
    vertical_shift: Parameter[float]  # [m]


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
    peaking_factor:
        (max. heat flux on fw)/(avg. heat flux on fw)
    temperature:
        plasma temperature (assumed to be uniform throughout the plasma)
    shaf_shift:
        Shafranov shift
        shift of the centre of flux surfaces, i.e.
        mean(min radius, max radius) of the LCFS,
        towards the outboard radial direction.
    vertical_shift:
        how far (upwards) in the z direction is the centre of the plasma
        shifted compared to the geometric center of the poloidal cross-section.
    plasma_physics_units:
        Plasma_physics_units converted variables
    """

    major_radius: float  # [m]
    aspect_ratio: float  # [dimensionless]
    elongation: float  # [dimensionless]
    triangularity: float  # [dimensionless]
    reactor_power: float  # [W]
    peaking_factor: float  # [dimensionless]
    temperature: float  # [K]
    shaf_shift: float  # [m]
    vertical_shift: float  # [m]
    plasma_physics_units: PlasmaSourceParameters | None = None

    def __post_init__(self):
        """Check dimensionless variables are sensible."""
        if self.peaking_factor < 1.0:
            raise ValueError(
                "Peaking factor (peak heat load/avg. heat load) "
                "must be larger than 1, by definition."
            )
        if self.aspect_ratio < 1.0:
            raise GeometryError(
                "By construction, tokamak aspect ratio " "can't be smaller than 1."
            )
        if self.elongation < 1.0:
            raise GeometryError("Elongation can't be smaller than 1")
        if abs(self.triangularity) > 1.0:
            # triangularity <0 is known as reversed/ negative triangularity.
            bluemira_warn(
                "Triangularity with magnitude >1 implies that the difference"
                "between the major radius and R_upper is larger than the minor radius."
            )

    @property
    def minor_radius(self):
        """Calculate minor radius from
        aspect_ratio = major_radius/minor_radius
        """
        return self.major_radius / self.aspect_ratio

    @classmethod
    def from_parameterframe(cls, params: ParameterFrame):
        """
        Convert from si units dataclass
        :class:`~bluemira.neutronics.params.PlasmaSourceParameters`

        This gives the illusion that self.cgs.x = scale_factor*self.x
        We rely on the 'frozen' nature of this dataclass so these links don't break.
        """
        conversion = {
            "major_radius": ("m", "cm"),
            "reactor_power": ("W", "MW"),
            "temperature": ("J", "keV"),
            "shaf_shift": ("m", "cm"),
            "vertical_shift": ("m", "cm"),
        }
        mapping = {
            "aspect_ratio": "A",
            "major_radius": "R_0",
            "elongation": "kappa",
            "triangularity": "delta",
            "temperature": "T_e",
        }
        param_convert_dict = {}
        param_dict = {}
        for k in fields(cls):
            if k.name == "plasma_physics_units":
                continue
            val = getattr(params, mapping.get(k.name, k.name)).value
            param_dict[k.name] = val
            if k.name in conversion:
                param_convert_dict[k.name] = raw_uc(val, *conversion[k.name])
            else:
                param_convert_dict[k.name] = val

        return cls(**param_dict, plasma_physics_units=cls(**param_convert_dict))


@dataclass
class BlanketThickness:
    """
    Give the depth of the interfaces between blanket layers.

    Parameters
    ----------
    surface
        Thickness of the surface layer of the blanket. Can be zero.
        Only used for tallying purpose, i.e. not a physical component.
    first_wall
        Thickness of the first wall.
    breeding_zone
        Thickness of the breedng zone. Could be zero if the breeding zone is absent.

    Note
    ----
    Thickness of the vacuum vessel is not required because we we assume it fills up the
    remaining space between the manifold's end and the outer_boundary.
    """

    surface: float
    first_wall: float
    breeding_zone: float
    manifold: float

    def get_interface_depths(self):
        """Return the depth of the interface layers"""
        return np.cumsum([
            self.surface,
            self.first_wall,
            self.breeding_zone,
        ])


@dataclass
class DivertorThickness:
    """
    Divertor dimensions.
    For now it only has 1 value: the surface layer thickness.

    Parameters
    ----------
    surface
        The surface layer of the divertor, which we expect to be made of a different
        material (e.g. Tungsten or alloy of Tungsten) from the bulk support & cooling
        structures of the divertor.
    """

    surface: float


@dataclass
class ToroidalFieldCoilDimension:
    """
    Gives the toroidal field coil diameters. Working with the simplest assumption, we
    assume that the tf coil is circular for now.

    Parameters
    ----------
    inner_diameter
        (i.e. inner diameter of the windings.)
    outer_diameter
        Outer diameter of the windings.
    """

    inner_diameter: float
    outer_diameter: float


@dataclass
class TokamakDimensions:
    """
    The dimensions of the simplest axis-symmetric case of the tokamak.

    Parameters
    ----------
    inboard
        thicknesses of the inboard blanket
    outboard
        thicknesses of the outboard blanket
    divertor
        thicknesses of the divertor components
    central_solenoid
        diameters of the toroidal field coil in the
    """

    inboard: BlanketThickness
    inboard_outboard_transition_radius: float
    outboard: BlanketThickness
    divertor: DivertorThickness
    central_solenoid: ToroidalFieldCoilDimension

    @classmethod
    def from_parameterframe(cls, params, r_inner_cut: float):
        """Setup tokamak dimensions"""
        return cls(
            BlanketThickness(
                params.blanket_surface_tk.value,
                params.inboard_fw_tk.value,
                params.inboard_breeding_tk.value,
                params.blk_ib_manifold.value,
            ),
            r_inner_cut,
            BlanketThickness(
                params.blanket_surface_tk.value,
                params.outboard_fw_tk.value,
                params.outboard_breeding_tk.value,
                params.blk_ob_manifold.value,
            ),
            DivertorThickness(params.divertor_surface_tk.value),
            ToroidalFieldCoilDimension(
                params.tf_inner_radius.value, params.tf_outer_radius.value
            ),
        )
