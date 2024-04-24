# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.geometry.error import GeometryError

if TYPE_CHECKING:
    from pathlib import Path


class BlanketLayers(Enum):
    """
    The five layers of the blanket as used in the neutronics simulation.

    Variables
    ---------
    Surface
        The surface layer of the first wall.
    First wall
        Typically made of tungsten or Eurofer
    BreedingZone
        Where tritium is bred
    Manifold
        The pipe works and supporting structure
    VacuumVessel
        The vacuum vessel keeping the plasma from mixing with outside air.
    """

    Surface = auto()
    FirstWall = auto()
    BreedingZone = auto()
    Manifold = auto()
    VacuumVessel = auto()


@dataclass
class OpenMCSimulationRuntimeParameters:
    """Parameters used in the actual simulation

    Parameters
    ----------
    particles:
        Number of neutrons emitted by the plasma source per batch.
    batches:
        How many batches to simulate.
    photon_transport:
        Whether to simulate the transport of photons (i.e. gamma-rays created) or not.
    electron_treatment:
        The way in which OpenMC handles secondary charged particles.
        'thick-target bremsstrahlung' or 'local energy deposition'
        'thick-target bremsstrahlung' accounts for the energy carried away by
        bremsstrahlung photons and deposited elsewhere, whereas 'local energy
        deposition' assumes electrons deposit all energies locally.
        (the latter is expected to be computationally faster.)
    run_mode:
        see below for details:
        https://docs.openmc.org/en/stable/usersguide/settings.html#run-modes
    openmc_write_summary:
        whether openmc should write a 'summary.h5' file or not.
    cross_section_xml:
        Where the xml file for cross-section is stored locally.
    """

    # Parameters used outside of setup_openmc()
    particles: int  # number of particles used in the neutronics simulation
    cross_section_xml: str | Path
    batches: int = 2
    photon_transport: bool = True
    electron_treatment: Literal["ttb", "led"] = (
        "led"  # Bremsstrahlung only matters for very thin objects
    )
    run_mode: str = openmc.settings.RunMode.FIXED_SOURCE.value
    openmc_write_summary: bool = False
    parametric_source: bool = True
    # number of particles used in the volume calculation.
    volume_calc_particles: int = int(4e8)


class BlanketType(Enum):
    """Types of allowed blankets, named by their acronyms."""

    DCLL = auto()
    HCPB = auto()
    WCLL = auto()


@dataclass
class BreederTypeParameters:
    """Dataclass to hold information about the breeder blanket material
    and design choices.
    """

    enrichment_fraction_Li6: float
    blanket_type: BlanketType


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
            "temperature": ("K", "keV"),
            "shaf_shift": ("m", "cm"),
            "vertical_shift": ("m", "cm"),
        }
        param_convert_dict = {}
        param_dict = {}
        for k in fields(cls):
            if k.name == "plasma_physics_units":
                continue
            val = getattr(params, k.name).value
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
        Unit = [m (if in TokamakGeometryBase) /cm (if in TokamakGeometry)]
    first_wall
        Thickness of the first wall.
        Unit = [m (if in TokamakGeometryBase) /cm (if in TokamakGeometry)]
    breeding_zone
        Thickness of the breedng zone. Could be zero if the breeding zone is absent.
        Unit = [m (if in TokamakGeometryBase) /cm (if in TokamakGeometry)]

    Note
    ----
    Thickness of the vacuum vessel is not required because we we assume it fills up the
    remaining space between the manifold's end and the outer_boundary.
    """

    surface: float
    first_wall: float
    breeding_zone: float

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
        Unit = [m (if in TokamakGeometryBase) /cm (if in TokamakGeometry)]
    outer_diameter
        Outer diameter of the windings.
        Unit = [m (if in TokamakGeometryBase) /cm (if in TokamakGeometry)]
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
    def from_tokamak_geometry_base(
        cls,
        tokamak_geometry_base: TokamakGeometry,
        major_radius,
        divertor_thickness,
        tf_inner_radius,
        tf_outer_radius,
    ):
        """Bodge method that can be deleted later once
        :func:`~get_preset_physical_properties` migrated over to use TokamakDimensions.
        """
        return cls(
            BlanketThickness(
                0.01,
                tokamak_geometry_base.inb_fw_thick,
                tokamak_geometry_base.inb_bz_thick,
            ),
            major_radius,
            BlanketThickness(
                0.01,
                tokamak_geometry_base.outb_fw_thick,
                tokamak_geometry_base.outb_bz_thick,
            ),
            DivertorThickness(divertor_thickness),
            ToroidalFieldCoilDimension(tf_inner_radius, tf_outer_radius),
        )


@dataclass(frozen=True)  # TODO: obsolete: remove when neutronics.ex.py is deleted from
# feature/neutronics.
class TokamakGeometryBase:
    """
    The thickness measurements for all of the generic components of the tokamak.

    Parameters
    ----------
    inb_fw_thick:     inboard first wall thickness [m]
    inb_bz_thick:     inboard breeding zone thickness [m]
    inb_mnfld_thick:  inboard manifold thickness [m]
    inb_vv_thick:     inboard vacuum vessel thickness [m]
    tf_thick:         toroidal field coil thickness [m]
    outb_fw_thick:    outboard first wall thickness [m]
    outb_bz_thick:    outboard breeding zone thickness [m]
    outb_mnfld_thick: outboard manifold thickness [m]
    outb_vv_thick:    outboard vacuum vessel thickness [m]
    inb_gap:          inboard gap [m]
    """

    inb_fw_thick: float
    inb_bz_thick: float
    inb_mnfld_thick: float
    inb_vv_thick: float
    tf_thick: float
    outb_fw_thick: float
    outb_bz_thick: float
    outb_mnfld_thick: float
    outb_vv_thick: float
    inb_gap: float


@dataclass(frozen=True)  # TODO: obsolete: remove when neutronics.ex.py is deleted from
# feature/neutronics.
class TokamakGeometry(TokamakGeometryBase):
    """See TokamakGeometryBase

    Addition of cgs converted variables
    """

    cgs: TokamakGeometryBase

    @classmethod
    def from_si(cls, tokamak_geometry_base: TokamakGeometryBase):
        """
        Convert from si units dataclass
        :class:`~bluemira.neutronics.params.TokamakGeometryBase`
        This gives the illusion that self.cgs.x = 100*self.x. We rely on the 'frozen'
        nature of this dataclass so these links don't break.
        """
        tg = asdict(tokamak_geometry_base)
        tgcgs = tg.copy()
        for k, v in tgcgs.items():
            tgcgs[k] = raw_uc(v, "m", "cm")
        return cls(**tg, cgs=TokamakGeometryBase(**tgcgs))


def get_preset_physical_properties(
    blanket_type: str | BlanketType,
) -> tuple[BreederTypeParameters, TokamakGeometryBase]:
    """
    Works as a switch-case for choosing the tokamak geometry
    and blankets for a given blanket type.
    The allowed list of blanket types are specified in BlanketType.
    Currently, the blanket types with pre-populated data in this function are:
    {'wcll', 'dcll', 'hcpb'}
    """
    if not isinstance(blanket_type, BlanketType):
        blanket_type = BlanketType[blanket_type.lower()]

    breeder_materials = BreederTypeParameters(
        blanket_type=blanket_type,
        enrichment_fraction_Li6=0.60,
    )

    # Geometry variables

    # Break down from here.
    # Paper inboard build ---
    # Nuclear analyses of solid breeder blanket options for DEMO:
    # Status,challenges and outlook,
    # Pereslavtsev, 2019
    #
    # 0.400,      # TF Coil inner
    # 0.200,      # gap                  from figures
    # 0.060,       # VV steel wall
    # 0.480,      # VV
    # 0.060,       # VV steel wall
    # 0.020,       # gap                  from figures
    # 0.350,      # Back Supporting Structure
    # 0.060,       # Back Wall and Gas Collectors   Back wall = 3.0
    # 0.350,      # breeder zone
    # 0.022        # fw and armour

    shared_building_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": 0.2,  # [m]
        "inb_vv_thick": 0.6,  # [m]
        "tf_thick": 0.4,  # [m]
        "outb_vv_thick": 0.6,  # [m]
    }
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometryBase(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.378,  # [m]
            inb_mnfld_thick=0.435,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.538,  # [m]
            outb_mnfld_thick=0.429,  # [m]
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometryBase(
            **shared_building_geometry,
            inb_fw_thick=0.022,  # [m]
            inb_bz_thick=0.300,  # [m]
            inb_mnfld_thick=0.178,  # [m]
            outb_fw_thick=0.022,  # [m]
            outb_bz_thick=0.640,  # [m]
            outb_mnfld_thick=0.248,  # [m]
        )
    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        tokamak_geometry = TokamakGeometryBase(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.460,  # [m]
            inb_mnfld_thick=0.560,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.460,  # [m]
            outb_mnfld_thick=0.560,  # [m]
        )

    return breeder_materials, tokamak_geometry
