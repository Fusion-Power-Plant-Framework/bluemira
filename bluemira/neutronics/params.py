# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
import openmc

from bluemira.base.constants import EPS, raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.error import GeometryError
from bluemira.neutronics.make_materials import BlanketType


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
        The vacuum vessel keeping out the atmospheric air.
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

    def __post_init__(self):
        """Check dimensionless variables are sensible."""
        if self.peaking_factor < 1.0:  # noqa: PLR2004
            raise ValueError(
                "Peaking factor (peak heat load/avg. heat load) "
                "must be larger than 1, by definition."
            )
        if self.aspect_ratio < 1.0:  # noqa: PLR2004
            raise GeometryError(
                "By construction, tokamak aspect ratio " "can't be smaller than 1."
            )
        if self.elongation < 1.0:  # noqa: PLR2004
            raise GeometryError("Elongation can't be smaller than 1")
        if abs(self.triangularity) > 1.0:  # noqa: PLR2004
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


@dataclass(frozen=True)
class PlasmaSourceParametersPPS(PlasmaSourceParameters):
    """See PlasmaSourceParameters

    Addition of plasma_physics_units converted variables
    """

    plasma_physics_units: PlasmaSourceParameters

    @classmethod
    def from_si(cls, op_params: PlasmaSourceParameters):
        """
        Convert from si units dataclass
        :class:`~bluemira.neutronics.params.PlasmaSourceParameters`
        """
        conversion = {
            "major_radius": ("m", "cm"),
            "reactor_power": ("W", "MW"),
            "temperature": ("K", "keV"),
            "shaf_shift": ("m", "cm"),
            "vertical_shift": ("m", "cm"),
        }
        op = asdict(op_params)
        op_pps = op.copy()
        for k, v in op_pps.items():
            if k in conversion:
                op_pps[k] = raw_uc(v, *conversion[k])
        return cls(**op, plasma_physics_units=PlasmaSourceParameters(**op_pps))


@dataclass(frozen=True)
class TokamakGeometryBase:
    """
    The thickness measurements for all of the generic components of the tokamak.

    Parameters
    ----------
    inb_fw_thick:     inboard first wall thickness [m]
    inb_bz_thick:     inboard breeding zone thickness [m]
    inb_mnfld_thick:  inboard manifold thickness [m]
    inb_vv_thick:     inboard vacuum vessel thickness [m]
    tf_thick:         toroidal field thickness [m]
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


@dataclass(frozen=True)
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
        """
        tg = asdict(tokamak_geometry_base)
        tgcgs = tg.copy()
        for k, v in tgcgs.items():
            tgcgs[k] = raw_uc(v, "m", "cm")
        return cls(**tg, cgs=TokamakGeometryBase(**tgcgs))


@dataclass
class WallThicknessFraction:
    """List of thickness of various sections of the blanket as fractions"""

    first_wall: float  # [dimensionless]
    breeding_zone: float  # [dimensionless]
    manifold: float  # [dimensionless]
    vacuum_vessel: float  # [dimensionless]

    def __post_init__(self):
        """Check fractions are between 0 and 1 and sums to unity."""
        for section in ("first_wall", "manifold", "vacuum_vessel"):
            if getattr(self, section) <= 0:
                raise GeometryError(f"Thickness fraction of {section} must be non-zero")
        if self.breeding_zone < 0:  # can be zero, but not negative.
            raise GeometryError(
                "Thickness fraction of breeding_zone must be nonnegative"
            )
        if not np.isclose(
            sum([
                self.first_wall,
                self.manifold,
                self.breeding_zone,
                self.vacuum_vessel,
            ]),
            1.0,
            rtol=0,
            atol=EPS,
        ):
            raise GeometryError(
                "Thickness fractions of all four sections " "must add up to unity!"
            )

    def extended_prefix_sums(self):
        """Return the exclusive and includsive sum"""
        zone_sequence = ("first_wall", "breeding_zone", "manifold", "vacuum_vessel")
        _each_thick = [getattr(self, zone) for zone in zone_sequence]
        return np.insert(np.cumsum(_each_thick).tolist(), 0, 0)


class InboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the inboard wall of the blanket"""

    pass


class OutboardThicknessFraction(WallThicknessFraction):
    """Thickness fraction list of the outboard wall of the blanket"""

    pass


@dataclass
class ThicknessFractions:
    """
    A dataclass containing info. on both
    inboard and outboard blanket thicknesses as fractions.
    """

    inboard: InboardThicknessFraction
    outboard: OutboardThicknessFraction
    inboard_outboard_transition_radius: float

    @classmethod
    def from_TokamakGeometry(
        cls, tokamak_geometry: TokamakGeometry, transition_radius: float = 8.0
    ):
        """
        Create this dataclass by
        translating from our existing tokamak_geometry dataclass.
        """
        inb_sum = sum([
            tokamak_geometry.inb_fw_thick,
            tokamak_geometry.inb_bz_thick,
            tokamak_geometry.inb_mnfld_thick,
            tokamak_geometry.inb_vv_thick,
        ])
        inb = InboardThicknessFraction(
            tokamak_geometry.inb_fw_thick / inb_sum,
            tokamak_geometry.inb_bz_thick / inb_sum,
            tokamak_geometry.inb_mnfld_thick / inb_sum,
            tokamak_geometry.inb_vv_thick / inb_sum,
        )
        outb_sum = sum([
            tokamak_geometry.outb_fw_thick,
            tokamak_geometry.outb_bz_thick,
            tokamak_geometry.outb_mnfld_thick,
            tokamak_geometry.outb_vv_thick,
        ])
        outb = OutboardThicknessFraction(
            tokamak_geometry.outb_fw_thick / outb_sum,
            tokamak_geometry.outb_bz_thick / outb_sum,
            tokamak_geometry.outb_mnfld_thick / outb_sum,
            tokamak_geometry.outb_vv_thick / outb_sum,
        )
        return cls(inb, outb, transition_radius)


class TokamakThicknesses(ThicknessFractions):
    """Placeholder class, to be filled in later."""

    pass


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
