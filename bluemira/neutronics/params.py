# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""dataclasses containing parameters used to set up the openmc model."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Tuple, Union

import openmc

from bluemira.base.constants import raw_uc
from bluemira.neutronics.make_materials import BlanketType


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
    parametric_source: bool  # to use the pps_isotropic module or not.
    particles: int  # number of particles used in the neutronics simulation
    cross_section_xml: Union[str, Path]
    batches: int = 2
    photon_transport: bool = True
    electron_treatment: Literal[
        "ttb", "led"
    ] = "led"  # Bremsstrahlung only matters for very thin objects
    run_mode: str = openmc.settings.RunMode.FIXED_SOURCE.value
    openmc_write_summary: bool = False
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
class TokamakOperationParametersBase:
    """Parameters describing how the tokamak is operated,
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

    reactor_power: float  # [W]
    peaking_factor: float  # [dimensionless]
    temperature: float  # [K]
    shaf_shift: float  # [m]
    vertical_shift: float  # [m]


@dataclass(frozen=True)
class TokamakOperationParameters(TokamakOperationParametersBase):
    """See TokamakOperationParametersBase

    Addition of plasma_physics_units converted variables
    """

    plasma_physics_units: TokamakOperationParametersBase

    @classmethod
    def from_si(cls, op_params: TokamakOperationParametersBase):
        """Convert from si units dataclass"""
        conversion = {
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
        return cls(**op, plasma_physics_units=TokamakOperationParametersBase(**op_pps))


@dataclass(frozen=True)
class PlasmaGeometryBase:
    """List of parameters describing the

    Parameters
    ----------
    major_r:
        major radius [m]
        how far is the origin in the poloidal view from the center of the torus.
    minor_r:
        minor radius [m] (R0 in equation referenced below)
        radius of the poloidal cross-section
    elong:
        elongation [dimensionless] (a in equation referenced below)
        how eccentric the poloidal ellipse is
    triang:
        triangularity [dimensionless] (δ in equation referenced below)
        second order eccentricity (best visualized by plotting R(θ) wrt.θ in eq. below)

    Notes
    -----
    R = R0 + a cos(θ + δ sin θ)
    https://hibp.ecse.rpi.edu/~connor/education/plasma/PlasmaEngineering/Miyamoto.pdf
    page.239 # noqa: W505
    """

    major_r: float
    minor_r: float
    elong: float
    triang: float


@dataclass(frozen=True)
class PlasmaGeometry(PlasmaGeometryBase):
    """See PlasmaGeometryBase

    Addition of cgs converted variables
    """

    cgs: PlasmaGeometryBase

    @classmethod
    def from_si(cls, plasma_geometry: PlasmaGeometryBase):
        """Convert from si units dataclass"""
        pg = asdict(plasma_geometry)
        pgcgs = pg.copy()
        for k, v in pgcgs.items():
            if k.endswith("_r"):  # minor and major radii
                pgcgs[k] = raw_uc(v, "m", "cm")
        return cls(**pg, cgs=PlasmaGeometryBase(**pgcgs))


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
    def from_si(cls, tokamak_geometry: TokamakGeometryBase):
        """Convert from si units dataclass"""
        tg = asdict(tokamak_geometry)
        tgcgs = tg.copy()
        for k, v in tgcgs.items():
            tgcgs[k] = raw_uc(v, "m", "cm")
        return cls(**tg, cgs=TokamakGeometryBase(**tgcgs))


def get_preset_physical_properties(
    blanket_type: Union[str, BlanketType],
) -> Tuple[BreederTypeParameters, PlasmaGeometryBase, TokamakGeometryBase]:
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

    plasma_geometry = PlasmaGeometryBase(
        major_r=8.938,  # [m]
        minor_r=2.883,  # [m]
        elong=1.65,  # [dimensionless]
        triang=0.333,  # [m]
    )

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

    return breeder_materials, plasma_geometry, tokamak_geometry
