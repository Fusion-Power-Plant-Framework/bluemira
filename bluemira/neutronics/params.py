# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
"""dataclasses containing parameters used to set up the openmc model."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Tuple, Union

import openmc

from bluemira.base.constants import raw_uc
from bluemira.neutronics.constants import energy_per_dt
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
    electron_treatment: Literal["ttb", "led"] = "ttb"
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
class TokamakOperationParameters:
    """Parameters describing how the tokamak is operated,
    i.e. where the plasma is positioned (and therefore where the power is concentrated),
    and what temperature the plasma is at.

    Parameters
    ----------
    reactor_power:
        total reactor (thermal) power when operating at 100%
    peaking_factor:
        (max. heat flux on fw)/(avg. heat flux on fw)
    shaf_shift:
        shafranov shift
        how far (towards the outboard direction) is the centre of the plasma shifted
        compared to the geometric center of the poloidal cross-section.
    vertical_shift:
        how far (upwards) in the z direction is the centre of the plasma
        shifted compared to the geometric center of the poloidal cross-section.
    """

    reactor_power: float  # [W]
    temperature: float  # [K]
    peaking_factor: float  # [dimensionless]
    shaf_shift: float  # [m]
    vertical_shift: float  # [m]

    def calculate_total_neutron_rate(self) -> float:  # [1/s]
        """Convert the reactor power to neutron rate
        (number of neutrons produced per second) assuming 100% efficiency.
        """
        return self.reactor_power / energy_per_dt


@dataclass(frozen=True)
class TokamakOperationParametersPPS(TokamakOperationParameters):
    """See TokamakOperationParameters

    Addition of plasma_physics_units converted variables
    """

    plasma_physics_units: TokamakOperationParameters

    @classmethod
    def from_si(cls, op_params: TokamakOperationParameters):
        """Convert from si units dataclass"""
        conversion = {
            "reactor_power": ("W", "MW"),
            "temperature": ("K", "keV"),
            "shaf_shift": ("m", "cm"),
            "vertical_shift": ("m", "cm"),
        }
        op = asdict(op_params)
        op_ppu = op.copy()
        for k, v in op_ppu.items():
            if k in conversion:
                op_ppu[k] = raw_uc(v, *conversion[k])
        return cls(**op, plasma_physics_units=TokamakOperationParameters(**op_ppu))


@dataclass(frozen=True)
class TokamakGeometry:
    """The measurements for all of the generic SOLID components of the tokamak.

    Parameters
    ----------
    major_r:
        major radius
        how far is the origin in the poloidal view from the center of the torus.
    minor_r:
        minor radius (R0 in equation referenced below)
        radius of the poloidal cross-section
    elong:
        elongation (a in equation referenced below)
        how eccentric the poloidal ellipse is
    triang:
        triangularity (δ in equation referenced below)
        second order eccentricity (best visualized by plotting R(θ) wrt.θ in eq. below)

    Notes
    -----
    R = R0 + a cos(θ + δ sin θ)
    https://hibp.ecse.rpi.edu/~connor/education/plasma/PlasmaEngineering/Miyamoto.pdf
    page.239 # noqa: W505

    Other terminologies:

        thick: thickness
        inb: inboard
        outb: outboard
        fw: first wall
        bz: breeding zone
        mnfld: manifold
        vv: vacuum vessel
        tf: toroidal field
    """

    major_r: float  # [m]
    minor_r: float  # [m]
    elong: float  # [dimensionless]
    triang: float  # [dimensionless]
    inb_fw_thick: float  # [m]
    inb_bz_thick: float  # [m]
    inb_mnfld_thick: float  # [m]
    inb_vv_thick: float  # [m]
    tf_thick: float  # [m]
    outb_fw_thick: float  # [m]
    outb_bz_thick: float  # [m]
    outb_mnfld_thick: float  # [m]
    outb_vv_thick: float  # [m]
    inb_gap: float  # [m]


@dataclass(frozen=True)
class TokamakGeometryCGS(TokamakGeometry):
    """See TokamakGeometry

    Addition of cgs converted variables
    """

    cgs: TokamakGeometry

    @classmethod
    def from_si(cls, tokamak_geometry: TokamakGeometry):
        """Convert from si units dataclass"""
        tg = asdict(tokamak_geometry)
        tgcgs = tg.copy()
        for k, v in tgcgs.items():
            if k not in ("elong", "triang"):
                tgcgs[k] = raw_uc(v, "m", "cm")
        return cls(**tg, cgs=TokamakGeometry(**tgcgs))


def get_preset_physical_properties(
    blanket_type: Union[str, BlanketType],
) -> Tuple[BreederTypeParameters, TokamakGeometry]:
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

    shared_poloidal_outline = {
        "major_r": 8.938,  # [m]
        "minor_r": 2.883,  # [m]
        "elong": 1.65,  # [dimensionless]
        "triang": 0.333,  # [m]
    }
    shared_building_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": 0.2,  # [m]
        "inb_vv_thick": 0.6,  # [m]
        "tf_thick": 0.4,  # [m]
        "outb_vv_thick": 0.6,  # [m]
    }
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.378,  # [m]
            inb_mnfld_thick=0.435,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.538,  # [m]
            outb_mnfld_thick=0.429,  # [m]
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
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
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.460,  # [m]
            inb_mnfld_thick=0.560,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.460,  # [m]
            outb_mnfld_thick=0.560,  # [m]
        )

    return breeder_materials, tokamak_geometry
