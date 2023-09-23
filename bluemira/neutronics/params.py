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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import openmc

import bluemira.neutronics.make_materials as mm
from bluemira.base.constants import raw_uc
from bluemira.neutronics.constants import energy_per_dt


@dataclass
class OpenMCSimulationRuntimeParameters:
    """Parameters used in the actual simulation

    Parameters
    ----------
    particles: int
        Number of neutrons emitted by the plasma source per batch.
    batches: int, default=2
        How many batches to simulate.
    photon_transport: bool, default=True
        Whether to simulate the transport of photons (i.e. gamma-rays created) or not.
    electron_treatment: {'ttb','led'}
        The way in which OpenMC handles secondary charged particles.
        'thick-target bremsstrahlung' or 'local energy deposition'
        'thick-target bremsstrahlung' accounts for the energy carried away by
            bremsstrahlung photons and deposited elsewhere, whereas 'local energy
            deposition' assumes electrons deposit all energies locally.
        (the latter is expected to be computationally faster.)
    run_mode: str, {'eigenvalue', 'fixed source', 'plot', 'volume', 'particle restart'}
        see below for details:
        https://docs.openmc.org/en/stable/usersguide/settings.html#run-modes
    openmc_write_summary: bool
        whether openmc should write a 'summary.h5' file or not.
    cross_section_xml:
        Where the xml file for cross-section is stored locally.
    """

    # Parameters used outside of setup_openmc()
    volume_calc_particles: int  # number of particles used in the stochastic volume calculation.
    # parameters used inside setup_openmc()
    parametric_source: bool  # to use the pps_isotropic module or not.
    particles: int  # number of particles used in the neutronics simulation
    cross_section_xml: Union[str, Path]
    batches: int = 2
    photon_transport: bool = True
    electron_treatment: Literal["ttb", "led"] = "ttb"
    run_mode: str = openmc.settings.RunMode.FIXED_SOURCE.value
    openmc_write_summary: bool = False


@dataclass
class BreederTypeParameters:
    """Dataclass to hold information about the breeder blanket material
    and design choices.
    """

    enrichment_fraction_Li6: float  # [dimensionless]
    blanket_type: mm.BlanketType


class DataclassUnitConverter:
    def __init__(self, parent, unit_converter_dict):
        self.parent = parent
        for attr, in_out_units in unit_converter_dict.items():
            setattr(self, attr, raw_uc(getattr(self.parent, attr), *in_out_units))


@dataclass(frozen=True)
class TokamakOperationParameters:
    """Parameters describing how the tokamak is operated,
    i.e. where the plasma is positioned (and therefore where the power is concentrated),
    and what temperature the plasma is at.

    Parameters
    ----------
    reactor_power: total reactor (thermal) power when operating at 100%
    peaking_factor: (max. heat flux on fw)/(avg. heat flux on fw)
    shaf_shift: shafranov shift
        how far (towards the outboard direction) is the centre of the plasma shifted
        compared to the geometric center of the poloidal cross-section.
    vertical_shift: how far (upwards) in the z direction is the centre of the plasma
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

    def __post_init__(self):
        object.__setattr__(  # hack to get around the frozen attributes
            self,
            "plasma_physics_units",
            DataclassUnitConverter(
                self,
                {
                    "reactor_power": ("W", "MW"),
                    "temperature": ("K", "keV"),
                    "peaking_factor": ("1", "1"),
                    "shaf_shift": ("m", "cm"),
                    "vertical_shift": ("m", "cm"),
                },
            ),
        )


@dataclass(frozen=True)
class TokamakGeometry:
    """The measurements for all of the generic SOLID components of the tokamak.

    Parameters
    ----------
    major_r: major radius
        how far is the origin in the poloidal view from the center of the torus.
    minor_r: minor radius (R0 in equation referenced below)
        radius of the poloidal cross-section
    elong: elongation (a in equation referenced below)
        how eccentric the poloidal ellipse is
    triang: triangularity (δ in equation referenced below)
        second order eccentricity (best visualized by plotting R(θ) wrt.θ in eq. below)
    Reference
    ---------
    R = R0 + a cos(θ + δ sin θ)
    https://hibp.ecse.rpi.edu/~connor/education/plasma/PlasmaEngineering/Miyamoto.pdf
        page.239 # noqa: W505

    Other terminologies
    -------------------
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

    def __post_init__(self):
        object.__setattr__(  # hack to get around the frozen attributes
            self,
            "cgs",
            DataclassUnitConverter(
                self,
                {
                    "major_r": ("m", "cm"),
                    "minor_r": ("m", "cm"),
                    "elong": ("1", "1"),
                    "triang": ("1", "1"),
                    "inb_fw_thick": ("m", "cm"),
                    "inb_bz_thick": ("m", "cm"),
                    "inb_mnfld_thick": ("m", "cm"),
                    "inb_vv_thick": ("m", "cm"),
                    "tf_thick": ("m", "cm"),
                    "outb_fw_thick": ("m", "cm"),
                    "outb_bz_thick": ("m", "cm"),
                    "outb_mnfld_thick": ("m", "cm"),
                    "outb_vv_thick": ("m", "cm"),
                    "inb_gap": ("m", "cm"),
                },
            ),
        )
