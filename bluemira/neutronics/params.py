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

import bluemira.neutronics.make_materials as mm
from bluemira.base.constants import raw_uc
from bluemira.neutronics.constants import energy_per_dt


@dataclass
class OpenMCSimulationRuntimeParameters:
    """Parameters used in the actual simulation"""

    # parameters used in setup_openmc()
    particles: int  # number of particles used in the neutronics simulation
    batches: int
    photon_transport: bool
    electron_treatment: str
    run_mode: str
    openmc_write_summary: bool
    # Parameters used elsewhere
    parametric_source: bool
    volume_calc_particles: int  # number of particles used in the volume calculation.
    cross_section_xml: str


@dataclass
class TokamakOperationParameters:
    """The tokamak's operational parameter, such as its power"""

    reactor_power: float  # [W]

    def calculate_total_neutron_rate(self) -> float:  # [1/s]
        """Convert the reactor power to neutron rate
        (number of neutrons produced per second) assuming 100% efficiency.
        """
        return self.reactor_power / energy_per_dt


@dataclass
class BreederTypeParameters:
    """Dataclass to hold information about the breeder blanket material
    and design choices.
    """

    li_enrich_percent: float  # [%]
    blanket_type: mm.BlanketType


@dataclass
class TokamakGeometry:
    """The measurements for all of the geneic components of the tokamak"""

    minor_r: float  # [m]
    major_r: float  # [m]
    elong: float  # [dimensionless]
    # TODO: Move 'shaf_shift', 'peaking_factor' and 'temperature' into TokamakOperationParameters
    shaf_shift: float  # [m]
    vertical_shift: float  # [m]
    peaking_factor: float  # [m]
    inb_fw_thick: float  # [m]
    inb_bz_thick: float  # [m]
    inb_mnfld_thick: float  # [m]
    inb_vv_thick: float  # [m]
    tf_thick: float  # [m]
    outb_fw_thick: float  # [m]
    outb_bz_thick: float  # [m]
    outb_mnfld_thick: float  # [m]
    outb_vv_thick: float  # [m]
    triang: float  # [dimensionless]
    inb_gap: float  # [m]


@dataclass
class TokamakGeometryCGS:
    """The measurements for all of the geneic components of the tokamak,
    provided in CGS (Centimeter, Grams, Seconds) units.
    """

    minor_r: float  # [cm]
    major_r: float  # [cm]
    elong: float  # [dimensionless]
    shaf_shift: float  # [cm]
    vertical_shift: float  # [cm]
    peaking_factor: float  # [cm]
    inb_fw_thick: float  # [cm]
    inb_bz_thick: float  # [cm]
    inb_mnfld_thick: float  # [cm]
    inb_vv_thick: float  # [cm]
    tf_thick: float  # [cm]
    outb_fw_thick: float  # [cm]
    outb_bz_thick: float  # [cm]
    outb_mnfld_thick: float  # [cm]
    outb_vv_thick: float  # [cm]
    triang: float  # [dimensionless]
    inb_gap: float  # [cm]

    @classmethod
    def from_SI(cls, tokamak_geometry: TokamakGeometry):
        return cls(
            raw_uc(tokamak_geometry.minor_r, "m", "cm"),
            raw_uc(tokamak_geometry.major_r, "m", "cm"),
            tokamak_geometry.elong,
            raw_uc(tokamak_geometry.shaf_shift, "m", "cm"),
            raw_uc(tokamak_geometry.vertical_shift, "m", "cm"),
            raw_uc(tokamak_geometry.peaking_factor, "m", "cm"),
            raw_uc(tokamak_geometry.inb_fw_thick, "m", "cm"),
            raw_uc(tokamak_geometry.inb_bz_thick, "m", "cm"),
            raw_uc(tokamak_geometry.inb_mnfld_thick, "m", "cm"),
            raw_uc(tokamak_geometry.inb_vv_thick, "m", "cm"),
            raw_uc(tokamak_geometry.tf_thick, "m", "cm"),
            raw_uc(tokamak_geometry.outb_fw_thick, "m", "cm"),
            raw_uc(tokamak_geometry.outb_bz_thick, "m", "cm"),
            raw_uc(tokamak_geometry.outb_mnfld_thick, "m", "cm"),
            raw_uc(tokamak_geometry.outb_vv_thick, "m", "cm"),
            tokamak_geometry.triang,
            raw_uc(tokamak_geometry.inb_gap, "m", "cm"),
        )
