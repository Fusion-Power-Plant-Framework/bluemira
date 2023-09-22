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
from bluemira.neutronics.constants import energy_per_dt_MeV


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

    reactor_power_MW: float  # MW

    def calculate_total_neutron_rate(self) -> float:
        """Convert the reactor power to neutron rate
        (number of neutrons produced per second)
        """
        reactor_power_in_MeV_per_s = raw_uc(self.reactor_power_MW, "MW", "MeV/s")
        return reactor_power_in_MeV_per_s / energy_per_dt_MeV


@dataclass
class BreederTypeParameters:
    """Dataclass to hold information about the breeder blanket material
    and design choices.
    """

    li_enrich_ao: float
    blanket_type: mm.BlanketType


@dataclass
class TokamakGeometry:
    """The measurements for all of the geneic components of the tokamak"""

    minor_r: float  # [cm]
    major_r: float  # [cm]
    elong: float  # [dimensionless]
    shaf_shift: float  # [cm]
    inb_fw_thick: float  # [cm]
    inb_bz_thick: float  # [cm]
    inb_mnfld_thick: float  # [cm]
    inb_vv_thick: float  # [cm]
    tf_thick: float  # [cm]
    outb_fw_thick: float  # [cm]
    outb_bz_thick: float  # [cm]
    outb_mnfld_thick: float  # [cm]
    outb_vv_thick: float  # [cm]
    triang: float = 0.333  # [dimensionless]
    inb_gap: float = 20.0  # [cm]
