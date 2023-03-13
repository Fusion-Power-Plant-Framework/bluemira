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

"""
The EUDEMO reactor design routine.

1. Radial build (using PROCESS)
2. Perform equilibria optimisation
3. Build plasma
4. Design scaffold for the IVCs
5. Build vacuum vessel
6. Build TF coils
7. Build PF coils
8. Build cryo thermal shield
9. Build cryostat
10. Build radiation shield
11. Produce power cycle report
"""

import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from bluemira.base.components import Component
from bluemira.base.designer import run_designer
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.base.reactor import Reactor
from bluemira.builders.cryostat import CryostatBuilder, CryostatDesigner
from bluemira.builders.divertor import DivertorBuilder
from bluemira.builders.pf_coil import PFCoilBuilder, PFCoilPictureFrame
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import CryostatTSBuilder, VVTSBuilder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from eudemo.blanket import Blanket, BlanketBuilder
from eudemo.comp_managers import (
    Cryostat,
    CryostatThermalShield,
    RadiationShield,
    VacuumVesselThermalShield,
)
from eudemo.equilibria import (
    FixedEquilibriumDesigner,
    FreeBoundaryEquilibriumFromFixedDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.maintenance.upper_port import UpperPortDesigner
from eudemo.params import EUDEMOReactorParams
from eudemo.pf_coils import PFCoil, PFCoilsDesigner, build_pf_coils_component
from eudemo.power_cycle import SteadyStatePowerCycleSolver
from eudemo.radial_build import radial_build
from eudemo.tf_coils.tf_coils import TFCoil, TFCoilBuilder, TFCoilDesigner
from eudemo.vacuum_vessel import VacuumVessel, VacuumVesselBuilder

CONFIG_DIR = Path(__file__).parent.parent / "config"
PARAMS_FILE = os.path.join(CONFIG_DIR, "params.json")


class EUDEMO(Reactor):
    """EUDEMO reactor definition."""

    plasma: Plasma
    vacuum_vessel: VacuumVessel
    divertor: Divertor
    blanket: Blanket
    tf_coils: TFCoil
    vv_thermal: VacuumVesselThermalShield
    pf_coils: PFCoil
    cryostat: Cryostat
    cryostat_thermal: CryostatThermalShield
    radiation_shield: RadiationShield


def build_plasma(build_config: Dict, eq: Equilibrium) -> Plasma:
    """Build EUDEMO plasma from an equilibrium."""
    lcfs_loop = eq.get_LCFS()
    lcfs_wire = make_polygon({"x": lcfs_loop.x, "z": lcfs_loop.z}, closed=True)
    builder = PlasmaBuilder(build_config, lcfs_wire)
    return Plasma(builder.build())


def build_vacuum_vessel(params, build_config, ivc_koz) -> VacuumVessel:
    """Build the vacuum vessel around the given IVC keep-out zone."""
    vv_builder = VacuumVesselBuilder(params, build_config, ivc_koz)
    return VacuumVessel(vv_builder.build())


def build_vacuum_vessel_thermal_shield(
    params, build_config, vv_koz
) -> VacuumVesselThermalShield:
    """Build the vacuum vessel thermal shield around the given  VV keep-out zone"""
    vvts_builder = VVTSBuilder(params, build_config, vv_koz)
    return VacuumVesselThermalShield(vvts_builder.build())


def build_divertor(params, build_config, div_silhouette) -> Divertor:
    """Build the divertor given a silhouette of a sector."""
    builder = DivertorBuilder(params, build_config, div_silhouette)
    return Divertor(builder.build())


def build_blanket(
    params, build_config, blanket_face, r_inner_cut: float, cut_angle: float
) -> Blanket:
    """Build the blanket given a silhouette of a sector."""
    builder = BlanketBuilder(params, build_config, blanket_face, r_inner_cut, cut_angle)
    return Blanket(builder.build())


def build_vvts(params, build_config, vv_boundary) -> VacuumVesselThermalShield:
    """Build the vacuum vessel thermal shield"""
    vv_thermal_shield = VVTSBuilder(
        params,
        build_config.get("Vacuum vessel", {}),
        keep_out_zone=vv_boundary,
    )
    return VacuumVesselThermalShield(vv_thermal_shield.build())


def build_tf_coils(
    params, build_config, separatrix, vacuum_vessel_cross_section
) -> TFCoil:
    """Design and build the TF coils for the reactor."""
    centreline, wp_cross_section = run_designer(
        TFCoilDesigner,
        params,
        build_config,
        separatrix=separatrix,
        keep_out_zone=vacuum_vessel_cross_section,
    )
    builder = TFCoilBuilder(
        params, build_config, centreline.create_shape(), wp_cross_section
    )
    return TFCoil(builder.build(), builder._make_field_solver())


def build_pf_coils(
    params, build_config, tf_coil_boundary, pf_coil_keep_out_zones=()
) -> PFCoil:
    """
    Design and build the PF coils for the reactor.
    """
    pf_designer = PFCoilsDesigner(
        params,
        build_config,
        tf_coil_boundary,
        pf_coil_keep_out_zones,
    )
    coilset = pf_designer.execute()
    component = build_pf_coils_component(params, build_config, coilset)
    return PFCoil(component, coilset)


def build_cryots(params, build_config, pf_kozs, tf_koz) -> CryostatThermalShield:
    """
    Build the Cryostat thermal shield for the reactor.
    """
    return CryostatThermalShield(
        CryostatTSBuilder(
            params,
            build_config.get("Cryostat", {}),
            reactor.pf_coils.xz_boundary(),
            reactor.tf_coils.boundary(),
        ).build()
    )


def build_cryostat(params, build_config, cryostat_thermal_koz) -> Cryostat:
    """
    Design and build the Cryostat for the reactor.
    """
    cryod = CryostatDesigner(params, cryostat_thermal_koz)
    return Cryostat(CryostatBuilder(params, build_config, *cryod.execute()).build())


def build_radiation_shield(params, build_config, cryostat_koz) -> RadiationShield:
    """
    Design and build the Radition shield for the reactor.
    """
    return RadiationShield(
        RadiationShieldBuilder(params, build_config, BluemiraFace(cryostat_koz)).build()
    )


def _read_json(file_path: str) -> Dict:
    """Read a JSON file to a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import time

    reactor = EUDEMO("EUDEMO")
    params = make_parameter_frame(PARAMS_FILE, EUDEMOReactorParams)
    if params is None:
        raise ValueError("Params cannot be None")
    build_config = _read_json(os.path.join(CONFIG_DIR, "build_config.json"))

    params = radial_build(params, build_config["Radial build"])
    fixed_boundary_eq = run_designer(
        FixedEquilibriumDesigner, params, build_config["Fixed boundary equilibrium"]
    )

    free_boundary_eq = run_designer(
        FreeBoundaryEquilibriumFromFixedDesigner,
        params,
        build_config["Free boundary equilibrium"],
    )

    reactor.plasma = build_plasma(build_config.get("Plasma", {}), free_boundary_eq)

    blanket_face, divertor_face, ivc_boundary = design_ivc(
        params, build_config["IVC"], equilibrium=free_boundary_eq
    )

    t1 = time.time()
    upper_port_designer = UpperPortDesigner(
        params, build_config.get("Upper Port", {}), blanket_face
    )
    t2 = time.time()
    print(f"{t2-t1}")
    upper_port_xz, r_inner_cut, cut_angle = upper_port_designer.execute()
    t3 = time.time()
    print(f"{t3-t2}")
    reactor.vacuum_vessel = build_vacuum_vessel(
        params, build_config.get("Vacuum vessel", {}), ivc_boundary
    )
    t4 = time.time()
    print(f"{t4-t3}")

    reactor.divertor = build_divertor(
        params, build_config.get("Divertor", {}), divertor_face
    )
    t5 = time.time()
    print(f"{t5-t4}")

    reactor.blanket = build_blanket(
        params, build_config.get("Blanket", {}), blanket_face, r_inner_cut, cut_angle
    )
    t6 = time.time()
    print(f"{t6-t5}")

    thermal_shield_config = build_config.get("Thermal shield", {})
    reactor.vv_thermal = build_vacuum_vessel_thermal_shield(
        params,
        thermal_shield_config.get("VVTS", {}),
        reactor.vacuum_vessel.xz_boundary(),
    )
    t7 = time.time()
    print(f"{t7-t6}")

    reactor.tf_coils = build_tf_coils(
        params,
        build_config.get("TF coils", {}),
        reactor.plasma.lcfs(),
        reactor.vv_thermal.xz_boundary(),
    )
    t8 = time.time()
    print(f"{t8-t7}")

    reactor.pf_coils = build_pf_coils(
        params,
        build_config.get("PF coils", {}),
        reactor.tf_coils.boundary(),
        pf_coil_keep_out_zones=[],
    )

    reactor.cryostat_thermal = build_cryots(
        params,
        build_config.get("Thermal shield", {}),
        reactor.pf_coils.xz_boundary(),
        reactor.tf_coils.boundary(),
    )

    reactor.cryostat = build_cryostat(
        params, build_config.get("Cryostat", {}), reactor.cryostat_thermal.xz_boundary()
    )

    reactor.radiation_shield = build_radiation_shield(
        params, build_config.get("RadiationShield", {}), reactor.cryostat.xz_boundary()
    )

    sspc_solver = SteadyStatePowerCycleSolver(params)
    sspc_result = sspc_solver.execute()
    sspc_solver.model.plot()
    plt.show()

    del reactor.pf_coils  # Because of bug

    reactor.show_cad()
