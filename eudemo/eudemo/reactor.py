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

from bluemira.base.designer import run_designer
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.base.reactor import Reactor
from bluemira.builders.divertor import DivertorBuilder
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.thermal_shield import VVTSBuilder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.tools import make_polygon
from eudemo.blanket import Blanket, BlanketBuilder
from eudemo.equilibria import (
    EquilibriumDesigner,
    FixedEquilibriumDesigner,
    FreeBoundaryEquilibriumFromFixedDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.params import EUDEMOReactorParams
from eudemo.pf_coils import PFCoilsDesigner
from eudemo.radial_build import radial_build
from eudemo.tf_coils import TFCoil, TFCoilBuilder, TFCoilDesigner
from eudemo.thermal_shield import VacuumVesselThermalShield
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


def build_divertor(params, build_config, div_silhouette) -> Divertor:
    """Build the divertor given a silhouette of a sector."""
    builder = DivertorBuilder(params, build_config, div_silhouette)
    return Divertor(builder.build())


def build_blanket(params, build_config, blanket_face) -> Blanket:
    """Build the blanket given a silhouette of a sector."""
    builder = BlanketBuilder(params, build_config, blanket_face)
    return Blanket(builder.build())


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


def _read_json(file_path: str) -> Dict:
    """Read a JSON file to a dictionary."""
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
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

    # reactor.plasma = build_plasma(build_config.get("Plasma", {}), eq)

    # blanket_face, divertor_face, ivc_boundary = design_ivc(
    #     params, build_config["IVC"], equilibrium=eq
    # )

    # reactor.vacuum_vessel = build_vacuum_vessel(
    #     params, build_config.get("Vacuum vessel", {}), ivc_boundary
    # )
    # reactor.divertor = build_divertor(
    #     params, build_config.get("Divertor", {}), divertor_face
    # )
    # reactor.blanket = build_blanket(
    #     params, build_config.get("Blanket", {}), blanket_face
    # )

    # thermal_shield_config = build_config.get("Thermal shield", {})
    # vv_thermal_shield = VVTSBuilder(
    #     params,
    #     thermal_shield_config.get("Vacuum vessel", {}),
    #     keep_out_zone=reactor.vacuum_vessel.xz_boundary(),
    # )
    # reactor.vv_thermal = vv_thermal_shield.build()

    # reactor.tf_coils = build_tf_coils(
    #     params,
    #     build_config.get("TF coils", {}),
    #     reactor.plasma.lcfs(),
    #     reactor.vacuum_vessel.xz_boundary(),
    # )

    # pf_coil_keep_out_zones = []  # Update when ports are added
    # pf_designer = PFCoilsDesigner(
    #     params,
    #     build_config.get("PF coils", {}),
    #     reactor.tf_coils.boundary(),
    #     pf_coil_keep_out_zones,
    # )
    # coilset = pf_designer.execute()

    # reactor.show_cad()
