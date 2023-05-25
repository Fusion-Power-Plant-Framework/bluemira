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

import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from bluemira.base.designer import run_designer
from bluemira.base.logs import set_log_level
from bluemira.base.reactor import Reactor
from bluemira.base.reactor_config import ReactorConfig
from bluemira.builders.cryostat import CryostatBuilder, CryostatDesigner
from bluemira.builders.divertor import DivertorBuilder
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import CryostatTSBuilder, VVTSBuilder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import interpolate_bspline
from eudemo.blanket import Blanket, BlanketBuilder, BlanketDesigner
from eudemo.coil_structure import build_coil_structures_component
from eudemo.comp_managers import (
    CoilStructures,
    Cryostat,
    CryostatThermalShield,
    RadiationShield,
    VacuumVesselThermalShield,
)
from eudemo.equilibria import (
    DummyFixedEquilibriumDesigner,
    FixedEquilibriumDesigner,
    ReferenceFreeBoundaryEquilibriumDesigner,
)
from eudemo.ivc import design_ivc
from eudemo.ivc.divertor_silhouette import Divertor
from eudemo.maintenance.lower_port import LowerPortBuilder, LowerPortDuctDesigner
from eudemo.maintenance.upper_port import UpperPortDesigner
from eudemo.params import EUDEMOReactorParams
from eudemo.pf_coils import PFCoil, PFCoilsDesigner, build_pf_coils_component
from eudemo.power_cycle import SteadyStatePowerCycleSolver
from eudemo.radial_build import radial_build
from eudemo.tf_coils import TFCoil, TFCoilBuilder, TFCoilDesigner
from eudemo.vacuum_vessel import VacuumVessel, VacuumVesselBuilder

CONFIG_DIR = Path(__file__).parent.parent / "config"
BUILD_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "build_config.json")


class EUDEMO(Reactor):
    """EUDEMO reactor definition."""

    plasma: Plasma
    vacuum_vessel: VacuumVessel
    vv_thermal: VacuumVesselThermalShield
    divertor: Divertor
    blanket: Blanket
    tf_coils: TFCoil
    pf_coils: PFCoil
    coil_structures: CoilStructures
    cryostat: Cryostat
    cryostat_thermal: CryostatThermalShield
    radiation_shield: RadiationShield


def build_plasma(params, build_config: Dict, eq: Equilibrium) -> Plasma:
    """Build EUDEMO plasma from an equilibrium."""
    lcfs_loop = eq.get_LCFS()
    lcfs_wire = interpolate_bspline({"x": lcfs_loop.x, "z": lcfs_loop.z}, closed=True)
    builder = PlasmaBuilder(params, build_config, lcfs_wire)
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


def build_lower_port(params, build_config, divertor_face, tf_coils_outer_boundary):
    """Builder for the Lower Port and Duct"""
    (
        lp_duct_xz_void_space,
        lp_duct_xz_koz,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
    ) = LowerPortDuctDesigner(
        params, build_config, divertor_face, tf_coils_outer_boundary
    ).execute()
    builder = LowerPortBuilder(
        params,
        build_config,
        lp_duct_xz_koz,
        lp_duct_angled_nowall_extrude_boundary,
        lp_duct_straight_nowall_extrude_boundary,
    )
    return builder.build(), lp_duct_xz_koz


def build_blanket(
    params,
    build_config: Dict,
    blanket_boundary,
    blanket_face,
    r_inner_cut: float,
    cut_angle: float,
) -> Blanket:
    """Build the blanket given a silhouette of a sector."""
    designer = BlanketDesigner(
        params, blanket_boundary, blanket_face, r_inner_cut, cut_angle
    )
    ib_silhouette, ob_silhouette = designer.execute()
    builder = BlanketBuilder(params, build_config, ib_silhouette, ob_silhouette)
    return Blanket(builder.build())


def build_tf_coils(params, build_config, separatrix, vvts_cross_section) -> TFCoil:
    """Design and build the TF coils for the reactor."""
    centreline, wp_cross_section = run_designer(
        TFCoilDesigner,
        params,
        build_config,
        separatrix=separatrix,
        keep_out_zone=vvts_cross_section,
    )
    builder = TFCoilBuilder(
        params, build_config, centreline.create_shape(), wp_cross_section
    )
    return TFCoil(builder.build(), builder._make_field_solver())


def build_pf_coils(
    params,
    build_config,
    reference_equilibrium,
    tf_coil_boundary,
    pf_coil_keep_out_zones=(),
) -> PFCoil:
    """
    Design and build the PF coils for the reactor.
    """
    pf_designer = PFCoilsDesigner(
        params,
        build_config,
        reference_equilibrium,
        tf_coil_boundary,
        pf_coil_keep_out_zones,
    )
    coilset = pf_designer.execute()
    component = build_pf_coils_component(params, build_config, coilset)
    return PFCoil(component, coilset)


def build_coil_structures(
    params,
    build_config,
    tf_coil_xz_face,
    pf_coil_xz_wires,
    pf_coil_keep_out_zones,
) -> CoilStructures:
    """
    Design and build the coil structures for the reactor.
    """
    component = build_coil_structures_component(
        params, build_config, tf_coil_xz_face, pf_coil_xz_wires, pf_coil_keep_out_zones
    )
    return CoilStructures(component)


def build_cryots(params, build_config, pf_kozs, tf_koz) -> CryostatThermalShield:
    """
    Build the Cryostat thermal shield for the reactor.
    """
    return CryostatThermalShield(
        CryostatTSBuilder(
            params,
            build_config,
            pf_kozs,
            tf_koz,
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
    Design and build the Radiation shield for the reactor.
    """
    return RadiationShield(
        RadiationShieldBuilder(params, build_config, BluemiraFace(cryostat_koz)).build()
    )


if __name__ == "__main__":
    set_log_level("INFO")
    reactor_config = ReactorConfig(
        BUILD_CONFIG_FILE_PATH, EUDEMOReactorParams, warn_on_empty_config=False
    )
    reactor = EUDEMO(
        "EUDEMO",
        n_sectors=reactor_config.global_params.n_TF.value,
    )

    radial_build(
        reactor_config.params_for("Radial build").global_params,
        reactor_config.config_for("Radial build"),
    )
    lcfs_coords, profiles = run_designer(
        FixedEquilibriumDesigner,
        reactor_config.params_for("Fixed boundary equilibrium"),
        reactor_config.config_for("Fixed boundary equilibrium"),
    )

    lcfs_coords, profiles = run_designer(
        DummyFixedEquilibriumDesigner,
        reactor_config.params_for("Dummy fixed boundary equilibrium"),
        reactor_config.config_for("Dummy fixed boundary equilibrium"),
    )

    reference_eq = run_designer(
        ReferenceFreeBoundaryEquilibriumDesigner,
        reactor_config.params_for("Free boundary equilibrium"),
        reactor_config.config_for("Free boundary equilibrium"),
        lcfs_coords=lcfs_coords,
        profiles=profiles,
    )

    reactor.plasma = build_plasma(
        reactor_config.params_for("Plasma"),
        reactor_config.config_for("Plasma"),
        reference_eq,
    )

    ivc_shapes = design_ivc(
        reactor_config.params_for("IVC").global_params,
        reactor_config.config_for("IVC"),
        equilibrium=reference_eq,
    )

    upper_port_designer = UpperPortDesigner(
        reactor_config.params_for("Upper Port"),
        reactor_config.config_for("Upper Port"),
        ivc_shapes.blanket_face,
    )
    upper_port_xz, r_inner_cut, cut_angle = upper_port_designer.execute()

    reactor.vacuum_vessel = build_vacuum_vessel(
        reactor_config.params_for("Vacuum vessel"),
        reactor_config.config_for("Vacuum vessel"),
        ivc_shapes.outer_boundary,
    )

    reactor.divertor = build_divertor(
        reactor_config.params_for("Divertor"),
        reactor_config.config_for("Divertor"),
        ivc_shapes.divertor_face,
    )

    reactor.blanket = build_blanket(
        reactor_config.params_for("Blanket"),
        reactor_config.config_for("Blanket"),
        ivc_shapes.inner_boundary,
        ivc_shapes.blanket_face,
        r_inner_cut,
        cut_angle,
    )
    reactor.vv_thermal = build_vacuum_vessel_thermal_shield(
        reactor_config.params_for("Thermal shield"),
        reactor_config.config_for("Thermal shield", "VVTS"),
        reactor.vacuum_vessel.xz_boundary(),
    )

    reactor.tf_coils = build_tf_coils(
        reactor_config.params_for("TF coils"),
        reactor_config.config_for("TF coils"),
        reactor.plasma.lcfs(),
        reactor.vv_thermal.xz_boundary(),
    )

    lower_port, lower_port_duct_xz_koz = build_lower_port(
        reactor_config.params_for("Lower Port"),
        reactor_config.config_for("Lower Port"),
        ivc_shapes.divertor_face,
        reactor.tf_coils.xz_outer_boundary(),
    )

    reactor.pf_coils = build_pf_coils(
        reactor_config.params_for("PF coils"),
        reactor_config.config_for("PF coils"),
        reference_eq,
        reactor.tf_coils.xz_outer_boundary(),
        pf_coil_keep_out_zones=[
            lower_port_duct_xz_koz,
        ],
    )

    reactor.cryostat_thermal = build_cryots(
        reactor_config.params_for("Thermal shield"),
        reactor_config.config_for("Thermal shield", "Cryostat"),
        reactor.pf_coils.xz_boundary(),
        reactor.tf_coils.xz_outer_boundary(),
    )

    reactor.coil_structures = build_coil_structures(
        reactor_config.params_for("Coil structures"),
        reactor_config.config_for("Coil structures"),
        tf_coil_xz_face=reactor.tf_coils.xz_face(),
        pf_coil_xz_wires=reactor.pf_coils.PF_xz_boundary(),
        pf_coil_keep_out_zones=[
            upper_port_xz,
            lower_port_duct_xz_koz,
        ],
    )

    reactor.cryostat = build_cryostat(
        reactor_config.params_for("Cryostat"),
        reactor_config.config_for("Cryostat"),
        reactor.cryostat_thermal.xz_boundary(),
    )

    reactor.radiation_shield = build_radiation_shield(
        reactor_config.params_for("RadiationShield"),
        reactor_config.config_for("RadiationShield"),
        reactor.cryostat.xz_boundary(),
    )

    reactor.show_cad("xz")
    reactor.show_cad(n_sectors=2)

    sspc_solver = SteadyStatePowerCycleSolver(reactor_config.global_params)
    sspc_result = sspc_solver.execute()
    sspc_solver.model.plot()
    plt.show()
