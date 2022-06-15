# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Perform the EU-DEMO design.
"""

import os

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.design import Reactor
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.cryostat import CryostatBuilder
from bluemira.builders.EUDEMO.blanket import BlanketBuilder
from bluemira.builders.EUDEMO.divertor import DivertorBuilder
from bluemira.builders.EUDEMO.ivc import InVesselComponentBuilder
from bluemira.builders.EUDEMO.ivc.ivc import build_ivc_xz_shapes
from bluemira.builders.EUDEMO.pf_coils import PFCoilsBuilder
from bluemira.builders.EUDEMO.plasma import PlasmaBuilder
from bluemira.builders.EUDEMO.power_cycle import SteadyStatePowerCycleSolver
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder
from bluemira.builders.EUDEMO.vacuum_vessel import VacuumVesselBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import (
    CryostatThermalShieldBuilder,
    VacuumVesselThermalShieldBuilder,
)
from bluemira.codes import systems_code_solver
from bluemira.codes.process import NAME as PROCESS


class EUDEMOReactor(Reactor):
    """
    The EU-DEMO Reactor object encapsulates the logic for performing an EU-DEMO tokamak
    design.
    """

    PLASMA = "Plasma"
    DIVERTOR = "Divertor"
    BLANKET = "Breeding Blanket"
    TF_COILS = "TF Coils"
    PF_COILS = "PF Coils"
    IVC = "In-Vessel Components"
    VACUUM_VESSEL = "Vacuum Vessel"
    THERMAL_SHIELD = "Thermal Shield"
    VVTS = "Vacuum Vessel Thermal Shield"
    CTS = "Cryostat Thermal Shield"
    CRYOSTAT = "Cryostat"
    RADIATION_SHIELD = "Radiation Shield"
    POWER_CYCLE = "Power Cycle"

    def run(self) -> Component:
        """
        Run the EU-DEMO reactor build process.
        """
        component = super().run()

        self.run_systems_code()
        component.add_child(self.build_plasma())
        (
            blanket_face,
            divertor_face,
            ivc_boundary,
        ) = self.build_in_vessel_component_shapes(component)
        component.add_child(self.build_vacuum_vessel(component, ivc_boundary))
        component.add_child(self.build_divertor(component, divertor_face))
        component.add_child(self.build_blanket(component, blanket_face))
        thermal_shield = Component(self.THERMAL_SHIELD, parent=component)
        thermal_shield.add_child(self.build_VV_thermal_shield(component))
        component.add_child(self.build_TF_coils(component))
        component.add_child(self.build_PF_coils(component))
        thermal_shield.add_child(self.build_cryo_thermal_shield(component))
        component.add_child(self.build_cryostat(component))
        component.add_child(self.build_radiation_shield(component))
        self.run_power_cycle()
        bluemira_print("Reactor Design Complete!")

        return component

    @Reactor.design_stage(PROCESS)
    def run_systems_code(self):
        """
        Run the systems code module in the requested run mode.
        """
        # Use the generated/reference data dirs as read/run defaults,
        # but these can be overridden for the specific systems code.
        default_config = {
            "read_dir": self._file_manager.reference_data_dirs["systems_code"],
            "run_dir": self._file_manager.generated_data_dirs["systems_code"],
        }
        config = self._process_design_stage_config(default_config)
        run_mode = config.pop("runmode")
        solver = systems_code_solver(self._params, config)

        self.register_solver(solver)
        solver.execute(solver.run_mode_cls.from_string(run_mode))
        self._params.update_kw_parameters(solver.params.to_dict())

    @Reactor.design_stage(PLASMA)
    def build_plasma(self):
        """
        Run the plasma build using the requested equilibrium problem.
        """
        default_eqdsk_dir = self._file_manager.reference_data_dirs["equilibria"]
        default_eqdsk_name = f"{self._params.Name.value}_eqref.json"
        default_eqdsk_path = os.path.join(default_eqdsk_dir, default_eqdsk_name)

        default_config = {"runmode": "run", "eqdsk_path": default_eqdsk_path}

        config = self._process_design_stage_config(default_config)

        builder = PlasmaBuilder(self._params.to_dict(), config)
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(TF_COILS)
    def build_TF_coils(self, component_tree: Component):
        """
        Run the TF Coils build using the requested mode.
        """
        default_variables_map = {
            "x1": {
                "value": "r_tf_in_centre",
                "fixed": True,
            },
            "x2": {
                "value": "r_tf_out_centre",
                "lower_bound": 14.0,
            },
            "dz": {
                "value": 0.0,
                "fixed": True,
            },
        }

        default_config = {
            "param_class": "PrincetonD",
            "variables_map": default_variables_map,
            "geom_path": None,
            "runmode": "run",
            "problem_class": "bluemira.builders.tf_coils::RippleConstrainedLengthGOP",
            "problem_settings": {},
            "opt_conditions": {
                "ftol_rel": 1e-3,
                "xtol_rel": 1e-6,
                "xtol_abs": 1e-6,
                "max_eval": 1000,
            },
            "opt_parameters": {},
        }

        config = self._process_design_stage_config(default_config)

        if config["geom_path"] is None:
            if config["runmode"] == "run":
                default_geom_dir = self._file_manager.generated_data_dirs["geometry"]
            else:
                default_geom_dir = self._file_manager.reference_data_dirs["geometry"]
            geom_name = f"tf_coils_{config['param_class']}_{self._params['n_TF']}.json"
            geom_path = os.path.join(default_geom_dir, geom_name)

            config["geom_path"] = geom_path

        plasma = component_tree.get_component(EUDEMOReactor.PLASMA)
        sep_comp: PhysicalComponent = plasma.get_component("xz").get_component("LCFS")
        sep_shape = sep_comp.shape.boundary[0]
        thermal_shield = component_tree.get_component(self.THERMAL_SHIELD)

        vvts_xz = (
            thermal_shield.get_component(self.VVTS)
            .get_component("xz")
            .get_component("VVTS")
            .shape.boundary[0]
        )

        builder = TFCoilsBuilder(
            self._params.to_dict(), config, separatrix=sep_shape, keep_out_zone=vvts_xz
        )
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(PF_COILS)
    def build_PF_coils(self, component_tree: Component):
        """
        Run the PF Coils build using the requested mode.
        """
        default_eqdsk_dir = self._file_manager.reference_data_dirs["equilibria"]
        default_eqdsk_name = f"{self._params.Name.value}_eqref.json"
        default_eqdsk_path = os.path.join(default_eqdsk_dir, default_eqdsk_name)

        default_config = {
            "runmode": "read",
            "eqdsk_path": default_eqdsk_path,
        }

        config = self._process_design_stage_config(default_config)

        builder = PFCoilsBuilder(self._params.to_dict(), config)
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(VVTS)
    def build_VV_thermal_shield(self, component_tree: Component):
        """
        Run the vacuum vessel thermal shield build.
        """
        vessel = component_tree.get_component(self.VACUUM_VESSEL).get_component("xz")

        vv_koz = vessel.get_component("Body").shape.boundary[0]

        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = VacuumVesselThermalShieldBuilder(
            self._params.to_dict(), config, vv_koz=vv_koz
        )

        self.register_builder(builder)
        return super()._build_stage()

    @Reactor.design_stage(CTS)
    def build_cryo_thermal_shield(self, component_tree: Component):
        """
        Run the cryostat thermal shield build.
        """
        # Prepare inputs
        pf_coils = component_tree.get_component(self.PF_COILS).get_component("xz")
        pf_kozs = [
            coil.get_component("casing").shape.boundary[0] for coil in pf_coils.children
        ]
        tf_coils = component_tree.get_component(self.TF_COILS).get_component("xz")
        tf_koz = (
            tf_coils.get_component("Casing").get_component("outer").shape.boundary[0]
        )

        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = CryostatThermalShieldBuilder(
            self._params.to_dict(),
            config,
            pf_coils_xz_kozs=pf_kozs,
            tf_xz_koz=tf_koz,
        )
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(IVC)
    def build_in_vessel_component_shapes(self, component_tree: Component):
        """
        Run the in-vessel component builder.
        """
        default_variables_map = {
            "x1": {"value": "r_fw_ib_in", "fixed": True},  # ib radius
            "x2": {"value": "r_fw_ob_in"},  # ob radius
        }

        default_config = {
            "algorithm_name": "SLSQP",
            "name": self.IVC,
            "opt_conditions": {
                "ftol_rel": 1e-6,
                "max_eval": 1000,
                "xtol_abs": 1e-8,
                "xtol_rel": 1e-8,
            },
            "param_class": "bluemira.builders.EUDEMO.ivc::WallPolySpline",
            "problem_class": "bluemira.geometry.optimisation::MinimiseLengthGOP",
            "runmode": "run",
            "variables_map": default_variables_map,
        }

        config = self._process_design_stage_config(default_config)

        plasma = component_tree.get_component(self.PLASMA)
        builder = InVesselComponentBuilder(
            self._params.to_dict(), build_config=config, equilibrium=plasma.equilibrium
        )
        self.register_builder(builder)

        component = super()._build_stage()

        return build_ivc_xz_shapes(component, self._params.c_rm.value)

    @Reactor.design_stage(DIVERTOR)
    def build_divertor(self, component_tree: Component, divertor_face):
        """
        Run the divertor build.
        """
        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = DivertorBuilder(
            self._params.to_dict(), config, divertor_silhouette=divertor_face
        )
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(BLANKET)
    def build_blanket(self, component_tree: Component, blanket_face):
        """
        Run the breeding blanket build.
        """
        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = BlanketBuilder(
            self._params.to_dict(), config, blanket_silhouette=blanket_face
        )
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(VACUUM_VESSEL)
    def build_vacuum_vessel(self, component_tree: Component, ivc_boundary):
        """
        Run the reactor vacuum vessel build.
        """
        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = VacuumVesselBuilder(
            self._params.to_dict(), config, ivc_koz=ivc_boundary
        )
        self.register_builder(builder)
        return super()._build_stage()

    @Reactor.design_stage(CRYOSTAT)
    def build_cryostat(self, component_tree: Component):
        """
        Run the cryostat vacuum vessel build.
        """
        thermal_shield = component_tree.get_component(self.THERMAL_SHIELD)
        cts = (
            thermal_shield.get_component(self.CTS)
            .get_component("xz")
            .get_component("Cryostat TS")
            .shape.boundary[0]
        )

        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = CryostatBuilder(self._params.to_dict(), config, cts_xz=cts)
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(RADIATION_SHIELD)
    def build_radiation_shield(self, component_tree: Component):
        """
        Run the radiation shield build.
        """
        cryostat = component_tree.get_component(self.CRYOSTAT).get_component("xz")
        cryo_vv_xz = cryostat.get_component("Cryostat VV").shape

        default_config = {}
        config = self._process_design_stage_config(default_config)

        builder = RadiationShieldBuilder(
            self._params.to_dict(),
            config,
            cryo_vv_xz=cryo_vv_xz,
        )
        self.register_builder(builder)

        return super()._build_stage()

    @Reactor.design_stage(POWER_CYCLE)
    def run_power_cycle(self):
        """
        Run the power balance for the reactor.
        """
        solver = SteadyStatePowerCycleSolver(self._params)
        self.register_solver(solver)
        result = solver.execute()
        self._params.update_kw_parameters(result)
