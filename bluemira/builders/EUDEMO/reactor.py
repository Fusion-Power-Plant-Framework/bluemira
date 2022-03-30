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
from bluemira.builders.EUDEMO.first_wall import FirstWallBuilder
from bluemira.builders.EUDEMO.pf_coils import PFCoilsBuilder
from bluemira.builders.EUDEMO.plasma import PlasmaBuilder
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import ThermalShieldBuilder
from bluemira.codes import systems_code_solver
from bluemira.codes.process import NAME as PROCESS


class EUDEMOReactor(Reactor):
    """
    The EU-DEMO Reactor object encapsulates the logic for performing an EU-DEMO tokamak
    design.
    """

    PLASMA = "Plasma"
    TF_COILS = "TF Coils"
    PF_COILS = "PF Coils"
    FIRST_WALL = "First Wall"
    THERMAL_SHIELD = "Thermal Shield"
    CRYOSTAT = "Cryostat"
    RADIATION_SHIELD = "Radiation Shield"

    def run(self) -> Component:
        """
        Run the EU-DEMO reactor build process.
        """
        component = super().run()

        self.run_systems_code()
        component.add_child(self.build_plasma())
        component.add_child(self.build_TF_coils(component))
        component.add_child(self.build_PF_coils(component))
        component.add_child(self.build_first_wall(component))
        component.add_child(self.build_thermal_shield(component))
        component.add_child(self.build_cryostat(component))
        component.add_child(self.build_radiation_shield(component))

        bluemira_print("Reactor Design Complete!")

        return component

    def run_systems_code(self):
        """
        Run the systems code module in the requested run mode.
        """
        name = PROCESS

        bluemira_print(f"Starting design stage: {name}")

        default_config = {"process_mode": "run"}

        config = self._process_design_stage_config(name, default_config)

        # TODO: This is needed to support backward compatibility with the old
        # process_mode configuration at the top level. Can be removed when the
        # run_systems_code interface is updated to have a more general runmode value.
        config["process_mode"] = config.pop("runmode")

        solver = systems_code_solver(
            self._params,
            config,
            self._file_manager.generated_data_dirs["systems_code"],
            self._file_manager.reference_data_dirs["systems_code"],
        )

        self.register_solver(solver, name)
        solver.run()
        self._params.update_kw_parameters(solver.params.to_dict())

        bluemira_print(f"Completed design stage: {name}")

    def build_plasma(self):
        """
        Run the plasma build using the requested equilibrium problem.
        """
        name = EUDEMOReactor.PLASMA

        bluemira_print(f"Starting design stage: {name}")

        default_eqdsk_dir = self._file_manager.reference_data_dirs["equilibria"]
        default_eqdsk_name = f"{self._params.Name.value}_eqref.json"
        default_eqdsk_path = os.path.join(default_eqdsk_dir, default_eqdsk_name)

        default_config = {"runmode": "run", "eqdsk_path": default_eqdsk_path}

        config = self._process_design_stage_config(name, default_config)

        builder = PlasmaBuilder(self._params.to_dict(), config)
        self.register_builder(builder, name)

        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")

        return component

    def build_TF_coils(self, component_tree: Component):
        """
        Run the TF Coils build using the requested mode.
        """
        name = EUDEMOReactor.TF_COILS

        bluemira_print(f"Starting design stage: {name}")

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
            "problem_class": "bluemira.builders.tf_coils::RippleConstrainedLengthOpt",
            "problem_settings": {},
            "opt_conditions": {
                "ftol_rel": 1e-3,
                "xtol_rel": 1e-6,
                "xtol_abs": 1e-6,
                "max_eval": 1000,
            },
            "opt_parameters": {},
        }

        config = self._process_design_stage_config(name, default_config)

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

        builder = TFCoilsBuilder(self._params.to_dict(), config, separatrix=sep_shape)
        self.register_builder(builder, name)

        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")

        return component

    def build_PF_coils(self, component_tree: Component):
        """
        Run the PF Coils build using the requested mode.
        """
        name = EUDEMOReactor.PF_COILS

        default_eqdsk_dir = self._file_manager.reference_data_dirs["equilibria"]
        default_eqdsk_name = f"{self._params.Name.value}_eqref.json"
        default_eqdsk_path = os.path.join(default_eqdsk_dir, default_eqdsk_name)

        default_config = {
            "runmode": "read",
            "eqdsk_path": default_eqdsk_path,
        }

        config = self._process_design_stage_config(name, default_config)

        builder = PFCoilsBuilder(self._params.to_dict(), config)
        self.register_builder(builder, name)

        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")
        return component

    def build_thermal_shield(self, component_tree: Component):
        """
        Run the thermal shield build.
        """
        name = self.THERMAL_SHIELD

        bluemira_print(f"Starting design stage: {name}")

        # Prepare inputs
        pf_coils = component_tree.get_component("PF Coils").get_component("xz")
        pf_kozs = [
            coil.get_component("casing").shape.boundary[0] for coil in pf_coils.children
        ]
        tf_coils = component_tree.get_component("TF Coils").get_component("xz")
        tf_koz = (
            tf_coils.get_component("Casing").get_component("outer").shape.boundary[0]
        )

        default_config = {}
        config = self._process_design_stage_config(name, default_config)

        builder = ThermalShieldBuilder(
            self._params.to_dict(),
            config,
            pf_coils_xz_kozs=pf_kozs,
            tf_xz_koz=tf_koz,
            vv_xz_koz=None,
        )
        self.register_builder(builder, name)
        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")

        return component

    def build_first_wall(self, component_tree: Component, **kwargs):
        """
        Run the first wall builder.
        """
        name = EUDEMOReactor.FIRST_WALL

        bluemira_print(f"Starting design stage: {name}")

        default_variables_map = {
            "x1": {"value": "r_fw_ib_in"},  # ib radius
            "x2": {"value": "r_fw_ob_in"},  # ob radius
        }

        default_config = {
            "algorithm_name": "SLSQP",
            "name": self.FIRST_WALL,
            "opt_conditions": {
                "ftol_rel": 1e-6,
                "max_eval": 100,
                "xtol_abs": 1e-8,
                "xtol_rel": 1e-8,
            },
            "param_class": "bluemira.builders.EUDEMO.first_wall::WallPrincetonD",
            "problem_class": "bluemira.geometry.optimisation::MinimiseLength",
            "runmode": "run",
            "variables_map": default_variables_map,
        }

        config = self._process_design_stage_config(name, default_config)

        plasma = component_tree.get_component(self.PLASMA)
        builder = FirstWallBuilder(
            self._params.to_dict(), build_config=config, equilibrium=plasma.equilibrium
        )
        self.register_builder(builder, name)

        component = super()._build_stage(name)
        bluemira_print(f"Completed design stage: {name}")
        return component

    def build_cryostat(self, component_tree: Component):
        """
        Run the cryostat vacuum vessel build.
        """
        name = EUDEMOReactor.CRYOSTAT

        bluemira_print(f"Starting design stage: {name}")

        thermal_shield = component_tree.get_component(
            EUDEMOReactor.THERMAL_SHIELD
        ).get_component("xz")
        cts = thermal_shield.get_component("Cryostat TS").shape.boundary[0]

        default_config = {}
        config = self._process_design_stage_config(name, default_config)

        builder = CryostatBuilder(self._params.to_dict(), config, cts_xz=cts)
        self.register_builder(builder, name)
        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")

        return component

    def build_radiation_shield(self, component_tree: Component):
        """
        Run the radiation shield build.
        """
        name = EUDEMOReactor.RADIATION_SHIELD

        bluemira_print(f"Starting design stage: {name}")

        cryostat = component_tree.get_component(EUDEMOReactor.CRYOSTAT).get_component(
            "xz"
        )
        cryo_vv_xz = cryostat.get_component("Cryostat VV").shape

        default_config = {}
        config = self._process_design_stage_config(name, default_config)

        builder = RadiationShieldBuilder(
            self._params.to_dict(),
            config,
            cryo_vv_xz=cryo_vv_xz,
        )
        self.register_builder(builder, name)
        component = super()._build_stage(name)

        bluemira_print(f"Completed design stage: {name}")

        return component
