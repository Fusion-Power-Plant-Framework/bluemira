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
from bluemira.base.parameter import ParameterFrame
from bluemira.base.design import Reactor
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.EUDEMO.plasma import PlasmaBuilder
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder
from bluemira.codes import run_systems_code
from bluemira.codes.process import NAME as PROCESS


class EUDEMOReactor(Reactor):
    """
    The EU-DEMO Reactor object encapsulates the logic for performing an EU-DEMO tokamak
    design.
    """

    def run(self) -> Component:
        """
        Run the EU-DEMO reactor build process. Performs the following tasks:

        - Run the (PROCESS) systems code
        """
        component = super().run()

        self.run_systems_code()
        component.add_child(self.build_plasma())
        component.add_child(self.build_TF_coils(component))

        return component

    def run_systems_code(self, **kwargs):
        """
        Run the systems code module in the requested run mode.
        """
        bluemira_print(f"Running: {PROCESS}")

        output: ParameterFrame = run_systems_code(
            self._params,
            self._build_config,
            self._file_manager.generated_data_dirs["systems_code"],
            self._file_manager.reference_data_dirs["systems_code"],
        )
        self._params.update_kw_parameters(output.to_dict())

    def build_plasma(self, **kwargs):
        """
        Run the plasma build using the requested equilibrium problem.
        """
        name = "Plasma"

        default_eqdsk_dir = self._file_manager.reference_data_dirs["equilibria"]
        default_eqdsk_name = f"{self._params.Name.value}_eqref.json"
        default_eqdsk_path = os.path.join(default_eqdsk_dir, default_eqdsk_name)

        plasma_config = {
            "name": name,
            "plot_flag": self._build_config.get("plot_flag", False),
            "run_mode": self._build_config.get("plasma_mode", "run"),
            "eqdsk_path": self._build_config.get("eqdsk_path", default_eqdsk_path),
        }

        builder = PlasmaBuilder(self._params.to_dict(), plasma_config)
        self.register_builder(builder, name)

        return super()._build_stage(name)

    def build_TF_coils(self, component_tree: Component, **kwargs):
        """
        Run the TF coil build using the requested mode.
        """
        name = "TF Coils"
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
        tf_config = {
            "name": name,
            "param_class": self._build_config.get("TF_param_class", "PrincetonD"),
            "variables_map": self._build_config.get(
                "TF_variables_map", default_variables_map
            ),
            "problem_class": self._build_config.get(
                "TF_problem_class",
                "bluemira.builders.tf_coils::RippleConstrainedLengthOpt",
            ),
            "opt_conditions": self._build_config.get(
                "TF_opt_conditions",
                {
                    "ftol_rel": 1e-3,
                    "xtol_rel": 1e-12,
                    "xtol_abs": 1e-12,
                    "max_eval": 1000,
                },
            ),
            "opt_parameters": self._build_config.get("TF_opt_parameters", {}),
            "run_mode": self._build_config.get("TF_mode", "run"),
        }

        plasma = component_tree.get_component("Plasma")
        sep_comp: PhysicalComponent = plasma.get_component("xz").get_component("LCFS")
        sep_shape = sep_comp.shape.boundary[0]

        builder = TFCoilsBuilder(self._params.to_dict(), tf_config)
        self.register_builder(builder, name)

        return super()._build_stage(name, separatrix=sep_shape)
