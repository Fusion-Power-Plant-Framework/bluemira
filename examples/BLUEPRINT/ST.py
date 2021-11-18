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
A typical spherical tokamak fusion power reactor.
"""
import os

from BLUEPRINT.base.file import make_BP_path, get_bluemira_root
from bluemira.base.look_and_feel import plot_defaults, print_banner
from BLUEPRINT.reactor import Reactor
from BLUEPRINT.systems.config import Spherical

plot_defaults()
KEY_TO_PLOT = False
PLOTFOLDER = make_BP_path("Data/plots")
if os.path.isdir(PLOTFOLDER) is False:
    KEY_TO_PLOT = False

REACTORNAME = "ST"


config = {
    "Name": REACTORNAME,
    "P_el_net": 1.0,
    "plasma_type": "DN",
    "reactor_type": "ST",
    "op_mode": "steady-state",
    "n_CS": 0,
    "n_PF": 8,
    "n_TF": 12,
    "f_ni": 0.1,
    "fw_psi_n": 1.05,
    "div_graze_angle": 1.5,
    "div_L2D_ib": 0.3,
    "div_L2D_ob": 2.5,
    "div_psi_o": 0.5,
    "div_Ltarg": 0.5,
    "tk_ts": 0.05,
    "tk_vv_in": 0.6,
    "tk_tf_side": 0.1,
    "tk_bb_ib": 0.7,
    "tk_sol_ib": 0.15,
    "LPangle": -25,
    "TF_ripple_limit": 1.0,
}

build_config = {
    "generated_data_root": "!BP_ROOT!/generated_data/BLUEPRINT",
    "plot_flag": False,
    "process_mode": "mock",
    "process_indat": os.path.join(
        get_bluemira_root(), "examples", "data", "codes", "process", "ST_IN.DAT"
    ),
    "plasma_mode": "read",
    "plasma_filepath": os.path.join(
        get_bluemira_root(), "data", "BLUEPRINT", "eqdsk", "step_v7_format.geqdsk"
    ),
    "reconstruct_jtor": True,
    "tf_mode": "run",
    "HCD_method": "power",
    "TF_objective": "L",
    "GS_type": "JT60SA",
    "BB_segmentation": "radial",
    "FW_parameterisation": "BS",
    "VV_parameterisation": "BS",
    "div_profile_class_name": "DivertorSilhouetteFlatDomePsiBaffle",
    "lifecycle_mode": "life",
    "HCD_method": "power",
    # TF coil config
    "TF_type": "P",
    "wp_shape": "N",
    "conductivity": "SC",
    # Equilibrium modes
    "rm_shuffle": False,
    "force": False,
    "swing": False,
    "plot_flag": False,
}

build_tweaks = {
    # Equilibrium solver tweakers
    "wref": 50,  # Seed flux swing [V.s]
    "rms_limit": 0.03,  # RMS convergence criterion [m]
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
    "nrippoints": 20,  # Number of points to check edge ripple on
}


class SphericalTokamak(Reactor):
    """
    A spherical tokamak fusion power reactor class.
    """

    config: dict
    build_config: dict
    build_tweaks: dict

    default_params = Spherical().to_records()

    def __init__(self, config, build_config, build_tweaks):
        super().__init__(config, build_config, build_tweaks)

        print_banner()

        for key, val in config.items():
            if not isinstance(val, (tuple, dict)):
                config[key] = (val, "Input")
        self.params.update_kw_parameters(config)

        self.derive_inputs()

    def build(self):
        """
        Build the reactor.
        """
        self.run_systems_code()

        if self.build_config["process_mode"] != "mock":
            self.build_0D_plasma()

        if self.build_config["plasma_mode"] == "run":
            self.create_equilibrium()
        elif self.build_config["plasma_mode"] == "read":
            self.load_equilibrium(
                self.build_config.get("plasma_filepath", None),
                self.build_config.get("reconstruct_jtor", False),
            )

        self.shape_firstwall()
        self.build_cross_section()

    #         self.build_ports()
    #         self.define_in_vessel_layout()
    #         self.build_containments()
    #         #self.specify_palette((self.palette)
    #         self.specify_palette2()
    #         self.power_balance(plot=False)
    #         self.analyse_maintenance()
    #         # self.get_params()
    #         self.life_cycle(mode=self.Build_Config['lifecycle_mode'])
    # =============================================================================


if __name__ == "__main__":
    SR = SphericalTokamak(config, build_config, build_tweaks)
    SR.build()
