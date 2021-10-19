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
A typical EU-DEMO-like single null tokamak fusion power reactor.
"""
import matplotlib.pyplot as plt
from BLUEPRINT.reactor import Reactor
from BLUEPRINT.systems.config import SingleNull
from BLUEPRINT.base.file import make_BP_path
from bluemira.base.look_and_feel import plot_defaults


plot_defaults()
KEY_TO_PLOT = False

REACTORNAME = "EU-DEMO"
config = {
    "Name": REACTORNAME,
    "P_el_net": 500,
    # TODO: Slightly shorter than 2 hr flat-top..
    "tau_flattop": 6900,
    "plasma_type": "SN",
    "reactor_type": "Normal",
    "blanket_type": "HCPB",
    "CS_material": "Nb3Sn",
    "PF_material": "NbTi",
    "A": 3.1,
    "n_CS": 5,
    "n_PF": 6,
    "n_TF": 18,
    "P_hcd_ss": 50,
    "f_ni": 0.1,
    "fw_psi_n": 1.06,
    "l_i": 0.8,
    # EU-DEMO has no shield component, so set thickness to 0.3 and subtract from VV.
    "tk_sh_in": 0.3,
    "tk_sh_out": 0.3,
    "tk_sh_top": 0.3,
    "tk_sh_bot": 0.3,
    "tk_vv_in": 0.3,
    "tk_vv_out": 0.8,
    "tk_vv_top": 0.3,
    "tk_vv_bot": 0.3,
    "tk_tf_side": 0.1,
    "tk_tf_front_ib": 0.05,
    "tk_bb_ib": 0.755,
    "tk_bb_ob": 1.275,
    "tk_sol_ib": 0.225,
    "tk_sol_ob": 0.225,
    "tk_ts": 0.05,
    "g_cs_tf": 0.05,
    "g_tf_pf": 0.05,
    "g_vv_bb": 0.02,
    "C_Ejima": 0.3,
    "e_nbi": 1000,
    "eta_nb": 0.4,
    "LPangle": -15,
    "bb_e_mult": 1.35,
    "w_g_support": 1.5,
}

build_config = {
    "generated_data_root": "!BP_ROOT!/generated_data/BLUEPRINT",
    "plot_flag": False,
    "process_mode": "mock",
    "plasma_mode": "run",
    "tf_mode": "run",
    # TF coil config
    "TF_type": "S",
    "wp_shape": "N",
    "conductivity": "SC",
    "TF_objective": "L",
    "GS_type": "ITER",
    # FW and VV config
    "VV_parameterisation": "S",
    "FW_parameterisation": "S",
    "BB_segmentation": "radial",
    "HCD_method": "power",
}


build_tweaks = {
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
    "nrippoints": 20,  # Number of points to check edge ripple on
}


class SingleNullReactor(Reactor):
    """
    A single-null fusion power reactor class.
    """

    config: dict
    build_config: dict
    build_tweaks: dict
    default_params = SingleNull().to_records()


if __name__ == "__main__":
    plt.close("all")

    LOAD = False

    if LOAD:
        filename = (
            make_BP_path(f"reactors/{REACTORNAME}", subfolder="data/BLUEPRINT")
            + "/"
            + REACTORNAME
            + ".pkl"
        )
        R = SingleNullReactor.load(
            filename, generated_data_root="generated_data/BLUEPRINT"
        )
        R.TF.cross_section()
    else:
        R = SingleNullReactor(config, build_config, build_tweaks)
        R.build()
        # R.run_systems_code()
    plot_defaults()
