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
PROCESS mappings
"""
from bluemira.base.parameter import ParameterMapping

mappings = {
    "P_el_net": ParameterMapping("pnetelin", False, True),
    "P_el_net_process": ParameterMapping("pnetelmw", True, False),
    "tau_flattop": ParameterMapping("tburn", False, False),
    "n_TF": ParameterMapping("n_tf", False, True),
    "TF_ripple_limit": ParameterMapping("ripmax", False, True),
    "A": ParameterMapping("aspect", True, True),
    "R_0": ParameterMapping("rmajor", True, False),
    "B_0": ParameterMapping("bt", True, False),
    "q_95": ParameterMapping("q95", False, False),
    "kappa_95": ParameterMapping("kappa95", True, False),
    "kappa": ParameterMapping("kappa", True, False),
    "delta_95": ParameterMapping("triang95", True, False),
    "delta": ParameterMapping("triang", True, False),
    "T_e": ParameterMapping("te", False, False),
    "Z_eff": ParameterMapping("zeff", False, False),
    "V_p": ParameterMapping("vol", False, False),
    "l_i": ParameterMapping("rli", False, False),
    "I_p": ParameterMapping("plascur/1d6", True, False),
    "P_fus": ParameterMapping("powfmw", True, False),
    "P_fus_DT": ParameterMapping("pdt", True, False),
    "P_fus_DD": ParameterMapping("pdd", True, False),
    "H_star": ParameterMapping("hfact", True, False),
    "P_sep": ParameterMapping("pdivt", True, False),
    "P_rad_core": ParameterMapping("pcoreradmw", True, False),
    "P_rad_edge": ParameterMapping("pedgeradmw", True, False),
    "P_rad": ParameterMapping("pradmw", True, False),
    "P_line": ParameterMapping("plinepv*vol", True, False),
    "P_sync": ParameterMapping("psyncpv*vol", True, False),
    "P_brehms": ParameterMapping("pbrempv*vol", True, False),
    "f_bs": ParameterMapping("bootipf", True, False),
    "beta_p": ParameterMapping("betap", True, False),
    "beta": ParameterMapping("beta", True, False),
    "tau_e": ParameterMapping("taueff", True, False),
    "v_burn": ParameterMapping("vburn", True, False),
    "C_Ejima": ParameterMapping("gamma", False, True),
    "f_ni": ParameterMapping("faccd", False, False),
    "e_nbi": ParameterMapping("enbeam", False, True),
    "P_hcd_ss": ParameterMapping("pinjalw", False, True),
    "eta_nb": ParameterMapping("etanbi", False, True),
    "bb_e_mult": ParameterMapping("emult", False, True),
    "tk_bb_ib": ParameterMapping("blnkith", True, True),
    "tk_bb_ob": ParameterMapping("blnkoth", True, True),
    "tk_sh_in": ParameterMapping("shldith", True, True),
    "tk_sh_out": ParameterMapping("shldoth", False, True),
    "tk_sh_top": ParameterMapping("shldtth", False, True),
    "tk_sh_bot": ParameterMapping("shldlth", False, True),
    "tk_vv_in": ParameterMapping("d_vv_in", True, True),
    "tk_vv_out": ParameterMapping("d_vv_out", False, True),
    "tk_vv_top": ParameterMapping("d_vv_top", False, True),
    "tk_vv_bot": ParameterMapping("d_vv_bot", False, True),
    "tk_sol_ib": ParameterMapping("scrapli", True, True),
    "tk_sol_ob": ParameterMapping("scraplo", True, True),
    "tk_ts": ParameterMapping("thshield", True, True),
    "tk_cr_vv": ParameterMapping("ddwex", False, True),
    "tk_fw_in": ParameterMapping("fwith", True, False),
    "tk_fw_out": ParameterMapping("fwoth", True, False),
    "tk_tf_inboard": ParameterMapping("tfcth", True, False),
    "tk_tf_outboard": ParameterMapping("tfthko", False, False),
    "tk_tf_nose": ParameterMapping("thkcas", True, False),
    "tk_tf_wp": ParameterMapping("dr_tf_wp", True, False),
    "tk_tf_front_ib": ParameterMapping("casthi", False, True),
    "tk_tf_side": ParameterMapping("casths", False, True),
    "tf_wp_depth": ParameterMapping("wwp1", False, False),
    "tk_cs": ParameterMapping("ohcth", True, False),
    "sigma_tf_max": ParameterMapping("alstrtf", False, False),
    "h_cp_top": ParameterMapping("h_cp_top", False, False),
    "h_tf_max_in": ParameterMapping("hmax", False, False),
    "r_cp_top": ParameterMapping("r_cp_top", True, False),
    "r_cs_in": ParameterMapping("bore", True, False),
    "r_tf_in": ParameterMapping("rtfin", True, False),
    "r_tf_inboard_out": ParameterMapping("r_tf_inboard_out", False, False),
    "r_tf_in_centre": ParameterMapping("r_tf_inboard_mid", True, False),
    "r_ts_ib_in": ParameterMapping("r_ts_ib_in", True, False),
    "r_vv_ib_in": ParameterMapping("r_vv_ib_in", True, False),
    "r_fw_ib_in": ParameterMapping("r_fw_ib_in", True, False),
    "r_fw_ob_in": ParameterMapping("r_fw_ob_in", True, False),
    "r_vv_ob_in": ParameterMapping("r_vv_ob_in", True, False),
    "r_tf_out_centre": ParameterMapping("r_tf_outboard_mid", True, False),
    "g_vv_ts": ParameterMapping("gapds", True, False),
    "g_cs_tf": ParameterMapping("gapoh", True, True),
    "g_ts_tf": ParameterMapping("tftsgap", True, True),
    "g_vv_bb": ParameterMapping("vvblgap", True, True),
}
