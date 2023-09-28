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
PROCESS mappings
"""
from bluemira.codes.utilities import Model, create_mapping

IN_mappings = {
    "P_el_net": ("pnetelin", "MW"),
    "n_TF": ("n_tf", "dimensionless"),
    "TF_ripple_limit": ("ripmax", "%"),
    "C_Ejima": ("gamma", "dimensionless"),
    "e_nbi": ("enbeam", "keV"),
    "P_hcd_ss": ("pinjalw", "MW"),
    "eta_nb": ("etanbi", "dimensionless"),
    "e_mult": ("emult", "dimensionless"),
    "tk_sh_out": ("shldoth", "m"),
    "tk_sh_top": ("shldtth", "m"),
    "tk_sh_bot": ("shldlth", "m"),
    "tk_vv_out": ("d_vv_out", "m"),
    "tk_vv_top": ("d_vv_top", "m"),
    "tk_vv_bot": ("d_vv_bot", "m"),
    "tk_cr_vv": ("ddwex", "m"),
    "tk_tf_front_ib": ("casthi", "m"),
    "tk_tf_side": ("casths", "m"),
    "PsepB_qAR_max": ("psepbqarmax", "MW.T/m"),
}

OUT_mappings = {
    "P_el_net_process": ("pnetelmw", "MW"),
    "R_0": ("rmajor", "m"),
    "B_0": ("bt", "T"),
    "kappa_95": ("kappa95", "dimensionless"),
    "kappa": ("kappa", "dimensionless"),
    "delta_95": ("triang95", "dimensionless"),
    "delta": ("triang", "dimensionless"),
    "I_p": ("plascur/1d6", "MA"),
    "P_fus": ("powfmw", "MW"),
    "P_fus_DT": ("pdt", "MW"),
    "P_fus_DD": ("pdd", "MW"),
    "H_star": ("hfact", "dimensionless"),
    "P_sep": ("pdivt", "MW"),
    "P_rad_core": ("pinnerzoneradmw", "MW"),
    "P_rad_edge": ("pouterzoneradmw", "MW"),
    "P_rad": ("pradmw", "MW"),
    "P_line": ("plinepv*vol", "MW"),
    "P_sync": ("psyncpv*vol", "MW"),
    "P_brehms": ("pbrempv*vol", "MW"),
    "f_bs": ("bootipf", "dimensionless"),
    "beta_p": ("betap", "dimensionless"),
    "beta": ("beta", "dimensionless"),
    "tau_e": ("taueff", "s"),
    "v_burn": ("vburn", "V"),
    "tk_fw_in": ("fwith", "m"),
    "tk_fw_out": ("fwoth", "m"),
    "tk_tf_inboard": ("tfcth", "m"),
    "tk_tf_nose": ("thkcas", "m"),
    "tf_wp_width": ("dr_tf_wp", "m"),
    "tf_wp_depth": ("wwp1", "m"),
    "tk_tf_ins": ("tinstf", "m"),
    "tk_tf_insgap": ("tfinsgap", "m"),
    "tk_cs": ("ohcth", "m"),
    "r_cp_top": ("r_cp_top", "m"),
    "r_cs_in": ("bore", "m"),
    "r_tf_in": ("rtfin", "m"),
    "r_tf_in_centre": ("r_tf_inboard_mid", "m"),
    "r_ts_ib_in": ("r_ts_ib_in", "m"),
    "r_vv_ib_in": ("r_vv_ib_in", "m"),
    "r_fw_ib_in": ("r_fw_ib_in", "m"),
    "r_fw_ob_in": ("r_fw_ob_in", "m"),
    "r_vv_ob_in": ("r_vv_ob_in", "m"),
    "r_tf_out_centre": ("r_tf_outboard_mid", "m"),
    "g_vv_ts": ("gapds", "m"),
    "TF_res_bus": ("tfbusres", "m"),
    "TF_res_tot": ("ztot", "ohm"),
    "TF_E_stored": ("estotftgj", "GJ"),
    "TF_respc_ob": ("tflegres", "ohm"),
    "TF_currpt_ob": ("cpttf", "A"),
    "P_bd_in": ("pinjmw", "MW"),
    "condrad_cryo_heat": ("qss/1.0d6", "MW"),
}

IO_mappings = {
    "A": ("aspect", "dimensionless"),
    "tk_bb_ib": ("blnkith", "m"),
    "tk_bb_ob": ("blnkoth", "m"),
    "tk_sh_in": ("shldith", "m"),
    "tk_vv_in": ("d_vv_in", "m"),
    "tk_sol_ib": ("scrapli", "m"),
    "tk_sol_ob": ("scraplo", "m"),
    "tk_ts": ("thshield_ob", "m"),
    "tk_ts": ("thshield_vb", "m"),
    "tk_ts": ("thshield_ib", "m"),
    "g_cs_tf": ("gapoh", "m"),
    "g_ts_tf": ("tftsgap", "m"),
    "g_vv_bb": ("vvblgap", "m"),
}

NONE_mappings = {
    "tau_flattop": ("tburn", "s"),
    "B_tf_peak": ("bmaxtfrp", "T"),
    "q_95": ("q95", "dimensionless"),
    "T_e": ("te", "keV"),
    "Z_eff": ("zeff", "amu"),
    "V_p": ("vol", "m^3"),
    "l_i": ("rli", "dimensionless"),
    "f_ni": ("faccd", "dimensionless"),
    "tk_tf_outboard": ("tfthko", "m"),
    "sigma_tf_case_max": ("sig_tf_case_max", "Pa"),
    "sigma_tf_wp_max": ("sig_tf_wp_max", "Pa"),
    "h_cp_top": ("h_cp_top", "m"),
    "h_tf_max_in": ("hmax", "m"),
    "r_tf_inboard_out": ("r_tf_inboard_out", "m"),
}

mappings = create_mapping(IN_mappings, OUT_mappings, IO_mappings, NONE_mappings)
