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
from bluemira.codes.utilities import Model, create_mapping


class CurrentDriveEfficiencyModel(Model):
    """
    Switch for current drive efficiency model:

    1 - Fenstermacher Lower Hybrid
    2 - Ion Cyclotron current drive
    3 - Fenstermacher ECH
    4 - Ehst Lower Hybrid
    5 - ITER Neutral Beam
    6 - new Culham Lower Hybrid model
    7 - new Culham ECCD model
    8 - new Culham Neutral Beam model
    10 - ECRH user input gamma
    11 - ECRH "HARE" model (E. Poli, Physics of Plasmas 2019)
    12 - EBW user scaling input. Scaling (S. Freethy)

    PROCESS variable name: "iefrf"
    """

    FENSTER_LH = 1
    ICYCCD = 2
    FENSTER_ECH = 3
    EHST_LH = 4
    ITER_NB = 5
    CUL_LH = 6
    CUL_ECCD = 7
    CUL_NB = 8
    ECRH_UI_GAM = 10
    ECRH_HARE = 11
    EBW_UI = 12


class TFCoilConductorTechnology(Model):
    """
    Switch for TF coil conductor model:

    0 - copper
    1 - superconductor
    2 - Cryogenic aluminium

    PROCESS variable name: "i_tf_sup"
    """

    COPPER = 0
    SC = 1
    CYRO_AL = 2


IN_mappings = {
    "P_el_net": ("pnetelin", "MW"),
    "n_TF": ("n_tf", "dimensionless"),
    "TF_ripple_limit": ("ripmax", "%"),
    "C_Ejima": ("gamma", "dimensionless"),
    "e_nbi": ("enbeam", "keV"),
    "P_hcd_ss": ("pinjalw", "MW"),
    "eta_nb": ("etanbi", "dimensionless"),
    "bb_e_mult": ("emult", "dimensionless"),
    "tk_sh_out": ("shldoth", "m"),
    "tk_sh_top": ("shldtth", "m"),
    "tk_sh_bot": ("shldlth", "m"),
    "tk_vv_out": ("d_vv_out", "m"),
    "tk_vv_top": ("d_vv_top", "m"),
    "tk_vv_bot": ("d_vv_bot", "m"),
    "tk_cr_vv": ("ddwex", "m"),
    "tk_tf_front_ib": ("casthi", "m"),
    "tk_tf_side": ("casths", "m"),
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
    "e_mult": ParameterMapping("emult", False, True),
    "bb_t_inlet": ParameterMapping("t_in_bb", False, True),
    "bb_t_outlet": ParameterMapping("t_out_bb", False, True),
    # TODO: What about water?
    "bb_p_outlet": ParameterMapping("p_he", False, True),
    # TODO: PROCESS uses dP as an input, not outlet-inlet
    "bb_pump_eta_el": ParameterMapping("etahtp", False, True),
    "bb_pump_eta_isen": ParameterMapping("etaiso", False, True),
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
    "P_rad_core": ("pcoreradmw", "MW"),
    "P_rad_edge": ("pedgeradmw", "MW"),
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
    "condrad_cryo_heat": ("qss/1.0D6", "MW"),
}

IO_mappings = {
    "A": ("aspect", "dimensionless"),
    "tk_bb_ib": ("blnkith", "m"),
    "tk_bb_ob": ("blnkoth", "m"),
    "tk_sh_in": ("shldith", "m"),
    "tk_vv_in": ("d_vv_in", "m"),
    "tk_sol_ib": ("scrapli", "m"),
    "tk_sol_ob": ("scraplo", "m"),
    "tk_ts": ("thshield", "m"),
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
