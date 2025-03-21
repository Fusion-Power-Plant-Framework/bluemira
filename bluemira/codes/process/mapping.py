# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS mappings
"""

from bluemira.codes.utilities import create_mapping

IN_mappings = {
    "n_TF": ("n_tf_coils", "dimensionless"),
    "TF_ripple_limit": ("ripmax", "%"),
    "C_Ejima": ("ejima_coeff", "dimensionless"),
    "e_nbi": ("beam_energy", "keV"),
    "P_hcd_ss": ("pinjalw", "MW"),
    "eta_nb": ("etanbi", "dimensionless"),
    "e_mult": ("emult", "dimensionless"),
    "tk_cr_vv": ("dr_cryostat", "m"),
    "tk_tf_front_ib": ("casthi", "m"),
    "tk_tf_side": ("casths", "m"),
    "PsepB_qAR_max": ("psepbqarmax", "MW.T/m"),
    "q_0": ("q0", "dimensionless"),
    "m_s_limit": ("m_s_limit", "dimensionless"),
    "delta": ("triang", "dimensionless"),
    "sigma_tf_case_max": ("sig_tf_case_max", "Pa"),
    "sigma_tf_wp_max": ("sig_tf_wp_max", "Pa"),
    "sigma_cs_wp_max": ("alstroh", "Pa"),
    "H_star": ("hfact", "dimensionless"),
    "bb_pump_eta_el": ("etahtp", "dimensionless"),
    "bb_pump_eta_isen": ("etaiso", "dimensionless"),
    "bb_t_inlet": ("temp_blkt_coolant_in", "K"),
    "bb_t_outlet": ("temp_blkt_coolant_out", "K"),
    "eta_ecrh": ("etaech", "dimensionless"),
    "gamma_ecrh": ("gamma_ecrh", "1e20 A/W/m^2"),
}

OUT_mappings = {
    "R_0": ("rmajor", "m"),
    "B_0": ("bt", "T"),
    "kappa_95": ("kappa95", "dimensionless"),
    "kappa": ("kappa", "dimensionless"),
    "delta_95": ("triang95", "dimensionless"),
    "delta": ("triang", "dimensionless"),
    "I_p": ("plasma_current_ma", "MA"),
    "P_fus": ("fusion_power", "MW"),
    "P_fus_DT": ("dt_power", "MW"),
    "P_fus_DD": ("dd_power", "MW"),
    "H_star": ("hfact", "dimensionless"),
    "P_sep": ("pdivt", "MW"),
    "P_rad_core": ("p_plasma_inner_rad_mw", "MW"),
    "P_rad_edge": ("p_plasma_outer_rad_mw", "MW"),
    "P_rad": ("p_plasma_rad_mw", "MW"),
    "P_line": ("plinepv*vol", "MW"),
    "P_sync": ("pden_plasma_sync_mw*vol", "MW"),
    "P_brehms": ("pbrempv*plasma_volume", "MW"),
    "f_bs": ("bootstrap_current_fraction", "dimensionless"),
    "beta_p": ("beta_poloidal", "dimensionless"),
    "beta": ("beta", "dimensionless"),
    "tau_e": ("t_energy_confinement", "s"),
    "v_burn": ("v_plasma_loop_burn", "V"),
    "tk_fw_in": ("dr_fw_inboard", "m"),
    "tk_fw_out": ("dr_fw_outboard", "m"),
    "tk_tf_inboard": ("dr_tf_inboard", "m"),
    "tk_tf_nose": ("thkcas", "m"),
    "tf_wp_width": ("dr_tf_wp", "m"),
    "tf_wp_depth": ("wwp1", "m"),
    "tk_tf_ins": ("tinstf", "m"),
    "tk_tf_insgap": ("tfinsgap", "m"),
    "tk_cs": ("dr_cs", "m"),
    "r_cp_top": ("r_cp_top", "m"),
    "r_cs_in": ("dr_bore", "m"),
    "r_tf_in": ("rtfin", "m"),
    "r_tf_in_centre": ("r_tf_inboard_mid", "m"),
    "r_ts_ib_in": ("r_ts_ib_in", "m"),
    "r_vv_ib_in": ("r_vv_ib_in", "m"),
    "r_fw_ib_in": ("r_fw_ib_in", "m"),
    "r_fw_ob_in": ("r_fw_ob_in", "m"),
    "r_vv_ob_in": ("r_vv_ob_in", "m"),
    "r_tf_out_centre": ("r_tf_outboard_mid", "m"),
    "g_vv_ts": ("dr_shld_vv_gap_inboard", "m"),
    "TF_res_bus": ("tfbusres", "m"),
    "TF_res_tot": ("res_tf_system_total", "ohm"),
    "TF_E_stored": ("estotftgj", "GJ"),
    "TF_respc_ob": ("res_tf_leg", "ohm"),
    "TF_currpt_ob": ("cpttf", "A"),
    "P_bd_in": ("pinjmw", "MW"),
    "condrad_cryo_heat": ("qss/1.0d6", "MW"),
}

IO_mappings = {
    "A": ("aspect", "dimensionless"),
    "tau_flattop": (("t_burn_min", "t_burn"), "s"),
    "P_el_net": (("pnetelin", "pnetelmw"), "MW"),
    "tk_bb_ib": ("dr_blkt_inboard", "m"),
    "tk_bb_ob": ("dr_blkt_outboard", "m"),
    "tk_vv_in": ("dr_vv_inboard", "m"),
    "tk_sol_ib": ("dr_fw_plasma_gap_inboard", "m"),
    "tk_sol_ob": ("dr_fw_plasma_gap_outboard", "m"),
    "g_cs_tf": ("dr_cs_tf_gap", "m"),
    "g_ts_tf": ("dr_tf_shld_gap", "m"),
    "g_vv_bb": ("dr_shld_blkt_gap", "m"),
}

NONE_mappings = {
    "B_tf_peak": ("bmaxtfrp", "T"),
    "T_e": ("te", "keV"),
    "Z_eff": ("zeff", "amu"),
    "V_p": ("plasma_volume", "m^3"),
    "l_i": ("ind_plasma_internal_norm", "dimensionless"),
    "f_ni": ("faccd", "dimensionless"),
    "tk_tf_outboard": ("dr_tf_outboard", "m"),
    "h_cp_top": ("h_cp_top", "m"),
    "h_tf_max_in": ("hmax", "m"),
    "r_tf_inboard_out": ("r_tf_inboard_out", "m"),
    # The following mappings are not 1:1
    "tk_sh_in": ("dr_shld_inboard", "m"),
    "tk_sh_out": ("dr_shld_outboard", "m"),
    "tk_sh_top": ("dz_shld_upper", "m"),
    "tk_sh_bot": ("dz_shld_lower", "m"),
    "tk_vv_out": ("dr_vv_outboard", "m"),
    "tk_vv_top": ("dz_vv_upper", "m"),
    "tk_vv_bot": ("dz_vv_lower", "m"),
    # Thermal shield thickness is a constant for us
    "tk_ts": ("dr_shld_thermal_inboard", "m"),
    # "tk_ts": ("dr_shld_thermal_outboard", "m"),
    # "tk_ts": ("dz_shld_thermal", "m"),
    "q_95": ("q95", "dimensionless"),
}

mappings = create_mapping(IN_mappings, OUT_mappings, IO_mappings, NONE_mappings)
