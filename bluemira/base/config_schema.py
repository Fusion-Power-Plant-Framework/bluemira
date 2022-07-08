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
Defines the master configuration schema, which controls the parameterisation of a given
analysis.
"""

from bluemira.base.parameter import Parameter


class ConfigurationSchema:
    """
    The master configuration schema.
    """

    Name: Parameter
    op_mode: Parameter
    plasma_type: Parameter
    reactor_type: Parameter

    A: Parameter
    Av: Parameter
    B_0: Parameter
    blanket_type: Parameter
    n_CS: Parameter
    n_PF: Parameter
    n_TF: Parameter
    P_el_net: Parameter
    P_el_net_process: Parameter
    R_0: Parameter
    tau_flattop: Parameter
    TF_ripple_limit: Parameter
    z_0: Parameter

    # Plasma
    beta: Parameter
    beta_N: Parameter  # noqa :N815 - mixed case to match PROCESS
    beta_p: Parameter
    C_Ejima: Parameter
    delta: Parameter
    delta_95: Parameter
    f_bs: Parameter
    f_DD_fus: Parameter
    H_star: Parameter
    I_p: Parameter
    kappa: Parameter
    kappa_95: Parameter
    l_i: Parameter
    m_s_limit: Parameter
    P_brehms: Parameter
    P_fus: Parameter
    P_fus_DD: Parameter
    P_fus_DT: Parameter
    P_LH: Parameter
    P_line: Parameter
    P_ohm: Parameter
    P_rad: Parameter
    P_rad_core: Parameter
    P_rad_edge: Parameter
    P_sep: Parameter
    P_sync: Parameter
    q_95: Parameter
    res_plasma: Parameter
    shaf_shift: Parameter
    T_e: Parameter
    T_e_ped: Parameter
    tau_e: Parameter
    v_burn: Parameter
    V_p: Parameter
    Z_eff: Parameter

    # Heating and current drive
    condrad_cryo_heat: Parameter
    e_nbi: Parameter
    eta_ec: Parameter
    eta_nb: Parameter
    f_cd_aux: Parameter
    f_cd_ohm: Parameter
    f_ni: Parameter
    g_cd_ec: Parameter
    g_cd_nb: Parameter
    P_bd_in: Parameter
    p_ec: Parameter
    P_hcd_ss: Parameter
    P_hcd_ss_el: Parameter
    p_nb: Parameter
    q_control: Parameter
    TF_currpt_ob: Parameter
    TF_E_stored: Parameter
    TF_res_bus: Parameter
    TF_res_tot: Parameter
    TF_respc_ob: Parameter

    # Radiation and charged particles
    f_core_rad_fw: Parameter
    f_fw_aux: Parameter
    f_sol_ch_fw: Parameter
    f_sol_rad: Parameter
    f_sol_rad_fw: Parameter

    # First wall profile
    f_p_sol_near: Parameter
    fw_a_max: Parameter
    fw_dL_max: Parameter  # noqa :N815 - mixed case to match PROCESS
    fw_dL_min: Parameter  # noqa :N815 - mixed case to match PROCESS
    fw_psi_n: Parameter
    hf_limit: Parameter
    P_sep_particle: Parameter

    # SN/DN variables for heat flux transport
    f_hfs_lower_target: Parameter
    f_hfs_upper_target: Parameter
    f_lfs_lower_target: Parameter
    f_lfs_upper_target: Parameter
    fw_lambda_q_far_imp: Parameter
    fw_lambda_q_far_omp: Parameter
    fw_lambda_q_near_imp: Parameter
    fw_lambda_q_near_omp: Parameter

    # Divertor profile
    div_graze_angle: Parameter
    div_L2D_ib: Parameter
    div_L2D_ob: Parameter
    div_Ltarg: Parameter  # noqa :N815 - mixed case to match PROCESS
    div_open: Parameter
    div_psi_o: Parameter
    g_vv_div_add: Parameter
    LPangle: Parameter
    n_div_cassettes: Parameter
    psi_norm: Parameter
    tk_inner_target_pfr: Parameter
    tk_inner_target_sol: Parameter
    tk_outer_target_pfr: Parameter
    tk_outer_target_sol: Parameter
    xpt_inner_gap: Parameter
    xpt_outer_gap: Parameter
    # ad hoc SN variables
    inner_strike_h: Parameter
    outer_strike_h: Parameter
    # ad hoc DN variables
    gamma_inner_target: Parameter
    gamma_outer_target: Parameter
    inner_strike_r: Parameter
    outer_strike_r: Parameter
    theta_inner_target: Parameter
    theta_outer_target: Parameter
    xpt_height: Parameter
    # Divertor cassette
    n_div_cassettes: Parameter
    tk_div_cass: Parameter
    tk_div_cass_in: Parameter

    # Blanket
    bb_min_angle: Parameter
    bb_p_inlet: Parameter
    bb_p_outlet: Parameter
    bb_pump_eta_el: Parameter
    bb_pump_eta_isen: Parameter
    bb_t_inlet: Parameter
    bb_t_outlet: Parameter
    n_bb_inboard: Parameter
    n_bb_outboard: Parameter
    tk_r_ib_bss: Parameter
    tk_r_ib_bz: Parameter
    tk_r_ib_manifold: Parameter
    tk_r_ob_bss: Parameter
    tk_r_ob_bz: Parameter
    tk_r_ob_manifold: Parameter

    # ST Breeding blanket
    g_bb_fw: Parameter
    tk_bb_bz: Parameter
    tk_bb_man: Parameter

    # Divertor
    div_pump_eta_el: Parameter
    div_pump_eta_isen: Parameter

    # Component radial thicknesses (some vertical)
    tk_bb_arm: Parameter
    tk_bb_fw: Parameter
    tk_bb_ib: Parameter
    tk_bb_ob: Parameter
    tk_cr_vv: Parameter
    tk_div: Parameter
    tk_fw_div: Parameter
    tk_fw_in: Parameter
    tk_fw_out: Parameter
    tk_ib_ts: Parameter
    tk_ob_ts: Parameter
    tk_rs: Parameter
    tk_sh_bot: Parameter
    tk_sh_in: Parameter
    tk_sh_out: Parameter
    tk_sh_top: Parameter
    tk_sol_ib: Parameter
    tk_sol_ob: Parameter
    tk_ts: Parameter
    tk_vv_bot: Parameter
    tk_vv_in: Parameter
    tk_vv_out: Parameter
    tk_vv_top: Parameter

    # TF coils
    B_tf_peak: Parameter
    h_cp_top: Parameter
    h_tf_max_in: Parameter
    h_tf_min_in: Parameter
    r_tf_curve: Parameter
    r_tf_inboard_corner: Parameter
    r_tf_outboard_corner: Parameter
    sigma_tf_case_max: Parameter
    sigma_tf_wp_max: Parameter
    tf_taper_frac: Parameter
    tf_wp_depth: Parameter
    tf_wp_width: Parameter
    tk_cs: Parameter
    tk_tf_case_out_in: Parameter
    tk_tf_case_out_out: Parameter
    tk_tf_front_ib: Parameter
    tk_tf_inboard: Parameter
    tk_tf_ins: Parameter
    tk_tf_insgap: Parameter
    tk_tf_nose: Parameter
    tk_tf_ob_casing: Parameter
    tk_tf_outboard: Parameter
    tk_tf_side: Parameter
    tk_tf_wp: Parameter
    tk_tf_wp_y: Parameter

    # PF coils
    r_cs_corner: Parameter
    r_pf_corner: Parameter
    tk_cs_casing: Parameter
    tk_cs_insulation: Parameter
    tk_pf_casing: Parameter
    tk_pf_insulation: Parameter

    # Coil structures
    gs_z_offset: Parameter
    h_cs_seat: Parameter
    min_OIS_length: Parameter
    tk_oic: Parameter
    tk_pf_support: Parameter
    w_g_support: Parameter
    x_g_support: Parameter

    # Component radii
    r_cp_top: Parameter
    r_cs_in: Parameter
    r_fw_ib_in: Parameter
    r_fw_ob_in: Parameter
    r_tf_in: Parameter
    r_tf_in_centre: Parameter
    r_tf_inboard_out: Parameter
    r_tf_out_centre: Parameter
    r_ts_ib_in: Parameter
    r_ts_joint: Parameter
    r_vv_ib_in: Parameter
    r_vv_joint: Parameter
    r_vv_ob_in: Parameter

    # Gaps and clearances
    c_rm: Parameter
    g_cr_rs: Parameter
    g_cr_ts: Parameter
    g_cr_vv: Parameter
    g_cs_mod: Parameter
    g_cs_tf: Parameter
    g_ib_ts_tf: Parameter
    g_ib_vv_ts: Parameter
    g_ob_ts_tf: Parameter
    g_ob_vv_ts: Parameter
    g_tf_pf: Parameter
    g_ts_pf: Parameter
    g_ts_tf: Parameter
    g_ts_tf_topbot: Parameter
    g_vv_bb: Parameter
    g_vv_ts: Parameter

    # Offsets
    o_p_cr: Parameter
    o_p_rs: Parameter

    # Vacuum vessel
    vv_dtk: Parameter
    vv_stk: Parameter
    vvpfrac: Parameter

    # Neutronics
    blk_1_dpa: Parameter
    blk_2_dpa: Parameter
    div_dpa: Parameter
    e_decay_mult: Parameter
    e_mult: Parameter
    tf_fluence: Parameter
    vv_dpa: Parameter

    # Central solenoid
    CS_bmax: Parameter
    CS_jmax: Parameter
    CS_material: Parameter
    F_cs_sepmax: Parameter
    F_cs_ztotmax: Parameter
    F_pf_zmax: Parameter

    # PF magnets
    PF_bmax: Parameter
    PF_jmax: Parameter
    PF_material: Parameter

    # Equilibria
    B_premag_stray_max: Parameter

    # Cryostat
    cr_l_d: Parameter
    n_cr_lab: Parameter
    r_cryo_ts: Parameter
    tk_cryo_ts: Parameter
    z_cryo_ts: Parameter

    # Radiation shield
    n_rs_lab: Parameter
    rs_l_d: Parameter
    rs_l_gap: Parameter

    # Lifecycle
    a_max: Parameter
    a_min: Parameter
    n_DD_reactions: Parameter
    n_DT_reactions: Parameter

    # Tritium fuelling and vacuum system
    m_gas: Parameter

    # Maintenance
    bmd: Parameter
    dmd: Parameter
    RMTFI: Parameter

    # Central column shield
    g_ccs_div: Parameter
    g_ccs_fw: Parameter
    g_ccs_vv_add: Parameter
    g_ccs_vv_inboard: Parameter
    r_ccs: Parameter
    tk_ccs_min: Parameter
