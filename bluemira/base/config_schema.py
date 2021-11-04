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
    plasma_type: Parameter
    reactor_type: Parameter
    op_mode: Parameter

    P_el_net: Parameter
    tau_flattop: Parameter
    blanket_type: Parameter
    n_TF: Parameter
    n_PF: Parameter
    n_CS: Parameter
    TF_ripple_limit: Parameter
    Av: Parameter
    A: Parameter
    R_0: Parameter
    B_0: Parameter

    # Plasma
    q_95: Parameter
    kappa_95: Parameter
    kappa: Parameter
    delta_95: Parameter
    delta: Parameter
    T_e: Parameter
    Z_eff: Parameter
    res_plasma: Parameter
    V_p: Parameter
    l_i: Parameter
    I_p: Parameter
    P_fus: Parameter
    P_fus_DT: Parameter
    P_fus_DD: Parameter
    f_DD_fus: Parameter
    H_star: Parameter
    P_sep: Parameter
    P_rad_core: Parameter
    P_rad_edge: Parameter
    P_rad: Parameter
    P_line: Parameter
    P_sync: Parameter
    P_brehms: Parameter
    f_bs: Parameter
    beta_N: Parameter  # noqa(N815) - mixed case to match PROCESS
    beta_p: Parameter
    beta: Parameter
    tau_e: Parameter
    v_burn: Parameter
    shaf_shift: Parameter
    C_Ejima: Parameter
    m_s_limit: Parameter

    # Heating and current drive
    f_ni: Parameter
    e_nbi: Parameter
    P_hcd_ss: Parameter
    q_control: Parameter
    g_cd_nb: Parameter
    eta_nb: Parameter
    p_nb: Parameter
    g_cd_ec: Parameter
    eta_ec: Parameter
    p_ec: Parameter
    f_cd_aux: Parameter
    f_cd_ohm: Parameter

    # First wall profile
    fw_dx: Parameter
    fw_psi_n: Parameter
    fw_dL_min: Parameter  # noqa(N815) - mixed case to match PROCESS
    fw_dL_max: Parameter  # noqa(N815) - mixed case to match PROCESS
    fw_a_max: Parameter
    fw_p_sol_near: Parameter
    fw_p_sol_far: Parameter
    hf_limit: Parameter
    # ad hoc SN variables
    fw_lambda_q_near: Parameter
    fw_lambda_q_far: Parameter
    f_outer_target: Parameter
    f_inner_target: Parameter
    # ad hoc DN variables
    fw_dpsi_n_near: Parameter
    fw_dpsi_n_far: Parameter
    fw_dx_omp: Parameter
    fw_dx_imp: Parameter
    p_rate_omp: Parameter
    p_rate_imp: Parameter
    fw_lambda_q_near_omp: Parameter
    fw_lambda_q_far_omp: Parameter
    fw_lambda_q_near_imp: Parameter
    fw_lambda_q_far_imp: Parameter
    dr_near_omp: Parameter
    dr_far_omp: Parameter
    f_lfs_lower_target: Parameter
    f_lfs_upper_target: Parameter
    f_hfs_lower_target: Parameter
    f_hfs_upper_target: Parameter

    # Divertor profile
    div_L2D_ib: Parameter
    div_L2D_ob: Parameter
    div_graze_angle: Parameter
    div_psi_o: Parameter
    div_Ltarg: Parameter  # noqa(N815) - mixed case to match PROCESS
    div_open: Parameter
    g_vv_div_add: Parameter
    LPangle: Parameter
    n_div_cassettes: Parameter
    psi_norm: Parameter
    xpt_outer_gap: Parameter
    xpt_inner_gap: Parameter
    tk_outer_target_sol: Parameter
    tk_outer_target_pfr: Parameter
    tk_inner_target_sol: Parameter
    tk_inner_target_pfr: Parameter
    # ad hoc SN variables
    outer_strike_h: Parameter
    inner_strike_h: Parameter
    # ad hoc DN variables
    outer_strike_r: Parameter
    inner_strike_r: Parameter
    theta_outer_target: Parameter
    theta_inner_target: Parameter
    xpt_height: Parameter

    # Blanket
    bb_e_mult: Parameter
    bb_min_angle: Parameter
    tk_r_ib_bz: Parameter
    tk_r_ib_manifold: Parameter
    tk_r_ib_bss: Parameter
    tk_r_ob_bz: Parameter
    tk_r_ob_manifold: Parameter
    tk_r_ob_bss: Parameter

    # ST Breeding blanket
    g_bb_fw: Parameter
    tk_bb_bz: Parameter
    tk_bb_man: Parameter

    # Component radial thicknesses (some vertical)
    tk_bb_ib: Parameter
    tk_bb_ob: Parameter
    tk_bb_fw: Parameter
    tk_bb_arm: Parameter
    tk_sh_in: Parameter
    tk_sh_out: Parameter
    tk_sh_top: Parameter
    tk_sh_bot: Parameter
    tk_vv_in: Parameter
    tk_vv_out: Parameter
    tk_vv_top: Parameter
    tk_vv_bot: Parameter
    tk_sol_ib: Parameter
    tk_sol_ob: Parameter
    tk_div: Parameter
    tk_ts: Parameter
    tk_ib_ts: Parameter
    tk_ob_ts: Parameter
    tk_cr_vv: Parameter
    tk_rs: Parameter
    tk_fw_in: Parameter
    tk_fw_out: Parameter
    tk_fw_div: Parameter
    tk_div_cass: Parameter
    tk_div_in: Parameter

    # TF coils
    tk_tf_inboard: Parameter
    tk_tf_outboard: Parameter
    tk_tf_nose: Parameter
    tk_tf_wp: Parameter
    tk_tf_front_ib: Parameter
    tk_tf_ins: Parameter
    tk_tf_insgap: Parameter
    tk_tf_side: Parameter
    tk_tf_case_out_in: Parameter
    tk_tf_case_out_out: Parameter
    tf_wp_width: Parameter
    tf_wp_depth: Parameter
    tk_cs: Parameter
    sigma_tf_max: Parameter
    h_cp_top: Parameter
    B_tf_peak: Parameter
    tf_taper_frac: Parameter
    r_tf_outboard_corner: Parameter
    r_tf_inboard_corner: Parameter
    r_tf_curve: Parameter
    h_tf_max_in: Parameter
    h_tf_min_in: Parameter

    # Coil structures
    x_g_support: Parameter
    w_g_support: Parameter
    tk_oic: Parameter
    tk_pf_support: Parameter
    gs_z_offset: Parameter
    h_cs_seat: Parameter
    min_OIS_length: Parameter

    # Component radii
    r_cp_top: Parameter
    r_cs_in: Parameter
    r_tf_in: Parameter
    r_tf_inboard_out: Parameter
    r_tf_in_centre: Parameter
    r_tf_out_centre: Parameter
    r_ts_joint: Parameter
    r_ts_ib_in: Parameter
    r_vv_ib_in: Parameter
    r_vv_ob_in: Parameter
    r_fw_ib_in: Parameter
    r_fw_ob_in: Parameter

    # Gaps and clearances
    g_cs_mod: Parameter
    g_vv_ts: Parameter
    g_cs_tf: Parameter
    g_ts_tf: Parameter
    g_ib_ts_tf: Parameter
    g_ob_ts_tf: Parameter
    g_vv_bb: Parameter
    g_tf_pf: Parameter
    g_ts_pf: Parameter
    g_ts_tf_topbot: Parameter
    g_cr_ts: Parameter
    g_cr_vv: Parameter
    g_cr_rs: Parameter
    c_rm: Parameter

    # Offsets
    o_p_rs: Parameter
    o_p_cr: Parameter

    # Vacuum vessel
    vv_dtk: Parameter
    vv_stk: Parameter
    vvpfrac: Parameter

    # Neutronics
    blk_1_dpa: Parameter
    blk_2_dpa: Parameter
    div_dpa: Parameter
    vv_dpa: Parameter
    tf_fluence: Parameter

    # Central solenoid
    F_pf_zmax: Parameter
    F_cs_ztotmax: Parameter
    F_cs_sepmax: Parameter
    CS_material: Parameter

    # PF magnets
    PF_material: Parameter

    # Cryostat
    n_cr_lab: Parameter
    cr_l_d: Parameter

    # Radiation shield
    n_rs_lab: Parameter
    rs_l_d: Parameter
    rs_l_gap: Parameter

    # Lifecycle
    n_DT_reactions: Parameter
    n_DD_reactions: Parameter
    a_min: Parameter
    a_max: Parameter

    # Tritium fuelling and vacuum system
    m_gas: Parameter

    # Maintenance
    bmd: Parameter
    dmd: Parameter
    RMTFI: Parameter

    # Central column shield
    g_ccs_vv_inboard: Parameter
    g_ccs_vv_add: Parameter
    g_ccs_fw: Parameter
    g_ccs_div: Parameter
    tk_ccs_min: Parameter
    r_ccs: Parameter
