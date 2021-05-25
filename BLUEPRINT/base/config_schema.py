# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Defines the master configuration schema, which controls the parameterisation of a given
analysis.
"""

from BLUEPRINT.base.parameter import Parameter


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

    # First wall and divertor profile
    fw_psi_n: Parameter
    fw_dx: Parameter
    div_L2D_ib: Parameter
    div_L2D_ob: Parameter
    div_graze_angle: Parameter
    div_psi_o: Parameter
    div_Ltarg: Parameter  # noqa(N815) - mixed case to match PROCESS
    div_open: Parameter
    fw_dL_min: Parameter  # noqa(N815) - mixed case to match PROCESS
    fw_dL_max: Parameter  # noqa(N815) - mixed case to match PROCESS
    fw_a_max: Parameter
    g_vv_div_add: Parameter
    LPangle: Parameter
    n_div_cassettes: Parameter

    # Blanket
    bb_e_mult: Parameter
    bb_min_angle: Parameter
    tk_r_ib_bz: Parameter
    tk_r_ib_manifold: Parameter
    tk_r_ib_bss: Parameter
    tk_r_ob_bz: Parameter
    tk_r_ob_manifold: Parameter
    tk_r_ob_bss: Parameter

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

    # TF coils
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
