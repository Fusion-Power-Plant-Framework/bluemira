"""EUDEMO reactor build parameters."""

from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame


@dataclass
class EUDEMOReactorParams(ParameterFrame):
    """All parameters for the EUDEMO reactor."""

    # Common parameters
    A: Parameter[float]
    B_0: Parameter[float]
    B_premag_stray_max: Parameter[float]
    B_tf_peak: Parameter[float]
    beta_p: Parameter[float]
    beta: Parameter[float]
    C_Ejima: Parameter[float]
    condrad_cryo_heat: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    delta_l: Parameter[float]
    delta_u: Parameter[float]
    e_mult: Parameter[float]
    e_nbi: Parameter[float]
    eta_nb: Parameter[float]
    f_bs: Parameter[float]
    f_ni: Parameter[float]
    g_cs_mod: Parameter[float]
    g_cs_tf: Parameter[float]
    g_ts_tf: Parameter[float]
    g_vv_bb: Parameter[float]
    g_vv_ts: Parameter[float]
    h_cp_top: Parameter[float]
    H_star: Parameter[float]
    h_tf_max_in: Parameter[float]
    I_p: Parameter[float]
    ib_offset_angle: Parameter[float]
    kappa_95: Parameter[float]
    kappa: Parameter[float]
    kappa_l: Parameter[float]
    kappa_u: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    n_TF: Parameter[float]
    ob_offset_angle: Parameter[float]
    P_bd_in: Parameter[float]
    P_brehms: Parameter[float]
    P_el_net_process: Parameter[float]
    P_el_net: Parameter[float]
    P_fus_DD: Parameter[float]
    P_fus_DT: Parameter[float]
    P_fus: Parameter[float]
    P_hcd_ss: Parameter[float]
    P_line: Parameter[float]
    P_rad_core: Parameter[float]
    P_rad_edge: Parameter[float]
    P_rad: Parameter[float]
    P_sep: Parameter[float]
    P_sync: Parameter[float]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    q_95: Parameter[float]
    R_0: Parameter[float]
    r_cp_top: Parameter[float]
    r_cs_in: Parameter[float]
    r_fw_ib_in: Parameter[float]
    r_fw_ob_in: Parameter[float]
    r_tf_in_centre: Parameter[float]
    r_tf_in: Parameter[float]
    r_tf_inboard_out: Parameter[float]
    r_tf_out_centre: Parameter[float]
    r_ts_ib_in: Parameter[float]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    sigma_tf_case_max: Parameter[float]
    sigma_tf_wp_max: Parameter[float]
    T_e: Parameter[float]
    tau_e: Parameter[float]
    tau_flattop: Parameter[float]
    TF_currpt_ob: Parameter[float]
    TF_E_stored: Parameter[float]
    TF_res_bus: Parameter[float]
    TF_res_tot: Parameter[float]
    TF_respc_ob: Parameter[float]
    TF_ripple_limit: Parameter[float]
    tf_wp_depth: Parameter[float]
    tf_wp_width: Parameter[float]
    tk_bb_ib: Parameter[float]
    tk_bb_ob: Parameter[float]
    tk_cr_vv: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]
    tk_fw_in: Parameter[float]
    tk_fw_out: Parameter[float]
    tk_sh_bot: Parameter[float]
    tk_sh_in: Parameter[float]
    tk_sh_out: Parameter[float]
    tk_sh_top: Parameter[float]
    tk_sol_ib: Parameter[float]
    tk_sol_ob: Parameter[float]
    tk_tf_front_ib: Parameter[float]
    tk_tf_inboard: Parameter[float]
    tk_tf_ins: Parameter[float]
    tk_tf_insgap: Parameter[float]
    tk_tf_nose: Parameter[float]
    tk_tf_outboard: Parameter[float]
    tk_tf_side: Parameter[float]
    tk_ts: Parameter[float]
    tk_vv_bot: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    tk_vv_top: Parameter[float]
    v_burn: Parameter[float]
    V_p: Parameter[float]
    Z_eff: Parameter[float]

    # PLASMOD
    pheat_max: Parameter[float]
    q_control: Parameter[float]

    # Equilibrium
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    shaf_shift: Parameter[float]

    # Wall designer
    fw_psi_n: Parameter[float]

    # Divertor silhouette
    div_type: Parameter[str]
    div_Ltarg: Parameter[float]  # noqa: N815
    div_open: Parameter[bool]

    # Plasma face
    c_rm: Parameter[float]

    # Vacuum vessel
    vv_in_off_deg: Parameter[float]
    vv_out_off_deg: Parameter[float]

    # Divertor
    n_div_cassettes: Parameter[int]

    # Blanket
    n_bb_inboard: Parameter[int]
    n_bb_outboard: Parameter[int]

    # TF Coils
    r_tf_current_ib: Parameter[float]
    tk_tf_wp_y: Parameter[float]
    tk_tf_wp: Parameter[float]
    z_0: Parameter[float]

    # PF Coils
    F_cs_sepmax: Parameter[float]
    F_cs_ztotmax: Parameter[float]
    F_pf_zmax: Parameter[float]
