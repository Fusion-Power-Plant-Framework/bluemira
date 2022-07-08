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

    A: Parameter
    B_0: Parameter
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
    H_star: Parameter
    I_p: Parameter
    kappa: Parameter
    kappa_95: Parameter
    l_i: Parameter
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
    T_e: Parameter
    T_e_ped: Parameter
    tau_e: Parameter
    v_burn: Parameter
    V_p: Parameter
    Z_eff: Parameter

    # Heating and current drive
    condrad_cryo_heat: Parameter
    e_nbi: Parameter
    eta_nb: Parameter
    f_ni: Parameter
    P_bd_in: Parameter
    P_hcd_ss: Parameter
    P_hcd_ss_el: Parameter
    q_control: Parameter
    TF_currpt_ob: Parameter
    TF_E_stored: Parameter
    TF_res_bus: Parameter
    TF_res_tot: Parameter
    TF_respc_ob: Parameter

    # First wall profile
    f_p_sol_near: Parameter
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

    # Component radial thicknesses (some vertical)
    tk_bb_ib: Parameter
    tk_bb_ob: Parameter
    tk_cr_vv: Parameter
    tk_fw_in: Parameter
    tk_fw_out: Parameter
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
    sigma_tf_case_max: Parameter
    sigma_tf_wp_max: Parameter
    tf_wp_depth: Parameter
    tf_wp_width: Parameter
    tk_cs: Parameter
    tk_tf_front_ib: Parameter
    tk_tf_inboard: Parameter
    tk_tf_ins: Parameter
    tk_tf_insgap: Parameter
    tk_tf_nose: Parameter
    tk_tf_outboard: Parameter
    tk_tf_side: Parameter

    # Coil structures
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
    r_vv_ib_in: Parameter
    r_vv_ob_in: Parameter

    # Gaps and clearances
    g_cr_rs: Parameter
    g_cr_ts: Parameter
    g_cr_vv: Parameter
    g_cs_tf: Parameter
    g_ts_pf: Parameter
    g_ts_tf: Parameter
    g_vv_bb: Parameter
    g_vv_ts: Parameter

    # Offsets
    o_p_cr: Parameter
    o_p_rs: Parameter

    # Neutronics
    e_mult: Parameter

    # Equilibria
    B_premag_stray_max: Parameter

    # Cryostat
    cr_l_d: Parameter
    n_cr_lab: Parameter
    tk_cryo_ts: Parameter

    # Radiation shield
    n_rs_lab: Parameter
    rs_l_d: Parameter
    rs_l_gap: Parameter

    # Tritium fuelling and vacuum system
    m_gas: Parameter
