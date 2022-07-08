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
Unused params
"""
Parameter = None


class UnusedConfigurationSchema:
    """
    Unused config
    """

    op_mode: Parameter
    reactor_type: Parameter

    Av: Parameter
    blanket_type: Parameter

    f_DD_fus: Parameter
    m_s_limit: Parameter

    eta_ec: Parameter
    f_cd_aux: Parameter
    f_cd_ohm: Parameter
    g_cd_ec: Parameter
    g_cd_nb: Parameter
    p_ec: Parameter
    p_nb: Parameter

    fw_a_max: Parameter
    fw_dL_max: Parameter  # noqa :N815 - mixed case to match PROCESS
    fw_dL_min: Parameter  # noqa :N815 - mixed case to match PROCESS
    hf_limit: Parameter

    # Divertor profile
    div_graze_angle: Parameter
    div_psi_o: Parameter
    g_vv_div_add: Parameter
    LPangle: Parameter
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

    tk_bb_arm: Parameter
    tk_bb_fw: Parameter
    tk_div: Parameter
    tk_fw_div: Parameter
    tk_ib_ts: Parameter
    tk_ob_ts: Parameter
    h_tf_min_in: Parameter
    r_tf_curve: Parameter
    r_tf_inboard_corner: Parameter
    r_tf_outboard_corner: Parameter
    tf_taper_frac: Parameter
    tk_tf_case_out_in: Parameter
    tk_tf_case_out_out: Parameter
    tk_tf_ob_casing: Parameter

    gs_z_offset: Parameter
    h_cs_seat: Parameter
    min_OIS_length: Parameter
    tk_oic: Parameter
    tk_pf_support: Parameter
    w_g_support: Parameter

    r_ts_joint: Parameter
    r_vv_joint: Parameter

    g_cs_mod: Parameter
    g_ib_ts_tf: Parameter
    g_ib_vv_ts: Parameter
    g_ob_ts_tf: Parameter
    g_ob_vv_ts: Parameter
    g_tf_pf: Parameter
    g_ts_tf_topbot: Parameter

    # Vacuum vessel
    vv_dtk: Parameter
    vv_stk: Parameter

    blk_1_dpa: Parameter
    blk_2_dpa: Parameter
    div_dpa: Parameter
    tf_fluence: Parameter
    vv_dpa: Parameter

    # Central solenoid
    CS_material: Parameter
    F_cs_sepmax: Parameter
    F_cs_ztotmax: Parameter
    F_pf_zmax: Parameter

    # PF magnets
    PF_material: Parameter

    r_cryo_ts: Parameter
    z_cryo_ts: Parameter

    # Lifecycle
    a_max: Parameter
    a_min: Parameter
    n_DD_reactions: Parameter
    n_DT_reactions: Parameter

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


# fmt:off
params = [
    ['op_mode', 'Mode of operation', 'Pulsed', 'dimensionless', None, 'Input'],
    ['reactor_type', 'Type of reactor', 'Normal', 'dimensionless', None, 'Input'],

    # Reactor
    ['Av', 'Reactor availability', 0.3, 'dimensionless', None, 'Input'],
    ['blanket_type', 'Blanket type', 'HCPB', 'dimensionless', None, 'Input'],

    # Plasma
    ['f_DD_fus', 'Fraction of D-D fusion in total fusion', 0.0025, 'dimensionless', None, 'Input'],
    ['m_s_limit', "Margin to vertical stability", 0.3, "dimensionless", None, "Input"],

    # Heating and current drive
    ['eta_ec', 'EC electrical efficiency', 0.35, 'dimensionless', 'Check units!', 'Input'],
    ['f_cd_aux', 'Auxiliary current drive fraction', 0.1, 'dimensionless', None, 'Input'],
    ['f_cd_ohm', 'Ohmic current drive fraction', 0.1, 'dimensionless', None, 'Input'],
    ['g_cd_ec', 'EC current drive efficiency', 0.15, 'MA/MW.m', 'Check units!', 'Input'],
    ['g_cd_nb', 'NB current drive efficiency', 0.4, 'MA/MW.m', 'Check units!', 'Input'],
    ['p_ec', 'EC launcher power', 10, 'MW', 'Maximum launcher power per sector', 'Input'],
    ['p_nb', 'NB launcher power', 1, 'MA', 'Maximum launcher current drive in a port', 'Input'],

    # First wall profile
    ['fw_a_max', 'Maximum angle between FW modules', 25, '°', None, 'Input'],
    ['fw_dL_max', 'Maximum FW module length', 2, 'm', None, 'Input'],
    ['fw_dL_min', 'Minimum FW module length', 0.75, 'm', None, 'Input'],
    ['hf_limit', 'heat flux material limit', 0.5, 'MW/m^2', None, 'Input'],

    # Divertor profile
    ['div_graze_angle', 'Divertor SOL grazing angle', 1.5, '°', None, 'Input'],
    ['div_psi_o', 'Divertor flux offset', 0.5, 'm', None, 'Input'],
    ['g_vv_div_add', 'Additional divertor/VV gap', 0, 'm', None, 'Input'],
    ['LPangle', 'Lower port inclination angle', -30, '°', None, 'Input'],
    ['psi_norm', 'Normalised flux value of strike-point contours', 1, 'dimensionless', None, 'Input'],
    ['tk_inner_target_pfr', 'Inner target length PFR side', 0.5, 'm', None, 'Input'],
    ['tk_inner_target_sol', 'Inner target length SOL side', 0.3, 'm', None, 'Input'],
    ['tk_outer_target_pfr', 'Outer target length PFR side', 0.3, 'm', None, 'Input'],
    ['tk_outer_target_sol', 'Outer target length SOL side', 0.7, 'm', None, 'Input'],
    ['xpt_inner_gap', 'Gap between x-point and inner wall', 0.4, 'm', None, 'Input'],
    ['xpt_outer_gap', 'Gap between x-point and outer wall', 2, 'm', None, 'Input'],
    # ad hoc SN variables
    ['inner_strike_h', 'Inner strike point height', 1, 'm', None, 'Input'],
    ['outer_strike_h', 'Outer strike point height', 2, 'm', None, 'Input'],
    # ad hoc DN variables
    ['gamma_inner_target', 'Angle between flux line and the outer divertor strike point defined in the 3D space', 3, 'deg', None, 'Input'],
    ['gamma_outer_target', 'Angle between flux line and the outer divertor strike point defined in the 3D space', 3, 'deg', None, 'Input'],
    ['inner_strike_r', 'Inner strike point major radius', 8, 'm', None, 'Input'],
    ['outer_strike_r', 'Outer strike point major radius', 10.3, 'm', None, 'Input'],
    ['theta_inner_target', 'Angle between flux line and the outer divertor strike point projected in the poloidal plane', 20, 'deg', None, 'Input'],
    ['theta_outer_target', 'Angle between flux line and the outer divertor strike point projected in the poloidal plane', 20, 'deg', None, 'Input'],
    ['xpt_height', 'x-point vertical_gap', 0.35, 'm', None, 'Input'],
    # Divertor cassette
    ['n_div_cassettes', 'Number of divertor cassettes per sector', 3, 'dimensionless', None, 'Input'],
    ['tk_div_cass', 'Minimum thickness between inner divertor profile and cassette', 0.3, 'm', None, 'Input'],
    ['tk_div_cass_in', 'Additional radial thickness on inboard side relative to to inner strike point', 0.1, 'm', None, 'Input'],

    # Blanket
    ["tk_r_ib_bss", "Thickness ratio of the inboard blanket back supporting structure", 0.577, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
    ["tk_r_ib_bz", "Thickness ratio of the inboard blanket breeding zone", 0.309, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
    ["tk_r_ib_manifold", "Thickness ratio of the inboard blanket manifold", 0.114, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
    ["tk_r_ob_bss", "Thickness ratio of the outboard blanket back supporting structure", 0.498, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
    ["tk_r_ob_bz", "Thickness ratio of the outboard blanket breeding zone", 0.431, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
    ["tk_r_ob_manifold", "Thickness ratio of the outboard blanket manifold", 0.071, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],

    # ST Breeding blanket
    ['g_bb_fw', 'Separation between the first wall and the breeding blanket', 0.05, 'm', None, 'Input'],
    ['tk_bb_bz', 'Breeding zone thickness', 1.0, 'm', None, 'Input'],
    ['tk_bb_man', 'Breeding blanket manifold thickness', 0.2, 'm', None, 'Input'],

    # Component radial thicknesses (some vertical)
    ['tk_bb_arm', 'Tungsten armour thickness', 0.002, 'm', None, 'Input'],
    ['tk_bb_fw', 'First wall thickness', 0.052, 'm', None, 'Input'],
    ['tk_div', 'Divertor thickness', 0.4, 'm', None, 'Input'],
    ['tk_fw_div', 'First wall thickness around divertor', 0.052, 'm', None, 'Input'],
    ['tk_ib_ts', 'Inboard TS thickness', 0.05, 'm', None, 'Input'],
    ['tk_ob_ts', 'Outboard TS thickness', 0.05, 'm', None, 'Input'],

    # TF coils
    ['h_tf_min_in', 'Plasma side TF coil min height', -6.5, 'm', None, 'Input'],
    ['r_tf_curve', "Start of the upper curve of domed picture frame shale", 3., 'm', None, 'Input'],
    ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
    ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
    ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'dimensionless', None, 'Input'],
    ['tk_tf_case_out_in', 'TF coil case thickness on the outboard inside', 0.35, 'm', None, 'Input'],
    ['tk_tf_case_out_out', 'TF coil case thickness on the outboard outside', 0.4, 'm', None, 'Input'],
    ['tk_tf_ob_casing', "TF leg conductor casing general thickness", 0.02, "m", None, "Input"],

    # Coil structures
    ['gs_z_offset', 'Gravity support vertical offset', -1, 'm', None, 'Input'],
    ['h_cs_seat', 'Height of the CS support', 2, 'm', None, 'Input'],
    ['min_OIS_length', 'Minimum length of an inter-coil structure', 0.5, 'm', None, 'Input'],
    ['tk_oic', 'Outer inter-coil structure thickness', 0.3, 'm', None, 'Input'],
    ['tk_pf_support', 'PF coil support plate thickness', 0.15, 'm', None, 'Input'],
    ['w_g_support', 'TF coil gravity support width', 0.75, 'm', None, 'Input'],

    # Component radii
    ['r_ts_joint', 'Radius of inboard/outboard TS joint', 2., 'm', None, 'Input'],
    ['r_vv_joint', 'Radius of inboard/outboard VV joint', 2., 'm', None, 'Input'],

    # Gaps and clearances
    ['g_cs_mod', 'Gap between CS modules', 0.1, 'm', None, 'Input'],
    ['g_ib_ts_tf', 'Inboard gap between TS and TF', 0.05, 'm', None, 'Input'],
    ['g_ib_vv_ts', 'Inboard gap between VV and TS', 0.05, 'm', None, 'Input'],
    ['g_ob_ts_tf', 'Outboard gap between TS and TF', 0.05, 'm', None, 'Input'],
    ['g_ob_vv_ts', 'Outboard gap between VV and TS', 0.05, 'm', None, 'Input'],
    ['g_tf_pf', 'Gap between TF and PF', 0.15, 'm', None, 'Input'],
    ['g_ts_tf_topbot', 'Vessel KOZ offset to TF coils on top and bottom edges', 0.11, 'm', None, 'Input'],

    # Vacuum vessel
    ['vv_dtk', 'VV double-walled thickness', 0.2, 'm', None, 'Input'],
    ['vv_stk', 'VV single-walled thickness', 0.06, 'm', None, 'Input'],

    # Neutronics
    ['blk_1_dpa', 'Starter blanket life limit (EUROfer)', 20, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
    ['blk_2_dpa', 'Second blanket life limit (EUROfer)', 50, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
    ['div_dpa', 'Divertor life limit (CuCrZr)', 5, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
    ['tf_fluence', 'Insulation fluence limit for ITER equivalent to 10 MGy', 3.2e21, '1/m^2', None, 'Input (https://ieeexplore.ieee.org/document/6374236/)'],
    ['vv_dpa', 'Vacuum vessel life limit (SS316-LN-IG)', 3.25, 'dpa', None, 'Input (RCC-Mx or whatever it is called)'],

    # Central solenoid
    ['CS_material', 'Conducting material to use for the CS modules', 'Nb3Sn', 'dimensionless', None, 'Input'],
    ['F_cs_sepmax', 'Maximum separation force between CS modules', 300, 'MN', None, 'Input'],
    ['F_cs_ztotmax', 'Maximum total vertical force in the CS stack', 350, 'MN', None, 'Input'],
    ['F_pf_zmax', 'Maximum vertical force on a single PF coil', 450, 'MN', None, 'Input'],

    # PF magnets
    ['PF_material', 'Conducting material to use for the PF coils', 'NbTi', 'dimensionless', None, 'Input'],

    # Cryostat
    ['r_cryo_ts', 'Radius of outboard cryo TS', 8, 'm', None, 'Input'],
    ['z_cryo_ts', 'Half height of outboard cryo TS', 8, 'm', None, 'Input'],

    # Lifecycle
    ['a_max', 'Maximum operational load factor', 0.5, 'dimensionless', 'Can be violated', 'Input'],
    ['a_min', 'Minimum operational load factor', 0.1, 'dimensionless', 'Otherwise nobody pays', 'Input'],
    ['n_DD_reactions', 'D-D fusion reaction rate', 8.5E18, '1/s', 'At full power', 'Input'],
    ['n_DT_reactions', 'D-T fusion reaction rate', 7.1E20, '1/s', 'At full power', 'Input'],

    # Maintenance
    ['bmd', 'Blanket maintenance duration', 150, 'days', 'Full replacement intervention duration', 'Input'],
    ['dmd', 'Divertor maintenance duration', 90, 'days', 'Full replacement intervention duration', 'Input'],
    ['RMTFI', 'RM Technical Feasibility Index', 1, 'dimensionless', 'Default value. Should not really be 1', 'Input'],

    # Central column shield
    ["g_ccs_div", "Gap between the central column shield and the divertor cassette", 0.05, "m", None, "Input"],
    ["g_ccs_fw", "Gap between the central column shield and the first wall", 0.05, "m", None, "Input"],
    ["g_ccs_vv_add", "Additional gap between the central column shield and the vacuum vessel", 0.0, "m", None, "Input"],
    ["g_ccs_vv_inboard", "Gap between central column shield and the vacuum vessel on the inboard side", 0.05, "m", None, "Input"],
    ["r_ccs", "Outer radius of the central column shield", 2.5, "m", None, "Input"],
    ["tk_ccs_min", "Minimum thickness of the central column shield", 0.1, "m", None, "Input"],
]

# fmt:on
