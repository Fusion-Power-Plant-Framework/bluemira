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
Configuration classes
"""
from bluemira.base.config_schema import ConfigurationSchema
from bluemira.base.parameter import ParameterFrame


class Configuration(ConfigurationSchema, ParameterFrame):
    """
    The base object for all variable names and metadata in bluemira.
    Variables specified here should be physical in some way, and not represent
    how the code is being run.
    Defaults are also specified here, and overridden later.
    New variables should be defined here, with a corresponding entry in the
    ConfigurationSchema, and passed onwards as Parameter objects.
    """

    # fmt: off
    params = [
        ['Name', 'Reactor name', 'Cambridge', 'dimensionless', None, 'Input'],
        ['plasma_type', 'Type of plasma', 'SN', 'dimensionless', None, 'Input'],
        ['reactor_type', 'Type of reactor', 'Normal', 'dimensionless', None, 'Input'],
        ['op_mode', 'Mode of operation', 'Pulsed', 'dimensionless', None, 'Input'],

        # Reactor
        ['P_el_net', 'Net electrical power output', 500, 'MW', None, 'Input'],
        ['P_el_net_process', 'Net electrical power output as provided by PROCESS', None, 'MW', None, 'Input'],
        ['tau_flattop', 'Flat-top duration', 2 * 3600, 's', None, 'Input'],
        ['blanket_type', 'Blanket type', 'HCPB', 'dimensionless', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['n_PF', 'Number of PF coils', 6, 'dimensionless', None, 'Input'],
        ['n_CS', 'Number of CS coil divisions', 5, 'dimensionless', None, 'Input'],
        ['TF_ripple_limit', 'TF coil ripple limit', 0.6, '%', None, 'Input'],
        ['Av', 'Reactor availability', 0.3, 'dimensionless', None, 'Input'],
        ['A', 'Plasma aspect ratio', 3.1, 'dimensionless', None, 'Input'],
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['z_0', 'Vertical offset of plasma centreline', 0, 'm', None, 'Input'],
        ['B_0', 'Toroidal field at R_0', 6, 'T', None, 'Input'],

        # Plasma
        ['q_95', 'Plasma safety factor', 3.5, 'dimensionless', None, 'Input'],
        ['kappa_95', '95th percentile plasma elongation', 1.6, 'dimensionless', None, 'Input'],
        ['kappa', 'Last closed surface plasma elongation', 1.792, 'dimensionless', None, 'Input'],
        ['delta_95', '95th percentile plasma triangularity', 0.333, 'dimensionless', None, 'Input'],
        ['delta', 'Last closed surface plasma triangularity', 0.5, 'dimensionless', None, 'Input'],
        ['T_e', 'Average plasma electron temperature', 13, 'keV', None, 'Input'],
        ['Z_eff', 'Effective particle radiation atomic mass', 2.2, 'amu', None, 'Input'],
        ['res_plasma', 'Plasma resistance', 0, 'ohm', None, 'Calculated'],
        ['V_p', 'Plasma volume', 2400, 'm^3', None, 'Calculated'],
        ['l_i', 'Normalised internal plasma inductance', 0.8, 'dimensionless', None, 'Input'],
        ['I_p', 'Plasma current', 19, 'MA', None, 'Input'],
        ['P_fus', 'Total fusion power', 2000, 'MW', None, 'Input'],
        ['P_fus_DT', 'D-T fusion power', 1995, 'MW', None, 'Input'],
        ['P_fus_DD', 'D-D fusion power', 5, 'MW', None, 'Input'],
        ['f_DD_fus', 'Fraction of D-D fusion in total fusion', 0.0025, 'dimensionless', None, 'Input'],
        ['H_star', 'H factor (radiation corrected)', 1, 'dimensionless', None, 'Input'],
        ['P_sep', 'Separatrix power', 150, 'MW', None, 'Input'],
        ['P_rad_core', 'Core radiation power', 0, 'MW', None, 'Input'],
        ['P_rad_edge', 'Edge radiation power', 400, 'MW', None, 'Input'],
        ['P_rad', 'Radiation power', 400, 'MW', None, 'Input'],
        ['P_line', 'Line radiation', 30, 'MW', None, 'Input'],
        ['P_sync', 'Synchrotron radiation', 50, 'MW', None, 'Input'],
        ['P_brehms', 'Bremsstrahlung', 80, 'MW', None, 'Input'],
        ['P_LH', 'LH transition power', 0, 'W', None, 'Input'],
        ['P_ohm', 'Ohimic heating power', 0, 'W', None, 'Input'],
        ['f_bs', 'Bootstrap fraction', 0.5, 'dimensionless', None, 'Input'],
        ['beta_N', 'Normalised ratio of plasma pressure to magnetic pressure', 2.7, 'dimensionless', None, 'Input'],
        ['beta_p', 'Ratio of plasma pressure to poloidal magnetic pressure', 0.04, 'dimensionless', None, 'Input'],
        ['beta', 'Total ratio of plasma pressure to magnetic pressure', 0.04, 'dimensionless', None, 'Input'],
        ['tau_e', 'Energy confinement time', 3, 's', None, 'Input'],
        ['v_burn', 'Loop voltage during burn', 0.05, 'V', None, 'Input'],
        ['shaf_shift', 'Shafranov shift of plasma (geometric=>magnetic)', 0.5, 'm', None, 'equilibria'],
        ["C_Ejima", "Ejima constant", 0.4, "dimensionless", None, "Input (Ejima, et al., Volt-second analysis and consumption in Doublet III plasmas, Nuclear Fusion 22, 1313 (1982))"],
        ["m_s_limit", "Margin to vertical stability", 0.3, "dimensionless", None, "Input"],

        # Heating and current drive
        ['f_ni', 'Non-inductive current drive fraction', 0.1, 'dimensionless', None, 'Input'],
        ['e_nbi', 'Neutral beam energy', 1000, 'keV', None, 'Input'],
        ['P_hcd_ss', 'Steady-state HCD power', 50, 'MW', None, 'Input'],
        ['q_control', 'Control HCD power', 50, 'MW', None, 'Input'],
        ['g_cd_nb', 'NB current drive efficiency', 0.4, 'MA/MW.m', 'Check units!', 'Input'],
        ['eta_nb', 'NB electrical efficiency', 0.3, 'dimensionless', 'Check units!', 'Input'],
        ['p_nb', 'NB launcher power', 1, 'MA', 'Maximum launcher current drive in a port', 'Input'],
        ['g_cd_ec', 'EC current drive efficiency', 0.15, 'MA/MW.m', 'Check units!', 'Input'],
        ['eta_ec', 'EC electrical efficiency', 0.35, 'dimensionless', 'Check units!', 'Input'],
        ['p_ec', 'EC launcher power', 10, 'MW', 'Maximum launcher power per sector', 'Input'],
        ['f_cd_aux', 'Auxiliary current drive fraction', 0.1, 'dimensionless', None, 'Input'],
        ['f_cd_ohm', 'Ohmic current drive fraction', 0.1, 'dimensionless', None, 'Input'],
        ['TF_res_bus', 'TF Bus resistance', 0, 'm' , None, 'Input'],
        ['TF_res_tot', 'Total resistance for TF coil set', 0, 'ohm' , None, 'Input'],
        ['TF_E_stored', 'total stored energy in the toroidal field', 0, 'GJ', None, 'Input'],
        ['TF_respc_ob', 'TF coil leg resistance', 0, 'ohm', None, 'Input'],
        ['TF_currpt_ob', 'TF coil current per turn' , 0, 'A', None, 'Input'],
        ['P_bd_in', 'total auxiliary injected power' , 0, 'MW', None, 'Input'],
        ['condrad_cryo_heat', "Conduction and radiation heat loads on cryogenic components", 0, 'MW', None, 'Input'],

        # First wall profile
        ['fw_psi_n', 'Normalised psi boundary to fit FW to', 1.07, 'dimensionless', None, 'Input'],
        ['fw_dL_min', 'Minimum FW module length', 0.75, 'm', None, 'Input'],
        ['fw_dL_max', 'Maximum FW module length', 2, 'm', None, 'Input'],
        ['fw_a_max', 'Maximum angle between FW modules', 25, '°', None, 'Input'],
        ['P_sep_particle', 'Separatrix power', 150, 'MW', None, 'Input'],
        ['f_p_sol_near', 'near scrape-off layer power rate', 0.65, 'dimensionless', None, 'Input'],
        ['hf_limit', 'heat flux material limit', 0.5, 'MW/m^2', None, 'Input'],

        # SN/DN variables for heat flux transport
        ['fw_lambda_q_near_omp', 'Lambda_q near SOL omp', 0.003, 'm', None, 'Input'],
        ['fw_lambda_q_far_omp', 'Lambda_q far SOL omp', 0.1, 'm', None, 'Input'],
        ['fw_lambda_q_near_imp', 'Lambda_q near SOL imp', 0.003, 'm', None, 'Input'],
        ['fw_lambda_q_far_imp', 'Lambda_q far SOL imp', 0.1, 'm', None, 'Input'],
        ["f_lfs_lower_target", "Fraction of SOL power deposited on the LFS lower target", 0.5, "dimensionless", None, "Input"],
        ["f_hfs_lower_target", "Fraction of SOL power deposited on the HFS lower target", 0.5, "dimensionless", None, "Input"],
        ["f_lfs_upper_target", "Fraction of SOL power deposited on the LFS upper target (DN only)", 0.5, "dimensionless", None, "Input"],
        ["f_hfs_upper_target", "Fraction of SOL power deposited on the HFS upper target (DN only)", 0.5, "dimensionless", None, "Input"],

        # Divertor profile
        ['div_L2D_ib', 'Inboard divertor leg length', 1.1, 'm', None, 'Input'],
        ['div_L2D_ob', 'Outboard divertor leg length', 1.45, 'm', None, 'Input'],
        ['div_graze_angle', 'Divertor SOL grazing angle', 1.5, '°', None, 'Input'],
        ['div_psi_o', 'Divertor flux offset', 0.5, 'm', None, 'Input'],
        ['div_Ltarg', 'Divertor target length', 0.5, 'm', None, 'Input'],
        ['div_open', 'Divertor open/closed configuration', False, 'dimensionless', None, 'Input'],
        ['g_vv_div_add', 'Additional divertor/VV gap', 0, 'm', None, 'Input'],
        ['LPangle', 'Lower port inclination angle', -30, '°', None, 'Input'],
        ['psi_norm', 'Normalised flux value of strike-point contours', 1, 'dimensionless', None, 'Input'],
        ['xpt_outer_gap', 'Gap between x-point and outer wall', 2, 'm', None, 'Input'],
        ['xpt_inner_gap', 'Gap between x-point and inner wall', 0.4, 'm', None, 'Input'],
        ['tk_outer_target_sol', 'Outer target length SOL side', 0.7, 'm', None, 'Input'],
        ['tk_outer_target_pfr', 'Outer target length PFR side', 0.3, 'm', None, 'Input'],
        ['tk_inner_target_sol', 'Inner target length SOL side', 0.3, 'm', None, 'Input'],
        ['tk_inner_target_pfr', 'Inner target length PFR side', 0.5, 'm', None, 'Input'],
        # ad hoc SN variables
        ['outer_strike_h', 'Outer strike point height', 2, 'm', None, 'Input'],
        ['inner_strike_h', 'Inner strike point height', 1, 'm', None, 'Input'],
        # ad hoc DN variables
        ['outer_strike_r', 'Outer strike point major radius', 10.3, 'm', None, 'Input'],
        ['inner_strike_r', 'Inner strike point major radius', 8, 'm', None, 'Input'],
        ['theta_outer_target', 'Angle between flux line tangent at outer strike point and SOL side of outer target', 20, 'deg', None, 'Input'],
        ['theta_inner_target', 'Angle between flux line tangent at inner strike point and SOL side of inner target', 20, 'deg', None, 'Input'],
        ['xpt_height', 'x-point vertical_gap', 0.35, 'm', None, 'Input'],
        # Divertor cassette
        ['tk_div_cass', 'Minimum thickness between inner divertor profile and cassette', 0.3, 'm', None, 'Input'],
        ['tk_div_cass_in', 'Additional radial thickness on inboard side relative to to inner strike point', 0.1, 'm', None, 'Input'],
        ['n_div_cassettes', 'Number of divertor cassettes per sector', 3, 'dimensionless', None, 'Input'],


        # Blanket
        ["bb_e_mult", "Energy multiplication factor", 1.35, "dimensionless", None, "HCPB classic"],
        ['bb_min_angle', 'Minimum module angle', 70, '°', 'Sharpest cut of a module possible', 'Input (Lorenzo Boccaccini said this in a meeting in 2015, Garching, Germany)'],
        ["tk_r_ib_bz", "Thickness ratio of the inboard blanket breeding zone", 0.309, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["tk_r_ib_manifold", "Thickness ratio of the inboard blanket manifold", 0.114, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["tk_r_ib_bss", "Thickness ratio of the inboard blanket back supporting structure", 0.577, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["tk_r_ob_bz", "Thickness ratio of the outboard blanket breeding zone", 0.431, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["tk_r_ob_manifold", "Thickness ratio of the outboard blanket manifold", 0.071, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["tk_r_ob_bss", "Thickness ratio of the outboard blanket back supporting structure", 0.498, "dimensionless", None, "Input (HCPB 2015 design description document 2MHDNB)"],
        ["n_bb_inboard", "Number of inboard blanket segments", 2, "dimensionless", None, "Input"],
        ["n_bb_outboard", "Number of outboard blanket segments", 3, "dimensionless", None, "Input"],

        # ST Breeding blanket
        ['g_bb_fw', 'Separation between the first wall and the breeding blanket', 0.05, 'm', None, 'Input'],
        ['tk_bb_bz', 'Breeding zone thickness', 1.0, 'm', None, 'Input'],
        ['tk_bb_man', 'Breeding blanket manifold thickness', 0.2, 'm', None, 'Input'],

        # Component radial thicknesses (some vertical)
        ['tk_bb_ib', 'Inboard blanket thickness', 0.8, 'm', None, 'Input'],
        ['tk_bb_ob', 'Outboard blanket thickness', 1.1, 'm', None, 'Input'],
        ['tk_bb_fw', 'First wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_bb_arm', 'Tungsten armour thickness', 0.002, 'm', None, 'Input'],
        ['tk_sh_in', 'Inboard shield thickness', 1E-6, 'm', "DO NOT USE - PROCESS has VV = VV + shield", 'Input'],
        ['tk_sh_out', 'Outboard shield thickness', 1E-6, 'm', "DO NOT USE - PROCESS has VV = VV + shield", 'Input'],
        ['tk_sh_top', 'Upper shield thickness', 1E-6, 'm', "DO NOT USE - PROCESS has VV = VV + shield", 'Input'],
        ['tk_sh_bot', 'Lower shield thickness', 1E-6, 'm', "DO NOT USE - PROCESS has VV = VV + shield", 'Input'],
        ['tk_vv_in', 'Inboard vacuum vessel thickness', 0.6, 'm', None, 'Input'],
        ['tk_vv_out', 'Outboard vacuum vessel thickness', 1.1, 'm', None, 'Input'],
        ['tk_vv_top', 'Upper vacuum vessel thickness', 0.6, 'm', None, 'Input'],
        ['tk_vv_bot', 'Lower vacuum vessel thickness', 0.6, 'm', None, 'Input'],
        ['tk_sol_ib', 'Inboard SOL thickness', 0.225, 'm', None, 'Input'],
        ['tk_sol_ob', 'Outboard SOL thickness', 0.225, 'm', None, 'Input'],
        ['tk_div', 'Divertor thickness', 0.4, 'm', None, 'Input'],
        ['tk_ts', 'TS thickness', 0.05, 'm', None, 'Input'],
        ['tk_ib_ts', 'Inboard TS thickness', 0.05, 'm', None, 'Input'],
        ['tk_ob_ts', 'Outboard TS thickness', 0.05, 'm', None, 'Input'],
        ['tk_cr_vv', 'Cryostat VV thickness', 0.3, 'm', None, 'Input'],
        ['tk_rs', 'Radiation shield thickness', 2.5, 'm', None, 'Input'],
        ['tk_fw_in', 'Inboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_out', 'Outboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_div', 'First wall thickness around divertor', 0.052, 'm', None, 'Input'],

        # TF coils
        ['tk_tf_inboard', 'TF coil inboard thickness', 1, 'm', None, 'Input'],
        ['tk_tf_outboard', 'TF coil outboard thickness', 1, 'm', None, 'Input'],
        ['tk_tf_nose', 'TF coil inboard nose thickness', 0.6, 'm', None, 'Input'],
        ['tk_tf_wp', 'TF coil winding pack radial thickness', 0.5, 'm', 'Excluding insulation', 'Input'],
        ['tk_tf_wp_y', 'TF coil winding pack toroidal thickness', 0.5, 'm', 'Excluding insulation', 'Input'],
        ['tk_tf_front_ib', 'TF coil inboard steel front plasma-facing', 0.04, 'm', None, 'Input'],
        ['tk_tf_ins', 'TF coil ground insulation thickness', 0.08, 'm', None, 'Input'],
        ['tk_tf_insgap', 'TF coil WP insertion gap', 0.1, 'm', 'Backfilled with epoxy resin (impregnation). This is an average value; can be less or more due to manufacturing tolerances', 'Input'],
        ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.1, 'm', None, 'Input'],
        ["tk_tf_ob_casing", "TF leg conductor casing general thickness", 0.02, "m", None, "Input"],
        ['tk_tf_case_out_in', 'TF coil case thickness on the outboard inside', 0.35, 'm', None, 'Input'],
        ['tk_tf_case_out_out', 'TF coil case thickness on the outboard outside', 0.4, 'm', None, 'Input'],
        ['tf_wp_width', 'TF coil winding pack radial width', 0.76, 'm', 'Including insulation', 'Input'],
        ['tf_wp_depth', 'TF coil winding pack depth (in y)', 1.05, 'm', 'Including insulation', 'Input'],
        ['tk_cs', 'Central Solenoid radial thickness', 0.8, 'm', None, 'Input'],
        ["sigma_tf_case_max", "Maximum von Mises stress in the TF coil case nose", 550e6, "Pa", None, "Input"],
        ["sigma_tf_wp_max", "Maximum von Mises stress in the TF coil winding pack nose", 550e6, "Pa", None, "Input"],
        ['h_cp_top', 'Height of the TF coil inboard Tapered section end', 6., 'm', None, 'Input'],
        ['h_tf_max_in', 'Plasma side TF coil maximum height', 6.5, 'm', None, 'Input'],
        ['h_tf_min_in', 'Plasma side TF coil min height', -6.5, 'm', None, 'Input'],
        ['B_tf_peak', 'Peak field inside the TF coil winding pack', 12, 'T', None, 'Input'],
        ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'dimensionless', None, 'Input'],
        ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
        ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
        ['r_tf_curve', "Start of the upper curve of domed picture frame shale", 3., 'm', None, 'Input'],

        # PF coils
        ['r_pf_corner', 'Corner radius of the PF coil winding pack', 0.05, 'm', None, 'Input'],
        ['tk_pf_insulation', 'Thickness of the PF coil insulation', 0.05, 'm', None, 'Input'],
        ['tk_pf_casing', 'Thickness of the PF coil casing', 0.07, 'm', None, 'Input'],
        ['r_cs_corner', 'Corner radius of the CS coil winding pack', 0.05, 'm', None, 'Input'],
        ['tk_cs_insulation', 'Thickness of the CS coil insulation', 0.05, 'm', None, 'Input'],
        ['tk_cs_casing', 'Thickness of the CS coil casing', 0.07, 'm', None, 'Input'],

        # Coil structures
        ['x_g_support', 'TF coil gravity support radius', 13, 'm', None, 'Input'],
        ['w_g_support', 'TF coil gravity support width', 0.75, 'm', None, 'Input'],
        ['tk_oic', 'Outer inter-coil structure thickness', 0.3, 'm', None, 'Input'],
        ['tk_pf_support', 'PF coil support plate thickness', 0.15, 'm', None, 'Input'],
        ['gs_z_offset', 'Gravity support vertical offset', -1, 'm', None, 'Input'],
        ['h_cs_seat', 'Height of the CS support', 2, 'm', None, 'Input'],
        ['min_OIS_length', 'Minimum length of an inter-coil structure', 0.5, 'm', None, 'Input'],

        # Component radii
        ['r_cp_top', 'Radial Position of Top of TF coil taper', 0.8934, 'm', None, 'Input'],
        ['r_cs_in', 'Central Solenoid inner radius', 2.2, 'm', None, 'Input'],
        ['r_tf_in', 'Inboard radius of the TF coil inboard leg', 3.2, 'm', None, 'Input'],
        ['r_tf_inboard_out', 'Outboard Radius of the TF coil inboard leg tapered region', 0.6265, "m", None, "Input"],
        ['r_tf_in_centre', 'Inboard TF leg centre radius', 3.7, 'm', None, 'Input'],
        ['r_ts_ib_in', 'Inboard TS inner radius', 4.3, 'm', None, 'Input'],
        ['r_vv_ib_in', 'Inboard vessel inner radius', 5.1, 'm', None, 'Input'],
        ['r_fw_ib_in', 'Inboard first wall inner radius', 5.8, 'm', None, 'Input'],
        ['r_fw_ob_in', 'Outboard first wall inner radius', 12.1, 'm', None, 'Input'],
        ['r_vv_ob_in', 'Outboard vessel inner radius', 14.5, 'm', None, 'Input'],
        ['r_tf_out_centre', 'Outboard TF leg centre radius', 16.2, 'm', None, 'Input'],
        ['r_ts_joint', 'Radius of inboard/outboard TS joint', 2., 'm', None, 'Input'],
        ['r_vv_joint', 'Radius of inboard/outboard VV joint', 2., 'm', None, 'Input'],

        # Gaps and clearances
        ['g_cs_mod', 'Gap between CS modules', 0.1, 'm', None, 'Input'],
        ['g_vv_ts', 'Gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_ib_vv_ts', 'Inboard gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_ob_vv_ts', 'Outboard gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_cs_tf', 'Gap between CS and TF', 0.05, 'm', None, 'Input'],
        ['g_ts_tf', 'Gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_ib_ts_tf', 'Inboard gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_ob_ts_tf', 'Outboard gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_vv_bb', 'Gap between VV and BB', 0.02, 'm', None, 'Input'],
        ['g_tf_pf', 'Gap between TF and PF', 0.15, 'm', None, 'Input'],
        ['g_ts_pf', 'Clearances to PFs', 0.075, 'm', None, 'Input'],
        ['g_ts_tf_topbot', 'Vessel KOZ offset to TF coils on top and bottom edges', 0.11, 'm', None, 'Input'],
        ['g_cr_ts', 'Gap between the Cryostat and CTS', 0.3, 'm', None, 'Input'],
        ['g_cr_vv', 'Gap between Cryostat and VV ports', 0.2, 'm', None, 'Input'],
        ['g_cr_rs', 'Cryostat VV offset to radiation shield', 0.5, 'm', 'Distance away from edge of cryostat VV in all directions', 'Input'],
        ['c_rm', 'Remote maintenance clearance', 0.02, 'm', 'Distance between IVCs', 'Input'],

        # Offsets
        ['o_p_rs', 'Port offset from VV to RS', 0.25, 'm', None, 'Input'],
        ['o_p_cr', 'Port offset from VV to CR', 0.1, 'm', None, 'Input'],

        # Vacuum vessel
        ['vv_dtk', 'VV double-walled thickness', 0.2, 'm', None, 'Input'],
        ['vv_stk', 'VV single-walled thickness', 0.06, 'm', None, 'Input'],
        ['vvpfrac', 'Fraction of neutrons deposited in VV', 0.04, 'dimensionless', 'simpleneutrons needs a correction for VV n absorbtion', 'Input (Bachmann, probably thanks to P. Pereslavtsev)'],

        # Neutronics
        ['blk_1_dpa', 'Starter blanket life limit (EUROfer)', 20, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
        ['blk_2_dpa', 'Second blanket life limit (EUROfer)', 50, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
        ['div_dpa', 'Divertor life limit (CuCrZr)', 5, 'dpa', None, 'Input (https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf)'],
        ['vv_dpa', 'Vacuum vessel life limit (SS316-LN-IG)', 3.25, 'dpa', None, 'Input (RCC-Mx or whatever it is called)'],
        ['tf_fluence', 'Insulation fluence limit for ITER equivalent to 10 MGy', 3.2e21, '1/m^2', None, 'Input (https://ieeexplore.ieee.org/document/6374236/)'],

        # Central solenoid
        ['F_pf_zmax', 'Maximum vertical force on a single PF coil', 450, 'MN', None, 'Input'],
        ['F_cs_ztotmax', 'Maximum total vertical force in the CS stack', 350, 'MN', None, 'Input'],
        ['F_cs_sepmax', 'Maximum separation force between CS modules', 300, 'MN', None, 'Input'],
        ['CS_material', 'Conducting material to use for the CS modules', 'Nb3Sn', 'dimensionless', None, 'Input'],

        # PF magnets
        ['PF_material', 'Conducting material to use for the PF coils', 'NbTi', 'dimensionless', None, 'Input'],

        # Cryostat
        ['n_cr_lab', 'Number of cryostat labyrinth levels', 2, 'dimensionless', None, 'Input'],
        ['cr_l_d', 'Cryostat labyrinth total delta', 0.2, 'm', None, 'Input'],
        ['tk_cryo_ts', 'Cryo TS thickness', 0.10, 'm', None, 'Input'],
        ['r_cryo_ts', 'Radius of outboard cryo TS', 8, 'm', None, 'Input'],
        ['z_cryo_ts', 'Half height of outboard cryo TS', 8, 'm', None, 'Input'],

        # Radiation shield
        ['n_rs_lab', 'Number of radiation shield labyrinth levels', 4, 'dimensionless', None, 'Input'],
        ['rs_l_d', 'Radiation shield labyrinth delta', 0.6, 'm', 'Thickness of a radiation shield penetration neutron labyrinth', 'Input'],
        ['rs_l_gap', 'Radiation shield labyrinth gap', 0.02, 'm', 'Gap between plug and radiation shield', 'Input'],

        # Lifecycle
        ['n_DT_reactions', 'D-T fusion reaction rate', 7.1E20, '1/s', 'At full power', 'Input'],
        ['n_DD_reactions', 'D-D fusion reaction rate', 8.5E18, '1/s', 'At full power', 'Input'],
        ['a_min', 'Minimum operational load factor', 0.1, 'dimensionless', 'Otherwise nobody pays', 'Input'],
        ['a_max', 'Maximum operational load factor', 0.5, 'dimensionless', 'Can be violated', 'Input'],

        # Tritium fuelling and vacuum system
        ['m_gas', 'Gas puff flow rate', 50, 'Pa m^3/s', 'To maintain detachment - no chance of fusion from gas injection', 'Input (Discussions with Chris Day and Yannick Hörstensmeyer)'],

        # Maintenance
        ['bmd', 'Blanket maintenance duration', 150, 'days', 'Full replacement intervention duration', 'Input'],
        ['dmd', 'Divertor maintenance duration', 90, 'days', 'Full replacement intervention duration', 'Input'],
        ['RMTFI', 'RM Technical Feasibility Index', 1, 'dimensionless', 'Default value. Should not really be 1', 'Input'],

        # Central column shield
        ["g_ccs_vv_inboard", "Gap between central column shield and the vacuum vessel on the inboard side", 0.05, "m", None, "Input"],
        ["g_ccs_vv_add", "Additional gap between the central column shield and the vacuum vessel", 0.0, "m", None, "Input"],
        ["g_ccs_fw", "Gap between the central column shield and the first wall", 0.05, "m", None, "Input"],
        ["g_ccs_div", "Gap between the central column shield and the divertor cassette", 0.05, "m", None, "Input"],
        ["tk_ccs_min", "Minimum thickness of the central column shield", 0.1, "m", None, "Input"],
        ["r_ccs", "Outer radius of the central column shield", 2.5, "m", None, "Input"]
    ]
    # fmt: on
    ParameterFrame.set_default_parameters(params)

    def __init__(self, custom_params=None):
        super().__init__(custom_params, with_defaults=True)

    def _ck_duplicates(self):
        """
        Check there are no duplicate parameters
        """
        if not len(set(self.keys())) == len(self.keys()):
            raise KeyError("Careful: there are duplicate parameters.")


class SingleNull(Configuration):
    """
    Single null tokamak default configuration. By default the same as
    Configuration.
    """

    pass


class Spherical(Configuration):
    """
    Spherical tokamak default configuration.
    """

    new_values = {
        "A": 1.67,
        "R_0": 2.5,
        "kappa_95": 2.857,
        "kappa": 3.2,
        "delta": 0.55,
        "delta_95": 0.367,
        "q_95": 4.509,
        "n_TF": 12,
    }

    def __init__(self, custom_params=new_values):
        super().__init__(custom_params)


class DoubleNull(Configuration):
    """
    Double null tokamak default configuration.
    """

    new_values = {"plasma_type": "DN"}

    def __init__(self, custom_params=new_values):
        super().__init__(custom_params)
