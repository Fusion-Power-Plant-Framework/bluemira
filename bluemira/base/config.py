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
        ["Name", "Reactor name", "Cambridge", "dimensionless", None, "Input"],
        ["plasma_type", "Type of plasma", "SN", "dimensionless", None, "Input"],

        # Reactor
        ["A", "Plasma aspect ratio", 3.1, "dimensionless", None, "Input"],
        ["B_0", "Toroidal field at R_0", 6, "T", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["P_el_net", "Net electrical power output", 500, "MW", None, "Input"],
        ["P_el_net_process", "Net electrical power output as provided by PROCESS", None, "MW", None, "Input"],
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["tau_flattop", "Flat-top duration", 2 * 3600, "s", None, "Input"],
        ["TF_ripple_limit", "TF coil ripple limit", 0.6, "%", None, "Input"],
        ["z_0", "Vertical offset of plasma centreline", 0, "m", None, "Input"],

        # Plasma
        ["beta", "Total ratio of plasma pressure to magnetic pressure", 0.04, "dimensionless", None, "Input"],
        ["beta_N", "Normalised ratio of plasma pressure to magnetic pressure", 2.7, "dimensionless", None, "Input"],
        ["beta_p", "Ratio of plasma pressure to poloidal magnetic pressure", 0.04, "dimensionless", None, "Input"],
        ["C_Ejima", "Ejima constant", 0.4, "dimensionless", None, "Input (Ejima, et al., Volt-second analysis and consumption in Doublet III plasmas, Nuclear Fusion 22, 1313 (1982))"],
        ["delta", "Last closed surface plasma triangularity", 0.5, "dimensionless", None, "Input"],
        ["delta_95", "95th percentile plasma triangularity", 0.333, "dimensionless", None, "Input"],
        ["f_bs", "Bootstrap fraction", 0.5, "dimensionless", None, "Input"],
        ["H_star", "H factor (radiation corrected)", 1, "dimensionless", None, "Input"],
        ["I_p", "Plasma current", 19, "MA", None, "Input"],
        ["kappa", "Last closed surface plasma elongation", 1.792, "dimensionless", None, "Input"],
        ["kappa_95", "95th percentile plasma elongation", 1.6, "dimensionless", None, "Input"],
        ["l_i", "Normalised internal plasma inductance", 0.8, "dimensionless", None, "Input"],
        ["P_brehms", "Bremsstrahlung", 80, "MW", None, "Input"],
        ["P_fus", "Total fusion power", 2000, "MW", None, "Input"],
        ["P_fus_DD", "D-D fusion power", 5, "MW", None, "Input"],
        ["P_fus_DT", "D-T fusion power", 1995, "MW", None, "Input"],
        ["P_LH", "LH transition power", 0, "W", None, "Input"],
        ["P_line", "Line radiation", 30, "MW", None, "Input"],
        ["P_ohm", "Ohimic heating power", 0, "W", None, "Input"],
        ["P_rad", "Radiation power", 400, "MW", None, "Input"],
        ["P_rad_core", "Core radiation power", 0, "MW", None, "Input"],
        ["P_rad_edge", "Edge radiation power", 400, "MW", None, "Input"],
        ["P_sep", "Separatrix power", 150, "MW", None, "Input"],
        ["P_sync", "Synchrotron radiation", 50, "MW", None, "Input"],
        ["q_95", "Plasma safety factor", 3.5, "dimensionless", None, "Input"],
        ["res_plasma", "Plasma resistance", 0, "ohm", None, "Calculated"],
        ["T_e", "Average plasma electron temperature", 13, "keV", None, "Input"],
        ["T_e_ped", "Electron temperature at the pedestal", 5.5, "keV", "Used in PLASMOD if fixed temperature pedestal model used", "Input"],
        ["tau_e", "Energy confinement time", 3, "s", None, "Input"],
        ["v_burn", "Loop voltage during burn", 0.05, "V", None, "Input"],
        ["V_p", "Plasma volume", 2400, "m^3", None, "Calculated"],
        ["Z_eff", "Effective particle radiation atomic mass", 2.2, "amu", None, "Input"],

        # Heating and current drive
        ["condrad_cryo_heat", "Conduction and radiation heat loads on cryogenic components", 0, "MW", None, "Input"],
        ["e_nbi", "Neutral beam energy", 1000, "keV", None, "Input"],
        ["eta_nb", "NB electrical efficiency", 0.3, "dimensionless", "Check units!", "Input"],
        ["f_ni", "Non-inductive current drive fraction", 0.1, "dimensionless", None, "Input"],
        ["P_bd_in", "total auxiliary injected power" , 0, "MW", None, "Input"],
        ["P_hcd_ss", "Steady-state HCD power", 50, "MW", None, "Input"],
        ["P_hcd_ss_el", "Steady-state heating and current drive electrical power", 150, "MW", None, "PROCESS"],
        ["q_control", "Control HCD power", 50, "MW", None, "Input"],
        ["TF_currpt_ob", "TF coil current per turn" , 0, "A", None, "Input"],
        ["TF_E_stored", "total stored energy in the toroidal field", 0, "GJ", None, "Input"],
        ["TF_res_bus", "TF Bus resistance", 0, "m" , None, "Input"],
        ["TF_res_tot", "Total resistance for TF coil set", 0, "ohm" , None, "Input"],
        ["TF_respc_ob", "TF coil leg resistance", 0, "ohm", None, "Input"],

        # First wall profile
        ["f_p_sol_near", "near scrape-off layer power rate", 0.65, "dimensionless", None, "Input"],
        ["P_sep_particle", "Separatrix power", 150, "MW", None, "Input"],

        # SN/DN variables for heat flux transport
        ["f_hfs_lower_target", "Fraction of SOL power deposited on the HFS lower target", 0.5, "dimensionless", None, "Input"],
        ["f_hfs_upper_target", "Fraction of SOL power deposited on the HFS upper target (DN only)", 0.5, "dimensionless", None, "Input"],
        ["f_lfs_lower_target", "Fraction of SOL power deposited on the LFS lower target", 0.5, "dimensionless", None, "Input"],
        ["f_lfs_upper_target", "Fraction of SOL power deposited on the LFS upper target (DN only)", 0.5, "dimensionless", None, "Input"],
        ["fw_lambda_q_far_imp", "Lambda_q far SOL imp", 0.1, "m", None, "Input"],
        ["fw_lambda_q_far_omp", "Lambda_q far SOL omp", 0.1, "m", None, "Input"],
        ["fw_lambda_q_near_imp", "Lambda_q near SOL imp", 0.003, "m", None, "Input"],
        ["fw_lambda_q_near_omp", "Lambda_q near SOL omp", 0.003, "m", None, "Input"],

        # Component radial thicknesses (some vertical)
        ["tk_bb_ib", "Inboard blanket thickness", 0.8, "m", None, "Input"],
        ["tk_bb_ob", "Outboard blanket thickness", 1.1, "m", None, "Input"],
        ["tk_cr_vv", "Cryostat VV thickness", 0.3, "m", None, "Input"],
        ["tk_fw_in", "Inboard first wall thickness", 0.052, "m", None, "Input"],
        ["tk_fw_out", "Outboard first wall thickness", 0.052, "m", None, "Input"],
        ["tk_rs", "Radiation shield thickness", 2.5, "m", None, "Input"],
        ["tk_sh_bot", "Lower shield thickness", 1E-6, "m", "DO NOT USE - PROCESS has VV = VV + shield", "Input"],
        ["tk_sh_in", "Inboard shield thickness", 1E-6, "m", "DO NOT USE - PROCESS has VV = VV + shield", "Input"],
        ["tk_sh_out", "Outboard shield thickness", 1E-6, "m", "DO NOT USE - PROCESS has VV = VV + shield", "Input"],
        ["tk_sh_top", "Upper shield thickness", 1E-6, "m", "DO NOT USE - PROCESS has VV = VV + shield", "Input"],
        ["tk_sol_ib", "Inboard SOL thickness", 0.225, "m", None, "Input"],
        ["tk_sol_ob", "Outboard SOL thickness", 0.225, "m", None, "Input"],
        ["tk_ts", "TS thickness", 0.05, "m", None, "Input"],
        ["tk_vv_bot", "Lower vacuum vessel thickness", 0.6, "m", None, "Input"],
        ["tk_vv_in", "Inboard vacuum vessel thickness", 0.6, "m", None, "Input"],
        ["tk_vv_out", "Outboard vacuum vessel thickness", 1.1, "m", None, "Input"],
        ["tk_vv_top", "Upper vacuum vessel thickness", 0.6, "m", None, "Input"],

        # TF coils
        ["B_tf_peak", "Peak field inside the TF coil winding pack", 12, "T", None, "Input"],
        ["h_cp_top", "Height of the TF coil inboard Tapered section end", 6., "m", None, "Input"],
        ["h_tf_max_in", "Plasma side TF coil maximum height", 6.5, "m", None, "Input"],
        ["sigma_tf_case_max", "Maximum von Mises stress in the TF coil case nose", 550e6, "Pa", None, "Input"],
        ["sigma_tf_wp_max", "Maximum von Mises stress in the TF coil winding pack nose", 550e6, "Pa", None, "Input"],
        ["tf_wp_depth", "TF coil winding pack depth (in y)", 1.05, "m", "Including insulation", "Input"],
        ["tf_wp_width", "TF coil winding pack radial width", 0.76, "m", "Including insulation", "Input"],
        ["tk_cs", "Central Solenoid radial thickness", 0.8, "m", None, "Input"],
        ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.04, "m", None, "Input"],
        ["tk_tf_inboard", "TF coil inboard thickness", 1, "m", None, "Input"],
        ["tk_tf_ins", "TF coil ground insulation thickness", 0.08, "m", None, "Input"],
        ["tk_tf_insgap", "TF coil WP insertion gap", 0.1, "m", "Backfilled with epoxy resin (impregnation). This is an average value; can be less or more due to manufacturing tolerances", "Input"],
        ["tk_tf_nose", "TF coil inboard nose thickness", 0.6, "m", None, "Input"],
        ["tk_tf_outboard", "TF coil outboard thickness", 1, "m", None, "Input"],
        ["tk_tf_side", "TF coil inboard case minimum side wall thickness", 0.1, "m", None, "Input"],

        # Coil structures
        ["x_g_support", "TF coil gravity support radius", 13, "m", None, "Input"],

        # Component radii
        ["r_cp_top", "Radial Position of Top of TF coil taper", 0.8934, "m", None, "Input"],
        ["r_cs_in", "Central Solenoid inner radius", 2.2, "m", None, "Input"],
        ["r_fw_ib_in", "Inboard first wall inner radius", 5.8, "m", None, "Input"],
        ["r_fw_ob_in", "Outboard first wall inner radius", 12.1, "m", None, "Input"],
        ["r_tf_in", "Inboard radius of the TF coil inboard leg", 3.2, "m", None, "Input"],
        ["r_tf_in_centre", "Inboard TF leg centre radius", 3.7, "m", None, "Input"],
        ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.6265, "m", None, "Input"],
        ["r_tf_out_centre", "Outboard TF leg centre radius", 16.2, "m", None, "Input"],
        ["r_ts_ib_in", "Inboard TS inner radius", 4.3, "m", None, "Input"],
        ["r_vv_ib_in", "Inboard vessel inner radius", 5.1, "m", None, "Input"],
        ["r_vv_ob_in", "Outboard vessel inner radius", 14.5, "m", None, "Input"],

        # Gaps and clearances
        ["g_cr_rs", "Cryostat VV offset to radiation shield", 0.5, "m", "Distance away from edge of cryostat VV in all directions", "Input"],
        ["g_cr_ts", "Gap between the Cryostat and CTS", 0.3, "m", None, "Input"],
        ["g_cr_vv", "Gap between Cryostat and VV ports", 0.2, "m", None, "Input"],
        ["g_cs_tf", "Gap between CS and TF", 0.05, "m", None, "Input"],
        ["g_ts_pf", "Clearances to PFs", 0.075, "m", None, "Input"],
        ["g_ts_tf", "Gap between TS and TF", 0.05, "m", None, "Input"],
        ["g_vv_bb", "Gap between VV and BB", 0.02, "m", None, "Input"],
        ["g_vv_ts", "Gap between VV and TS", 0.05, "m", None, "Input"],

        # Offsets
        ["o_p_cr", "Port offset from VV to CR", 0.1, "m", None, "Input"],
        ["o_p_rs", "Port offset from VV to RS", 0.25, "m", None, "Input"],

        # Neutronics
        ["e_mult", "Energy multiplication factor", 1.35, "dimensionless", "Instantaneous energy multiplication due to neutron multiplication and the like", "Input (HCPB classic)"],

        # Equilibria
        ["B_premag_stray_max", "Maximum stray field inside the breakdown zone during premagnetisation", 0.003, "T", None, "Input"],

        # Cryostat
        ["cr_l_d", "Cryostat labyrinth total delta", 0.2, "m", None, "Input"],
        ["n_cr_lab", "Number of cryostat labyrinth levels", 2, "dimensionless", None, "Input"],
        ["tk_cryo_ts", "Cryo TS thickness", 0.10, "m", None, "Input"],

        # Radiation shield
        ["n_rs_lab", "Number of radiation shield labyrinth levels", 4, "dimensionless", None, "Input"],
        ["rs_l_d", "Radiation shield labyrinth delta", 0.6, "m", "Thickness of a radiation shield penetration neutron labyrinth", "Input"],
        ["rs_l_gap", "Radiation shield labyrinth gap", 0.02, "m", "Gap between plug and radiation shield", "Input"],

        # Tritium fuelling and vacuum system
        ["m_gas", "Gas puff flow rate", 50, "Pa m^3/s", "To maintain detachment - no chance of fusion from gas injection", "Input (Discussions with Chris Day and Yannick Hörstensmeyer)"],

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
