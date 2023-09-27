# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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


class PlasmaGeometryModel(Model):
    """
    Switch for plasma geometry

    PROCESS variable name: "ishape"
    """

    HENDER_K_D_100 = 0
    GALAMBOS_K_D_95 = 1
    ZOHM_ITER = 2
    ZOHM_ITER_D_95 = 3
    HENDER_K_D_95 = 4
    MAST_95 = 5
    MAST_100 = 6
    FIESTA_95 = 7
    FIESTA_100 = 8


class PlasmaProfileModel(Model):
    """
    Switch for plasma profile model

    PROCESS variable name: "ipedestal"
    """

    NO_PEDESTAL = 0
    PEDESTAL_GW = 1
    PLASMOD_GW = 2
    PLASMOD = 3


class BetaLimitModel(Model):
    """
    Switch for the plasma beta limit model

    PROCESS variable name: "iculbl"
    """

    TOTAL = 0  # Including fast ion contribution
    THERMAL = 1
    THERMAL_NBI = 2
    TOTAL_TF = 3  # Calculated using only the toroidal field


class BetaGScalingModel(Model):
    """
    Switch for the beta g coefficient dnbeta model

    PROCESS variable name: "gtscale"

    NOTE: Over-ridden if iprofile = 1
    """

    INPUT = 0  # dnbeta is an input
    CONVENTIONAL = 1
    MENARD_ST = 2


class AlphaPressureModel(Model):
    """
    Switch for the pressure contribution from fast alphas

    PROCESS variable name: "ifalphap"
    """

    HENDER = 0
    LUX = 1


class DensityLimitModel(Model):
    """
    Switch for the density limit model

    PROCESS variable name: "idensl"
    """

    ASDEX = 1
    BORRASS_ITER_I = 2
    BORRASS_ITER_II = 3
    JET_RADIATION = 4
    JET_SIMPLE = 5
    HUGILL_MURAKAMI = 6
    GREENWALD = 7


class PlasmaCurrentScalingLaw(Model):
    """
    Switch for plasma current scaling law

    PROCESS variable name: "icurr"
    """

    PENG = 1
    PENG_DN = 2
    ITER_SIMPLE = 3
    ITER_REVISED = 4  # Recommended for iprofile = 1
    TODD_I = 5
    TODD_II = 6
    CONNOR_HASTIE = 7
    SAUTER = 8
    FIESTA = 9


class ConfinementTimeScalingLaw(Model):
    """
    Switch for the energy confinement time scaling law

    PROCESS variable name: "isc"
    """

    NEO_ALCATOR_OHMIC = 1
    MIRNOV_H_MODE = 2
    MEREZHKIN_MUHKOVATOV_L_MODE = 3
    SHIMOMURA_H_MODE = 4
    KAYE_GOLDSTON_L_MODE = 5
    ITER_89_P_L_MODE = 6
    ITER_89_O_L_MODE = 7
    REBUT_LALLIA_L_MODE = 8
    GOLDSTON_L_MODE = 9
    T10_L_MODE = 10
    JAERI_88_L_MODE = 11
    KAYE_BIG_COMPLEX_L_MODE = 12
    ITER_H90_P_H_MODE = 13
    ITER_MIX = 14  # Minimum of 6 and 7
    RIEDEL_L_MODE = 15
    CHRISTIANSEN_L_MODE = 16
    LACKNER_GOTTARDI_L_MODE = 17
    NEO_KAYE_L_MODE = 18
    RIEDEL_H_MODE = 19
    ITER_H90_P_H_MODE_AMENDED = 20
    LHD_STELLARATOR = 21
    GRYO_RED_BOHM_STELLARATOR = 22
    LACKNER_GOTTARDI_STELLARATOR = 23
    ITER_93H_H_MODE = 24
    TITAN_RFP = 25
    ITER_H97_P_NO_ELM_H_MODE = 26
    ITER_H97_P_ELMY_H_MODE = 27
    ITER_96P_L_MODE = 28
    VALOVIC_ELMY_H_MODE = 29
    KAYE_PPPL98_L_MODE = 30
    ITERH_PB98P_H_MODE = 31
    IPB98_Y_H_MODE = 32
    IPB98_Y1_H_MODE = 33
    IPB98_Y2_H_MODE = 34
    IPB98_Y3_H_MODE = 35
    IPB98_Y4_H_MODE = 36
    ISS95_STELLARATOR = 37
    ISS04_STELLARATOR = 38
    DS03_H_MODE = 39
    MURARI_H_MODE = 40
    PETTY_H_MODE = 41
    LANG_H_MODE = 42
    HUBBARD_NOM_I_MODE = 43
    HUBBARD_LOW_I_MODE = 44
    HUBBARD_HI_I_MODE = 45
    NSTX_H_MODE = 46
    NSTX_PETTY_H_MODE = 47
    NSTX_GB_H_MODE = 48
    INPUT = 49  # tauee_in


class BootstrapCurrentScalingLaw(Model):
    """
    Switch for the model to calculate bootstrap fraction

    PROCESS variable name: "ibss"
    """

    ITER = 1
    GENERAL = 2
    NUMERICAL = 3
    SAUTER = 4


class LHThreshholdScalingLaw(Model):
    """
    Switch for the model to calculate the L-H power threshhold

    PROCESS variable name: "ilhthresh"
    """

    ITER_1996_NOM = 1
    ITER_1996_LOW = 2
    ITER_1996_HI = 3
    ITER_1997 = 4
    ITER_1997_K = 5
    MARTIN_NOM = 6
    MARTIN_HI = 7
    MARTIN_LOW = 8
    SNIPES_NOM = 9
    SNIPES_HI = 10
    SNIPES_LOW = 11
    SNIPES_CLOSED_DIVERTOR_NOM = 12
    SNIPES_CLOSED_DIVERTOR_HI = 13
    SNIPES_CLOSED_DIVERTOR_LOW = 14
    HUBBARD_LI_NOM = 15
    HUBBARD_LI_HI = 16
    HUBBARD_LI_LOW = 17
    HUBBARD_2017_LI = 18
    MARTIN_ACORRECT_NOM = 19
    MARTIN_ACORRECT_HI = 20
    MARTIN_ACORRECT_LOW = 21


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
    CRYO_AL = 2


IN_mappings = {
    "P_el_net": ("pnetelin", "MW"),
    "n_TF": ("n_tf", "dimensionless"),
    "TF_ripple_limit": ("ripmax", "%"),
    "C_Ejima": ("gamma", "dimensionless"),
    "e_nbi": ("enbeam", "keV"),
    "P_hcd_ss": ("pinjalw", "MW"),
    "eta_nb": ("etanbi", "dimensionless"),
    "e_mult": ("emult", "dimensionless"),
    "tk_sh_out": ("shldoth", "m"),
    "tk_sh_top": ("shldtth", "m"),
    "tk_sh_bot": ("shldlth", "m"),
    "tk_vv_out": ("d_vv_out", "m"),
    "tk_vv_top": ("d_vv_top", "m"),
    "tk_vv_bot": ("d_vv_bot", "m"),
    "tk_cr_vv": ("ddwex", "m"),
    "tk_tf_front_ib": ("casthi", "m"),
    "tk_tf_side": ("casths", "m"),
    "PsepB_qAR_max": ("psepbqarmax", "MW.T/m"),
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
    "P_rad_core": ("pinnerzoneradmw", "MW"),
    "P_rad_edge": ("pouterzoneradmw", "MW"),
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
    "condrad_cryo_heat": ("qss/1.0d6", "MW"),
}

IO_mappings = {
    "A": ("aspect", "dimensionless"),
    "tk_bb_ib": ("blnkith", "m"),
    "tk_bb_ob": ("blnkoth", "m"),
    "tk_sh_in": ("shldith", "m"),
    "tk_vv_in": ("d_vv_in", "m"),
    "tk_sol_ib": ("scrapli", "m"),
    "tk_sol_ob": ("scraplo", "m"),
    "tk_ts": ("thshield_ob", "m"),
    "tk_ts": ("thshield_vb", "m"),
    "tk_ts": ("thshield_ib", "m"),
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
