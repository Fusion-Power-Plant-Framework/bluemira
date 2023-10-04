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
Death to PROCESS integers
"""

from bluemira.codes.utilities import Model


class Objective(Model):
    """
    Enum for PROCESS optimisation objective
    """

    MAJOR_RADIUS = 1
    # 2 NOT USED
    NEUTRON_WALL_LOAD = 3
    MAX_COIL_POWER = 4
    FUSION_GAIN = 5
    ELECTRICITY_COST = 6
    CAPITAL_COST = 7
    ASPECT_RATIO = 8
    DIVERTOR_HEAT_LOAD = 9
    TOROIDAL_FIELD = 10
    INJECTED_POWER = 11
    # 12, 13 NOT USED
    PULSE_LENGTH = 14
    AVAILABILITY = 15
    MAJOR_RADIUS_PULSE_LENGTH = 16
    NET_ELECTRICITY = 17
    NULL = 18
    FUSION_GAIN_PULSE_LENGTH = 19


OBJECTIVE_MIN_ONLY = (16, 19)


class Constraint(Model):
    """
    Enum for PROCESS constraints
    """

    BETA_CONSISTENCY = 1
    GLOBAL_POWER_CONSISTENCY = 2
    ION_POWER_CONSISTENCY = 3
    ELECTRON_POWER_CONSISTENCY = 4
    DENSITY_UPPER_LIMIT = 5
    EPS_BETA_POL_UPPER_LIMIT = 6
    HOT_BEAM_ION_DENSITY = 7
    NWL_UPPER_LIMIT = 8
    FUSION_POWER_UPPER_LIMIT = 9
    # 10 NOT USED
    RADIAL_BUILD_CONSISTENCY = 11
    VS_LOWER_LIMIT = 12
    BURN_TIME_LOWER_LIMIT = 13
    NBI_LAMBDA_CENTRE = 14
    LH_THRESHHOLD_LIMIT = 15
    NET_ELEC_UPPER_LIMIT = 16
    RAD_POWER_UPPER_LIMIT = 17
    DIVERTOR_HEAT_UPPER_LIMIT = 18
    MVA_UPPER_LIMIT = 19
    NBI_TANGENCY_UPPER_LIMIT = 20
    AMINOR_LOWER_LIMIT = 21
    DIV_COLL_CONN_UPPER_LIMIT = 22
    COND_SHELL_R_RATIO_UPPER_LIMIT = 23
    BETA_UPPER_LIMIT = 24
    PEAK_TF_UPPER_LIMIT = 25
    CS_EOF_DENSITY_LIMIT = 26
    CS_BOP_DENSITY_LIMIT = 27
    Q_LOWER_LIMIT = 28
    IB_RADIAL_BUILD_CONSISTENCY = 29
    PINJ_UPPER_LIMIT = 30
    TF_CASE_STRESS_UPPER_LIMIT = 31
    TF_JACKET_STRESS_UPPER_LIMIT = 32
    TF_JCRIT_RATIO_UPPER_LIMIT = 33
    TF_DUMP_VOLTAGE_UPPER_LIMIT = 34
    TF_CURRENT_DENSITY_UPPER_LIMIT = 35
    TF_T_MARGIN_LOWER_LIMIT = 36
    CD_GAMMA_UPPER_LIMIT = 37
    # 38 NOT USED
    FW_TEMP_UPPER_LIMIT = 39
    PAUX_LOWER_LIMIT = 40
    IP_RAMP_LOWER_LIMIT = 41
    CYCLE_TIME_LOWER_LIMIT = 42
    CENTREPOST_TEMP_AVERAGE = 43
    CENTREPOST_TEMP_UPPER_LIMIT = 44
    QEDGE_LOWER_LIMIT = 45
    IP_IROD_UPPER_LIMIT = 46
    TF_TOROIDAL_TK_UPPER_LIMIT = 47  # 47 NOT USED (or maybe it is, WTF?!)
    BETAPOL_UPPER_LIMIT = 48
    # 49 NOT USED  / SCARES ME
    REP_RATE_UPPER_LIMIT = 50
    CS_FLUX_CONSISTENCY = 51
    TBR_LOWER_LIMIT = 52
    NFLUENCE_TF_UPPER_LIMIT = 53
    PNUCL_TF_UPPER_LIMIT = 54
    HE_VV_UPPER_LIMIT = 55
    PSEPR_UPPER_LIMIT = 56
    # 57, 58 NOT USED
    NBI_SHINETHROUGH_UPPER_LIMIT = 59
    CS_T_MARGIN_LOWER_LIMIT = 60
    AVAIL_LOWER_LIMIT = 61
    CONFINEMENT_RATIO_LOWER_LIMIT = 62
    NITERPUMP_UPPER_LIMIT = 63
    ZEFF_UPPER_LIMIT = 64
    DUMP_TIME_LOWER_LIMIT = 65
    PF_ENERGY_RATE_UPPER_LIMIT = 66
    WALL_RADIATION_UPPER_LIMIT = 67
    PSEPB_QAR_UPPER_LIMIT = 68
    PSEP_KALLENBACH_UPPER_LIMIT = 69
    TSEP_CONSISTENCY = 70
    NSEP_CONSISTENCY = 71
    CS_STRESS_UPPER_LIMIT = 72
    PSEP_LH_AUX_CONSISTENCY = 73
    TF_CROCO_T_UPPER_LIMIT = 74
    TF_CROCO_CU_AREA_CONSTRAINT = 75
    EICH_SEP_DENSITY_CONSTRAINT = 76
    TF_TURN_CURRENT_UPPER_LIMIT = 77
    REINKE_IMP_FRAC_LOWER_LIMIT = 78
    BMAX_CS_UPPER_LIMIT = 79
    PDIVT_LOWER_LIMIT = 80
    DENSITY_PROFILE_CONSISTENCY = 81
    STELLARATOR_COIL_CONSISTENCY = 82
    STELLARATOR_RADIAL_BUILD_CONSISTENCY = 83
    BETA_LOWER_LIMIT = 84
    CP_LIFETIME_LOWER_LIMIT = 85
    TURN_SIZE_UPPER_LIMIT = 86
    CRYOPOWER_UPPER_LIMIT = 87
    TF_STRAIN_UPPER_LIMIT = 88
    OH_CROCO_CU_AREA_CONSTRAINT = 89
    CS_FATIGUE = 90
    ECRH_IGNITABILITY = 91


# The dreaded f-values
FV_CONSTRAINT_ITVAR_MAPPING = {
    5: 9,
    6: 8,
    8: 14,
    9: 26,
    12: 15,
    13: 21,
    15: 103,
    16: 25,
    17: 28,
    18: 27,
    19: 30,
    20: 33,
    21: 32,
    22: 34,
    23: 104,
    24: 36,
    25: 35,
    26: 38,
    27: 39,
    28: 45,
    30: 46,
    31: 48,
    32: 49,
    33: 50,
    34: 51,
    35: 53,
    36: 54,
    37: 40,
    38: 62,
    39: 63,
    40: 64,
    41: 66,
    42: 67,
    44: 68,
    45: 71,
    46: 72,
    48: 79,
    50: 86,
    52: 89,
    53: 92,
    54: 95,
    55: 96,
    56: 97,
    59: 105,
    60: 106,
    61: 107,
    62: 110,
    63: 111,
    64: 112,
    65: 113,
    66: 115,
    67: 116,
    68: 117,
    69: 118,
    73: 137,
    74: 141,
    75: 143,
    76: 144,
    77: 146,
    78: 146,
    83: 160,
    84: 161,
    89: 166,
    91: 168,
}

ITERATION_VAR_MAPPING = {
    "aspect": 1,
    "bt": 2,
    "rmajor": 3,
    "te": 4,
    "beta": 5,
    "dene": 6,
    "rnbeam": 7,
    "fbeta": 8,
    "fdene": 9,
    "hfact": 10,
    "pheat": 11,
    "oacdp": 12,
    "tfcth": 13,
    "fwalld": 14,
    "fvs": 15,
    "ohcth": 16,
    "tdwell": 17,
    "q": 18,
    "enbeam": 19,
    "tcpav": 20,
    "ftburn": 21,
    # 22 NOT USED
    "fcoolcp": 23,
    # 24 NOT USED
    "fpnetel": 25,
    "ffuspow": 26,
    "fhldiv": 27,
    "fradpwr": 28,
    "bore": 29,
    "fmva": 30,
    "gapomin": 31,
    "frminor": 32,
    "fportsz": 33,
    "fdivcol": 34,
    "fpeakb": 35,
    "fbetatry": 36,
    "coheof": 37,
    "fjohc": 38,
    "fjohc0": 39,
    "fgamcd": 40,
    "fcohbop": 41,
    "gapoh": 42,
    # 43 NOT USED
    "fvsbrnni": 44,
    "fqval": 45,
    "fpinj": 46,
    "feffcd": 47,
    "fstrcase": 48,
    "fstrcond": 49,
    "fiooic": 50,
    "fvdump": 51,
    "vdalw": 52,
    "fjprot": 53,
    "ftmargtf": 54,
    # 55 NOT USED
    "tdmptf": 56,
    "thkcas": 57,
    "thwcndut": 58,
    "fcutfsu": 59,
    "cpttf": 60,
    "gapds": 61,
    "fdtmp": 62,
    "ftpeak": 63,
    "fauxmn": 64,
    "tohs": 65,
    "ftohs": 66,
    "ftcycl": 67,
    "fptemp": 68,
    "rcool": 69,
    "vcool": 70,
    "fq": 71,
    "fipir": 72,
    "scrapli": 73,
    "scraplo": 74,
    "tfootfi": 75,
    # 76, 77, 78 NOT USED
    "fbetap": 79,
    # 80 NOT USED
    "edrive": 81,
    "drveff": 82,
    "tgain": 83,
    "chrad": 84,
    "pdrive": 85,
    "frrmax": 86,
    # 87, 88 NOT USED
    "ftbr": 89,
    "blbuith": 90,
    "blbuoth": 91,
    "fflutf": 92,
    "shldith": 93,
    "shldoth": 94,
    "fptfnuc": 95,
    "fvvhe": 96,
    "fpsepr": 97,
    "li6enrich": 98,
    # 99, 100, 101 NOT USED
    "fimpvar": 102,
    "flhthresh": 103,
    "fcwr": 104,
    "fnbshinef": 105,
    "ftmargoh": 106,
    "favail": 107,
    "breeder_f": 108,
    "ralpne": 109,
    "ftaulimit": 110,
    "fniterpump": 111,
    "fzeffmax": 112,
    "ftaucq": 113,
    "fw_channel_length": 114,
    "fpoloidalpower": 115,
    "fradwall": 116,
    "fpsepbqar": 117,
    "fpsep": 118,
    "tesep": 119,
    "ttarget": 120,
    "neratio": 121,
    "oh_steel_frac": 122,
    "foh_stress": 123,
    "qtargettotal": 124,
    "fimp(3)": 125,  # Beryllium
    "fimp(4)": 126,  # Carbon
    "fimp(5)": 127,  # Nitrogen
    "fimp(6)": 128,  # Oxygen
    "fimp(7)": 129,  # Neon
    "fimp(8)": 130,  # Silicon
    "fimp(9)": 131,  # Argon
    "fimp(10)": 132,  # Iron
    "fimp(11)": 133,  # Nickel
    "fimp(12)": 134,  # Krypton
    "fimp(13)": 135,  # Xenon
    "fimp(14)": 136,  # Tungsten
    "fplhsep": 137,
    "rebco_thickness": 138,
    "copper_thick": 139,
    "dr_tf_wp": 140,  # TODO: WTF
    "fcqt": 141,
    "nesep": 142,
    "f_coppera_m2": 143,
    "fnesep": 144,
    "fgwped": 145,
    "fcpttf": 146,
    "freinke": 147,
    "fzactual": 148,
    "fbmaxcs": 149,
    # 150, 151 NOT USED
    "fgwsep": 152,
    "fpdivlim": 153,
    "fne0": 154,
    "pfusife": 155,
    "rrin": 156,
    "fvssu": 157,
    "croco_thick": 158,
    "ftoroidalgap": 159,
    "f_avspace": 160,
    "fbetatry_lower": 161,
    "r_cp_top": 162,
    "f_t_turn_tf": 163,
    "f_crypmw": 164,
    "fstr_wp": 165,
    "f_copperaoh_m2": 166,
    "fncycle": 167,
    "fecrh_ignition": 168,
    "te0_ecrh_achievable": 169,
    "beta_div": 170,
}


VAR_ITERATION_MAPPING = {v: k for k, v in ITERATION_VAR_MAPPING.items()}
