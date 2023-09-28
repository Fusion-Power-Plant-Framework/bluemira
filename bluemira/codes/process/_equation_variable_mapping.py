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

OBJECTIVE_EQ_MAPPING = {
    "rmajor": 1,  # major radius
    # 2 NOT USED
    "NWL": 3,  # neutron wall load
    "P_max": 4,  # P_tf + P_pf
    "Q": 5,  # fusion gain Q
    "electricity_cost": 6,  # cost of electricity
    "captial_cost": 7,  # capital cost (direct cost if ireactor=0, constructed cost otherwise)
    "aspect": 8,  # aspect ratio
    "divertor_heat_load": 9,  # divertor heat load
    "bt": 10,  # toroidal field
    "pinj": 11,  # total injected power
    # 12, 13 NOT USED
    "tpulse": 14,  # pulse length
    "avail": 15,  # plant availability factor (N.B. requires iavail=1 to be set)
    "rmajor_tpulse": 16,  # linear combination of major radius (minimised) and pulse length (maximised)
    # note: FoM should be minimised only!
    "pnetel": 17,  # net electrical output
    "NULL": 18,  # Null Figure of Merit
    "Q_tpulse": 19,  # linear combination of big Q and pulse length (maximised)
    # note: FoM should be minimised only!
}

OBJECTIVE_MIN_ONLY = [16, 19]

CONSTRAINT_EQ_MAPPING = {
    "beta_consistency": 1,  # Beta Consistency
    "global_power_consistency": 2,  # Global Power Balance Consistency
    "ion_power_consistency": 3,  # Global power balance equation for ions
    "electron_power_consistency": 4,  # Global power balance equation for electrons
    "density_upper_limit": 5,  # Density Upper Limit (Greenwald)
    "eps_beta_pol_upper_limit": 6,  # Equation for epsilon beta-poloidal upper limit
    "hot_beam_ion_density": 7,  # Equation for hot beam ion density
    "NWL_upper_limit": 8,  # Neutron wall load upper limit
    "fusioN_power_upper_limit": 9,  # Equation for fusion power upper limit
    # 10 NOT USED
    "radial_build_consistency": 11,  # Radial Build Consistency
    "vs_lower_limit": 12,  # Equation for volt-second capability lower limit
    "burn_time_lower_limit": 13,  # Burn time lower limit
    "NBI_lambda_centre": 14,  # Equation to fix number of NBI decay lengths to plasma centre
    "LH_threshhold_limit": 15,  # L-H Power Threshold Limit
    "net_electric_lower_limit": 16,  # Net electric power lower limit
    "Prad_upper_limit": 17,  # Equation for radiation power upper limit
    "divertor_heat_upper_limit": 18,  # Equation for divertor heat load upper limit
    "MVA_upper_limit": 19,  # Equation for MVA upper limit
    "NBI_tangency_upper_limit": 20,  # Equation for neutral beam tangency radius upper limit
    "aminor_lower_limit": 21,  # Equation for minor radius lower limit
    "div_coll_conn_ratio_upper_limit": 22,  # Equation for divertor collision/connection length ratio upper limit
    "cond_shell_r_aminor_upper_limit": 23,  # Equation for conducting shell radius / rminor upper limit
    "beta_upper_limit": 24,  # Beta Upper Limit
    "peak_TF_upper_limit": 25,  # Max TF field
    "CS_EOF_density_limit": 26,  # Central solenoid EOF current density upper limit
    "CS_BOP_density_limit": 27,  # Central solenoid BOP current density upper limit
    "Q_lower_limit": 28,  # Equation for fusion gain (big Q) lower limit
    "inboard_major_radius": 29,  # Equation for inboard major radius
    "Pinj_upper_limit": 30,  # Injection Power Upper Limit
    "TF_case_stress_upper_limit": 31,  # TF coil case stress upper limit
    "TF_jacket_stress_upper_limit": 32,  # TF WP steel jacket/conduit stress upper limit
    "TF_jcrit_ratio_upper_limit": 33,  # TF superconductor operating current / critical current density
    "TF_dump_voltage_upper_limit": 34,  # Dump voltage upper limit
    "TF_current_density_upper_limit": 35,  # J_winding pack
    "TF_temp_margin_lower_limit": 36,  # TF temperature marg
    "CD_gamma_upper_limit": 37,  # Equation for current drive gamma upper limit
    # 38 NOT USED
    "FW_temp_upper_limit": 39,  # Equation for first wall temperature upper limit
    "Paux_lower_limit": 40,  # Equation for auxiliary power lower limit
    "Ip_ramp_lower_limit": 41,  # Equation for plasma current ramp-up time lower limit
    "cycle_time_lower_limit": 42,  # Equation for cycle time lower limit
    "centrepost_temp_average": 43,  # Equation for average centrepost temperature
    "centrepost_temp_upper_limit": 44,  # Equation for centrepost temperature upper limit (TART)
    "qedge_lower_limit": 45,  # Equation for edge safety factor lower limit (TART)
    "Ip_Irod_upper_limit": 46,  # Equation for Ip/Irod upper limit (TART)
    "TF_toroidal_tk_upper_limit": 47,  # Equation for TF coil toroidal thickness upper limit
    "betapol_upper_limit": 48,  # Equation for poloidal beta upper limit
    # 49 SCARES ME
    "rep_rate_upper_limit": 50,  # Equation for repetition rate upper limit
    "CS_flux_consistency": 51,  # Equation to enforce startup flux = available startup flux
    "TBR_lower_limit": 52,  # Equation for tritium breeding ratio lower limit
    "nfluence_TF_upper_limit": 53,  # Equation for fast neutron fluence on TF coil upper limit
    "Pnucl_TF_upper_limit": 54,  # Equation for peak TF coil nuclear heating upper limit
    "He_VV_upper_limit": 55,  # Equation for helium concentration in vacuum vessel upper limit
    "PsepR_upper_limit": 56,  # Equation for power through separatrix / major radius upper limit
    # 57, 58 NOT USED
    "NBI_shinethrough_upper_limit": 59,  # Equation for neutral beam shine-through fraction upper limit
    "CS_temp_margin_lower_limit": 60,  # Equation for Central Solenoid s/c temperature margin lower limit
    "availability_lower_limit": 61,  # Equation for availability limit
    "confinement_ratio_lower_limit": 62,  # Lower limit on taup/taueff the ratio of alpha particle to energy confinement times
    "niterpump_upper_limit": 63,  # Upper limit on niterpump (vacuum_model = simple)
    "Zeff_upper_limit": 64,  # Upper limit on Zeff
    "dump_time_lower_limit": 65,  # Limit TF dump time to calculated quench time
    "PF_energy_rate_upper_limit": 66,  # Limit on rate of change of energy in poloidal field
    "wall_radiation_upper_limit": 67,  # Simple upper limit on radiation wall load
    "PsepBqAR_upper_limit": 68,  # Pseparatrix Bt / q A R upper limit
    "Psep_kallenbach_upper_limit": 69,  # Ensure separatrix power is less than value from Kallenbach divertor
    "tsep_consistency": 70,  # Separatrix temperature consistency
    "nsep_consistency": 71,  # Separatrix density consistency
    "CS_stress_upper_limit": 72,  # Central Solenoid Tresca yield criterion
    "Psep_LH_aux_consistency": 73,  # ensure separatrix power is greater than the L-H power + auxiliary power
    "TF_CROCO_temp_upper_limit": 74,  # ensure TF coil quench temperature < tmax_croco
    "TF_CROCO_Cu_area_constraint": 75,  # ensure that TF coil current / copper area < Maximum value ONLY used for croco HTS coil
    "eich_sep_density_constraint": 76,  # Eich critical separatrix density model
    "TF_turn_current_upper_limit": 77,  # Equation for maximum TF current per turn upper limit
    "reinke_imp_frac_lower_limit": 78,  # Equation for Reinke criterion, divertor impurity fraction lower limit
    "Bmax_CS_upper_limit": 79,  # Equation for maximum CS field
    "pdivt_lower_limit": 80,  # Lower limit pdivt
    "density_profile_sanity": 81,  # ne(0) > ne(ped) constraint
    "stellarator_coil_consistency": 82,  # Constraint equation making sure that stellarator coils dont touch in toroidal direction
    "stellarator_radial_build_consistency": 83,  # Constraint ensuring radial build consistency for stellarators
    "beta_lower_limit": 84,  # Constraint for lower limit of beta
    "CP_lifetime_lower_limit": 85,  # Constraint for CP lifetime
    "turn_size_upper_limit": 86,  # Constraint for turn dimension
    "cryopower_upper_limit": 87,  # Constraint for cryogenic power
    "TF_strain_upper_limit": 88,  # Constraint for TF coil strain
    "OH_CROCO_Cu_area_constraint": 89,  # ensure that OH coil current / copper area < Maximum value ONLY used for croco HTS coil
    "CS_fatigue": 90,  # CS fatigue constraints
    "ECRH_ignitability": 91,  # Constraint for indication of ECRH ignitability
}

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
