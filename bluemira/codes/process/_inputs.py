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
Parameter classes/structures for Process
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Generator, List, Optional, Tuple, Union

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.process.api import _INVariable


@dataclass
class ProcessInputs:
    """
    Process Inputs dataclass

    Notes
    -----
    All entries get wrapped in an INVariable class to enable easy InDat writing.

    Units for these are available in bluemira.codes.process.mapping for mapped
    variables otherwise
    `process.io.python_fortran_dicts.get_dicts()["DICT_DESCRIPTIONS"]`
    """

    bounds: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "2": {"u": "20.0"},
            "3": {"u": "13"},
            "4": {"u": "150.0"},
            "9": {"u": "1.2"},
            "18": {"l": "3.5"},
            "29": {"l": "0.1"},
            "38": {"u": "1.0"},
            "39": {"u": "1.0"},
            "42": {"l": "0.05", "u": "0.1"},
            "50": {"u": "1.0"},
            "52": {"u": "10.0"},
            "61": {"l": "0.02"},
            "103": {"u": "10.0"},
            "60": {"l": "6.0e4", "u": "9.0e4"},
            "59": {"l": "0.50", "u": "0.94"},
        }
    )
    # fmt: off
    icc: List[int] = field(default_factory=lambda: [1, 2, 5, 8, 11, 13, 15, 16, 24, 25,
                                                    26, 27, 30, 31, 32, 33, 34, 35, 36,
                                                    60, 62, 65, 68, 72])
    ixc: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 9, 13, 14, 16, 18,
                                                    29, 36, 37, 38, 39, 41, 42, 44, 48,
                                                    49, 50, 51, 52, 53, 54, 56, 57, 58,
                                                    59, 60, 61, 102, 103, 106, 109, 110,
                                                    113, 117, 122, 123])
    # fmt: on
    abktflnc: float = 15.0
    adivflnc: float = 20.0
    alphan: float = 1.0
    alphat: float = 1.45
    alstroh: float = 660000000.0
    aspect: float = 3.1
    beta: float = 0.031421
    blnkith: float = 0.755
    blnkoth: float = 0.982
    bmxlim: float = 11.2
    bore: float = 2.3322
    bscfmax: float = 0.99
    bt: float = 5.3292
    casths: float = 0.05
    cfactr: float = 0.75
    coheof: float = 20726000.0
    coreradiationfraction: float = 0.6
    coreradius: float = 0.75
    cost_model: int = 0
    cptdin: List[float] = field(
        default_factory=lambda: [*([42200.0] * 4), *([43000.0] * 4)]
    )
    cpttf: float = 65000.0
    d_vv_bot: float = 0.6
    d_vv_in: float = 0.6
    d_vv_out: float = 1.1
    d_vv_top: float = 0.6
    ddwex: float = 0.15
    dene: float = 7.4321e19
    dhecoil: float = 0.01
    dintrt: float = 0.0
    discount_rate: float = 0.06
    divdum: int = 1
    divfix: float = 0.621
    dnbeta: float = 3.0
    thkcas: float = 0.52465
    casthi: float = 0.06
    emult: float = 1.35
    enbeam: float = 1e3
    epsvmc: float = 1e-08
    etaech: float = 0.4
    etahtp: float = 0.87
    etaiso: float = 0.9
    etanbi: float = 0.3
    etath: float = 0.375
    fbetatry: float = 0.48251
    fcap0: float = 1.15
    fcap0cp: float = 1.06
    fcohbop: float = 0.93176
    fcontng: float = 0.15
    fcr0: float = 0.065
    fcuohsu: float = 0.7
    fcutfsu: float = 0.80884
    fdene: float = 1.2
    ffuspow: float = 1.0
    fgwped: float = 0.85
    fimp: List[float] = field(
        default_factory=lambda: [1.0, 0.1, *([0.0] * 10), 0.00044, 5e-05]
    )
    fimpvar: float = 0.00037786
    fiooic: float = 0.63437
    fjohc0: float = 0.53923
    fjohc: float = 0.57941
    fjprot: float = 1.0
    fkind: float = 1.0
    fkzohm: float = 1.0245
    flhthresh: float = 1.4972
    fncycle: float = 1.0
    fne0: float = 0.9
    foh_stress: float = 1.0
    fpeakb: float = 1.0
    fpinj: float = 1.0
    fpnetel: float = 1.0
    fpsepbqar: float = 1.0
    fstrcase: float = 1.0
    fstrcond: float = 0.92007
    ftaucq: float = 0.91874
    ftaulimit: float = 1.0
    ftburn: float = 1.0
    ftmargoh: float = 1.0
    ftmargtf: float = 1.0
    fvdump: float = 1.0
    fvsbrnni: float = 0.39566
    fwalld: float = 0.131
    gamma: float = 0.3
    gamma_ecrh: float = 0.3
    gapds: float = 0.02
    gapoh: float = 0.05
    gapomin: float = 0.2
    hfact: float = 1.1
    hldivlim: float = 10.0
    i_single_null: int = 1
    i_tf_sc_mat: int = 5
    i_tf_turns_integer: int = 1
    iavail: int = 0
    ibss: int = 4
    iculbl: int = 1
    icurr: int = 4
    idensl: int = 7
    iefrf: int = 10
    ieped: int = 1
    ifalphap: int = 1
    ifispact: int = 0
    ifueltyp: int = 1
    iinvqd: int = 1
    impvar: int = 13
    inuclear: int = 1
    iohcl: int = 1
    ioptimz: int = 1
    ipedestal: int = 1
    ipfloc: List[int] = field(default_factory=lambda: [2, 2, 3, 3])
    ipowerflow: int = 0
    iprimshld: int = 1
    iprofile: int = 1
    isc: int = 34
    ishape: int = 0
    isumatoh: int = 5
    isumatpf: int = 3
    kappa: float = 1.848
    ksic: float = 1.4
    lpulse: int = 1
    lsa: int = 2
    minmax: int = 1
    n_layer: int = 10
    n_pancake: int = 20
    n_tf: int = 16
    ncls: List[int] = field(default_factory=lambda: [1, 1, 2, 2])
    neped: float = 6.78e19
    nesep: float = 2e19
    ngrp: int = 4
    oacdcp: float = 8673900.0
    oh_steel_frac: float = 0.57875
    ohcth: float = 0.55242
    ohhghf: float = 0.9
    output_costs: int = 0
    pheat: float = 50.0
    pinjalw: float = 51.0
    plasma_res_factor: float = 0.66
    pnetelin: float = 500.0
    primary_pumping: int = 3
    prn1: float = 0.4
    psepbqarmax: float = 9.2
    pulsetimings: float = 0.0
    q0: float = 1.0
    q: float = 3.5
    qnuc: float = 12920.0
    ralpne: float = 0.06894
    rhopedn: float = 0.94
    rhopedt: float = 0.94
    ripmax: float = 0.6
    rjconpf: List[float] = field(
        default_factory=lambda: [1.1e7, 1.1e7, 6e6, 6e6, 8e6, 8e6, 8e6, 8e6]
    )
    rmajor: float = 8.8901
    rpf2: float = -1.825
    scrapli: float = 0.225
    scraplo: float = 0.225
    secondary_cycle: int = 2
    shldith: float = 1e-06
    shldlth: float = 1e-06
    shldoth: float = 1e-06
    shldtth: float = 1e-06
    sig_tf_case_max: float = 580000000.0
    sig_tf_wp_max: float = 580000000.0
    ssync: float = 0.6
    tbeta: float = 2.0
    tbrnmn: float = 7200.0
    tburn: float = 10000.0
    tdmptf: float = 25.829
    tdwell: float = 0.0
    te: float = 12.33
    teped: float = 5.5
    tesep: float = 0.1
    tfcth: float = 1.208
    tftmp: float = 4.75
    tftsgap: float = 0.05
    thicndut: float = 0.002
    thshield_ib: float = 0
    thshield_ob: float = 0
    thshield_vb: float = 0
    thwcndut: float = 0.008
    tinstf: float = 0.008
    tlife: float = 40.0
    tmargmin: float = 1.5
    tramp: float = 500.0
    triang: float = 0.5
    ucblvd: float = 280.0
    ucdiv: float = 500000.0
    ucme: float = 300000000.0
    vdalw: float = 10.0
    vfshld: float = 0.6
    vftf: float = 0.3
    vgap2: float = 0.05
    vvblgap: float = 0.02
    walalw: float = 8.0
    zeffdiv: float = 3.5
    zref: List[float] = field(
        default_factory=lambda: [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    iblkt_life: int = 1
    life_dpa: float = 70.0  # Not used??
    n_cycle_min: int = 20000
    t_crack_vertical: float = 0.65e-3
    sf_vertical_crack: float = 1.0
    sf_radial_crack: float = 1.0
    sf_fast_fracture: float = 1.0
    residual_sig_hoop: float = 1.50e8
    paris_coefficient: float = 3.86e-11
    paris_power_law: float = 2.394
    walker_coefficient: float = 0.5
    fracture_toughness: float = 150.0
    m_s_limit: float = 0.2
    gap_ds: float = 0.02

    def __iter__(self) -> Generator[Tuple[str, Union[float, List, Dict]], None, None]:
        """
        Iterate over this dataclass

        The order is based on the order in which the values were
        declared.
        """
        for _field in fields(self):
            yield _field.name, getattr(self, _field.name)

    def to_invariable(self) -> Dict[str, _INVariable]:
        """
        Wrap each value in an INVariable object

        Needed for compatibility with PROCESS InDat writer
        """
        out_dict = {}
        for name, value in self:
            if name not in ["icc", "ixc", "bounds"]:
                new_val = _INVariable(name, value, "Parameter", "", "")
                out_dict[name] = new_val
        out_dict["icc"] = _INVariable(
            "icc",
            self.icc,
            "Constraint Equation",
            "Constraint Equation",
            "Constraint Equations",
        )
        out_dict["ixc"] = _INVariable(
            "ixc",
            self.ixc,
            "Iteration Variable",
            "Iteration Variable",
            "Iteration Variables",
        )
        out_dict["bounds"] = _INVariable(
            "bounds", self.bounds, "Bound", "Bound", "Bounds"
        )
        return out_dict

    def to_dict(self) -> Dict[str, Union[float, List, Dict]]:
        """
        A dictionary representation of the dataclass

        """
        return dict(self)
        return {name: value for name, value in self}


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
    "ochth": 16,
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


class PROCESSTemplateBuilder:
    """
    An API patch to make PROCESS a little easier to work with before
    the PROCESS team write a Python API.
    """

    def __init__(self):
        self.values: Dict[str, float] = {}
        self.bounds: Dict[str, Dict[str, str]] = {}
        self.icc: List[int] = []
        self.ixc: List[int] = []
        self.minmax: int = 1

    def set_minimisation_objective(self, name: str):
        """
        Set the minimisation objective equation to use when running PROCESS
        """
        minmax = OBJECTIVE_EQ_MAPPING.get(name, None)
        if not minmax:
            raise ValueError(f"There is no objective equation: '{name}'")

        self.minmax = minmax

    def set_maximisation_objective(self, name: str):
        """
        Set the maximisation objective equation to use when running PROCESS
        """
        minmax = OBJECTIVE_EQ_MAPPING.get(name, None)
        if not minmax:
            raise ValueError(f"There is no objective equation: '{name}'")
        if minmax in OBJECTIVE_MIN_ONLY:
            raise ValueError(
                f"Equation {name} can only be used as a minimisation objective."
            )
        self.minmax = -minmax

    def add_constraint(self, name: str):
        """
        Add a constraint to the PROCESS run
        """
        constraint = CONSTRAINT_EQ_MAPPING.get(name, None)
        if not constraint:
            raise ValueError(f"There is no constraint equation: '{name}'")
        if constraint in self.icc:
            bluemira_warn(f"Constraint {name} is already in the constraint list.")

        if constraint in FV_CONSTRAINT_ITVAR_MAPPING.keys():
            # Sensible (?) defaults. bounds are standard PROCESS for f-values for _most_
            # f-value constraints.
            self.add_fvalue_constraint(name, 0.5, 1e-3, 1.0)
        else:
            self.icc.append(constraint)

    def add_fvalue_constraint(
        self,
        name: str,
        value: float,
        lower_bound: float = 1e-3,
        upper_bound: float = 1.0,
    ):
        """
        Add an f-value constraint to the PROCESS run
        """
        constraint = CONSTRAINT_EQ_MAPPING.get(name, None)
        if not constraint:
            raise ValueError(f"There is no constraint equation: '{name}'")

        if constraint not in FV_CONSTRAINT_ITVAR_MAPPING.keys():
            raise ValueError(f"Constraint '{name}' is not an f-value constraint.")

        itvar = FV_CONSTRAINT_ITVAR_MAPPING[constraint]
        if itvar not in self.ixc:
            self.add_variable(
                VAR_ITERATION_MAPPING[itvar], value, lower_bound, upper_bound
            )

    def add_variable(
        self,
        name: str,
        value: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        """
        Add an iteration variable to the PROCESS run
        """
        itvar = ITERATION_VAR_MAPPING.get(name, None)
        if not itvar:
            raise ValueError(f"There is no iteration variable: '{name}'")

        if itvar in self.ixc:
            bluemira_warn(
                "Iterable variable {name} is already in the variable list. Updating value and bounds."
            )
            self.values[name] = value
            if lower_bound:
                self.bounds[str(itvar)]["l"] = lower_bound
            if upper_bound:
                self.bounds[str(itvar)]["u"] = upper_bound

        else:
            self.ixc.append(itvar)
            self.values[name] = value

        if lower_bound or upper_bound:
            var_bounds = {}
            if lower_bound:
                var_bounds["l"] = lower_bound
            if upper_bound:
                var_bounds["u"] = upper_bound
            self.bounds[str(itvar)] = var_bounds

    def make_inputs(self) -> ProcessInputs:
        """
        Make the ProcessInputs for the specified template
        """
        return ProcessInputs(
            bounds=self.bounds,
            icc=self.icc,
            ixc=self.ixc,
            ioptimz=self.minmax,
            **self.values,
        )
