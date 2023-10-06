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
from typing import Dict, Generator, List, Tuple, Union

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
    # Settings
    maxcal: int = 1000
    minmax: int = 1
    epsvmc: float = 1e-08
    ioptimz: int = 1
    output_costs: int = 0

    # Top down of PROCESS variables list

    # Times
    tburn: float = 1000.0
    tdwell: float = 1800.0
    theat: float = 10.0
    tohs: float = 30.0
    tqnch: float = 15.0
    tramp: float = 15.0

    # FWBS
    ibkt_life: int = 1
    denstl: float = 7800.0
    denw: float = 19250.0
    emult: float = 1.269
    fblss: float = 0.09705
    fdiv: float = 0.115
    fwbsshape: int = 2
    fw_armour_thickness: float = 0.005
    iblanket: int = 1
    iblnkith: int = 1
    li6enrich: float = 30.0
    breeder_f: float = 0.5
    breeder_multiplier: float = 0.75
    vfcblkt: float = 0.05295
    vfpblkt: float = 0.1
    blktmodel: int = 1  # Listed as an output...
    # f_neut_shield: float = # -1.0 the documentation defaults cannot be right...
    breedmat: int = 1
    fblbe: float = 0.6
    fblbreed: float = 0.154
    fblhebmi: float = 0.4
    fblhebmo: float = 0.4
    fblhebpi: float = 0.6595
    fblhebpo: float = 0.6713
    hcdportsize: int = 1
    npdiv: int = 2
    nphcdin: int = 2
    nphcdout: int = 2
    wallpf: float = 1.21
    iblanket_thickness: int = 2
    secondary_cycle: int = 2  # Listed as an output...
    secondary_cycle_liq: int = 4
    afwi: float = 0.008
    afwo: float = 0.008
    fw_wall: float = 0.003
    afw: float = 0.006
    pitch: float = 0.02
    fwinlet: float = 573.0
    fwoutlet: float = 823.0
    fwpressure: float = 15500000.0
    roughness: float = 1.0e-6
    fw_channel_length: float = 4.0
    peaking_factor: float = 1.0
    blpressure: float = 15500000.0
    inlet_temp: float = 573.0
    outlet_temp: float = 823.0
    coolp: float = 15500000.0
    nblktmodpo: int = 8
    nblktmodpi: int = 7
    nblktmodto: int = 48
    nblktmodti: int = 32
    tfwmatmax: float = 823.0
    fw_th_conductivity: float = 28.34
    fvoldw: float = 1.74
    fvolsi: float = 1.0
    fvolso: float = 0.64
    fwclfr: float = 0.15
    rpf2dewar: float = 0.5
    vfshld: float = 0.25
    irefprop: int = 1
    fblli2o: float = 0.08
    fbllipb: float = 0.68
    vfblkt: float = 0.25
    declblkt: float = 0.075
    declfw: float = 0.075
    declshld: float = 0.075
    blkttype: int = 3
    etaiso: float = 0.85
    etahtp: float = 0.95
    n_liq_recirc: int = 10
    bz_channel_conduct_liq: float = 833000.0
    blpressure_liq: float = 1700000.0
    inlet_temp_liq: float = 570.0
    outlet_temp_liq: float = 720.0
    f_nuc_pow_bz_struct: float = 0.34
    pnuc_fw_ratio_dcll: float = 0.14

    # TF coil
    sig_tf_case_max: float = 600000000.0
    sig_tf_wp_max: float = 600000000.0
    bcritsc: float = 24.0
    casthi_fraction: float = 0.05
    casths_fraction: float = 0.06
    f_t_turn_tf: float = 1.0
    t_turn_tf_max: float = 0.05000000074505806
    cpttf: float = 70000.0
    cpttf_max: float = 90000.0
    dcase: float = 8000.0
    dcond: List[float] = field(
        default_factory=lambda: [
            6080.0,
            6080.0,
            6070.0,
            6080.0,
            6080.0,
            8500.0,
            6070.0,
            8500.0,
            8500.0,
        ]
    )
    dcondins: float = 1800.0
    dhecoil: float = 0.005
    farc4tf: float = 0.7
    b_crit_upper_nbti: float = 14.86
    t_crit_nbti: float = 9.04
    fcutfsu: float = 0.69
    fhts: float = 0.5
    i_tf_stress_model: int = 1
    i_tf_wp_geom: int = -1
    i_tf_case_geom: int = 1  # Listed as an output
    i_tf_turns_integer: int = 1  # Listed as an output
    i_tf_sc_mat: int = 1
    i_tf_sup: int = 1
    i_tf_shape: int = 1  # Listed as an output
    i_tf_cond_eyoung_trans: int = 1
    n_pancake: int = 20
    n_layer: int = 10
    n_rad_per_layer: int = 100
    i_tf_bucking = -1
    n_tf_graded_layers: int = 1
    jbus: float = 1250000.0
    eyoung_ins: float = 100000000.0
    eyoung_steel: float = 205000000000.0
    eyong_cond_axial: float = 660000000.0
    eyoung_res_tf_buck: float = 150000000000.0
    # eyoung_al: float = 69000000000.0 # defaults  cannot be right
    poisson_steel: float = 0.3
    poisson_copper: float = 0.35
    poisson_al: float = 0.35
    str_cs_con_res: float = -0.005
    str_pf_con_res: float = -0.005
    str_tf_con_res: float = -0.005
    str_wp_max: float = 0.007
    i_str_wp: int = 1
    quench_model: str = b"exponential"
    tcritsc: float = 16.0
    tdmptf: float = 10.0
    tfinsgap: float = 0.01
    # rhotfbus: float = -1.0 # defaults cannot be right
    frhocp: float = 1.0
    frholeg: float = 1.0
    # i_cp_joints: int = -1 # defaults cannot be right
    rho_tf_joints: float = 2.5e-10
    n_tf_joints_contact: int = 6
    n_tf_joints: int = 4
    th_joint_contact: float = 0.03
    # eff_tf_cryo: float = -1.0 # defaults cannot be right
    n_tf: int = 16
    tftmp: float = 4.75
    thicndut: float = 0.0008
    thkcas: float = 0.3
    thwcndut: float = 0.008
    tinstf: float = 0.018
    tmaxpro: float = 150.0
    tmax_croco: float = 200.0
    tmpcry: float = 4.5
    vdalw: float = 20.0
    f_vforce_inboard: float = 0.5
    vftf: float = 0.4
    etapump: float = 0.8
    fcoolcp: float = 0.3
    fcoolleg: float = 0.2
    ptempalw: float = 473.15
    rcool: float = 0.005
    tcoolin: float = 313.15
    tcpav: float = 373.15
    vcool: float = 20.0
    theta1_coil: float = 45.0
    theta1_vv: float = 1.0
    max_vv_stress: float = 143000000.0
    inuclear: int = 1
    qnuc: float = 12920.0
    ripmax: float = 1.0
    tf_in_cs: int = 0
    tfcth: float = 1.208
    tftsgap: float = 0.05
    casthi: float = 0.06
    casths: float = 0.05
    tmargmin: float = 1.5

    # PF Power
    iscenr: int = 2
    maxpoloidalpower: float = 1000.0

    # Cost variables
    abktflnc: float = 5.0
    adivflnc: float = 7.0
    cconfix: float = 80.0
    cconshpf: float = 70.0
    cconshtf: float = 75.0
    cfactr: float = 0.75
    cfind: List[float] = field(default_factory=lambda: [0.244, 0.244, 0.244, 0.29])
    cland: float = 19.2
    costexp: float = 0.8
    costexp_pebbles: float = 0.6
    cost_factor_buildings: float = 1.0
    cost_factor_land: float = 1.0
    cost_factor_tf_coils: float = 1.0
    cost_factor_fwbs: float = 1.0
    cost_factor_tf_rh: float = 1.0
    cost_factor_tf_vv: float = 1.0
    cost_factor_tf_bop: float = 1.0
    cost_factor_tf_misc: float = 1.0
    maintenance_fwbs: float = 0.2
    maintenance_gen: float = 0.05
    amortization: float = 13.6
    cost_model: int = 1
    cowner: float = 0.15
    cplife_input: float = 2.0
    cpstflnc: float = 10.0
    csi: float = 16.0
    # cturbb: float = 38.0 # defaults cannot be right
    decomf: float = 0.1
    fcap0: float = 1.165
    fcap0cp: float = 1.08
    fcdfuel: float = 0.1
    fcontng: float = 0.195
    fcr0: float = 0.0966
    fkind: float = 1.0
    iavail: int = 2
    life_dpa: float = 50.0
    avail_min: float = 0.75
    favail: float = 1.0
    num_rh_systems: int = 4
    conf_mag: float = 0.99
    div_prob_fail: float = 0.0002
    div_umain_time: float = 0.25
    div_nref: float = 7000.0
    div_nu: float = 14000.0
    fwbs_nref: float = 20000.0
    fwbs_nu: float = 40000.0
    fwbs_prob_fail: float = 0.0002
    fwbs_umain_time: float = 0.25
    redun_vacp: float = 25.0
    tbktrepl: float = 0.5
    tcomrepl: float = 0.5
    tdivrepl: float = 0.25
    uubop: float = 0.02
    uucd: float = 0.02
    uudiv: float = 0.04
    uufuel: float = 0.02
    uufw: float = 0.04
    uumag: float = 0.02
    uuves: float = 0.04
    ifueltyp: int = 1
    ucblvd: float = 280.0
    ucdiv: float = 500000.0
    ucme: float = 300000000.0
    ireactor: int = 1
    lsa: int = 4
    discount_rate: float = 0.0435
    startupratio: float = 1.0
    tlife: float = 30.0
    # ...

    # CS fatigue
    residual_sig_hoop: float = 240000000.0
    n_cycle_min: int = 20000
    t_crack_vertical: float = 0.00089
    t_crack_radial: float = 0.006
    t_structural_radial: float = 0.07
    t_structural_vertical: float = 0.022
    sf_vertical_crack: float = 2.0
    sf_radial_crack: float = 2.0
    sf_fast_fracture: float = 1.5
    paris_coefficient: float = 6.5e-13
    paris_power_law: float = 3.5
    walker_coefficient: float = 0.436
    fracture_toughness: float = 200.0

    # REBCO
    rebco_thickness: float = 1e-6
    copper_thick: float = 0.0001
    hastelloy_thickness: float = 5e-5
    tape_width: float = 0.004
    tape_thickness: float = 6.5e-5
    croco_thick: float = 0.0025
    copper_rrr: float = 100.0
    copper_m2_max: float = 100000000.0
    f_coppera_m2: float = 1.0
    copperaoh_m2_max: float = 100000000.0
    f_copperaoh_m2: float = 1.0

    # Primary pumping
    primary_pumping: int = 2
    gamma_he: float = 1.667
    t_in_bb: float = 573.13
    t_out_bb: float = 773.13
    p_he: float = 8000000.0
    dp_he: float = 550000.0

    # Constraint variables
    auxmin: float = 0.1
    betpmx: float = 0.19
    bigqmin: float = 10.0
    bmxlim: float = 12.0
    fauxmn: float = 1.0
    fbeta: float = 1.0
    fbetap: float = 1.0
    fbetatry: float = 1.0
    fbetatry_lower: float = 1.0
    fcwr: float = 1.0
    fdene: float = 1.0
    fdivcol: float = 1.0
    fdtmp: float = 1.0
    fecrh_ignition: float = 1.0
    fflutf: float = 1.0
    ffuspow: float = 1.0
    fgamcd: float = 1.0
    fhldiv: float = 1.0
    fiooic: float = 1.0
    fipir: float = 1.0
    fjohc: float = 1.0
    fjohc0: float = 1.0
    fjprot: float = 1.0
    flhthresh: float = 1.0
    fmva: float = 1.0
    fnbshinef: float = 1.0
    fncycle: float = 1.0
    fnesep: float = 1.0
    foh_stress: float = 1.0
    fpeakb: float = 1.0
    fpinj: float = 1.0
    fpnetel: float = 1.0
    fportsz: float = 1.0
    fpsepbqar: float = 1.0
    fpsepr: float = 1.0
    fptemp: float = 1.0
    fq: float = 1.0
    fqval: float = 1.0
    fradwall: float = 1.0
    freinke: float = 1.0
    fstrcase: float = 1.0
    fstrcond: float = 1.0
    fstr_wp: float = 1.0
    fmaxvvstress: float = 1.0
    ftbr: float = 1.0
    ftburn: float = 1.0
    ftcycl: float = 1.0
    ftmargoh: float = 1.0
    ftmargtf: float = 1.0
    ftohs: float = 1.0
    ftpeak: float = 1.0
    fvdump: float = 1.0
    fvs: float = 1.0
    fvvhe: float = 1.0
    fwalld: float = 1.0
    fzeffmax: float = 1.0
    gammax: float = 2.0
    maxradwallload: float = 1.0
    mvalim: float = 4.0
    nbshinefmax: float = 0.001
    nflutfmax: float = 1.0e23
    pdivtlim: float = 150.0
    peakfactrad: float = 3.33
    pnetelin: float = 1000.0
    powfmax: float = 1500.0
    psepbqarmax: float = 9.5
    pseprmax: float = 25.0
    ptfnucmax: float = 0.001
    tbrmin: float = 1.1
    tbrnmn: float = 1.0
    vvhealw: float = 1.0
    walalw: float = 8.0
    taulimit: float = 5.0
    ftaulimit: float = 1.0
    fniterpump: float = 1.0
    zeffmax: float = 3.6
    fpoloidalpower: float = 1.0
    fpsep: float = 1.0
    fcqt: float = 1.0

    # Build variables
    aplasmin: float = 0.25
    blbmith: float = 0.17
    blbmoth: float = 0.27
    blbpith: float = 0.3
    blbpoth: float = 0.35
    blbuith: float = 0.365
    blbuoth: float = 0.465
    blnkith: float = 0.115
    blnkoth: float = 0.235
    bore: float = 1.42
    clhsf: float = 4.268
    ddwex: float = 0.07
    d_vv_in: float = 0.07
    d_vv_out: float = 0.07
    d_vv_top: float = 0.07
    d_vv_bot: float = 0.07
    f_avspace: float = 1.0
    fcspc: float = 0.6
    fseppc: float = 350000000.0
    gapds: float = 0.155
    gapoh: float = 0.08
    gapomin: float = 0.234
    iohcl: int = 1
    iprecomp: int = 1
    ohcth: float = 0.811
    rinboard: float = 0.651
    f_r_cp: float = 1.4
    scrapli: float = 0.14
    scraplo: float = 0.15
    shldith: float = 0.69
    shldlth: float = 0.7
    shldoth: float = 1.05
    shldtth: float = 0.6
    sigallpc: float = 300000000.0
    tfoofti: float = 1.19
    thshield_ib: float = 0.05
    thshield_ob: float = 0.05
    thshield_vb: float = 0.05
    vgap2: float = 0.163
    vgaptop: float = 0.6
    vvblgap: float = 0.05
    plleni: float = 1.0
    plsepi: float = 1.0
    plsepo: float = 1.5

    # Buildings

    # Current drive
    beamwd: float = 0.58
    bscfmax: float = 0.99
    cboot: float = 1.0
    harnum: float = 1.0
    enbeam: float = 1e3
    etaech: float = 0.3
    etanbi: float = 0.3
    feffcd: float = 1.0
    frbeam: float = 1.05
    ftritbm: float = 1e-6
    gamma_ecrh: float = 0.35
    rho_ecrh: float = 0.1
    xi_ebw: float = 0.8
    iefrf: int = 5
    irfcf: int = 1
    nbshield: float = 0.5
    pheat: float = 0.0  # Listed as an output
    pinjalw: float = 150.0
    tbeamin: float = 3.0

    # Impurity radiation
    coreradius: float = 0.6
    coreradiationfraction: float = 1.0
    fimp: List[float] = field(default_factory=lambda: [1.0, 0.1, *([0.0] * 12)])
    fimpvar: float = 0.001
    impvar: int = 9

    # Reinke
    impvardiv: int = 9
    lhat: float = 4.33
    fzactual: float = 0.001

    # Divertor
    divdum: int = 0
    anginc: float = 0.262
    beta_div: float = 1.0
    betai: float = 1.0
    betao: float = 1.0
    bpsout: float = 0.6
    c1div: float = 0.45
    c2div: float = -7.0
    c3div: float = 0.54
    c4div: float = -3.6
    c5div: float = 0.7
    delld: float = 1.0
    divclfr: float = 0.3
    divdens: float = 10000.0
    divfix: float = 0.2
    divleg_profile_inner: float = 0.563
    divleg_profile_outer: float = 2.596
    divplt: float = 0.035
    fdfs: float = 10.0
    fdiva: float = 1.11
    fgamp: float = 1.0
    fififi: float = 0.004
    flux_exp: float = 2.0
    frrp: float = 0.4
    hldivlim: float = 5.0
    ksic: float = 0.8
    omegan: float = 1.0
    prn1: float = 0.285
    rlenmax: float = 0.5
    tdiv: float = 2.0
    xparain: float = 2100.0
    xpertin: float = 2.0
    zeffdiv: float = 1.0

    # Pulse
    bctmp: float = 320.0
    dtstor: float = 300.0
    istore: float = 1
    itcycl: float = 1
    lpulse: int = 1  # Listed as an output

    # IFE

    # Heat transport
    baseel: float = 5000000.0
    crypw_max: float = 50.0
    f_crypmw: float = 1.0
    etatf: float = 0.9
    etath: float = 0.375
    fpumpblkt: float = 0.005
    fpumpdiv: float = 0.005
    fpumpfw: float = 0.005
    fpumpshld: float = 0.005
    ipowerflow: int = 1
    iprimshld: int = 1
    pinjmax: float = 120.0
    pwpm2: float = 150.0
    trithtmw: float = 15.0
    vachtmw: float = 0.5

    # Water usage

    # Vacuum
    ntype: int = 1
    pbase: float = 0.0005
    prdiv: float = 0.36
    pumptp: float = 1.2155e22
    rat: float = 1.3e-8
    tn: float = 300.0
    pumpareafraction: float = 0.0203
    pumpspeedmax: float = 27.3
    pumpspeedfactor: float = 0.167
    initialpressure: float = 1.0
    outgasindex: float = 1.0
    outgasfactor: float = 0.0235

    # PF coil
    alfapf: float = 5.0e-10
    alstroh: float = 400000000.0
    coheof: float = 18500000.0
    cptdin: List[float] = field(default_factory=lambda: [40000.0] * 22)
    etapsu: float = 0.9
    fcohbop: float = 0.9
    fcuohsu: float = 0.7
    fcupfsu: float = 0.69
    fvssu: float = 1.0
    ipfloc: List[int] = field(
        default_factory=lambda: [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    ipfres: int = 0  # Listed as an output
    isumatoh: int = 1
    isumatpf: int = 1
    i_pf_current: int = 1
    ncls: List[int] = field(default_factory=lambda: [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    nfxfh: int = 7
    ngrp: int = 3
    ohhghf: float = 0.71
    oh_steel_frac: float = 0.5
    pfclres: float = 2.5e-8
    rjconpf: List[float] = field(default_factory=lambda: [30000000] * 22)
    routr: float = 1.5
    rpf2: float = -1.63
    rref: List[float] = field(default_factory=lambda: [7] * 10)
    sigpfcalw: float = 500.0
    sigpfcf: float = 0.666
    vf: List[float] = field(default_factory=lambda: [0.3] * 22)
    vhohc: float = 0.3
    zref: List[float] = field(
        default_factory=lambda: [3.6, 1.2, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    bmaxcs_lim: float = 13.0
    fbmaxcs: float = 1.0
    ld_ratio_cst: float = 3.0

    # Physics
    alphaj: float = 1.0
    alphan: float = 0.25
    alphat: float = 0.5
    aspect: float = 2.907
    beamfus0: float = 1.0
    beta: float = 0.042
    betbm0: float = 1.5
    bt: float = 5.68
    csawth: float = 1.0
    cvol: float = 1.0
    cwrmax: float = 1.35
    dene: float = 9.8e19
    dnbeta: float = 3.5
    epbetmax: float = 1.38
    falpha: float = 0.95
    fdeut: float = 0.5
    ftar: float = 1.0
    ffwal: float = 0.92
    fgwped: float = 0.85
    fgwsep: float = 0.5
    fkzohm: float = 1.0
    fpdivlim: float = 1.0
    fne0: float = 1.0
    ftrit: float = 0.5
    fvsbrnni: float = 1.0
    gamma: float = 0.4
    hfact: float = 1.0
    taumax: float = 10.0
    ibss: int = 3
    iculbl: int = 1  # listed as an output...
    icurr: int = 4
    idensl: int = 7
    ifalphap: int = 1
    ifispact: int = 0  # listed as an output...
    iinvqd: int = 1
    ipedestal: int = 1
    ieped: int = 1  # listed as an output...
    eped_sf: float = 1.0
    neped: float = 4.0e19
    nesep: float = 3.0e19
    plasma_res_factor: float = 1.0
    rhopedn: float = 1.0
    rhopedt: float = 1.0
    tbeta: float = 2.0
    teped: float = 1.0
    tesep: float = 0.1
    iprofile: int = 1
    iradloss: int = 1
    isc: int = 34
    iscrp: int = 1
    ishape: int = 0  # listed as an output...
    itart: int = 0  # listed as an output...
    itartpf: int = 1  # listed as an output...
    iwalld: int = 1
    kappa: float = 1.792
    kappa95: float = 1.6
    m_s_limit: float = 0.3
    ilhthresh: int = 19
    q: float = 3.0
    q0: float = 1.0
    tauratio: float = 1.0
    rad_fraction_sol: float = 0.8
    ralpne: float = 0.1
    rli: float = 0.9
    rmajor: float = 8.14
    rnbeam: float = 0.005
    i_single_null: int = 1
    ssync: float = 0.6
    te: float = 12.9
    ti: float = 12.9
    tratio: float = 1.0
    triang: float = 0.36
    triang95: float = 0.24

    # Stellarator

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
