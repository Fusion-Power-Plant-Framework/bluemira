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

from dataclasses import dataclass, fields
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

    runtitle: str = ""

    # Optimisation problem setup
    bounds: Dict[str, Dict[str, str]] = None
    icc: List[int] = None
    ixc: List[int] = None

    # Settings
    maxcal: int = None
    minmax: int = None
    epsvmc: float = None
    ioptimz: int = None
    output_costs: int = None
    isweep: int = None
    nsweep: int = None
    sweep: List[float] = None
    pulsetimings: int = None
    # Top down of PROCESS variables list

    # Times
    tburn: float = None
    tdwell: float = None
    theat: float = None
    tohs: float = None
    tqnch: float = None
    tramp: float = None

    # FWBS
    ibkt_life: int = None
    denstl: float = None
    denw: float = None
    emult: float = None
    fblss: float = None
    fdiv: float = None
    fwbsshape: int = None
    fw_armour_thickness: float = None
    iblanket: int = None
    iblnkith: int = None
    li6enrich: float = None
    breeder_f: float = None
    breeder_multiplier: float = None
    vfcblkt: float = None
    vfpblkt: float = None
    blktmodel: int = None  # Listed as an output...
    # f_neut_shield: float = # -1.0 the documentation defaults cannot be right...
    breedmat: int = None
    fblbe: float = None
    fblbreed: float = None
    fblhebmi: float = None
    fblhebmo: float = None
    fblhebpi: float = None
    fblhebpo: float = None
    hcdportsize: int = None
    npdiv: int = None
    nphcdin: int = None
    nphcdout: int = None
    wallpf: float = None
    iblanket_thickness: int = None
    secondary_cycle: int = None  # Listed as an output...
    secondary_cycle_liq: int = None
    afwi: float = None
    afwo: float = None
    fw_wall: float = None
    afw: float = None
    pitch: float = None
    fwinlet: float = None
    fwoutlet: float = None
    fwpressure: float = None
    roughness: float = None
    fw_channel_length: float = None
    peaking_factor: float = None
    blpressure: float = None
    inlet_temp: float = None
    outlet_temp: float = None
    coolp: float = None
    nblktmodpo: int = None
    nblktmodpi: int = None
    nblktmodto: int = None
    nblktmodti: int = None
    tfwmatmax: float = None
    fw_th_conductivity: float = None
    fvoldw: float = None
    fvolsi: float = None
    fvolso: float = None
    fwclfr: float = None
    rpf2dewar: float = None
    vfshld: float = None
    irefprop: int = None
    fblli2o: float = None
    fbllipb: float = None
    vfblkt: float = None
    declblkt: float = None
    declfw: float = None
    declshld: float = None
    blkttype: int = None
    etaiso: float = None
    etahtp: float = None
    n_liq_recirc: int = None
    bz_channel_conduct_liq: float = None
    blpressure_liq: float = None
    inlet_temp_liq: float = None
    outlet_temp_liq: float = None
    f_nuc_pow_bz_struct: float = None
    pnuc_fw_ratio_dcll: float = None

    # TF coil
    sig_tf_case_max: float = None
    sig_tf_wp_max: float = None
    bcritsc: float = None
    casthi_fraction: float = None
    casths_fraction: float = None
    f_t_turn_tf: float = None
    t_turn_tf_max: float = None
    cpttf: float = None
    cpttf_max: float = None
    dcase: float = None
    dcond: List[float] = None
    dcondins: float = None
    dhecoil: float = None
    farc4tf: float = None
    b_crit_upper_nbti: float = None
    t_crit_nbti: float = None
    fcutfsu: float = None
    fhts: float = None
    i_tf_stress_model: int = None
    i_tf_wp_geom: int = None
    i_tf_case_geom: int = None  # Listed as an output
    i_tf_turns_integer: int = None  # Listed as an output
    i_tf_sc_mat: int = None
    i_tf_sup: int = None
    i_tf_shape: int = None  # Listed as an output
    i_tf_cond_eyoung_trans: int = None
    n_pancake: int = None
    n_layer: int = None
    n_rad_per_layer: int = None
    i_tf_bucking: int = None
    n_tf_graded_layers: int = None
    jbus: float = None
    eyoung_ins: float = None
    eyoung_steel: float = None
    eyong_cond_axial: float = None
    eyoung_res_tf_buck: float = None
    # eyoung_al: float = 69000000000.0 # defaults  cannot be right
    poisson_steel: float = None
    poisson_copper: float = None
    poisson_al: float = None
    str_cs_con_res: float = None
    str_pf_con_res: float = None
    str_tf_con_res: float = None
    str_wp_max: float = None
    i_str_wp: int = None
    quench_model: str = None
    tcritsc: float = None
    tdmptf: float = None
    tfinsgap: float = None
    # rhotfbus: float = -1.0 # defaults cannot be right
    frhocp: float = None
    frholeg: float = None
    # i_cp_joints: int = -1 # defaults cannot be right
    rho_tf_joints: float = None
    n_tf_joints_contact: int = None
    n_tf_joints: int = None
    th_joint_contact: float = None
    # eff_tf_cryo: float = -1.0 # defaults cannot be right
    n_tf: int = None
    tftmp: float = None
    thicndut: float = None
    thkcas: float = None
    thwcndut: float = None
    tinstf: float = None
    tmaxpro: float = None
    tmax_croco: float = None
    tmpcry: float = None
    vdalw: float = None
    f_vforce_inboard: float = None
    vftf: float = None
    etapump: float = None
    fcoolcp: float = None
    fcoolleg: float = None
    ptempalw: float = None
    rcool: float = None
    tcoolin: float = None
    tcpav: float = None
    vcool: float = None
    theta1_coil: float = None
    theta1_vv: float = None
    max_vv_stress: float = None
    inuclear: int = None
    qnuc: float = None
    ripmax: float = None
    tf_in_cs: int = None
    tfcth: float = None
    tftsgap: float = None
    casthi: float = None
    casths: float = None
    tmargmin: float = None
    oacdcp: float = None

    # PF Power
    iscenr: int = None
    maxpoloidalpower: float = None

    # Cost variables
    abktflnc: float = None
    adivflnc: float = None
    cconfix: float = None
    cconshpf: float = None
    cconshtf: float = None
    cfactr: float = None
    cfind: List[float] = None
    cland: float = None
    costexp: float = None
    costexp_pebbles: float = None
    cost_factor_buildings: float = None
    cost_factor_land: float = None
    cost_factor_tf_coils: float = None
    cost_factor_fwbs: float = None
    cost_factor_tf_rh: float = None
    cost_factor_tf_vv: float = None
    cost_factor_tf_bop: float = None
    cost_factor_tf_misc: float = None
    maintenance_fwbs: float = None
    maintenance_gen: float = None
    amortization: float = None
    cost_model: int = None
    cowner: float = None
    cplife_input: float = None
    cpstflnc: float = None
    csi: float = None
    # cturbb: float = 38.0 # defaults cannot be right
    decomf: float = None
    dintrt: float = None
    fcap0: float = None
    fcap0cp: float = None
    fcdfuel: float = None
    fcontng: float = None
    fcr0: float = None
    fkind: float = None
    iavail: int = None
    life_dpa: float = None
    avail_min: float = None
    favail: float = None
    num_rh_systems: int = None
    conf_mag: float = None
    div_prob_fail: float = None
    div_umain_time: float = None
    div_nref: float = None
    div_nu: float = None
    fwbs_nref: float = None
    fwbs_nu: float = None
    fwbs_prob_fail: float = None
    fwbs_umain_time: float = None
    redun_vacp: float = None
    tbktrepl: float = None
    tcomrepl: float = None
    tdivrepl: float = None
    uubop: float = None
    uucd: float = None
    uudiv: float = None
    uufuel: float = None
    uufw: float = None
    uumag: float = None
    uuves: float = None
    ifueltyp: int = None
    ucblvd: float = None
    ucdiv: float = None
    ucme: float = None
    ireactor: int = None
    lsa: int = None
    discount_rate: float = None
    startupratio: float = None
    tlife: float = None
    bkt_life_csf: int = None
    # ...

    # CS fatigue
    residual_sig_hoop: float = None
    n_cycle_min: int = None
    t_crack_vertical: float = None
    t_crack_radial: float = None
    t_structural_radial: float = None
    t_structural_vertical: float = None
    sf_vertical_crack: float = None
    sf_radial_crack: float = None
    sf_fast_fracture: float = None
    paris_coefficient: float = None
    paris_power_law: float = None
    walker_coefficient: float = None
    fracture_toughness: float = None

    # REBCO
    rebco_thickness: float = None
    copper_thick: float = None
    hastelloy_thickness: float = None
    tape_width: float = None
    tape_thickness: float = None
    croco_thick: float = None
    copper_rrr: float = None
    copper_m2_max: float = None
    f_coppera_m2: float = None
    copperaoh_m2_max: float = None
    f_copperaoh_m2: float = None

    # Primary pumping
    primary_pumping: int = None
    gamma_he: float = None
    t_in_bb: float = None
    t_out_bb: float = None
    p_he: float = None
    dp_he: float = None

    # Constraint variables
    auxmin: float = None
    betpmx: float = None
    bigqmin: float = None
    bmxlim: float = None
    fauxmn: float = None
    fbeta: float = None
    fbetap: float = None
    fbetatry: float = None
    fbetatry_lower: float = None
    fcwr: float = None
    fdene: float = None
    fdivcol: float = None
    fdtmp: float = None
    fecrh_ignition: float = None
    fflutf: float = None
    ffuspow: float = None
    fgamcd: float = None
    fhldiv: float = None
    fiooic: float = None
    fipir: float = None
    fjohc: float = None
    fjohc0: float = None
    fjprot: float = None
    flhthresh: float = None
    fmva: float = None
    fnbshinef: float = None
    fncycle: float = None
    fnesep: float = None
    foh_stress: float = None
    fpeakb: float = None
    fpinj: float = None
    fpnetel: float = None
    fportsz: float = None
    fpsepbqar: float = None
    fpsepr: float = None
    fptemp: float = None
    fq: float = None
    fqval: float = None
    fradwall: float = None
    freinke: float = None
    fstrcase: float = None
    fstrcond: float = None
    fstr_wp: float = None
    fmaxvvstress: float = None
    ftbr: float = None
    ftburn: float = None
    ftcycl: float = None
    ftmargoh: float = None
    ftmargtf: float = None
    ftohs: float = None
    ftpeak: float = None
    fvdump: float = None
    fvs: float = None
    fvvhe: float = None
    fwalld: float = None
    fzeffmax: float = None
    gammax: float = None
    maxradwallload: float = None
    mvalim: float = None
    nbshinefmax: float = None
    nflutfmax: float = None
    pdivtlim: float = None
    peakfactrad: float = None
    pnetelin: float = None
    powfmax: float = None
    psepbqarmax: float = None
    pseprmax: float = None
    ptfnucmax: float = None
    tbrmin: float = None
    tbrnmn: float = None
    vvhealw: float = None
    walalw: float = None
    taulimit: float = None
    ftaulimit: float = None
    fniterpump: float = None
    zeffmax: float = None
    fpoloidalpower: float = None
    fpsep: float = None
    fcqt: float = None

    # Build variables
    aplasmin: float = None
    blbmith: float = None
    blbmoth: float = None
    blbpith: float = None
    blbpoth: float = None
    blbuith: float = None
    blbuoth: float = None
    blnkith: float = None
    blnkoth: float = None
    bore: float = None
    clhsf: float = None
    ddwex: float = None
    d_vv_in: float = None
    d_vv_out: float = None
    d_vv_top: float = None
    d_vv_bot: float = None
    f_avspace: float = None
    fcspc: float = None
    fhole: float = None
    fseppc: float = None
    gapds: float = None
    gapoh: float = None
    gapomin: float = None
    iohcl: int = None
    iprecomp: int = None
    ohcth: float = None
    rinboard: float = None
    f_r_cp: float = None
    scrapli: float = None
    scraplo: float = None
    shldith: float = None
    shldlth: float = None
    shldoth: float = None
    shldtth: float = None
    sigallpc: float = None
    tfoofti: float = None
    thshield_ib: float = None
    thshield_ob: float = None
    thshield_vb: float = None
    vgap: float = None
    vgap2: float = None
    vgaptop: float = None
    vvblgap: float = None
    plleni: float = None
    plsepi: float = None
    plsepo: float = None

    # Buildings

    # Current drive
    beamwd: float = None
    bscfmax: float = None
    cboot: float = None
    harnum: float = None
    enbeam: float = None
    etaech: float = None
    etanbi: float = None
    feffcd: float = None
    frbeam: float = None
    ftritbm: float = None
    gamma_ecrh: float = None
    rho_ecrh: float = None
    xi_ebw: float = None
    iefrf: int = None
    irfcf: int = None
    nbshield: float = None
    pheat: float = None  # Listed as an output
    pinjalw: float = None
    tbeamin: float = None

    # Impurity radiation
    coreradius: float = None
    coreradiationfraction: float = None
    fimp: List[float] = None
    fimpvar: float = None
    impvar: int = None

    # Reinke
    impvardiv: int = None
    lhat: float = None
    fzactual: float = None

    # Divertor
    divdum: int = None
    anginc: float = None
    beta_div: float = None
    betai: float = None
    betao: float = None
    bpsout: float = None
    c1div: float = None
    c2div: float = None
    c3div: float = None
    c4div: float = None
    c5div: float = None
    delld: float = None
    divclfr: float = None
    divdens: float = None
    divfix: float = None
    divleg_profile_inner: float = None
    divleg_profile_outer: float = None
    divplt: float = None
    fdfs: float = None
    fdiva: float = None
    fgamp: float = None
    fififi: float = None
    flux_exp: float = None
    frrp: float = None
    hldivlim: float = None
    ksic: float = None
    omegan: float = None
    prn1: float = None
    rlenmax: float = None
    tdiv: float = None
    xparain: float = None
    xpertin: float = None
    zeffdiv: float = None

    # Pulse
    bctmp: float = None
    dtstor: float = None
    istore: int = None
    itcycl: int = None
    lpulse: int = None  # Listed as an output

    # IFE

    # Heat transport
    baseel: float = None
    crypw_max: float = None
    f_crypmw: float = None
    etatf: float = None
    etath: float = None
    fpumpblkt: float = None
    fpumpdiv: float = None
    fpumpfw: float = None
    fpumpshld: float = None
    ipowerflow: int = None
    iprimshld: int = None
    pinjmax: float = None
    pwpm2: float = None
    trithtmw: float = None
    vachtmw: float = None
    irfcd: int = None

    # Water usage

    # Vacuum
    ntype: int = None
    pbase: float = None
    prdiv: float = None
    pumptp: float = None
    rat: float = None
    tn: float = None
    pumpareafraction: float = None
    pumpspeedmax: float = None
    pumpspeedfactor: float = None
    initialpressure: float = None
    outgasindex: float = None
    outgasfactor: float = None

    # PF coil
    alfapf: float = None
    alstroh: float = None
    coheof: float = None
    cptdin: List[float] = None
    etapsu: float = None
    fcohbop: float = None
    fcuohsu: float = None
    fcupfsu: float = None
    fvssu: float = None
    ipfloc: List[int] = None
    ipfres: int = None  # Listed as an output
    isumatoh: int = None
    isumatpf: int = None
    i_pf_current: int = None
    ncls: List[int] = None
    nfxfh: int = None
    ngrp: int = None
    ohhghf: float = None
    oh_steel_frac: float = None
    pfclres: float = None
    rjconpf: List[float] = None
    routr: float = None
    rpf2: float = None
    rref: List[float] = None
    sigpfcalw: float = None
    sigpfcf: float = None
    vf: List[float] = None
    vhohc: float = None
    zref: List[float] = None
    bmaxcs_lim: float = None
    fbmaxcs: float = None
    ld_ratio_cst: float = None

    # Physics
    alphaj: float = None
    alphan: float = None
    alphat: float = None
    aspect: float = None
    beamfus0: float = None
    beta: float = None
    betbm0: float = None
    bt: float = None
    csawth: float = None
    cvol: float = None
    cwrmax: float = None
    dene: float = None
    dnbeta: float = None
    epbetmax: float = None
    falpha: float = None
    fdeut: float = None
    ftar: float = None
    ffwal: float = None
    fgwped: float = None
    fgwsep: float = None
    fkzohm: float = None
    fpdivlim: float = None
    fne0: float = None
    ftrit: float = None
    fvsbrnni: float = None
    gamma: float = None
    hfact: float = None
    taumax: float = None
    ibss: int = None
    iculbl: int = None  # listed as an output...
    icurr: int = None
    idensl: int = None
    ifalphap: int = None
    ifispact: int = None  # listed as an output...
    iinvqd: int = None
    ipedestal: int = None
    ieped: int = None  # listed as an output...
    eped_sf: float = None
    neped: float = None
    nesep: float = None
    plasma_res_factor: float = None
    rhopedn: float = None
    rhopedt: float = None
    tbeta: float = None
    teped: float = None
    tesep: float = None
    iprofile: int = None
    iradloss: int = None
    isc: int = None
    iscrp: int = None
    ishape: int = None  # listed as an output...
    itart: int = None  # listed as an output...
    itartpf: int = None  # listed as an output...
    iwalld: int = None
    kappa: float = None
    kappa95: float = None
    m_s_limit: float = None
    ilhthresh: int = None
    q: float = None
    q0: float = None
    tauratio: float = None
    rad_fraction_sol: float = None
    ralpne: float = None
    rli: float = None
    rmajor: float = None
    rnbeam: float = None
    i_single_null: int = None
    ssync: float = None
    te: float = None
    ti: float = None
    tratio: float = None
    triang: float = None
    triang95: float = None

    # Stellarator
    fblvd: float = None

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
            if name not in ["icc", "ixc", "bounds"] and value is not None:
                new_val = _INVariable(name, value, "Parameter", "", "")
                out_dict[name] = new_val
        out_dict["icc"] = _INVariable(
            "icc",
            [] if self.icc is None else self.icc,
            "Constraint Equation",
            "Constraint Equation",
            "Constraint Equations",
        )
        out_dict["ixc"] = _INVariable(
            "ixc",
            [] if self.ixc is None else self.ixc,
            "Iteration Variable",
            "Iteration Variable",
            "Iteration Variables",
        )
        out_dict["bounds"] = _INVariable(
            "bounds",
            {} if self.bounds is None else self.bounds,
            "Bound",
            "Bound",
            "Bounds",
        )
        return out_dict

    def to_dict(self) -> Dict[str, Union[float, List, Dict]]:
        """
        A dictionary representation of the dataclass

        """
        return dict(self)
