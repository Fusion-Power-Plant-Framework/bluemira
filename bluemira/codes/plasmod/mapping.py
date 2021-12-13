from enum import Enum

from bluemira.base.parameter import ParameterMapping


class ImpurityModel(Enum):
    # [-] impurity model selector:
    # 0 - fixed concentration,
    # 1 - concentration fixed at pedestal top, then fixed density.
    # "i_impmodel"
    FIXED = 0
    PED_FIXED = 1


class TransportModel(Enum):
    # [-] selector for transport model
    # 1 - simple gyrobohm scaling with imposed H factor,
    # 555 - H factor scaling from F. Palermo
    # 111 - another model based on gyro-Bohm transport
    # 2 - no reference in the source code
    # "i_modeltype"
    GYROBOHM_1 = 1
    GYROBOHM_2 = 111
    UNKNOWN = 2
    H_FACTOR = 555


class EquilibriumModel(Enum):
    # [-] equilibrium model selector:
    # 1 - EMEQ solves equilibrium with given q95, with sawteeth.
    # 2 - EMEQ solves with given Ip, with sawteeth
    # "i_equiltype"
    q95_sawtooth = 1
    Ip_sawtooth = 2


class PedestalModel(Enum):
    # [-] pedestal model selector:
    # 1 - fixed pedestal temperature (Teped_in),
    # 2 - Saarelma scaling
    # "i_pedestal"
    FIX_TEMP = 1
    SAARELMA = 2


class SOLModel(Enum):
    # [-] SOL model selector:
    # 0 - fit based on Eich scaling
    # 1 - Mattia Siccinio's model
    # "isiccir"
    EICH_FIT = 0
    SICCINIO = 1


class Profiles(Enum):
    # [A/m²] Bootstrap parallel current density profile
    cubb = "cubb"
    # [-] Triangularity profile
    delta = "delta"
    # [m³] Volume increment profile
    dV = "dV"
    # [(m*T) * (m*T) / Wb == T] FF' profile
    ffprime = "ffprime"
    # [m⁴] < |grad V|²> g1 metric coefficient's profile
    g1 = "g1"
    # [m²] < |grad V|²/r²> g2 metric coefficient's profile
    g2 = "g2"
    # [m⁻²] < 1/r²> g3 metric coefficient's profile
    g3 = "g3"
    # [m*T] Poloidal current profile
    ipol = "i_pol"
    # [A/m²] Parallel current density profile
    jpar = "jpar"
    # [A/m²] CD parallel current density profile
    jcdr = "jcdr"
    # [-] Elongation profile
    kappa = "kappa"
    # [Pa/Wb] p' profile
    pprime = "pprime"
    # [10¹⁹/m3] argon density profile
    nar = "n_Ar"
    # [10¹⁹/m3] deuterium density profile
    ndeut = "n_D"
    # [10¹⁹/m3] electron density profile
    nepr = "n_e"
    # [10¹⁹/m3] fuel density profile
    nfuel = "n_fuel"
    # [10¹⁹/m3] helium density profile
    nhe = "n_He"
    # [10¹⁹/m³] ion density profile
    nions = "n_ion"
    # [10¹⁹/m3] tritium density profile
    ntrit = "n_T"
    # [10¹⁹/m3] xenon density profile
    nxe = "n_Xe"
    # [Wb] Toroidal flux profile
    phi = "phi"
    # [Pa] Plasma pressure profile
    pressure = "pressure"
    # [Wb] Poloidal flux profile
    psi = "psi"
    # [W/m³] fusion power density profile (DT + DT)
    q_fus = "qfus"
    # [W/m³] neutron power density profile
    qneut = "q_neut"
    # [-] Safety factor profile
    qprf = "q"
    # [W/m³] radiation power density profile
    qrad = "qrad"
    # [m] Grad-Shafranov shift profile
    shif = "GS"
    # [keV] Electron temperature profile
    Tepr = "Te"
    # [keV] Ion temperature profile
    Tipr = "Ti"
    # [-] normalized toroidal flux coordinate (Phi/Phi_b)
    x = "x"
    # [m³] Volume profile
    V = "V"


# TODO
# define all build tweaks properly
# Link all BM parameters
# Link all plasmod outputs

PLASMOD_INPUTS = {
    ############################
    # list geometry properties
    ############################
    # [-] plasma aspect ratio
    "A": "A",
    # [T] Toroidal field at plasma center
    "B_0": "Bt",
    # [-] plasma edge triangularity (used only for first iteration,
    # then iterated to constrain delta95)
    "delta": "deltaX",
    # [-] plasma triangularity at 95 % flux
    "delta_95": "delta95",
    # [-] plasma edge elongation (used only for first iteration,
    # then iterated to constrain kappa95)
    "kappa": "kappaX",
    # [-] plasma elongation at 95 % flux
    "kappa_95": "kappa95",
    # [-] safety factor at 95% flux surface
    "q_95": "q95",
    # [m] plasma major radius
    "R_0": "R0",
    # [m3] constrained plasma volume (set zero to disable volume constraining)
    "V_p": "V_in",
    ############################
    # list numerics properties
    #############################
    # [-] Newton differential
    "BM_INP": "dgy",
    # [-] min time step between iterations
    "BM_INP": "dtmin",
    # [-] max time step between iterations
    "BM_INP": "dtmax",
    # [-] number of interpolated grid points
    "BM_INP": "nx",
    # [-] number of reduced grid points
    "BM_INP": "nxt",
    # [-] max number of iteration
    "BM_INP": "test",
    # [-] max iteration error between transport/equilibrium iterations
    "BM_INP": "tol",
    ############################
    # list transport & confinement properties
    #############################
    # [-] Greenwald density fraction at pedestal
    "BM_INP": "f_gwped",
    # [-] Greenwald density fraction at separatrix
    "BM_INP": "f_gws",
    # [-] fraction of NBI power to ions
    "BM_INP": "fpion",
    # [-] tauparticle / tauE for D
    "BM_INP": "fp2e_d",
    # [-] tauparticle / tauE for T
    "BM_INP": "fp2e_t",
    # [-] tauparticle / tauE for He
    "BM_INP": "fp2e_he",
    # [-] tauparticle / tauE for Xe
    "BM_INP": "fp2e_xe",
    # [-] tauparticle / tauE for Ar
    "BM_INP": "fp2e_ar",
    # [-] input H-factor:if i_modeltype > 1 H factor calculated
    "BM_INP": "hfact_inp",
    # [-] normalized coordinate of pedestal density
    "BM_INP": "rho_n",
    # [-] normalized coordinate of pedestal temperature
    "BM_INP": "rho_T",
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    "BM_INP": "Teped_inp",
    # [keV] electrons/ions temperature at separatrix
    "BM_INP": "Tesep",
    ############################
    # list composition properties
    #############################
    # [-] Tungsten concentration
    "BM_INP": "cwol",
    # [-] fuel mix D/T
    "BM_INP": "fuelmix",
    ############################
    # list control & transport settings
    #############################
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for NBI heating (CD) to control Vloop or f_ni
    "BM_INP": "dx_cd_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed NBI heating
    "BM_INP": "dx_control_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control fusion power
    "BM_INP": "dx_fus_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control H-mode
    "BM_INP": "dx_heat_nbi",
    # [-] required fraction of non inductive current, if 0, dont use CD
    "BM_INP": "f_ni",
    # [m*MA/MW] Normalized CD efficiency
    "BM_INP": "nbcdeff",
    # [MW] max allowed power for control (fusion power, H-mode)
    "BM_INP": "Pheat_max",
    # [MW] required fusion power.
    # 0. - ignored
    # > 0 - Auxiliary heating is calculated to match Pfus_req
    "BM_INP": "Pfus_req",
    # [MW*T/m] Divertor challenging criterion Psep * Bt / (q95 * A R0)
    # if PsepBt_qAR > PsepBt_qAR_max seed Xenon
    "BM_INP": "PsepBt_qAR_max",
    # [-] max P_sep/P_LH. if Psep/PLH > Psep/PLH_max -> use Xe
    "BM_INP": "Psep_PLH_max",
    # [-] min P_sep/P_LH. if Psep/PLH < Psep/PLH_max -> use heating
    "BM_INP": "Psep_PLH_min",
    # [MW/m] Divertor challenging criterion Psep / R0
    # if Psep/R0 > Psep_R0_max seed Xenon
    "BM_INP": "Psep_R0_max",
    # [MW] fixed auxiliary heating power required for control
    "BM_INP": "q_control",
    # [MW/m2] max divertor heat flux -->
    # if qdivt > qdivt_max -> seed argon
    "BM_INP": "qdivt_max",
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    "BM_INP": "x_cd_nbi",
    # [-]  normalized mean location of fixed NBI heating
    "BM_INP": "x_control_nbi",
    # [-]  normalized mean location of NBI heating for
    # controlling fusion power (Pfus = Pfus_req)
    "BM_INP": "x_fus_nbi",
    # [-]  normalized mean location of aux. heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    "BM_INP": "x_heat_nbi",
    # [V] target loop voltage (if lower than -1e-3, ignored)
    "BM_INP": "v_loop_in",
}

#


PLASMOD_OUTPUTS = {
    ############################
    # list scalar outputs
    #############################
    # [m²] plasma poloidal cross section area
    "BM_OUT": "area_pol",
    # [m²] plasma toroidal surface
    "BM_OUT": "area_tor",
    # [-] poloidal beta
    "BM_OUT": "beta_p",
    # [-] normalized beta
    "BM_OUT": "beta_n",
    # [-] toroidal beta
    "BM_OUT": "beta_t",
    # [T] average poloidal field
    "BM_OUT": "Bpav",
    # [-] Argon concentration (ratio nAr/ne)
    "BM_OUT": "c_ar",
    # [-] Hydrogen concentration (ratio nH/ne)
    "BM_OUT": "c_h",
    # [-] Helium concentration (ratio nH/ne)
    "BM_OUT": "c_he",
    # [-] Xenon concentration (ratio nH/ne)
    "BM_OUT": "c_xe",
    # [-] plasma edge triangularity
    "BM_OUT": "delta_e",
    # [-] tolerance on kinetic profiles
    "BM_OUT": "etol",
    # [-] plasma bootstrap current fraction
    "BM_OUT": "f_bs",
    # [-] plasma current drive fraction
    "BM_OUT": "f_cd",
    # [-] plasma current inductive fraction
    "BM_OUT": "f_ind",
    # [-] H-factor
    "BM_OUT": "H",
    # [MA] plasma current
    "BM_OUT": "Ip",
    # [-] plasma edge elongation
    "BM_OUT": "kappa_e",
    # [-] plasma internal inductance
    "BM_OUT": "li",
    # [-] number of iterations
    "BM_OUT": "niter",
    # [1E19/m3] electron/ion density at pedestal height
    "BM_OUT": "nped",
    # [1E19/m3] electron/ion density at separatrix
    "BM_OUT": "nsep",
    # [W] additional heating power
    "BM_OUT": "Padd",
    # [W] alpha power
    "BM_OUT": "Palpha",
    # [W] Bremsstrahlung radiation power
    "BM_OUT": "Pbrem",
    # [W] Fusion power
    "BM_OUT": "Pfus",
    # [W] DD fusion power
    "BM_OUT": "PfusDD",
    # [W] DT fusion power
    "BM_OUT": "PfusDT",
    # [m] plasma perimeter
    "BM_OUT": "perim",
    # [W] Line radiation power
    "BM_OUT": "Pline",
    # [W] LH transition power
    "BM_OUT": "PLH",
    # [W] neutron fusion power
    "BM_OUT": "Pneut",
    # [W] Ohimic heating power
    "BM_OUT": "Pohm",
    # [W] total radiation power
    "BM_OUT": "Prad",
    # [W] total power across plasma separatrix
    "BM_OUT": "Psep",
    # [MW/m] Divertor challenging criterion Psep/R0
    "BM_OUT": "Psep_R0",
    # [MW * T/ m] Divertor challenging criterion Psep * Bt /(q95 * a)
    "BM_OUT": "Psep_Bt_q95_A_R0",
    # [W] Synchrotron radiation power
    "BM_OUT": "Psync",
    # [W/m2] divertor heat flux
    "BM_OUT": "qdivt",
    # [-] Edge safety factor
    "BM_OUT": "q_sep",
    # [m] Plasma minor radius
    "BM_OUT": "rpminor",
    # [Ohm] plasma resistance
    "BM_OUT": "rplas",
    # [s] energy confinement time
    "BM_OUT": "tau_e",
    # [keV] Ions/Electrons at pedestal
    "BM_OUT": "Teped",
    # [-] tolerance on safety factor profile
    "BM_OUT": "toleq",
    # [-] overall tolerance
    "BM_OUT": "tolfin",
    # [V] plasma loop voltage
    "BM_OUT": "Vloop",
    # [J] plasma thermal energy
    "BM_OUT": "Wth",
    # [-] plasma effective charge
    "BM_OUT": "Zeff",
}


def set_default_mappings():
    mappings = {}
    send = True
    recv = False
    for puts in [PLASMOD_INPUTS, PLASMOD_OUTPUTS]:
        for bm_key, pl_key in puts.items():
            mappings[bm_key] = ParameterMapping(pl_key, send=send, recv=recv)
        send = not send
        recv = not recv

    return mappings
