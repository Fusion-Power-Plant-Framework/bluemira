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
PLASMOD mappings
"""
from enum import Enum

from bluemira.base.parameter import ParameterMapping


class ImpurityModel(Enum):
    """
    Impurity Model selector

    0 - fixed concentration,
    1 - concentration fixed at pedestal top, then fixed density.

    Plasmod variable name: "i_impmodel"
    """

    FIXED = 0
    PED_FIXED = 1


class TransportModel(Enum):
    """
    Transport Model Selector

    1 - simple gyrobohm scaling with imposed H factor,
    555 - H factor scaling from F. Palermo
    111 - another model based on gyro-Bohm transport
    2 - no reference in the source code

    Plasmod variable name: "i_modeltype"
    """

    GYROBOHM_1 = 1
    GYROBOHM_2 = 111
    UNKNOWN = 2
    H_FACTOR = 555


class EquilibriumModel(Enum):
    """
    Equilibrium Model Selector

    1 - EMEQ solves equilibrium with given q95, with sawteeth.
    2 - EMEQ solves with given Ip, with sawteeth

    Plasmod variable name: "i_equiltype"
    """

    q95_sawtooth = 1
    Ip_sawtooth = 2


class PedestalModel(Enum):
    """
    Pedestal Model Selector

    1 - fixed pedestal temperature (Teped_in),
    2 - Saarelma scaling

    Plasmod variable name: "i_pedestal"
    """

    FIX_TEMP = 1
    SAARELMA = 2


class SOLModel(Enum):
    """
    SOL Model Selector:

    0 - fit based on Eich scaling
    1 - Mattia Siccinio's model

    Plasmod variable name: "isiccir"
    """

    EICH_FIT = 0
    SICCINIO = 1


class Profiles(Enum):
    """
    Profile Selector
    """

    cubb = "cubb"  # [A/m²] Bootstrap parallel current density profile
    delta = "delta"  # [-] Triangularity profile
    dV = "dV"  # noqa :N815 # [m³] Volume increment profile
    ffprime = "ffprime"  # [(m*T) * (m*T) / Wb == T] FF' profile
    g1 = "g1"  # [m⁴] < |grad V|²> g1 metric coefficient's profile
    g2 = "g2"  # [m²] < |grad V|²/r²> g2 metric coefficient's profile
    g3 = "g3"  # [m⁻²] < 1/r²> g3 metric coefficient's profile
    ipol = "i_pol"  # [m*T] Poloidal current profile
    jpar = "jpar"  # [A/m²] Parallel current density profile
    jcdr = "jcdr"  # [A/m²] CD parallel current density profile
    kappa = "kappa"  # [-] Elongation profile
    pprime = "pprime"  # [Pa/Wb] p' profile
    nar = "n_Ar"  # [10¹⁹/m3] argon density profile
    ndeut = "n_D"  # [10¹⁹/m3] deuterium density profile
    nepr = "n_e"  # [10¹⁹/m3] electron density profile
    nfuel = "n_fuel"  # [10¹⁹/m3] fuel density profile
    nhe = "n_He"  # [10¹⁹/m3] helium density profile
    nions = "n_ion"  # [10¹⁹/m³] ion density profile
    ntrit = "n_T"  # [10¹⁹/m3] tritium density profile
    nxe = "n_Xe"  # [10¹⁹/m3] xenon density profile
    phi = "phi"  # [Wb] Toroidal flux profile
    pressure = "pressure"  # [Pa] Plasma pressure profile
    psi = "psi"  # [Wb] Poloidal flux profile
    q_fus = "qfus"  # [W/m³] fusion power density profile (DT + DT)
    qneut = "q_neut"  # [W/m³] neutron power density profile
    qprf = "q"  # [-] Safety factor profile
    qrad = "qrad"  # [W/m³] radiation power density profile
    shif = "GS"  # [m] Grad-Shafranov shift profile
    Tepr = "Te"  # [keV] Electron temperature profile
    Tipr = "Ti"  # [keV] Ion temperature profile
    x = "x"  # [-] normalized toroidal flux coordinate (Phi/Phi_b)
    V = "V"  # [m³] Volume profile


# TODO
# define all build tweaks properly
# Link all BM parameters
# Link all plasmod outputs

PLASMOD_INPUTS = {

    ############################
    # list numerics properties
    #############################
    # [-] max iteration error between transport/equilibrium iterations
    # ###### "BM_INP": "tol",
    # [-] min time step between iterations
    # ###### "BM_INP": "dtmin",
    # [-] max time step between iterations
    # ###### "BM_INP": "dtmax",
    # [-] exponent of jipperdo2
    # ###### "BM_INP": "dtmaxmin",
    # [-] stabilizing diff for TGLF in m²/s
    # ###### "BM_INP": "dtmaxmax",
    # [-] tolerance above which TGLF should be always called
    # ###### "BM_INP": "dtminmax",
    # [-] !time step
    # ###### "BM_INP": "dt",
    # [-] !decrease of dt
    # ###### "BM_INP": "dtinc",
    # [-] !increase of dt
    # ###### "BM_INP": "Ainc",
    # [-] max number of iteration
    # ###### "BM_INP": "test",
    # [-] ! multiplier of etolm that should not be overcome
    # ###### "BM_INP": "tolmin",
    # [-] Newton differential
    # ###### "BM_INP": "dgy",
    # [-] !exponent of jipperdo
    # ###### "BM_INP": "eopt",
    # [-] first radial grid point
    # ###### "BM_INP": "capA",
    # [-] diagnostics for ASTRA (0 or 1)
    # ###### "BM_INP": "i_diagz",
    # [-] SOL model selector:
    # ###### "BM_INP": "isiccir": 0,
    # [-] sawtooth correction of q
    # ###### "BM_INP": "isawt",
    # [-] number of interpolated grid points
    # ###### "BM_INP": "nx",
    # [-] number of reduced grid points
    # ###### "BM_INP": "nxt",
    # [-] number of unknowns in the transport solver
    # ###### "BM_INP": "nchannels",
    # [-] impurity model selector:
    # ###### "BM_INP": "i_impmodel",
    # [-] selector for transport model
    # ###### "BM_INP": "i_modeltype",
    # [-] equilibrium model selector:
    # ###### "BM_INP": "i_equiltype",
    # [-] pedestal model selector:
    # ###### "BM_INP": "i_pedestal",
    # [-] number of tglf points, below positions
    # ###### "BM_INP": "ntglf",
    # [-] tglf points, position 1
    # ###### "BM_INP": "xtglf_1",
    # [-] tglf points, position 2
    # ###### "BM_INP": "xtglf_2",
    # [-] tglf points, position 3
    # ###### "BM_INP": "xtglf_3",
    # [-] tglf points, position 4,
    # ###### "BM_INP": "xtglf_4",
    # [-] tglf points, position 5
    # ###### "BM_INP": "xtglf_5",
    # [-] tglf points, position 6
    # ###### "BM_INP": "xtglf_6",
    # [-] tglf points, position 7
    # ###### "BM_INP": "xtglf_7",
    # [-] tglf points, position 8
    # ###### "BM_INP": "xtglf_8",
    # [-] tglf points, position 9
    # ###### "BM_INP": "xtglf_9",
    # [-] tglf points, position 10
    # ###### "BM_INP": "xtglf_10",
    # [-] tglf points, position 11
    # ###### "BM_INP": "xtglf_11",

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
    # list transport & confinement properties
    #############################
    # [-] Greenwald density fraction at pedestal
    # ###### "BM_INP": "f_gwped",
    # [-] Greenwald density fraction at separatrix
    # ###### "BM_INP": "f_gws",
    # [-] fraction of NBI power to ions
    # ###### "BM_INP": "fpion",
    # [-] tauparticle / tauE for D
    # ###### "BM_INP": "fp2e_d",
    # [-] tauparticle / tauE for T
    # ###### "BM_INP": "fp2e_t",
    # [-] tauparticle / tauE for He
    # ###### "BM_INP": "fp2e_he",
    # [-] tauparticle / tauE for Xe
    # ###### "BM_INP": "fp2e_xe",
    # [-] tauparticle / tauE for Ar
    # ###### "BM_INP": "fp2e_ar",
    # [-] normalized coordinate of pedestal density
    # ###### "BM_INP": "rho_n",
    # [-] normalized coordinate of pedestal temperature
    # ###### "BM_INP": "rho_T",
    # [keV] electrons/ions temperature at separatrix
    # ###### "BM_INP": "Tesep",
    ############################
    # list composition properties
    #############################
    # [-] Tungsten concentration
    # ###### "BM_INP": "cwol",
    # [-] fuel mix D/T
    # ###### "BM_INP": "fuelmix",
    ############################
    # list control & transport settings
    #############################
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for NBI heating (CD) to control Vloop or f_ni
    # ###### "BM_INP": "dx_cd_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed NBI heating
    # ###### "BM_INP": "dx_control_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control fusion power
    # ###### "BM_INP": "dx_fus_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control H-mode
    # ###### "BM_INP": "dx_heat_nbi",
    # [-] required fraction of non inductive current, if 0, dont use CD
    "f_ni": "f_ni",
    # [m*MA/MW] Normalized CD efficiency
    # ###### "BM_INP": "nbcdeff",  # tentative g_cd_nb but normalise wrt to what?
    # [MW] max allowed power for control (fusion power, H-mode)
    # ###### "BM_INP": "Pheat_max",
    # [MW] required fusion power.
    # 0. - ignored
    # > 0 - Auxiliary heating is calculated to match Pfus_req
    # ###### "BM_INP": "Pfus_req",
    # [MW*T/m] Divertor challenging criterion Psep * Bt / (q95 * A R0)
    # if PsepBt_qAR > PsepBt_qAR_max seed Xenon
    # ###### "BM_INP": "PsepBt_qAR_max",
    # [-] max P_sep/P_LH. if Psep/PLH > Psep/PLH_max -> use Xe
    # ###### "BM_INP": "Psep_PLH_max",
    # [-] min P_sep/P_LH. if Psep/PLH < Psep/PLH_max -> use heating
    # ###### "BM_INP": "Psep_PLH_min",
    # [MW/m] Divertor challenging criterion Psep / R0
    # if Psep/R0 > Psep_R0_max seed Xenon
    # ###### "BM_INP": "Psep_R0_max",
    # [MW] fixed auxiliary heating power required for control
    "q_control": "q_control",
    # [MW/m2] max divertor heat flux -->
    # if qdivt > qdivt_max -> seed argon
    # ###### "BM_INP": "qdivt_max",
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    # ###### "BM_INP": "x_cd_nbi",
    # [-]  normalized mean location of fixed NBI heating
    # ###### "BM_INP": "x_control_nbi",
    # [-]  normalized mean location of NBI heating for
    # controlling fusion power (Pfus = Pfus_req)
    # ###### "BM_INP": "x_fus_nbi",
    # [-]  normalized mean location of aux. heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    # ###### "BM_INP": "x_heat_nbi",
}

#


PLASMOD_OUTPUTS = {
    ############################
    # list scalar outputs
    #############################
    # [m²] plasma poloidal cross section area
    # ##### "BM_OUT": "area_pol",
    # [m²] plasma toroidal surface
    # ##### "BM_OUT": "area_tor",
    # [-] poloidal beta
    "beta_p": "beta_p",
    # [-] normalized beta
    "beta_N": "betan",
    # [-] toroidal beta
    # ##### "BM_OUT": "beta_t",
    # [T] average poloidal field
    # ##### "BM_OUT": "Bpav",
    # [-] Argon concentration (ratio nAr/ne)
    # ##### "BM_OUT": "c_ar",
    # [-] Hydrogen concentration (ratio nH/ne)
    # ##### "BM_OUT": "c_h",
    # [-] Helium concentration (ratio nH/ne)
    # ##### "BM_OUT": "c_he",
    # [-] Xenon concentration (ratio nH/ne)
    # ##### "BM_OUT": "c_xe",
    # [-] plasma edge triangularity
    "delta": "delta_e",
    # [-] tolerance on kinetic profiles
    # ##### "BM_OUT": "etol",
    # [-] plasma bootstrap current fraction
    "f_bs": "f_bs",
    # [-] plasma current drive fraction
    # ##### "BM_OUT": "f_cd",
    # [-] plasma current inductive fraction
    # ##### "BM_OUT": "f_ind",
    # [MA] plasma current
    "I_p": "Ip",
    # [-] plasma edge elongation
    "kappa": "kappa_e",
    # [-] plasma internal inductance
    "l_i": "li",
    # [-] number of iterations
    # ##### "BM_OUT": "niter",
    # [1E19/m3] electron/ion density at pedestal height
    # ##### "BM_OUT": "nped",
    # [1E19/m3] electron/ion density at separatrix
    # ##### "BM_OUT": "nsep",
    # [W] additional heating power
    # ##### "BM_OUT": "Padd",
    # [W] alpha power
    # ##### "BM_OUT": "Palpha",
    # [W] Bremsstrahlung radiation power
    "P_brehms": "Pbrem",
    # [W] Fusion power
    "P_fus": "Pfus",
    # [W] DD fusion power
    "P_fus_DD": "PfusDD",
    # [W] DT fusion power
    "P_fus_DT": "PfusDT",
    # [m] plasma perimeter
    # ##### "BM_OUT": "perim",
    # [W] Line radiation power
    "P_line": "Pline",
    # [W] LH transition power
    "P_LH": "PLH",
    # [W] neutron fusion power
    # ##### "BM_OUT": "Pneut",
    # [W] Ohimic heating power
    "P_ohm": "Pohm",
    # [W] total radiation power
    "P_rad": "Prad",
    # [W] total power across plasma separatrix
    "P_sep": "Psep",
    # [MW/m] Divertor challenging criterion Psep/R0
    # ##### "BM_OUT": "Psep_R0",
    # [MW * T/ m] Divertor challenging criterion Psep * Bt /(q95 * a)
    # ##### "BM_OUT": "Psep_Bt_q95_A_R0",
    # [W] Synchrotron radiation power
    "P_sync": "Psync",
    # [W/m2] divertor heat flux
    # ##### "BM_OUT": "qdivt",
    # [-] Edge safety factor
    # ##### "BM_OUT": "q_sep",
    # [m] Plasma minor radius
    # ##### "BM_OUT": "rpminor",
    # [Ohm] plasma resistance
    "res_plasma": "rplas",
    # [s] energy confinement time
    "tau_e": "tau_e",
    # [-] tolerance on safety factor profile
    # ##### "BM_OUT": "toleq",
    # [-] overall tolerance
    # ##### "BM_OUT": "tolfin",
    # [J] plasma thermal energy
    # ##### "BM_OUT": "Wth",
    # [-] plasma effective charge
    "Z_eff": "Zeff",
}

PLASMOD_INOUTS = {
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    # ##### "BM_IO": "Teped",
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    # ##### "BM_IO": "Teped_inp",
    # [-] input H-factor:if i_modeltype > 1 H factor calculated
    # ##### "BM_INP": "hfact_inp",
    # [-] H-factor
    # ##### "BM_OUT": "H",
    # [-] H-factor (radiation corrected)
    "H_star": "Hcorr",
    # [V] target loop voltage (if lower than -1e-3, ignored)-> plasma loop voltage
    "v_burn": "Vloop",
}


def create_mapping():
    """
    Creates mappings for plasmod

    Returns
    -------
    mappings: Dict
        A mapping from bluemira names to a plasmod ParameterMapping

    """
    mappings = {}
    ins = {"send": True, "recv": False}
    outs = {"send": False, "recv": True}
    inouts = {"send": True, "recv": True}
    for puts, sr in [
        [PLASMOD_INPUTS, ins],
        [PLASMOD_OUTPUTS, outs],
        [PLASMOD_INOUTS, inouts],
    ]:
        for bm_key, pl_key in puts.items():
            mappings[bm_key] = ParameterMapping(pl_key, send=sr["send"], recv=sr["recv"])

    return mappings
