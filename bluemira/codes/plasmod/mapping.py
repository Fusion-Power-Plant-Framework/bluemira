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

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter import ParameterMapping


class Model(Enum):
    """
    Base Model Enum
    """

    @classmethod
    def info(cls):
        """
        Show Model options
        """
        infostr = f"{cls.__doc__}\n" + "\n".join(repr(l_) for l_ in list(cls))
        bluemira_print(infostr)


class ImpurityModel(Model):
    """
    Impurity Model selector

    0 - fixed concentration,
    1 - concentration fixed at pedestal top, then fixed density.

    Plasmod variable name: "i_impmodel"
    """

    FIXED = 0
    PED_FIXED = 1


class TransportModel(Model):
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


class EquilibriumModel(Model):
    """
    Equilibrium Model Selector

    1 - EMEQ solves equilibrium with given q95, with sawteeth.
    2 - EMEQ solves with given Ip, with sawteeth

    Plasmod variable name: "i_equiltype"
    """

    q95_sawtooth = 1
    Ip_sawtooth = 2


class PedestalModel(Model):
    """
    Pedestal Model Selector

    1 - fixed pedestal temperature (Teped_in),
    2 - Saarelma scaling

    Plasmod variable name: "i_pedestal"
    """

    FIX_TEMP = 1
    SAARELMA = 2


class SOLModel(Model):
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

    x = "x"  # [-] normalized toroidal flux coordinate (Phi/Phi_b)
    ne = "n_e"  # [10¹⁹/m3] electron density profile
    Te = "Te"  # [keV] Electron temperature profile
    Ti = "Ti"  # [keV] Ion temperature profile
    psi = "psi"  # [Wb] Poloidal flux profile
    phi = "phi"  # [Wb] Toroidal flux profile
    press = "pressure"  # [Pa] Plasma pressure profile
    pprime = "pprime"  # [Pa/Wb] p' profile
    ffprime = "ffprime"  # [(m*T) * (m*T) / Wb == T] FF' profile
    kprof = "kappa"  # [-] Elongation profile
    dprof = "delta"  # [-] Triangularity profile
    shif = "GS"  # [m] Grad-Shafranov shift profile
    g2 = "g2"  # [m²] < |grad V|²/r²> g2 metric coefficient's profile
    g3 = "g3"  # [m⁻²] < 1/r²> g3 metric coefficient's profile
    volprof = "V"  # [m³] Volume profile
    vprime = "Vprime"  # [m³] Volume profile
    ipol = "i_pol"  # [m*T] Poloidal current profile
    qprof = "q"  # [-] Safety factor profile
    jpar = "jpar"  # [A/m²] Parallel current density profile
    jbs = "jbs"  # [A/m²] Bootstrap parallel current density profile
    jcd = "jcd"  # [A/m²] CD parallel current density profile
    nions = "n_ion"  # [10¹⁹/m³] ion density profile
    nfuel = "n_fuel"  # [10¹⁹/m3] fuel density profile
    ndeut = "n_D"  # [10¹⁹/m3] deuterium density profile
    ntrit = "n_T"  # [10¹⁹/m3] tritium density profile
    nalf = "n_He"  # [10¹⁹/m3] helium density profile
    # Not yet enabled in plasmod
    # qrad = "q_rad" # radiation density profile
    # qneut = "q_neut" # nuetron fusion power density profile


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
    # ###########################
    # list geometry properties
    # ###########################
    # [-] plasma aspect ratio
    "A": "A",
    # [T] Toroidal field at plasma center
    "B_0": "Bt",
    # [-] plasma triangularity at 95 % flux
    "delta_95": "d95",
    # [-] plasma elongation at 95 % flux
    "kappa_95": "k95",
    # [m] plasma major radius
    "R_0": "R",
    # [m3] constrained plasma volume (set zero to disable volume constraining)
    "V_p": "volume_in",
    # ###########################
    # list composition properties
    # ############################
    # [-] fuel mix D/T
    # ###### "BM_INP": "fuelmix",
    # [-] He3 as fuel concentration
    # ###### "BM_INP": "fuelhe3",
    # [-] tauparticle / tauE for D
    # ###### "BM_INP": "globtau_d",
    # [-] tauparticle / tauE for T
    # ###### "BM_INP": "globtau_t",
    # [-] tauparticle / tauE for He
    # ###### "BM_INP": "globtau_he",
    # [-] tauparticle / tauE for Xe
    # ###### "BM_INP": "globtau_xe",
    # [-] tauparticle / tauE for Ar
    # ###### "BM_INP": "globtau_ar",
    # [-] Tungsten concentration
    # ##### "BM_INP": "cwol": 0.0,
    # [-] min P_sep/P_LH. if Psep/PLH < Psep/PLH_max -> use heating
    # ###### "BM_INP": "psepplh_inf",
    # [-] max P_sep/P_LH. if Psep/PLH > Psep/PLH_max -> use Xe
    # ###### "BM_INP": "psepplh_sup",
    # [-] position after which radiation is "edge"
    # ###### "BM_INP": "pradpos",
    # [-] radiation fraction used for core transport
    # ###### "BM_INP": "pradfrac",
    # [MW*T/m] Divertor challenging criterion Psep * Bt / (q95 * A R0)
    # if PsepBt_qAR > PsepBt_qAR_max seed Xenon
    # ###### "BM_INP": "psepb_q95AR_sup",
    # [MW/m] Divertor challenging criterion Psep / R0
    # if Psep/R0 > Psep_R0_max seed Xenon
    # ###### "BM_INP": "psep_r_sup",
    # [-] ratio of Pline(Xe)/(Psep0 - Psepcrit), or -1 to ignore
    # ###### "BM_INP": "fcoreraditv",
    # [MW/m2] max divertor heat flux -->
    # if qdivt > qdivt_sup -> seed argon
    # ###### "BM_INP": "qdivt_sup",
    # [-] compression factor between sol and div
    # ###### "BM_INP": "c_car",
    # ###########################
    # list pedestal properties
    # ############################
    # [-] normalized coordinate of pedestal density
    # ###### "BM_INP": "rho_n",
    # [-] normalized coordinate of pedestal temperature
    # ###### "BM_INP": "rho_T",
    # [keV] electrons/ions temperature at separatrix
    # ###### "BM_INP": "Tesep",
    # [-] scaling factor for p_ped scaling formula
    # ###### "BM_INP": "pedscal",
    # ###########################
    # list general inputs: control, confinement, B.C., etc
    # ############################
    # [-] Greenwald density fraction at pedestal
    # ###### "BM_INP": "f_gw",
    # [-] Greenwald density fraction at separatrix
    # ###### "BM_INP": "f_gws",
    # [-] fraction of NBI power to ions
    # ###### "BM_INP": "fpion",
    # [m*MA/MW] Normalized CD efficiency
    # ###### "BM_INP": "nbcdeff",  # tentative g_cd_nb but normalise wrt to what?
    # [m*MA/MW] Normalized EC efficiency
    # ###### "BM_INP": "eccdeff",  # tentative g_cd_nb but normalise wrt to what?
    # [-]  normalized mean location of fixed NBI heating
    # ###### "BM_INP": "x_control_nbi",
    # [-]  normalized mean location of fixed EC heating
    # ###### "BM_INP": "x_control_ech",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed NBI heating
    # ###### "BM_INP": "dx_control_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed EC heating
    # ###### "BM_INP": "dx_control_ech",
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    # ###### "BM_INP": "x_cd_nbi",
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    # ###### "BM_INP": "x_cd_ech",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for NBI heating (CD) to control Vloop or f_ni
    # ###### "BM_INP": "dx_cd_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for EC heating to control Vloop or f_ni
    # ###### "BM_INP": "dx_cd_ech",
    # [-]  normalized mean location of NBI heating for
    # controlling fusion power (Pfus = Pfus_req)
    # ###### "BM_INP": "x_fus_nbi",
    # [-]  normalized mean location of EC heating for
    # controlling fusion power (Pfus = Pfus_req)
    # ###### "BM_INP": "x_fus_ech",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control fusion power
    # ###### "BM_INP": "dx_fus_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for EC heating to control fusion power
    # ###### "BM_INP": "dx_fus_ech",
    # [-]  normalized mean location of aux. NBI heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    # ###### "BM_INP": "x_heat_nbi",
    # [-]  normalized mean location of aux. EC heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    # ###### "BM_INP": "x_heat_ech",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control H-mode
    # ###### "BM_INP": "dx_heat_nbi",
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for EC heating to control H-mode
    # ###### "BM_INP": "dx_heat_ech",
    # [keV] NBI energy
    # ###### "BM_INP": "nbi_energy",
    # [MW] required fusion power.
    # 0. - ignored
    # > 0 - Auxiliary heating is calculated to match Pfus_req
    # ###### "BM_INP": "pfus_req",
    # [-] required fraction of non inductive current, if 0, dont use CD
    "f_ni": "f_ni",
    # [MW] max allowed power for control (fusion power, H-mode)
    # ###### "BM_INP": "pheat_max",
    # [MW] fixed auxiliary heating power required for control
    "q_control": "q_control",
    # [MW] total auxiliary power  (0.) DO NOT CHANGE
    # ###### "BM_INP": "q_heat",
    # [MW] total auxiliary current drive power (0.) DO NOT CHANGE
    # ###### "BM_INP": "q_cd",
    # [MW] total fusion power (0.) DO NOT CHANGE
    # ###### "BM_INP": "q_fus",
    # [MW] ECH power (not used)
    # ###### "BM_INP": "pech": 0.0,
    # [MW] NBI power (not used)
    # ###### "BM_INP": "pnbi": 0.0,
    # [-] ratio of PCD-Pothers over Pmax - Pothers
    # ###### "BM_INP": "fcdp": -1.0,
    # [-] maximum Paux/R allowed
    # ###### "BM_INP": "maxpauxor",
    # [-] type of PLH threshold.  6 - Martin scaling. Use 6 only
    # ###### "BM_INP": "plh",
    # [-] scaling factor for newton scheme on NBI (100.)
    # ###### "BM_INP": "qnbi_psepfac",
    # [-] scale factor for newton scheme on Xe (1.e-3)
    # ###### "BM_INP": "cxe_psepfac",
    # [-] scale factor for newton scheme on Ar (1.e-4)
    # ###### "BM_INP": "car_qdivt",
    # [MW / m²] Pcontrol / S_lateral(0.)
    # ###### "BM_INP": "contrpovs",
    # [MW / m²] Pcontrol / R(0.)
    # ###### "BM_INP": "contrpovr",
}


PLASMOD_OUTPUTS = {
    # ###########################################
    # list geometry properties (geom type)
    # ###########################################
    # [m] plasma perimeter
    # ##### "BM_OUT": "perim",
    # ###########################################
    # list MHD equilibrium properties (MHD type)
    # ###########################################
    # [T] average poloidal field
    # ##### "BM_OUT": "bpolavg",
    # [-] toroidal beta
    # ##### "BM_OUT": "betator",
    # [-] poloidal beta
    "beta_p": "betapol",
    # [-] normalized beta
    "beta_N": "betan",
    # [-] plasma bootstrap current fraction
    "f_bs": "fbs",
    # [-] plasma current drive fraction
    # ##### "BM_OUT": "fcd",
    # [-] Edge safety factor
    # ##### "BM_OUT": "q_sep",
    # [-] cylindrical safety factor
    # ##### "BM_OUT": "qstar",
    # [-] plasma internal inductance
    "l_i": "rli",
    # [m²] plasma poloidal cross section area
    # ##### "BM_OUT": "Sp",
    # [m²] plasma toroidal surface
    # ##### "BM_OUT": "torsurf",
    # [m³] plasma volume
    # ##### "BM_OUT": "Vp",
    # ###########################################
    # list confinement properties (loss type)
    # ###########################################
    # [-] radiation-corrected H-factor
    "H_star": "Hcorr",
    # [s] global energy confinement time
    "tau_e": "taueff",
    # [s] electrons energy confinement time
    # ##### "BM_OUT": "tauee",
    # [s] ions energy confinement time
    # ##### "BM_OUT": "tauei",
    # [J] plasma thermal energy
    # ##### "BM_OUT": "Wth",
    # [Ohm] plasma resistance
    "res_plasma": "rplas",
    # ###########################################
    # list power properties (loss type)
    # ###########################################
    # [W] DD fusion power
    "P_fus_DD": "Pfusdd",
    # [W] DT fusion power
    "P_fus_DT": "Pfusdt",
    # [W] Fusion power
    "P_fus": "Pfus",
    # [W] neutron fusion power
    # ##### "BM_OUT": "Pneut",
    # [W] total auxiliary heating power
    # ##### "BM_OUT": "Paux",
    # [W] auxiliary heating power to electrons
    # ##### "BM_OUT": "Peaux",
    # [W] auxiliary heating power to ions
    # ##### "BM_OUT": "Piaux",
    # [W] alpha power
    # ##### "BM_OUT": "Palpha",
    # [W] total radiation power
    "P_rad": "Prad",
    # [W] core radiation power
    # ##### "BM_OUT": "Pradcore",
    # [W] core radiation power
    # ##### "BM_OUT": "Pradedge",
    # [W] total power across plasma separatrix
    "P_sep": "Psep",
    # [W] Synchrotron radiation power
    "P_sync": "Psync",
    # [W] Bremsstrahlung radiation power
    "P_brehms": "Pbrehms",
    # [W] Line radiation power
    "P_line": "Pline",
    # [W] LH transition power
    "P_LH": "PLH",
    # [W] Ohimic heating power
    "P_ohm": "Pohm",
    # [W/m2] divertor heat flux
    # ##### "BM_OUT": "qdivt",
    # [MW/m] Divertor challenging criterion Psep/R0
    # ##### "BM_OUT": "psep_r",
    # [MW * T/ m] Divertor challenging criterion Psep * Bt /(q95 * a)
    # ##### "BM_OUT": "psepb_q95AR",
    # ###########################
    # list composition properties (type comp)
    # ############################
    # [-] plasma effective charge
    "Z_eff": "Zeff",
    # ###########################
    # list pedestal properties (type ped)
    # ############################
    # [1E19/m3] electron/ion density at pedestal height
    # ##### "BM_OUT": "nped",
    # [1E19/m3] electron/ion density at separatrix
    # ##### "BM_OUT": "nsep",
    # ###########################
    # list average properties for profiles (type radp)
    # ############################
    # [1E19/m3] volume-averaged ion density
    # ##### "BM_OUT": "av_ni",
    # [1E19/m3] volume-averaged fuel density
    # ##### "BM_OUT": "av_nd",
    # [1E19/m3] volume-averaged plasma impurities density
    # ##### "BM_OUT": "av_nz",
    # [1E19/m3] volume-averaged helium density
    # ##### "BM_OUT": "av_nhe",
    # [keV] volume-averaged ions temperature
    # ##### "BM_OUT": "av_Ti"
    # [keV] volume-averaged electrons temperature
    # ##### "BM_OUT": "av_Te",
    # [keV] density-averaged electrons temperature
    # ##### "BM_OUT": "av_Ten",
}

PLASMOD_INOUTS = {
    # ###########################################
    # list geometry properties (geome type)
    # ###########################################
    # [-] plasma edge triangularity (used only for first iteration,
    # then iterated to constrain delta95)
    "delta": "d",
    # [-] plasma edge elongation (used only for first iteration,
    # then iterated to constrain kappa95)
    "kappa": "k",
    # [-] plasma minor radius
    # "BM_INP": "amin",
    # ###########################################
    # list MHD equilibrium properties (mhd type)
    # ###########################################
    # [MA] plasma current
    "I_p": "Ip",
    # [-] safety factor at 95% flux surface
    "q_95": "q95",
    # [-] plasma current inductive fraction
    # ##### "BM_OUT": "f_ni",
    # [V] target loop voltage (if lower than -1e-3, ignored)-> plasma loop voltage
    "v_burn": "v_loop",
    # ###########################
    # list composition properties
    # ############################
    # [-] Hydrogen concentration
    # ##### "BM_OUT": "cprotium",
    # [-] helium concentration
    # ##### "BM_IO": "che",
    # [-] He3 concentration
    # ##### "BM_IO": "che3",
    # [-] Argon concentration
    # ###### "BM_IO": "car",
    # [-] Xenon concentration
    # #### "BM_IO": "cxe",
    # ###########################
    # list pedestal properties
    # ############################
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    # ##### "BM_IO": "teped",
    # ###########################
    # list onfinement properties (type loss)
    # ############################
    # [-] H-factor:if i_modeltype > 1 H factor calculated
    # ##### "BM_IO": "Hfact",
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
