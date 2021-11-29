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
API for the transport code PLASMOD and related functions
"""

import copy
import csv
import os
import pprint
import subprocess
import sys
from typing import Dict, Union
from enum import auto

import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_debug
import bluemira.codes.interface as interface
from bluemira.codes.plasmod.constants import NAME as PLASMOD
from bluemira.codes.plasmod.mapping import Profiles

# Absolute path to plasmod excutable
PLASMOD_PATH = "../../../../plasmod_bluemira"

# Todo: both INPUTS and OUTPUTS must to be completed.
# DEFAULT_PLASMOD_INPUTS is the dictionary containing all the inputs as requested by Plasmod
DEFAULT_PLASMOD_INPUTS = {
    ############################
    # list geometry properties
    ############################
    # [-] plasma aspect ratio
    "A": 3.1,
    # [T] Toroidal field at plasma center
    "Bt": 5.8,
    # [-] plasma edge triangularity (used only for first iteration,
    # then iterated to constrain delta95)
    "deltaX": 0.5,
    # [-] plasma triangularity at 95 % flux
    "delta95": 0.33,
    # [-] plasma edge elongation (used only for first iteration,
    # then iterated to constrain kappa95)
    "kappaX": 1.8,
    # [-] plasma elongation at 95 % flux
    "kappa95": 1.65,
    # [-] safety factor at 95% flux surface
    "q95": 3.5,
    # [m] plasma major radius
    "R0": 9.0,
    # [m3] constrained plasma volume (set zero to disable volume constraining)
    "V_in": 0,
    ############################
    # list numerics properties
    #############################
    # [-] Newton differential
    "dgy": 1e-5,
    # [-] min time step between iterations
    "dtmin": 1e-2,
    # [-] max time step between iterations
    "dtmax": 1e-2,
    # [-] number of interpolated grid points
    "nx": 41,
    # [-] number of reduced grid points
    "nxt": 5,
    # [-] max number of iteration
    "test": 10000,
    # [-] max iteration error between transport/equilibrium iterations
    "tol": 1e-10,
    ############################
    # list transport & confinement properties
    #############################
    # [-] Greenwald density fraction at pedestal
    "f_gwped": 0.85,
    # [-] Greenwald density fraction at separatrix
    "f_gws": 0.5,
    # [-] fraction of NBI power to ions
    "fpion": 0.5,
    # [-] tauparticle / tauE for D
    "fp2e_d": 5.0,
    # [-] tauparticle / tauE for T
    "fp2e_t": 5.0,
    # [-] tauparticle / tauE for He
    "fp2e_he": 5.0,
    # [-] tauparticle / tauE for Xe
    "fp2e_xe": 5.0,
    # [-] tauparticle / tauE for Ar
    "fp2e_ar": 5.0,
    # [-] input H-factor:if i_modeltype > 1 H factor calculated
    "hfact_inp": 1.0,
    # [-] normalized coordinate of pedestal density
    "rho_n": 0.94,
    # [-] normalized coordinate of pedestal temperature
    "rho_T": 0.94,
    # [keV] electrons/ions temperature at pedestal (ignored if i_pedestal = 2)
    "Teped_inp": 5.5,
    # [keV] electrons/ions temperature at separatrix
    "Tesep": 0.1,
    ############################
    # list composition properties
    #############################
    # [-] Tungsten concentration
    "cwol": 0.0,
    # [-] fuel mix D/T
    "fuelmix": 0.5,
    ############################
    # list control & transport settings
    #############################
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for NBI heating (CD) to control Vloop or f_ni
    "dx_cd_nbi": 0.2,
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x for fixed NBI heating
    "dx_control_nbi": 0.2,
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control fusion power
    "dx_fus_nbi": 0.2,
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x, for NBI heating to control H-mode
    "dx_heat_nbi": 0.2,
    # [-] required fraction of non inductive current, if 0, dont use CD
    "f_ni": 0.0,
    # [-] equilibrium model selector:
    # 1 - EMEQ solves equilibrium with given q95, with sawteeth.
    # 2 - EMEQ solves with given Ip, with sawteeth
    "i_equiltype": 1,
    # [-] impurity model selector:
    # 0 - fixed concentration,
    # 1 - concentration fixed at pedestal top, then fixed density.
    "i_impmodel": 1,
    # [-] selector for transport model: (see trmodel.f90)
    # 1 - simple gyrobohm scaling with imposed H factor,
    # 555 - H factor scaling from F. Palermo
    # 111 - another model based on gyro-Bohm transport
    # 2 - no reference in the source code
    "i_modeltype": 1,
    # [-] pedestal model selector:
    # 1 - fixed pedestal temperature (Teped_in),
    # 2 - Saarelma scaling
    "i_pedestal": 1,
    # [-] SOL model selector:
    # 0 - fit based on Eich scaling
    # 1 - Mattia Siccinio's model
    "isiccir": 0,
    # [m*MA/MW] Normalized CD efficiency
    "nbcdeff": 0.3,
    # [MW] max allowed power for control (fusion power, H-mode)
    "Pheat_max": 100.0,
    # [MW] required fusion power.
    # 0. - ignored
    # > 0 - Auxiliary heating is calculated to match Pfus_req
    "Pfus_req": 0.0,
    # [MW*T/m] Divertor challenging criterion Psep * Bt / (q95 * A R0)
    # if PsepBt_qAR > PsepBt_qAR_max seed Xenon
    "PsepBt_qAR_max": 9.2,
    # [-] max P_sep/P_LH. if Psep/PLH > Psep/PLH_max -> use Xe
    "Psep_PLH_max": 50.0,
    # [-] min P_sep/P_LH. if Psep/PLH < Psep/PLH_max -> use heating
    "Psep_PLH_min": 1.1,
    # [MW/m] Divertor challenging criterion Psep / R0
    # if Psep/R0 > Psep_R0_max seed Xenon
    "Psep_R0_max": 17.5,
    # [MW] fixed auxiliary heating power required for control
    "q_control": 50.0,
    # [MW/m2] max divertor heat flux -->
    # if qdivt > qdivt_max -> seed argon
    "qdivt_max": 10.0,
    # [-]  normalized mean location of NBI power for
    # controlling loop voltage or f_ni
    "x_cd_nbi": 0.0,
    # [-]  normalized mean location of fixed NBI heating
    "x_control_nbi": 0.0,
    # [-]  normalized mean location of NBI heating for
    # controlling fusion power (Pfus = Pfus_req)
    "x_fus_nbi": 0.0,
    # [-]  normalized mean location of aux. heating for
    # controlling H-mode operation (P_sep/P_LH > P_sep_P_LH_min)
    "x_heat_nbi": 0.0,
    # [V] target loop voltage (if lower than -1e-3, ignored)
    "v_loop_in": -1.0e-6,
}

#


DEFAULT_PLASMOD_OUTPUTS = {
    ############################
    # list scalar outputs
    #############################
    # [m²] plasma poloidal cross section area
    "_area_pol": [],
    # [m²] plasma toroidal surface
    "_area_tor": [],
    # [-] poloidal beta
    "_beta_p": [],
    # [-] normalized beta
    "_beta_n": [],
    # [-] toroidal beta
    "_beta_t": [],
    # [T] average poloidal field
    "_Bpav": [],
    # [-] Argon concentration (ratio nAr/ne)
    "_c_ar": [],
    # [-] Hydrogen concentration (ratio nH/ne)
    "_c_h": [],
    # [-] Helium concentration (ratio nH/ne)
    "_c_he": [],
    # [-] Xenon concentration (ratio nH/ne)
    "_c_xe": [],
    # [-] plasma edge triangularity
    "_delta_e": [],
    # [-] tolerance on kinetic profiles
    "_etol": [],
    # [-] plasma bootstrap current fraction
    "_f_bs": [],
    # [-] plasma current drive fraction
    "_f_cd": [],
    # [-] plasma current inductive fraction
    "_f_ind": [],
    # [-] H-factor
    "_H": [],
    # [-] exit flag
    #  1: PLASMOD converged successfully
    # -1: Max number of iterations achieved
    # (equilibrium oscillating, pressure too high, reduce H)
    # 0: transport solver crashed (abnormal parameters
    # or too large dtmin and/or dtmin
    # -2: Equilibrium solver crashed: too high pressure
    "_i_flag": [],
    # [MA] plasma current
    "_Ip": [],
    # [-] plasma edge elongation
    "_kappa_e": [],
    # [-] plasma internal inductance
    "_li": [],
    # [-] number of iterations
    "_niter": [],
    # [1E19/m3] electron/ion density at pedestal height
    "_nped": [],
    # [1E19/m3] electron/ion density at separatrix
    "_nsep": [],
    # [W] additional heating power
    "_Padd": [],
    # [W] alpha power
    "_Palpha": [],
    # [W] Bremsstrahlung radiation power
    "_Pbrem": [],
    # [W] Fusion power
    "_Pfus": [],
    # [W] DD fusion power
    "_PfusDD": [],
    # [W] DT fusion power
    "_PfusDT": [],
    # [m] plasma perimeter
    "_perim": [],
    # [W] Line radiation power
    "_Pline": [],
    # [W] LH transition power
    "_PLH": [],
    # [W] neutron fusion power
    "_Pneut": [],
    # [W] Ohimic heating power
    "_Pohm": [],
    # [W] total radiation power
    "_Prad": [],
    # [W] total power across plasma separatrix
    "_Psep": [],
    # [MW/m] Divertor challenging criterion Psep/R0
    "_Psep_R0": [],
    # [MW * T/ m] Divertor challenging criterion Psep * Bt /(q95 * a)
    "_Psep_Bt_q95_A_R0": [],
    # [W] Synchrotron radiation power
    "_Psync": [],
    # [W/m2] divertor heat flux
    "_qdivt": [],
    # [-] Edge safety factor
    "_q_sep": [],
    # [m] Plasma minor radius
    "_rpminor": [],
    # [Ohm] plasma resistance
    "_rplas": [],
    # [s] energy confinement time
    "_tau_e": [],
    # [keV] Ions/Electrons at pedestal
    "_Teped": [],
    # [-] tolerance on safety factor profile
    "_toleq": [],
    # [-] overall tolerance
    "_tolfin": [],
    # [V] plasma loop voltage
    "_Vloop": [],
    # [J] plasma thermal energy
    "_Wth": [],
    # [-] plasma effective charge
    "_Zeff": [],
    ############################
    # list profiles
    #############################
    # [A/m²] Bootstrap parallel current density profile
    "_cubb": [],
    # [-] Triangularity profile
    "_delta": [],
    # [m³] Volume increment profile
    "_dV": [],
    # [(m*T) * (m*T) / Wb == T] FF' profile
    "_ffprime": [],
    # [m⁴] < |grad V|²> g1 metric coefficient's profile
    "_g1": [],
    # [m²] < |grad V|²/r²> g2 metric coefficient's profile
    "_g2": [],
    # [m⁻²] < 1/r²> g3 metric coefficient's profile
    "_g3": [],
    # [m*T] Poloidal current profile
    "_ipol": [],
    # [A/m²] Parallel current density profile
    "_jpar": [],
    # [A/m²] CD parallel current density profile
    "_jcdr": [],
    # [-] Elongation profile
    "_kappa": [],
    # [Pa/Wb] p' profile
    "_pprime": [],
    # [10¹⁹/m3] argon density profile
    "_nar": [],
    # [10¹⁹/m3] deuterium density profile
    "_ndeut": [],
    # [10¹⁹/m3] electron density profile
    "_nepr": [],
    # [10¹⁹/m3] fuel density profile
    "_nfuel": [],
    # [10¹⁹/m3] helium density profile
    "_nhe": [],
    # [10¹⁹/m³] ion density profile
    "_nions": [],
    # [10¹⁹/m3] tritium density profile
    "_ntrit": [],
    # [10¹⁹/m3] xenon density profile
    "_nxe": [],
    # [Wb] Toroidal flux profile
    "_phi": [],
    # [Pa] Plasma pressure profile
    "_pressure": [],
    # [Wb] Poloidal flux profile
    "_psi": [],
    # [W/m³] fusion power density profile (DT + DT)
    "_qfus": [],
    # [W/m³] neutron power density profile
    "_qneut": [],
    # [-] Safety factor profile
    "_qprf": [],
    # [W/m³] radiation power density profile
    "_qrad": [],
    # [m] Grad-Shafranov shift profile
    "_shif": [],
    # [keV] Electron temperature profile
    "_Tepr": [],
    # [keV] Ion temperature profile
    "_Tipr": [],
    # [-] normalized toroidal flux coordinate (Phi/Phi_b)
    "_x": [],
    # [m³] Volume profile
    "_V": [],
}


def get_default_plasmod_inputs():
    """
    Returns a copy of the default plasmo inputs
    """
    return copy.deepcopy(DEFAULT_PLASMOD_INPUTS)


def get_default_plasmod_outputs():
    """
    Returns a copy of the defaults plasmod outputs.
    """
    return copy.deepcopy(DEFAULT_PLASMOD_OUTPUTS)


class PlasmodParameters:
    """
    A class to mandage plasmod parameters
    """

    _options = None

    def __init__(self, **kwargs):
        self.modify(**kwargs)
        for k, v in self._options.items():
            setattr(self, k, v)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def modify(self, **kwargs):
        """
        Function to override parameters value.
        """
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]
                    setattr(self, k, self._options[k])

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._options)}" + "\n)"


# Plasmod Inputs and Outputs have been separated to make easier the writing of plasmod
# input file and the reading of outputs from file. However, other strategies could be
# applied to make use of a single PlasmodParameters instance.
class Inputs(PlasmodParameters):
    """Class for Plasmod inputs"""

    def __init__(self, **kwargs):
        self._options = get_default_plasmod_inputs()
        super().__init__(**kwargs)


class Outputs(PlasmodParameters):
    """Class for Plasmod outputs"""

    def __init__(self, **kwargs):
        self._options = get_default_plasmod_outputs()
        super().__init__(**kwargs)


def write_input_file(params: Union[PlasmodParameters, dict], filename: str):
    """Write a set of PlasmodParameters into a file"""
    print(filename)
    # open input file
    fid = open(filename, "w")

    # print all input parameters
    print_parameter_list(params, fid)

    # close file
    fid.close()


def print_parameter_list(params: Union[PlasmodParameters, dict], fid=sys.stdout):
    """
    Print a set of parameter to screen or into an open file

    Parameters
    ----------
    params: Union[PlasmodParameters, dict]
        set of parameters to be printed
    fid:
        object where to direct the output of print. Default sys.stdout (print to
        screen)

    Notes
    -----
    Used format: %d for integer, %5.4e for float, default format for other instances.
    """
    if isinstance(params, PlasmodParameters):
        print_parameter_list(params.as_dict(), fid)
    elif isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, int):
                print(k + " %d" % v, file=fid)
            if isinstance(v, float):
                print(k + " % 5.4e" % v, file=fid)
            else:
                print(f"{k} {v}", file=fid)
    else:
        raise ValueError("Wrong input")


class RunMode(interface.RunMode):
    RUN = auto()
    MOCK = auto()


class Setup(interface.Setup):
    """Setup class for Plasmod"""

    def __init__(self, parent, input_file, output_file, profiles_file):
        super().__init__(parent)
        self.input_file = input_file
        self.output_file = output_file
        self.profiles_file = profiles_file

    def _run(self, *args, **kwargs):
        """batch setup function"""
        print(self.parent._parameters)
        write_input_file(self.parent._parameters, self.parent.setup_obj.input_file)

    def _mock(self, *args, **kwargs):
        """Mock setup function"""
        print(self.parent._parameters)


class Run(interface.Run):
    def _run(self, *args, **kwargs):
        print("run batch")
        super()._run_subprocess(
            [
                f"{PLASMOD_PATH}/plasmod.o",
                f"{self.parent.setup_obj.input_file}",
                f"{self.parent.setup_obj.output_file}",
                f"{self.parent.setup_obj.profiles_file}",
            ]
        )

    def _mock(self, *args, **kwargs):
        print("run mock")
        write_input_file(self.parent._parameters, self.parent.setup_obj.input_file)
        print(
            f"{PLASMOD_PATH}/plasmod.o {self.parent.setup_obj.input_file} "
            f"{self.parent.setup_obj.output_file} "
            f"{self.parent.setup_obj.profiles_file}"
        )


class Teardown(interface.Teardown):
    def _run(self, *args, **kwargs):
        output = self.read_output_files(self.parent.setup_obj.output_file)
        self.parent._out_params.modify(**output)
        self._check_return_value()
        output = self.read_output_files(self.parent.setup_obj.profiles_file)
        self.parent._out_params.modify(**output)
        print_parameter_list(self.parent._out_params)

    def _mock(self, *args, **kwargs):
        output = self.ead_output_files(self.parent.setup_obj.output_file)
        self.parent._out_params.modify(**output)
        output = self.read_output_files(self.parent.setup_obj.profiles_file)
        self.parent._out_params.modify(**output)
        print_parameter_list(self.parent._out_params)

    @staticmethod
    def read_output_files(output_file):
        """Read the Plasmod output parameters from the output file"""
        output = {}
        with open(output_file, "r") as fd:
            reader = csv.reader(fd, delimiter="\t")
            for row in reader:
                arr = row[0].split()
                output_key = "_" + arr[0]
                output_value = arr[1:]
                if len(output_value) > 1:
                    output[output_key] = np.array(arr[1:], dtype=np.float)
                else:
                    output[output_key] = float(arr[1])
        return output

    def _check_return_value(self):
        # [-] exit flag
        #  1: PLASMOD converged successfully
        # -1: Max number of iterations achieved
        # (equilibrium oscillating, pressure too high, reduce H)
        # 0: transport solver crashed (abnormal parameters
        # or too large dtmin and/or dtmin
        # -2: Equilibrium solver crashed: too high pressure
        exit_flag = self.parent._out_params._i_flag
        if exit_flag != 1:
            if exit_flag == -2:
                raise CodesError(
                    "PLASMOD error" "Equilibrium solver crashed: too high pressure"
                )
            elif exit_flag == -1:
                raise CodesError(
                    "PLASMOD error"
                    "Max number of iterations reached"
                    "equilibrium oscillating probably as a result of the pressure being too high"
                    "reducing H may help"
                )
            elif not exit_flag:
                raise CodesError(
                    "PLASMOD error" "Abnormal paramters, possibly dtmax/dtmin too large"
                )
        else:
            bluemira_debug("PLASMOD converged successfully")


class PlasmodSolver(interface.FileProgramInterface):
    """Plasmod solver class"""

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self,
        runmode="run",
        params=None,
        input_file="plasmod_input.dat",
        output_file="outputs.dat",
        profiles_file="profiles.dat",
    ):
        # todo: add a path variable where files are stored
        if params is None:
            self._parameters = Inputs()
        elif isinstance(params, Inputs):
            self._parameters = params
        elif isinstance(params, Dict):
            self._parameters = Inputs(**params)
        self._out_params = Outputs()
        super().__init__(
            runmode, params, PLASMOD, input_file, output_file, profiles_file
        )

    def get_profile(self, profile):
        return getattr(self._out_params, Profiles(profile).name)

    def get_profiles(self, profiles):
        profiles_dict = {}
        for profile in profiles:
            profiles_dict[profile] = get_profile(profile)
        return profiles_dict[profile]
