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

import os
import sys

import numpy as np

from typing import Union, Dict
import copy

import csv
import pprint

from ..external_code import ExternalCode

# Absolute path to plasmod excutable
PLASMOD_PATH = ""

# Todo: both INPUTS and OUTPUTS must to be completed.
#DEFAULT_PLASMOD_INPUTS is the dictionary containing all the inputs as requested by Plasmod
DEFAULT_PLASMOD_INPUTS = {
    ############################
    # list geometry properties
    ############################
    # [m] plasma major radius
    "R0": 9,
    # [m3] constrained plasma volume (set zero to disable volume constraining)
    "V_in": 0,
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
    "q95": 4,
    # [-] plasma aspect ratio
    "A": 3.1,
    # [T] Toroidal field at plasma center
    "Bt": 5.8,
    # [-] Tungsten concentration
    "cW": 0.0,
    # [-] Newton differential
    "dgy": 1e-5,
    # [-] max time step between iterations
    "dtmax": 1e-2,
    # [-] min time step between iterations
    "dtmin": 1e-2,
    # [-] variance of heat deposition, assimung Gaussian distribution on
    # normalized coordinate x [nbi, ech]
    "dx_control": 0.2,
    # [-] Greenwald density fraction at pedestal
    "f_gwped": 0.85,
    # [-] Greenwald density fraction at separatrix
    "f_gws": 0.5,
    # [-] required fraction of non inductive current, if 0, dont use CD
    "f_ni": 0.0,
    # [-] fraction of NBI power to ions
    "fpion": 0.5,
    # [-] tauparticle / tauE for D, T, He, Xe
    "fP2E": 7.1,
    # [-] tauparticle / tauE for Ar particles
    "fP2EAr": 1.0,
    # [-] fuel mix D/T
    "fuelmix": 0.5,
}


#
# H = 1.0                 # [-] H-factor: input if i_modeltype = 1, calculated if i_modeltype > 1
#
# i_equiltype = 1         # [-] equilibrium model: 1 - EMEQ, solve equilibrium with given q95, with sawteeth. 2- EMEQ, solve with given Ip, with sawteeth
# i_impmodel = 1          # [-] impurity model: 0 - fixed concentration, 1 - concentration fixed at pedestal top, then fixed density.
# i_modeltype = 1         # [-] 1 - simple gyrobohm scaling with imposed H factor, > 1, other models with H in output
# i_pedestal = 1          # [-] pedestal model: 1 fixed pedestal temperature, 2 Saarelma scaling
# isiccir = 0             # [-] SOL routine: 0 - fit, 1 - Mattia Siccinio's model
#

# nbcdeff = 0.3           # [m*MA/MW] * CD = this * PCD   units: m*MA/MW (MA/m^2 * m^3/MW)
# name = 'DEMO'           # [-] plasmod instance name
# nx = 41                 # [-] number of interpolated grid points (better not modify this parameter)
# nxt = 5                 # [-] number of reduced grid points
#
# Pheat_max = 50.         # [MW] max allowed power for heating + CD + fusion control
# Pfus_req = 0.           # [MW] target fusion power. If 0., not used (otherwise it would be controlled with Pauxheat)
# PsepBt_qAR_max = 9.2    # [MW * T / m] Psep * Bt / (q95 * A R0) max
# Psep_PLH_max = 50.0     # [-] Psep/PLH max -> use Xe
# Psep_PLH_min = 1.1      # [-] Psep/PLH min -> H mode
# Psep_R0_max = 17.5      # [MW/m] Psep/R0 max
#
# q_control = 50.         # [MW] minimum power required for control, e.g. auxiliary power in MW
# qdivt_max = 10.         # [MW/m2] max divertor heat flux --> calculate argon concentration
#
# rho_n = 0.94            # [-] normalized coordinate of pedestal height
# rho_T = 0.94            # [-] normalized coordinate of pedestal height
# R0 = 9.                 # [m] plasma major radius
#
# test = 10000            # [-] max number of iteration
# tol = 1E-10             # [-] iteration error between transport/equilibrium iterations
# Teped = 5.5             # [keV] electron temperature at pedestal (overridden by calculated value from Samuli scaling if i_pedestal = 2)
# Tesep = 0.1             # [keV] electron temperature at separatrix
#
# x_control = 0.0         # [-]  normalized location of heat deposition [nbi, ech]
#
# V_in = 0                # [m3] constrained plasma volume (set zero to disable volume constraining)



DEFAULT_PLASMOD_OUTPUTS = {
    # [-] poloidal beta
    "_beta_p": [],
    # [-] normalized beta
    "_beta_n": [],
    # [-] toroidal beta
    "_beta_t": [],
    # [T] average poloidal field
    "_Bpav": [],
    # [(m*T) * (m*T) / Wb == T] FF' profile
    "_FFprime": [],
    # [-] Argon concentration (ratio nAr/ne)
    "_cAr": [],
    # [-] Hydrogen concentration (ratio nH/ne)
    "_cH": [],
    # [-] Helium concentration (ratio nH/ne)
    "_cHe": [],
    # [-] Xenon concentration (ratio nH/ne)
    "_cXe": [],
}


#
# # ******************** private scalar attributes (plasmod outputs) ******************** #

# _deltaE = []            # [-] plasma edge triangularity
#
# _fBS = []               # [-] plasma bootstrap current fraction
# _fCD = []               # [-] plasma current drive fraction
# _find = []              # [-] plasma current inductive fraction
#
# _kappaE = []            # [-] plasma edge elongation
#
# _Ip = []                # [MA] plasma current
#
# _li = []                # [-] plasma internal inductance
# _Lp = []                # [H] plasma inductance
#
# _nped = []              # [1E19/m3] electron/ion density at pedestal height
# _nsep = []              # [1E19/m3] electron/ion density at separatrix
#
# _Padd = []              # [MW] additional heating power
# _Palpha = []            # [MW] alpha power
# _Pbrem = []             # [MW] Bremsstrahlung radiation power
# _Pfus = []              # [MW] Fusion power
# _PfusDD = []            # [MW] DD fusion power
# _PfusDT = []            # [MW] DT fusion power
# _Pline = []             # [MW] Line radiation power
# _PLH = []               # [MW] LH transition power
# _Pneut = []             # [MW] neutron fusion power
# _Poh = []               # [MW] Ohimic heating power
# _Prad = []              # [MW] total radiation power
# _Psep = []              # [MW] total power across plasma separatrix
# _Psep_R0 = []           # [MW/m] Psep/R0 ratio
# _Psep_Bt_q95_A_R0 = []  # [MW * T/ m] Psep * Bt /(q95 * a)
# _Psync = []             # [MW] Synchrotron radiation power
#
# _qEdge = []             # [-] Edge safety factor
#
# _rpminor = []           # [m] plasma minor radius
# _Rp = []                # [Ohm] plasma resistance
#
# _Sp = []                # [m2] plasma poloidal cross section area
# _St = []                # [m2] plasma toroidal surface
#
# _tauE = []              # [s] energy confinement time
#
# _Vloop = []             # [V] plasma loop voltage
#
# _Wth = []               # [MJ] plasma thermal energy
#
# _Zeff = []              # [-] plasma effective charge
#
# # ******************** private array  (plasmod outputs) ******************** #
# _FFprime = []           # [(m*T) * (m*T) / Wb == T] FF' profile
#
# _pprime = []            # [Pa/Wb] pprime profile
# _Psi = []               # [Wb] Poloidal flux profile
#
# _x = []                 # [-] normalized toroidal flux coordinate (Phi/Phi_b)
# _X = []                 # [-] normalized poloidal flux coordinate (sqrt(Psi_ax - Psi)/(Psi_ax - Psi_b)))

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
                output[output_key] = np.array(arr[1:])
                output[output_key] = output[output_key].astype(np.float)
            else:
                output[output_key] = float(arr[1])
    return output



class PlasmodSolver(ExternalCode):
    """Plasmod solver class"""

    def __init__(
        self,
        runmode="BATCH",
        input_params=None,
        input_file="plasmod_input.dat",
        output_file="outputs.dat",
        profiles_file="profiles.dat",
    ):
        #todo: add a path variable where files are stored
        if input_params is None:
            self._parameters = Inputs()
        elif isinstance(input_params, Inputs):
            self._parameters = input_params
        elif isinstance(input_params, Dict):
            self._parameters = Inputs(**input_params)
        self._out_params = Outputs()
        super().__init__(runmode, input_file, output_file, profiles_file)

    class Setup(ExternalCode.Setup):
        """Setup class for Plasmod"""
        def __init__(self, outer, input_file, output_file, profiles_file):
            super().__init__(outer)
            self.input_file = input_file
            self.output_file = output_file
            self.profiles_file = profiles_file

        def _batch(self, *args, **kwargs):
            """batch setup function"""
            print(self.outer._parameters)

        def _mock(self, *args, **kwargs):
            """Mock setup function"""
            print(self.outer._parameters)

    class Run(ExternalCode.Run):
        def _batch(self, *args, **kwargs):
            print("run batch")
            write_input_file(self.outer._parameters, self.outer.setup_obj.input_file)
            os.system(
                f"{PLASMOD_PATH}/plasmod.o '{self.outer.setup_obj.input_file}' '"
                f"{self.outer.setup_obj.output_file}' '"
                f"{self.outer.setup_obj.profiles_file}'"
            )

        def _mock(self, *args, **kwargs):
            print("run mock")
            write_input_file(self.outer._parameters, self.outer.setup_obj.input_file)
            print(
                f"{PLASMOD_PATH}/plasmod.o '{self.outer.setup_obj.input_file}' '"
                f"{self.outer.setup_obj.output_file}' '"
                f"{self.outer.setup_obj.profiles_file}'"
            )

    class Teardown(ExternalCode.Teardown):
        def _batch(self, *args, **kwargs):
            output = read_output_files(self.outer.setup_obj.output_file)
            self.outer._out_params.modify(**output)
            output = read_output_files(self.outer.setup_obj.profiles_file)
            self.outer._out_params.modify(**output)
            print_parameter_list(self.outer._out_params)

        def _mock(self, *args, **kwargs):
            output = read_output_files(self.outer.setup_obj.output_file)
            self.outer._out_params.modify(**output)
            output = read_output_files(self.outer.setup_obj.profiles_file)
            self.outer._out_params.modify(**output)
            print_parameter_list(self.outer._out_params)
