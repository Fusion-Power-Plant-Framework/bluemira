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
from io import StringIO
from typing import List, Any

import numpy as np

from typing import Optional, Union, List
import copy

import csv

from ..external_code import ExternalCode

DEFAULT_INPUTS = {
    ############################
    # list geometry properties
    ############################
    # [m] plasma major radius
    "R0": 9,
    # [m3] constrained plasma volume (set zero to disable volume constraining)
    "V_in": 0,
    # [-] plasma edge triangularity (used only for first iteration,
    # then iterated to constrain delta95)
    "deltaX": 1,
    # [-] plasma triangularity at 95 % flux
    "delta95": 2,
    # [-] plasma edge elongation (used only for first iteration,
    # then iterated to constrain kappa95)
    "kappaX": 3,
    # [-] plasma elongation at 95 % flux
    "kappa95": 4,
    # [-] safety factor at 95% flux surface
    "q95": 7,
    # [-] plasma aspect ratio
    "A": 3.1,
    # [T] Toroidal field at plasma center
    "Bt": 5.8,
    # [-] Tungsten concentration
    "cW": 0.0,
}

DEFAULT_OUTPUTS = {
    # [-] poloidal beta
    "_beta_p": [],
    # [-] normalized beta
    "_beta_n": [],
    # [-] toroidal beta
    "_beta_t": [],
    # [T] average poloidal field
    "_Bpav": [],
}


def get_default_inputs():
    """
    Returns the instance as a dictionary.
    """
    return copy.deepcopy(DEFAULT_INPUTS)


def get_default_outputs():
    """
    Returns the instance as a dictionary.
    """
    return copy.deepcopy(DEFAULT_OUTPUTS)


class PlasmodParameters:
    """
    List of plasmod inputs
    """

    def __init__(self, **kwargs):
        self.__options = {**get_default_inputs(), **get_default_outputs()}
        self.modify(**kwargs)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def modify(self, **kwargs):
        """
        Function to override plotting options.
        """
        if kwargs:
            for k in kwargs:
                if hasattr(self, k) and self.k.fset is not None:
                    self.k = kwargs[k]

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self.__options)}" + "\n)"

    # @property
    # def name(self):
    #     return self._options["name"]
    #
    # @name.setter
    # def name(self, val):
    #     self._options["name"] = val

    @property
    def A(self):
        return self.__options["A"]

    @A.setter
    def A(self, val):
        self.__options["A"] = val

    @property
    def Bt(self):
        return self.__options["Bt"]

    @Bt.setter
    def Bt(self, val):
        self.__options["Bt"] = val

    @property
    def cW(self):
        return self.__options["cW"]

    @cW.setter
    def cW(self, val):
        self.__options["cW"] = val

    ##### OUTPUT PARAMETERS #######
    @property
    def _beta_p(self):
        return self._options["_beta_p"]

    @property
    def _beta_n(self):
        return self._options["_beta_n"]

    @property
    def _beta_t(self):
        return self._options["_beta_t"]

    @property
    def _Bpav(self):
        return self._options["_Bpav"]


def write_input_file(params: Union[PlasmodParameters, dict], filename: str):
    '''Write a set of PlasmodParameters into a file'''

    # open input file
    fid = open(filename, "w")

    # print all input parameters
    self.print_parameter_list(fid, params)

    # close file
    fid.close()


def print_parameter_list(params: Union[PlasmodParameters, dict], fid = sys.stdout):
    '''Print a set of parameter to screen or into an open file'''
    if isinstance(params, PlasmodParameters):
        print_parameter_list(fid, params.as_dict())
    elif isinstance(params, dict):
        for k,v in params.items():
            if isinstance(v, int):
                print(params[i] + " %d" % v, file=fid)
            elif isinstance(v, float):
                print(params[i] + " % 5.4e" % v, file=fid)
    else:
        raise ValueError("Wrong input")


def read_output_files(output_file, profiles_file):
    '''Read the Plasmod profiles from the output file'''
    output = {}
    with open(profiles_file, 'r') as fd:
        reader = csv.reader(fd, delimiter='\t')
        for row in reader:
            arr = row[0].split()
            output[arr[0]] = np.array(arr[1:])
            output[arr[0]] = output[arr[0]].astype(np.float)
    return output


class PlasmodSolver(ExternalCode):
    """Plasmod solver class"""

    def __init__(self, runmode="BATCH", parameters=None, **kwargs):
        if parameters is None:
            self._parameters = PlasmodParameters()
        elif isinstance(parameters, PlasmodParameters):
            self._parameters = parameters
        elif isinstance(parameters, Dict):
            self._parameters = PlasmodParameters(**parameters)
        super().__init__(runmode)

    class Setup(ExternalCode.Setup):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test = "Test"

        def _batch(self, *args, **kwargs):
            print(f"running new {self.__class__.__name__} _batch")

    # # ******************** public attributes (plasmod inputs) ******************** #
    #
    # A = 3.1                 # [-] plasma aspect ratio
    #
    # Bt = 5.8                # [T] Toroidal field at plasma center
    #
    # cW = 0.0                # [-] Tungsten concentration
    #
    # delta95 = 0.33          # [-] plasma triangularity at 95 % flux
    # deltaX = 0.5            # [-] plasma edge triangularity (used only for first iteration, then iterated to constrain delta95)
    # dgy = 1E-5              # [-] Newton differential
    # dtmax = 1E-2            # [-] max time step between iterations
    # dtmin = 1E-2            # [-] min time step between iterations
    # dx_control = 0.2        # [-] variance of heat deposition, assimung Gaussian distribution on normalized coordinate x [nbi, ech]
    #
    # f_gwped = 0.85          # [-] Greenwald density fraction at pedestal
    # f_gws = 0.5             # [-] Greenwald density fraction at separatrix
    # f_ni = 0.0              # [-] required fraction of non inductive current, if 0, dont use CD
    # fpion = 0.5             # [-] fraction of NBI power to ions
    # fP2E = 7.1              # [-] tauparticle / tauE for D, T, He, Xe
    # fP2EAr = 1.0            # [-] tauparticle / tauE for Ar particles
    # fuelmix = 0.5           # [-] fuel mix D/T
    #
    # H = 1.0                 # [-] H-factor: input if i_modeltype = 1, calculated if i_modeltype > 1
    #
    # i_equiltype = 1         # [-] equilibrium model: 1 - EMEQ, solve equilibrium with given q95, with sawteeth. 2- EMEQ, solve with given Ip, with sawteeth
    # i_impmodel = 1          # [-] impurity model: 0 - fixed concentration, 1 - concentration fixed at pedestal top, then fixed density.
    # i_modeltype = 1         # [-] 1 - simple gyrobohm scaling with imposed H factor, > 1, other models with H in output
    # i_pedestal = 1          # [-] pedestal model: 1 fixed pedestal temperature, 2 Saarelma scaling
    # isiccir = 0             # [-] SOL routine: 0 - fit, 1 - Mattia Siccinio's model
    #
    # kappa95 = 1.65          # [-] plasma elongation at 95 % flux
    # kappaX = 1.8            # [-] plasma edge elongation (used only for first iteration, then iterated to constrain kappa95)
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
    # q95 = 4.0               # [-] safety factor at 95% flux surface
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
    #
    # # ******************** private scalar attributes (plasmod outputs) ******************** #
    #
    # _beta_p = []            # [-] poloidal beta
    # _beta_n = []            # [%] normalized beta
    # _beta_t = []            # [-] toroidal beta
    # _Bpav = []              # [T] average poloidal field
    #
    # _cAr = []               # [-] Argon concentration (ratio nAr/ne)
    # _cH = []                # [-] Hydrogen concentration (ratio nH/ne)
    # _cHe = []               # [-] Helium concentration (ratio nHe/ne)
    # _cXe = []               # [-] Xenon concentration (ratio nXe/ne)
    #
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
    #
    # def __init__(self, *args):
    #     # args is a tuple
    #     for i in range(0, len(args)-1, 2):
    #         self.__setattr__(args[i], args[i + 1])
    #
    # def write_input_file(self, filename):
    #     # open input file
    #     fid = open(filename, "w")
    #
    #     # list geometry properties
    #     geom_params = ["R0", "A", "V_in", "Bt", "deltaX", "delta95", "kappaX", "kappa95", "q95"]
    #
    #     # list numerics properties
    #     numeric_params = ["tol", "dtmin", "dtmax", "dgy", "nx", "nxt", "isiccir", "test"]
    #
    #     # list composition properties
    #     comp_params = ["fuelmix", "fP2E", "fP2EAr", "cW"]
    #
    #     # list pedestal properties
    #     pedestal_params = ["Tesep", "Teped", "rho_T", "rho_n", "Psep_PLH_min",
    #                        "Psep_PLH_max", "PsepBt_qAR_max", "Psep_R0_max", "qdivt_max"]
    #
    #     # list control parameters
    #     control_params = ["x_control", "dx_control", "Pfus_req", "q_control", "f_ni", "Pheat_max",
    #                       "i_impmodel", "i_modeltype", "i_equiltype", "i_pedestal"]
    #
    #     # confinement parameters
    #     confin_params = ["f_gwped", "f_gws", "H", "nbcdeff", "fpion"]
    #
    #     # print all input parameters
    #     params = geom_params + numeric_params + comp_params + pedestal_params + control_params + confin_params
    #     self.print_parameter_list(fid, params)
    #
    #     # close file
    #     fid.close()
    #
    # def print_parameter_list(self, fid, props):
    #     for i in range(0, len(props)-1):
    #         v = self.__getattribute__(props[i])
    #         if isinstance(v, int):
    #             print(props[i] + ' %d' % v, file=fid)
    #         elif isinstance(v, float):
    #             print(props[i] + ' % 5.4e' % v, file=fid)
    #
    # def read_output_files(self, output_file, profiles_file):
    #
    #     # read scalar output parameters
    #     with open(output_file) as f:  # open the file for reading
    #         for line in f:  # iterate over each line
    #             name, value = line.split()  # split it by whitespace
    #             self.__setattr__('_' + name, float(value))
    #
    #     # read output profiles
    #     c = StringIO(profiles_file)
    #     # n = np.loadtxt(c)
    #     # print(n)
    #     with open(profiles_file) as f:  # open the file for reading
    #         for line in f:  # iterate over each line
    #             print(line)
    #
    # @staticmethod
    # def run(plasmod_path, input_file, output_file, profiles_file):
    #     os.system('"./' + plasmod_path + '/plasmod.o" "'
    #               + input_file + '" "' + output_file + '" "' + profiles_file + '"')
    #
    # def plasma_current(self):
    #     return self._Ip
    #
    # def radiation_power(self):
    #     return self._Prad
    #
    # def fusion_power(self):
    #     return self._Pfus
    #
    # def norm_flux(self):
    #     self._X = np.linspace(0, 1, num=self.nx)
    #     return self._X
    #
    # def get_pprime(self):
    #     # dummy function to return pprime profile for testing
    #     self._pprime = np.array([-0.38578705020E+06, -0.16559348334E+06, -0.53773798366E+05, -0.15636259430E+05,   0.37115679778E+04,
    #                              0.15277180814E+05, 0.22773261901E+05, 0.27805313581E+05, 0.31180320154E+05, 0.33348369284E+05,
    #                              0.34580669336E+05, 0.35051567393E+05, 0.34879058937E+05, 0.34149804458E+05, 0.32919451691E+05,
    #                              0.25327268513E+05, 0.17462137639E+05, 0.15067715316E+05, 0.13187390251E+05, 0.11557198657E+05,
    #                              0.10139185019E+05, 0.88930594283E+04, 0.78152848718E+04, 0.69192379795E+04, 0.62389046877E+04,
    #                              0.57089866417E+04, 0.52264539695E+04, 0.47852568858E+04, 0.43806717571E+04, 0.40086262884E+04,
    #                              0.36503205347E+04, 0.33162177585E+04, 0.30202212281E+04, 0.27454397517E+04, 0.24867854095E+04,
    #                              0.22404973524E+04, 0.20071509148E+04, 0.32762743759E+04, 0.66336450411E+04, 0.76605442754E+04,
    #                              0.58171105731E+04])
    #     return self._pprime
    #
    # def get_FFrime(self):
    #     # dummy function to return FFprime profile for testing
    #     self._FFrime = np.array([+0.45011200133E+02, +0.20400731167E+02, +0.78752279995E+01, +0.35899451530E+01,
    #                              +0.14169589568E+01, +0.12630500956E+00, -0.69830414609E+00, -0.12373693924E+01,
    #                              -0.15825815769E+01, -0.17861196905E+01, -0.18804593337E+01, -0.18872157538E+01,
    #                              -0.18229437854E+01, -0.16901352943E+01, -0.15251418897E+01, -0.72696598500E+00,
    #                              +0.11697205834E+00, +0.38835770876E+00, +0.58411751279E+00, +0.76244095068E+00,
    #                              +0.89814153688E+00, +0.10488052331E+01, +0.62015709295E+00, -0.11804033821E-01,
    #                              -0.10357947441E+00, -0.94503246588E-01, -0.86074407534E-01, -0.78466874423E-01,
    #                              -0.71553303134E-01, -0.66373285594E-01, -0.61178511152E-01, -0.55163189093E-01,
    #                              -0.49757964213E-01, -0.44803021204E-01, -0.38593498630E-01, -0.30836338558E-01,
    #                              +0.20358940870E-01, +0.25914515185E-01, -0.48432312866E-01, -0.29158618440E+00, -0.60453861422E+00])
    #     return self._FFrime
