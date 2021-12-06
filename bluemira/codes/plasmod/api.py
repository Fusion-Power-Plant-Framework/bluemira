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
import json
import pprint
import sys
from enum import Enum, auto
from typing import Dict, Union

import numpy as np

import bluemira.codes.interface as interface
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import NAME as PLASMOD
from bluemira.codes.plasmod.mapping import (
    EquilibriumModel,
    ImpurityModel,
    PedestalModel,
    Profiles,
    SOLModel,
    TransportModel,
    set_default_mappings,
)
from bluemira.utilities.tools import CommentJSONDecoder

# Todo: both INPUTS and OUTPUTS must to be completed. Moved to json files
# DEFAULT_PLASMOD_INPUTS is the dictionary containing all the inputs as requested by Plasmod


def get_default_plasmod_inputs():
    """
    Returns a copy of the default plasmo inputs
    """
    path = get_bluemira_path("codes/plasmod")
    with open(path + "/PLASMOD_DEFAULT_IN.json") as jfh:
        return json.load(jfh, cls=CommentJSONDecoder)


def get_default_plasmod_outputs():
    """
    Returns a copy of the defaults plasmod outputs.
    """
    path = get_bluemira_path("codes/plasmod")
    with open(path + "/PLASMOD_DEFAULT_OUT.json") as jfh:
        return json.load(jfh, cls=CommentJSONDecoder)


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
            if isinstance(v, Enum):
                print(f"{k} {v.value:d}", file=fid)
            if isinstance(v, int):
                print(f"{k} {v:d}", file=fid)
            elif isinstance(v, float):
                print(f"{k} {v:5.4e}", file=fid)
            else:
                print(f"{k} {v}", file=fid)
    else:
        raise ValueError("Wrong input")


class RunMode(interface.RunMode):
    RUN = auto()
    MOCK = auto()


class Setup(interface.Setup):
    """Setup class for Plasmod"""

    def __init__(self, parent, input_file, output_file, profiles_file, **kwargs):
        super().__init__(parent)
        self.input_file = input_file
        self.output_file = output_file
        self.profiles_file = profiles_file

    def _run(self, *args, **kwargs):
        """Run setup function"""
        print(self.parent._parameters)
        write_input_file(self.parent._parameters, self.parent.setup_obj.input_file)

    def _mock(self, *args, **kwargs):
        """Mock setup function"""
        print(self.parent._parameters)
        write_input_file(self.parent._parameters, self.parent.setup_obj.input_file)


class Run(interface.Run):
    _binary = "transporz"  # Who knows why its not called plasmod

    def __init__(self, parent, **kwargs):
        super().__init__(parent, kwargs.pop("binary", self._binary))

    def _run(self, *args, **kwargs):
        bluemira_debug("Mode: run")
        super()._run_subprocess(
            [
                self._binary,
                self.parent.setup_obj.input_file,
                self.parent.setup_obj.output_file,
                self.parent.setup_obj.profiles_file,
            ]
        )

    def _mock(self, *args, **kwargs):
        bluemira_debug("Mode: mock")
        print(
            f"{self._binary} {self.parent.setup_obj.input_file} "
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
                    f"{PLASMOD} error" "Equilibrium solver crashed: too high pressure"
                )
            elif exit_flag == -1:
                raise CodesError(
                    f"{PLASMOD} error"
                    "Max number of iterations reached"
                    "equilibrium oscillating probably as a result of the pressure being too high"
                    "reducing H may help"
                )
            elif not exit_flag:
                raise CodesError(
                    f"{PLASMOD} error"
                    "Abnormal paramters, possibly dtmax/dtmin too large"
                )
        else:
            bluemira_debug(f"{PLASMOD} converged successfully")


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
        build_tweaks=None,
        input_file="plasmod_input.dat",
        output_file="outputs.dat",
        profiles_file="profiles.dat",
        binary="transporz",
    ):
        # todo: add a path variable where files are stored
        if params is None:
            self._parameters = Inputs()
        elif isinstance(params, Inputs):
            self._parameters = params
        elif isinstance(params, Dict):
            self._parameters = Inputs(**params)
        self._check_models()
        self._out_params = Outputs()
        super().__init__(
            PLASMOD,
            params,
            runmode,
            # default_mappings=set_default_mappings(),
            input_file=input_file,
            output_file=output_file,
            profiles_file=profiles_file,
            binary=binary,
        )

    def _check_models(self):
        self._parameters.i_impmodel = ImpurityModel(self._parameters.i_impmodel)
        self._parameters.i_modeltype = TransportModel(self._parameters.i_modeltype)
        self._parameters.i_equiltype = EquilibriumModel(self._parameters.i_equiltype)
        self._parameters.i_pedestal = PedestalModel(self._parameters.i_pedestal)
        self._parameters.isiccir = SOLModel(self._parameters.isiccir)

    def get_profile(self, profile):
        return getattr(self._out_params, Profiles(profile).name)

    def get_profiles(self, profiles):
        profiles_dict = {}
        for profile in profiles:
            profiles_dict[profile] = self.get_profile(profile)
        return profiles_dict[profile]
