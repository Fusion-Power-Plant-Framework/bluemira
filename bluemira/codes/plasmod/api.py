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
import os
import pprint
from enum import Enum, auto
from typing import Dict, Iterable, Union

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
    with open(path + "/PLASMOD_DEFAULT_IN.json", "r") as jfh:
        return json.load(jfh, cls=CommentJSONDecoder)


def get_default_plasmod_outputs():
    """
    Returns a copy of the defaults plasmod outputs.
    """
    path = get_bluemira_path("codes/plasmod")
    with open(path + "/PLASMOD_DEFAULT_OUT.json", "r") as jfh:
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

    def items(self):
        for k, v in self._options.items():
            yield k, v


class Outputs(PlasmodParameters):
    """Class for Plasmod outputs"""

    def __init__(self, **kwargs):
        self._options = get_default_plasmod_outputs()
        super().__init__(**kwargs)


class RunMode(interface.RunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    MOCK = auto()


class Setup(interface.Setup):
    """
    Setup class for Plasmod

    Parameters
    ----------
    parent
        Parent solver class instance
    input_file: str
        input file save location
    output_file: str
        output file save location
    profiles_file: str
        profiles file save location
    kwargs: Dict
        passed to parent setup task

    """

    def __init__(self, parent, input_file, output_file, profiles_file, **kwargs):
        super().__init__(parent, **kwargs)
        self._check_models()
        self.profiles_file = profiles_file
        self.filepath = get_bluemira_path("codes/plasmod")
        self.def_outfile = "PLASMOD_DEFAULT_OUT.json"
        self.def_infile = "PLASMOD_DEFAULT_IN.json"
        self.input_file = input_file
        self.output_file = output_file

    def _write(self, params, filename):
        """
        Plasmod input file writer

        Parameters
        ----------
        params: Dict
            dictionary to write
        filename: str
            file location
        """
        with open(filename, "w") as fid:
            for k, v in params.items():
                if isinstance(v, Enum):
                    fid.write(f"{k} {v.value:d}\n")
                elif isinstance(v, int):
                    fid.write(f"{k} {v:d}\n")
                elif isinstance(v, float):
                    fid.write(f"{k} {v:5.4e}\n")
                else:
                    fid.write(f"{k} {v}\n")

    def write_input(self):
        """
        Write input file
        """
        self._write(self.parent.params, os.path.join(self.filepath, self.input_file))

    def _check_models(self):
        """
        Check selected plasmod models are known
        """
        self.params.i_impmodel = ImpurityModel(self.params.i_impmodel)
        self.params.i_modeltype = TransportModel(self.params.i_modeltype)
        self.params.i_equiltype = EquilibriumModel(self.params.i_equiltype)
        self.params.i_pedestal = PedestalModel(self.params.i_pedestal)
        self.params.isiccir = SOLModel(self.params.isiccir)

    def _run(self, *args, **kwargs):
        """
        Run plasmod setup
        """
        self.write_input()

    def _mock(self, *args, **kwargs):
        """
        Mock plasmod setup
        """
        self.write_input()

    def get_default_plasmod_outputs(self):
        """
        Returns a copy of the defaults plasmod outputs.
        """
        return self._load_default_from_json(
            os.path.join(self.filepath, self.def_outfile)
        )

    def get_default_plasmod_inputs(self):
        """
        Returns a copy of the default plasmod inputs
        """
        return self._load_default_from_json(os.path.join(self.filepath, self.def_infile))

    @staticmethod
    def _load_default_from_json(filepath: str):
        """
        Load json file

        Parameters
        ----------
        filepath: str
            json file to load
        """
        bluemira_debug(filepath)
        with open(filepath) as jfh:
            return json.load(jfh, cls=CommentJSONDecoder)


class Run(interface.Run):
    """
    Run class for plasmod

    Parameters
    ----------
    parent
        Parent solver class instance
    kwargs: Dict
        passed to parent setup task

    """

    _binary = "transporz"  # Who knows why its not called plasmod

    def __init__(self, parent, **kwargs):
        super().__init__(parent, kwargs.pop("binary", self._binary))

    def _run(self, *args, **kwargs):
        """
        Run plasmod runner
        """
        bluemira_debug("Mode: run")
        super()._run_subprocess(
            [
                self._binary,
                self.parent.setup_obj.input_file,
                self.parent.setup_obj.output_file,
                self.parent.setup_obj.profiles_file,
            ]
        )


class Teardown(interface.Teardown):
    def _run(self, *args, **kwargs):
        """
        Run plasmod teardown
        """
        output = self.read_output_files(self.parent.setup_obj.output_file)
        self.parent._out_params.modify(**output)
        self._check_return_value()
        profiles = self.read_output_files(self.parent.setup_obj.profiles_file)
        self.parent._out_params.modify(**profiles)
        # print_parameter_list(self.parent._out_params)

    def _mock(self, *args, **kwargs):
        """
        Mock plasmod teardown
        """
        output = self.read_output_files(self.parent.setup_obj.output_file)
        self.parent._out_params.modify(**output)
        profiles = self.read_output_files(self.parent.setup_obj.profiles_file)
        self.parent._out_params.modify(**profiles)
        # print_parameter_list(self.parent._out_params)

    @staticmethod
    def read_output_files(output_file: str):
        """
        Read the Plasmod output parameters from the output file

        Parameters
        ----------
        output_file: str
            Read a plasmod output filename

        Returns
        -------
        output: dict

        """
        output = {}
        with open(output_file, "r") as fd:
            for row in csv.reader(fd, delimiter="\t"):
                output_key, *output_value = row[0].split()
                output[output_key] = (
                    np.array(output_value, dtype=np.float)
                    if len(output_value) > 1
                    else float(output_value[0])
                )
        return output

    def _check_return_value(self):
        """
        Check the return value of plasmod

         1: PLASMOD converged successfully
        -1: Max number of iterations achieved
            (equilibrium oscillating, pressure too high, reduce H)
         0: transport solver crashed (abnormal parameters
            or too large dtmin and/or dtmin
        -2: Equilibrium solver crashed: too high pressure

        """
        exit_flag = self.parent._out_params.i_flag
        if exit_flag == 1:
            bluemira_debug(f"{PLASMOD} converged successfully")
        elif exit_flag == -2:
            raise CodesError(
                f"{PLASMOD} error: Equilibrium solver crashed: too high pressure"
            )
        elif exit_flag == -1:
            raise CodesError(
                f"{PLASMOD} error: "
                "Max number of iterations reached "
                "equilibrium oscillating probably as a result of the pressure being too high "
                "reducing H may help"
            )
        elif not exit_flag:
            raise CodesError(
                f"{PLASMOD} error: " "Abnormal paramters, possibly dtmax/dtmin too large"
            )
        else:
            raise CodesError(f"{PLASMOD} error: " f"Unknown error code {exit_flag}")


class Solver(interface.FileProgramInterface):
    """
    Plasmod solver class

    Parameters
    ----------
    runmode: str
        Plasmod runmode
    params: ParameterFrame
        ParameterFrame for plasmod
    build_config: Dict
        build configuration dictionary
    input_file: str
        input file save location
    output_file: str
        output file save location
    profiles_file: str
        profiles file save location
    binary: str
        plasmod binary name

    """

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self,
        runmode="run",
        params=None,
        build_config=None,
        input_file="plasmod_input.dat",
        output_file="outputs.dat",
        profiles_file="profiles.dat",
        binary="transporz",
    ):
        self._out_params = Outputs()
        super().__init__(
            PLASMOD,
            self._getInputs(params),
            runmode,
            # default_mappings=set_default_mappings(),
            input_file=input_file,
            output_file=output_file,
            profiles_file=profiles_file,
            binary=binary,
        )

    @staticmethod
    def _getInputs(params):
        # TODO remove the need for this
        if params is None:
            params = Inputs()
        elif isinstance(params, Inputs):
            params = params
        elif isinstance(params, Dict):
            params = Inputs(**params)
        return params

    def get_profile(self, profile: str):
        """
        Get a single profile

        Parameters
        ----------
        profile: str
            A profile to get the data for

        Returns
        -------
        A profile data

        """
        return getattr(self._out_params, Profiles(profile).name)

    def get_profiles(self, profiles: Iterable):
        """
        Get list of profiles

        Parameters
        ----------
        profiles: Iterable
            A list of profiles to get data for

        Returns
        -------
        dictionary of the profiles request

        """
        profiles_dict = {}
        for profile in profiles:
            profiles_dict[profile] = self.get_profile(profile)
        return profiles_dict
