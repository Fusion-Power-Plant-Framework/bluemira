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
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import fortranformat as ff
import numpy as np

import bluemira.codes.interface as interface
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD
from bluemira.codes.plasmod.mapping import (
    EquilibriumModel,
    ImpurityModel,
    PedestalModel,
    Profiles,
    SOLModel,
    TransportModel,
    create_mapping,
)
from bluemira.utilities.tools import CommentJSONDecoder


class PlasmodParameters:
    """
    A class to manage plasmod parameters
    """

    filepath = get_bluemira_path("codes/plasmod")
    def_outfile = Path(filepath, "PLASMOD_DEFAULT_OUT.json")
    def_infile = Path(filepath, "PLASMOD_DEFAULT_IN.json")
    _options = None

    def __getattribute__(self, attr):
        """
        Get attribute but look in _options if not found.

        Avoids pollution of namespace but parameters still accessible

        Parameters
        ----------
        attr: str
            Attribute to get

        Returns
        -------
        requested attribute

        """
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            try:
                return self._options[attr]
            except KeyError as k:
                raise AttributeError(k)

    def __setattr__(self, attr, value):
        """
        Sets attribute if it already exists otherwise
        add it to _options dictionary

        Parameters
        ----------
        attr: str
            attribute name
        value:
            value to set for attribute

        """
        try:
            super().__getattribute__(attr)
            super().__setattr__(attr, value)
        except AttributeError:
            self._options[attr] = value

    def modify(self, new_options):
        """
        Function to override parameters value.
        """
        if new_options:
            for n_o in new_options:
                if n_o in self._options:
                    self._options[n_o] = new_options[n_o]

    @staticmethod
    def _load_default_from_json(filepath: str):
        """
        Load json file

        Parameters
        ----------
        filepath: str
            json file to load
        """
        bluemira_debug(f"Loading default values from json: {filepath}")
        with open(filepath) as jfh:
            return json.load(jfh, cls=CommentJSONDecoder)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def __repr__(self):
        """
        Representation string of the PlasmodParameters.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._options)}" + "\n)"


class Inputs(PlasmodParameters):
    """
    Class for Plasmod inputs
    """

    f_int = ff.FortranRecordWriter("a20,  i10")
    f_float = ff.FortranRecordWriter("a20, e17.9")

    def __init__(self, new_inputs=None):
        self._options = self.get_default_plasmod_inputs()

        self.modify(new_inputs)

    def modify(self, new_inputs):
        """
        Modify and check models

        Parameters
        ----------
        new_inputs: dict
        """
        super().modify(new_inputs)
        self._check_models()

    def _write(self, filename):
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
            for k, v in self._options.items():
                if isinstance(v, Enum):
                    line = self.f_int.write([k, v.value])
                elif isinstance(v, int):
                    line = self.f_int.write([k, v])
                elif isinstance(v, float):
                    line = self.f_float.write([k, v])
                else:
                    bluemira_warn(f"May produce fortran read errors, type: {type(v)}")
                    line = f"{k} {v}"
                fid.write(line)
                fid.write("\n")

    def _check_models(self):
        """
        Check selected plasmod models are known
        """
        models = [
            ["i_impmodel", ImpurityModel],
            ["i_modeltype", TransportModel],
            ["i_equiltype", EquilibriumModel],
            ["i_pedestal", PedestalModel],
            ["isiccir", SOLModel],
        ]

        for name, model_cls in models:
            val = getattr(self, name)
            model = model_cls[val] if isinstance(val, str) else model_cls(val)
            setattr(self, name, model)

    def get_default_plasmod_inputs(self):
        """
        Returns a copy of the default plasmod inputs
        """
        return self._load_default_from_json(self.def_infile)


class Outputs(PlasmodParameters):
    """Class for Plasmod outputs"""

    def __init__(self, use_defaults=False):
        self._options = self.get_default_plasmod_outputs()
        if not use_defaults:
            for k in self._options.keys():
                self._options[k] = None

    def get_default_plasmod_outputs(self):
        """
        Returns a copy of the defaults plasmod outputs.
        """
        return self._load_default_from_json(self.def_outfile)

    def read_output_files(self, scalar_file, profile_file):
        """
        Read and process plasmod output files

        Parameters
        ----------
        scalar_file: str
            scalar filename
        profile_file: str
            profile filename

        """
        scalars = self.read_file(scalar_file)
        self.modify(scalars)
        self._check_return_value(self.i_flag)
        profiles = self.read_file(profile_file)
        self.modify(profiles)

    @staticmethod
    def read_file(output_file: str) -> Dict[str, Union[float, np.ndarray]]:
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
                    np.array(output_value, dtype=float)
                    if len(output_value) > 1
                    else float(output_value[0])
                )
        return output

    @staticmethod
    def _check_return_value(exit_flag: int):
        """
        Check the return value of plasmod

         1: PLASMOD converged successfully
        -1: Max number of iterations achieved
            (equilibrium oscillating, pressure too high, reduce H)
         0: transport solver crashed (abnormal parameters
            or too large dtmin and/or dtmin
        -2: Equilibrium solver crashed: too high pressure

        """
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
            raise CodesError(f"{PLASMOD} error: Unknown error code {exit_flag}")


class RunMode(interface.RunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
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

    def __init__(self, parent, *args, problem_settings=None, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._problem_settings = problem_settings if problem_settings is not None else {}
        self.input_file = "plasmod_input.dat"
        self.output_file = "plasmod_outputs.dat"
        self.profiles_file = "plasmod_profiles.dat"
        self.io_manager = Inputs({**self._get_new_inputs(), **self._problem_settings})

    def update_inputs(self):
        """
        Update plasmod inputs
        """
        self.io_manager.modify({**self._get_new_inputs(), **self._problem_settings})

    def _get_new_inputs(self):
        """
        Get new key mappings from the ParameterFrame.
        """
        _inputs = {}
        for pl_key, bm_key in self._send_mapping.items():
            _inputs[pl_key] = self.params.get(bm_key)
        return _inputs

    def write_input(self):
        """
        Write input file
        """
        self.io_manager._write(Path(self.parent.run_dir, self.input_file))

    def _run(self):
        """
        Run plasmod setup
        """
        self.update_inputs()
        self.write_input()


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

    _binary = BINARY

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, kwargs.pop("binary", self._binary), *args, **kwargs)

    def _run(self):
        """
        Run plasmod runner
        """
        bluemira_debug("Mode: run")
        super()._run_subprocess(
            [
                self._binary,
                Path(self.parent.run_dir, self.parent.setup_obj.input_file),
                Path(self.parent.run_dir, self.parent.setup_obj.output_file),
                Path(self.parent.run_dir, self.parent.setup_obj.profiles_file),
            ]
        )


class Teardown(interface.Teardown):
    """
    Plasmod Teardown Task
    """

    def _run(self):
        """
        Run plasmod teardown
        """
        self.io_manager = Outputs()
        self.io_manager.read_output_files(
            Path(self.parent.run_dir, self.parent.setup_obj.output_file),
            Path(self.parent.run_dir, self.parent.setup_obj.profiles_file),
        )
        self.prepare_outputs()

    def _mock(self):
        """
        Mock plasmod teardown
        """
        self.io_manager = Outputs(use_defaults=True)
        self.prepare_outputs()

    def _read(self):
        """
        Read plasmod teardown
        """
        self.io_manager = Outputs()
        self.io_manager.read_output_files(
            Path(self.parent.read_dir, self.parent.setup_obj.output_file),
            Path(self.parent.read_dir, self.parent.setup_obj.profiles_file),
        )
        self.prepare_outputs()

    def prepare_outputs(self):
        """
        Prepare outputs for ParameterFrame
        """
        self.parent.params.update_kw_parameters(
            {
                bm_key: getattr(self.io_manager, pl_key)
                for pl_key, bm_key in self.parent._recv_mapping.items()
            },
            source=PLASMOD,
        )


class Solver(interface.FileProgramInterface):
    """
    Plasmod solver class

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for plasmod
    build_config: Dict
        build configuration dictionary
    run_dir: str
        Plasmod run directory
    read_dir: str
        Directory to read in previous run

    Notes
    -----
    build config keys: mode, binary, problem_settings
    """

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self,
        params,
        build_config=None,
        run_dir: Optional[str] = None,
        read_dir: Optional[str] = None,
    ):
        super().__init__(
            PLASMOD,
            params,
            build_config.get("mode", "run"),
            binary=build_config.get("binary", BINARY),
            run_dir=run_dir,
            read_dir=read_dir,
            mappings=create_mapping(),
            problem_settings=build_config.get("problem_settings", None),
        )

    @property
    def problem_settings(self):
        """
        Get problem settings dictionary
        """
        return self.setup_obj._problem_settings

    def get_scalar(self, scalar):
        """
        Get scalar values for unmapped variables.

        Please use params for mapped variables

        Parameters
        ----------
        scalar: str
            scalar value to get

        Returns
        -------
        scalar value

        """
        return getattr(self.teardown_obj.io_manager, scalar)

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
        return getattr(self.teardown_obj.io_manager, Profiles(profile).name)

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
