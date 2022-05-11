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
The API for the plasmod solver.
"""

from enum import auto
from typing import Any, Dict, Iterable, List

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import RunMode as BaseRunMode
from bluemira.base.solver import SolverABC
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.mapping import Profiles
from bluemira.codes.plasmod.mapping import mappings as plasmod_mappings
from bluemira.codes.plasmod.solver._outputs import PlasmodOutputs
from bluemira.codes.plasmod.solver._run import Run
from bluemira.codes.plasmod.solver._setup import Setup
from bluemira.codes.plasmod.solver._teardown import Teardown
from bluemira.codes.utilities import add_mapping


class RunMode(BaseRunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class Solver(SolverABC):
    """
    Plasmod solver class.

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for plasmod.
    build_config: Dict[str, Any]
        Build configuration dictionary.
        Expected keys include:
            - binary: str, path to the plasmod binary.
            - problem_settings: Dict[str, Any], any plasmod specific
              parameters (i.e., parameters that bluemira does not have
              direct mappings to through the ParameterFrame).
            - input_file: str, the path to write the plasmod input file
              to, this can be a relative path.
            - output_file: str, the path to write the plasmod scalar
              output file to.
            - profiles_file: str, the path to write the plasmod profiles
              output file to.
    """

    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown

    DEFAULT_INPUT_FILE = "plasmod_input.dat"
    DEFAULT_OUTPUT_FILE = "plasmod_output.dat"
    DEFAULT_PROFILES_FILE = "plasmod_profiles.dat"

    def __init__(self, params: ParameterFrame, build_config: Dict[str, Any] = None):
        # Init task objects on execution so parameters can be edited
        # between separate 'execute' calls.
        self._setup: Setup
        self._run: Run
        self._teardown: Teardown

        self.params = params
        add_mapping(PLASMOD_NAME, self.params, plasmod_mappings)

        self.build_config = {} if build_config is None else build_config

        self.binary = self.build_config.get("binary", PLASMOD_BINARY)
        self.problem_settings = self.build_config.get("problem_settings", {})
        self.input_file = self.build_config.get("input_file", self.DEFAULT_INPUT_FILE)
        self.output_file = self.build_config.get("output_file", self.DEFAULT_OUTPUT_FILE)
        self.profiles_file = self.build_config.get(
            "profiles_file", self.DEFAULT_PROFILES_FILE
        )

        # TODO(hsaunders): sanity check file paths are not equal?

    def execute(self, run_mode: RunMode) -> ParameterFrame:
        """
        Execute this plasmod solver.

        This solver:
            1. writes a plasmod input file using the given bluemira and
               problem parameters.
            2. processes that file using a shell call to plasmod.
            3. reads the plasmod output files, and updates this object's
               ParameterFrame with the results.

        Parameters
        ----------
        run_mode: RunMode
            The mode to execute this solver in.
        """
        self._setup = Setup(self.params, self.problem_settings, self.input_file)
        self._run = Run(
            self.params,
            self.input_file,
            self.output_file,
            self.profiles_file,
            self.binary,
        )
        self._teardown = Teardown(self.params, self.output_file, self.profiles_file)

        setup = self._get_execution_method(self._setup, run_mode)
        run = self._get_execution_method(self._run, run_mode)
        teardown = self._get_execution_method(self._teardown, run_mode)

        if setup:
            setup()
        if run:
            run()
        if teardown:
            teardown()

        return self.params

    def get_profile(self, profile: str) -> np.ndarray:
        # TODO(hsaunders1904): should this use the Profiles enum?
        """
        Get a single plasmod profile.

        Parameters
        ----------
        profile: str
            A profile to get the data for.

        Returns
        -------
        profile_values: np.ndarray
            A plasmod profile.
        """
        return getattr(self.plasmod_outputs(), Profiles(profile).name)

    def get_profiles(self, profiles: Iterable[str]) -> Dict[str, np.ndarray]:
        """
        Get a dictionary of plasmod profiles.

        Parameters
        ----------
        profiles: Iterable[str]
            An iterable of profile names.

        Returns
        -------
        profiles_dict: Dict[str, np.ndarray]
            A dictionary mapping profile names to values.
        """
        profiles_dict = {}
        for profile in profiles:
            profiles_dict[profile] = self.get_profile(profile)
        return profiles_dict

    def plasmod_outputs(self) -> PlasmodOutputs:
        """
        Return a structure of unmapped plasmod outputs.

        Use :code:`params` attribute for mapped outputs.

        Returns
        -------
        outputs: PlasmodOutputs
            The scalar plasmod outputs.
        """
        try:
            return self._teardown.outputs
        except AttributeError as attr_error:
            raise CodesError(
                "Cannot get outputs before the solver has been executed."
            ) from attr_error

    def modify_mappings(self, send_recv: Dict[str, Dict[str, bool]]):
        """
        Modify the send/receive truth values of a parameter.

        If a parameter's 'send' is set to False, its value will not be
        passed to plasmod (a default will be used). Likewise, if a
        parameter's 'recv' is False, its value will not be updated from
        plasmod's outputs.

        Parameters
        ----------
        mappings: dict
            A dictionary of variables to change mappings.
        Notes
        -----
            Only one of send or recv is needed. The mappings dictionary could look like:
               {"var1": {"send": False, "recv": True}, "var2": {"recv": False}}
        """
        for key, val in send_recv.items():
            try:
                p_map = getattr(self.params, key).mapping[PLASMOD_NAME]
            except (AttributeError, KeyError):
                bluemira_warn(f"No mapping known for {key} in {PLASMOD_NAME}")
            else:
                for sr_key, sr_val in val.items():
                    setattr(p_map, sr_key, sr_val)
