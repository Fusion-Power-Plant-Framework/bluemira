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
"""The API for the plasmod solver."""

from enum import auto
from typing import Any, Dict, Iterable

import numpy as np

from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import RunMode as BaseRunMode
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesSolver
from bluemira.codes.plasmod.api._outputs import PlasmodOutputs
from bluemira.codes.plasmod.api._run import Run
from bluemira.codes.plasmod.api._setup import Setup
from bluemira.codes.plasmod.api._teardown import Teardown
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.mapping import Profiles
from bluemira.codes.plasmod.mapping import mappings as plasmod_mappings
from bluemira.codes.utilities import add_mapping


class RunMode(BaseRunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class Solver(CodesSolver):
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

    name = PLASMOD_NAME
    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown
    run_mode_cls = RunMode

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
        if setup := self._get_execution_method(self._setup, run_mode):
            setup()
        if run := self._get_execution_method(self._run, run_mode):
            run()
        if teardown := self._get_execution_method(self._teardown, run_mode):
            teardown()

        return self.params

    def get_profile(self, profile: Profiles) -> np.ndarray:
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
        return getattr(self.plasmod_outputs(), profile.name)

    def get_profiles(self, profiles: Iterable[Profiles]) -> Dict[Profiles, np.ndarray]:
        """
        Get a dictionary of plasmod profiles.

        Parameters
        ----------
        profiles: Iterable[Profiles]
            An iterable of Profiles enum values.

        Returns
        -------
        profiles_dict: Dict[Profiles, np.ndarray]
            A dictionary mapping profile enum to values.
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
