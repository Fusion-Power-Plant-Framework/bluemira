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
PROCESS run functions
"""

from __future__ import annotations

import os
import subprocess  # noqa :S404
from enum import auto
from typing import Dict, List, Optional, Union

import bluemira.codes.interface as interface
from bluemira.base.builder import BuildConfig
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.process.api import DEFAULT_INDAT, ENABLED
from bluemira.codes.process.constants import BINARY
from bluemira.codes.process.constants import NAME as PROCESS
from bluemira.codes.process.mapping import mappings
from bluemira.codes.process.setup import Setup
from bluemira.codes.process.teardown import Teardown


class RunMode(interface.RunMode):
    """
    Enum class to pass args and kwargs to the PROCESS functions corresponding to the
    chosen PROCESS runmode (Run, Runinput, Read, Readall, or Mock).
    """

    RUN = auto()
    RUNINPUT = auto()
    READ = auto()
    READALL = auto()
    MOCK = auto()
    NONE = auto()


class Run(interface.Run):
    """
    Run task for process
    """

    _binary = BINARY

    def _run(self):
        self.run_PROCESS()

    def _runinput(self):
        self.run_PROCESS()

    def run_PROCESS(self):
        """
        Run the systems code to get an initial reactor solution (radial build).

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use bluemira values as PROCESS inputs. Used to re-run PROCESS
            within a bluemira run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        bluemira_print(f"Running {PROCESS} systems code")

        # Run PROCESS
        self._clear_PROCESS_output()
        self._run_subprocess()

    def _clear_PROCESS_output(self):
        """
        Clear the output files from PROCESS run directory.
        """
        for filename in self.parent.output_files:
            filepath = os.sep.join([self.parent.run_dir, filename])
            if os.path.exists(filepath):
                os.remove(filepath)

    def _run_subprocess(self):
        super()._run_subprocess(self._binary)


class Solver(interface.FileProgramInterface):
    """
    PROCESS Run functions. Runs, loads or mocks PROCESS to generate the reactor's radial
    build as an input for the bluemira run.

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for PROCESS
    build_config: Dict
        build configuration dictionary
    run_dir: str
        Path to the PROCESS run directory, where the main PROCESS executable is located
        and the input/output files will be written.
    read_dir: str
        Path to the PROCESS read directory, where the output files from a PROCESS run are
        read in
    template_indat: str
        Path to the template IN.DAT file to be used for the run.
        Default, the value specified by DEFAULT_INDAT.

    Notes
    -----
    - "run": Run PROCESS within a bluemira run to generate an radial build.
        Creates a new input file from a template IN.DAT modified with updated parameters
        from the bluemira run mapped with send=True.
    - "runinput": Run PROCESS from an unmodified input file (IN.DAT), generating the
        radial build to use as the input to the bluemira run. Overrides the send
        mapping of all parameters to be False.
    - "read": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        only the parameters mapped with recv=True.
    - "readall": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        all values with a bluemira mapping regardless of the mapping.recv bool.
        Overrides the recv mapping of all parameters to be True.
    - "mock": Run bluemira without running PROCESS, using the default radial build based
        on EU-DEMO. This option should not be used if PROCESS is installed, except for
        testing purposes.
    - "none": Do nothing. Useful when loading results from previous runs of Bluemira,
        when overwriting data with PROCESS output would be undesirable.

    Raises
    ------
    CodesError
        If PROCESS is not being mocked and is not installed.
    """

    _params: ParameterFrame
    _run_dir: str
    _read_dir: str
    _template_indat: str
    _parameter_mapping: Dict[str, str]
    _recv_mapping: Dict[str, str]
    _send_mapping: Dict[str, str]

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    output_files: List[str] = [
        "OUT.DAT",
        "MFILE.DAT",
        "OPT.DAT",
        "SIG_TF.DAT",
    ]

    def __init__(
        self,
        params: ParameterFrame,
        build_config: BuildConfig,
        run_dir: Optional[str] = None,
        read_dir: Optional[str] = None,
        template_indat: Optional[str] = None,
    ):

        self.read_dir = read_dir

        self._template_indat = (
            build_config.get("process_indat", DEFAULT_INDAT)
            if template_indat is None
            else template_indat
        )

        super().__init__(
            PROCESS,
            params,
            build_config.get("mode", "run"),
            binary=build_config.get("binary", BINARY),
            run_dir=run_dir,
            read_dir=read_dir,
            mappings=mappings,
            problem_settings=build_config.get("problem_settings", None),
        )

        self._enabled_check(build_config.get("mode", "run").lower())

    @staticmethod
    def _enabled_check(mode):
        if (not ENABLED) and (mode != "mock"):
            raise CodesError(f"{PROCESS} not (properly) installed")

    def get_raw_variables(self, params: Union[List, str]):
        """
        Get raw variables from an MFILE
        (mapped bluemira parameters will have bluemira names)

        Parameters
        ----------
        params: Union[List, str]
            parameter names to access

        Returns
        -------
        values list
        """
        return self.teardown_obj.bm_file.extract_outputs(params)
