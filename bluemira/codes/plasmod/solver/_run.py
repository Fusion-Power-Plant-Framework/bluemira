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
Defines the 'Run' stage of the plasmod solver.
"""


from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.solver._task import PlasmodTask
from bluemira.codes.utilities import run_subprocess


class Run(PlasmodTask):
    """
    The 'Run' class for plasmod transport solver.
    """

    def __init__(
        self,
        params: ParameterFrame,
        input_file: str,
        output_file: str,
        profiles_file: str,
        binary=PLASMOD_BINARY,
    ):
        super().__init__(params)
        self.binary = binary
        self.input_file = input_file
        self.output_file = output_file
        self.profiles_file = profiles_file

    def run(self):
        """
        Run the plasmod shell task.

        Runs plasmod on the command line using the given input files and
        output path.

        Raises
        ------
        CodesError
            If the subprocess returns a non-zero exit code or raises an
            OSError (e.g., the plasmod binary does not exist).
        """
        command = [self.binary, self.input_file, self.output_file, self.profiles_file]
        bluemira_debug("Mode: run")
        try:
            self._run_subprocess(command)
        except OSError as os_error:
            raise CodesError(f"Failed to run plasmod: {os_error}") from os_error

    def _run_subprocess(self, command, **kwargs):
        """
        Run a subprocess command and raise CodesError if it returns a
        non-zero exit code.
        """
        return_code = run_subprocess(command, **kwargs)
        if return_code != 0:
            raise CodesError("plasmod 'Run' task exited with a non-zero error code.")
