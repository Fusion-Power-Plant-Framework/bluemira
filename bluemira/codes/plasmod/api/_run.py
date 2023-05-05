# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from bluemira.base.file import working_dir
from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesTask
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.params import PlasmodSolverParams


class Run(CodesTask):
    """
    The 'Run' class for plasmod transport solver.

    Parameters
    ----------
    params:
        The bluemira parameters for the task. Note that this task does
        not apply any mappings to the ParameterFrame, so they should
        already be set. Most likely by a solver.
    input_file:
        The path to the plasmod input file.
    output_file:
        The path to which the plasmod scalar output file should be
        written.
    profiles_file:
        The path to which the plasmod profiles output file should be
        written.
    directory:
        The directory to run the code in
    binary:
        The name of, or path to, the plasmod binary. If this is not an
        absolute path, the binary must be on the system path.
    """

    params: PlasmodSolverParams

    def __init__(
        self,
        params: PlasmodSolverParams,
        input_file: str,
        output_file: str,
        profiles_file: str,
        directory: str = "./",
        binary=PLASMOD_BINARY,
    ):
        super().__init__(params, PLASMOD_NAME)
        self.binary = binary
        self.input_file = input_file
        self.output_file = output_file
        self.profiles_file = profiles_file
        self.directory = directory

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
        bluemira_print(f"Running '{PLASMOD_NAME}' systems code")
        command = [self.binary, self.input_file, self.output_file, self.profiles_file]
        with working_dir(self.directory):
            try:
                self._run_subprocess(command)
            except OSError as os_error:
                raise CodesError(f"Failed to run plasmod: {os_error}") from os_error
