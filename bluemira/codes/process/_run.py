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
PROCESS run functions
"""

from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes.interface import CodesTask
from bluemira.codes.process.constants import BINARY as PROCESS_BINARY
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.params import ProcessSolverParams


class Run(CodesTask):
    """
    Run task for PROCESS.

    Parameters
    ----------
    params:
        The bluemira parameters for this task.
    in_dat_path:
        The path to an existing PROCESS IN.DAT input file.
    run_directory:
        The directory in which to run PROCESS. This is where the output
        files will be written to. Default is current working directory.
    binary:
        The path, or name, of the PROCESS executable. The default is
        'process', which requires the executable to be on the system
        path.
    """

    def __init__(
        self,
        params: ProcessSolverParams,
        in_dat_path: str,
        binary: str = PROCESS_BINARY,
    ):
        super().__init__(params, PROCESS_NAME)

        self.in_dat_path = in_dat_path
        self.binary = binary

    def run(self):
        """
        Run the PROCESS executable on the IN.DAT file.

        This will run process using the :code:`in_dat_path` file.
        PROCESS's output files will be written to the same directory
        that :code:`in_dat_path` is in.
        """
        self._run_process()

    def runinput(self):
        """
        Run the PROCESS executable on the IN.DAT file, equivalent to
        'run' method.
        """
        self._run_process()

    def _run_process(self):
        bluemira_print(f"Running '{PROCESS_NAME}' systems code")
        command = [self.binary, "-i", self.in_dat_path]
        self._run_subprocess(command)
