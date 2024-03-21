# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS run functions
"""

import numpy as np
from rich.progress import Progress, TextColumn

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
        solver: str | None = None,
    ):
        super().__init__(params, PROCESS_NAME)

        self.in_dat_path = in_dat_path
        self.binary = binary
        self.solver = solver

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

    @staticmethod
    def flush_callable(line: str) -> bool:
        """Callable for flushed output"""
        try:
            int(line.split("|")[0])
        except ValueError:
            return False
        else:
            return True

    def flush_printer(self, line: str):
        description, convergence = line.rsplit(":", 1)
        description = f"{convergence} | {description.split('|')[0]}"
        self._progress.update(
            self._task,
            description=description,
            completed=np.clip(np.log(1 / float(convergence)), 0, None),
            visible=True,
        )

    def _get_epsvmc(self):
        with open(self.in_dat_path) as f:
            for line in f:
                if line.startswith("epsvmc"):
                    self._epsvmc = float(line.split("=")[1].split("*")[0])
                    break

    def _run_process(self):
        bluemira_print(f"Running '{PROCESS_NAME}' systems code")
        command = [self.binary, "-i", self.in_dat_path]
        if self.solver is not None:
            command.extend(["--solver", self.solver])

        self._get_epsvmc()

        cols = list(Progress.get_default_columns())
        tc = cols[0]
        cols[0] = TextColumn("Convergence")
        cols[-1] = tc

        with Progress(*cols) as self._progress:
            self._task = self._progress.add_task(
                "",
                completed=0,
                total=np.log(1 / getattr(self, "_epsvmc", 1)),
                visible=False,
            )

            self._run_subprocess(
                command,
                flush_callable=self.flush_callable,
                flush_printer=self.flush_printer,
            )
