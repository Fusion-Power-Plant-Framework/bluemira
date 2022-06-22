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
Defines the 'Teardown' stage for the plasmod solver.
"""

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesTeardown
from bluemira.codes.plasmod.api._outputs import PlasmodOutputs
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME


class Teardown(CodesTeardown):
    """
    Plasmod teardown task.

    In "RUN" and "READ" mode, this loads in plasmod results files and
    updates :code:`params` with the values.

    Parameters
    ----------
    params: ParameterFrame
        The bluemira parameters for the task. Note that this task does
        not apply any mappings to the ParameterFrame, so they should
        already be set. Most likely by a solver.
    output_file: str
        The path to the plasmod output file.
    profiles_file: str
        The path to the plasmod profiles file.
    """

    def __init__(self, params: ParameterFrame, output_file: str, profiles_file: str):
        super().__init__(params, PLASMOD_NAME)
        self.outputs = PlasmodOutputs()
        self.output_file = output_file
        self.profiles_file = profiles_file

    def run(self):
        """
        Load the plasmod results files and update this object's params
        with the read values.
        """
        self.read()

    def mock(self):
        """
        Update this object's plasmod params with default values.
        """
        self.outputs = PlasmodOutputs()
        self._update_params_with_outputs(vars(self.outputs))

    def read(self):
        """
        Load the plasmod results files and update this object's params
        with the read values.

        Raises
        ------
        CodesError
            If any of the plasmod files cannot be opened.
        """
        try:
            with open(self.output_file, "r") as scalar_file:
                with open(self.profiles_file, "r") as profiles_file:
                    self.outputs = PlasmodOutputs.from_files(scalar_file, profiles_file)
        except OSError as os_error:
            raise CodesError(
                f"Could not read plasmod output file: {os_error}."
            ) from os_error
        self._raise_on_plasmod_error_code(self.outputs.i_flag)
        self._update_params_with_outputs(vars(self.outputs))

    @staticmethod
    def _raise_on_plasmod_error_code(exit_code: int):
        """
        Check the returned exit code of plasmod.

        1: PLASMOD converged successfully
        -1: Max number of iterations achieved
            (equilibrium oscillating, pressure too high, reduce H)
            0: transport solver crashed (abnormal parameters
            or too large dtmin and/or dtmin
        -2: Equilibrium solver crashed: too high pressure

        Raises
        ------
        CodesError
            If the exit flag is an error code, or its value is not a known
            code.
        """
        if exit_code == 1:
            bluemira_debug("plasmod converged successfully.")
        elif exit_code == -2:
            raise CodesError(
                "plasmod error: Equilibrium solver crashed: too high pressure."
            )
        elif exit_code == -1:
            raise CodesError(
                "plasmod error: "
                "Max number of iterations reached equilibrium oscillating probably as a "
                "result of the pressure being too high reducing H may help."
            )
        elif not exit_code:
            raise CodesError(
                "plasmod error: Abnormal parameters, possibly dtmax/dtmin too large."
            )
        else:
            raise CodesError(f"plasmod error: Unknown error code '{exit_code}'.")
