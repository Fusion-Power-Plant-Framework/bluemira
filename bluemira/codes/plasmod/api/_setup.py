# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Defines the 'Setup' stage of the plasmod solver.
"""

import copy
import dataclasses
import enum
from typing import Any

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesSetup
from bluemira.codes.plasmod.api._inputs import PlasmodInputs
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.params import PlasmodSolverParams


class Setup(CodesSetup):
    """
    Setup task for a plasmod solver.

    On run, this task writes a plasmod input file using the input values
    defined in this class.

    Parameters
    ----------
    params:
        The bluemira parameters for the task. Note that this task does
        not apply any mappings to the ParameterFrame, so they should
        already be set. Most likely by a solver.
    problem_settings:
        Any non-bluemira parameters that should be passed to plasmod.
    plasmod_input_file:
        The path where the plasmod input file should be written.
    """

    params: PlasmodSolverParams

    def __init__(
        self,
        params: PlasmodSolverParams,
        problem_settings: dict[str, Any],
        plasmod_input_file: str,
    ):
        super().__init__(params, PLASMOD_NAME)

        self.inputs = PlasmodInputs()
        self.plasmod_input_file = plasmod_input_file
        self.update_inputs(problem_settings)

    def run(self):
        """
        Run plasmod setup.
        """
        self._write_input()

    def mock(self):
        """
        Run plasmod setup in mock mode.

        No need to generate an input file as results will be mocked.
        """

    def read(self):
        """
        Run plasmod setup in read mode.

        No need to generate an input file as results will be read from
        file.
        """

    def update_inputs(self, new_inputs: dict[str, float | enum.Enum] | None = None):
        """
        Update plasmod inputs using the given values.

        This also pulls input values from the task's ParameterFrame and
        uses them to update the inputs attributes. The inputs to this
        method take precedence over inputs in the ParameterFrame.

        Parameters
        ----------
        new_inputs:
            The new inputs to update with.

        Notes
        -----
        Updates this class's :code:`inputs` attribute.
        """
        new_inputs = {} if new_inputs is None else new_inputs
        new_inputs = self._remove_non_plasmod_inputs(new_inputs)
        new = self._get_new_inputs()
        new.update(new_inputs)
        # Create a new PlasmodInputs object so we still benefit from
        # the __post_init__ processing (converts models to enums)
        self.inputs = PlasmodInputs(**new)

    @staticmethod
    def _remove_non_plasmod_inputs(_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Remove non-plasmod inputs from a dictionary. Warn that the
        removed inputs will be ignored.

        This copies the original dictionary, the input dictionary is not
        modified.
        """
        inputs = copy.deepcopy(_inputs)
        fields = {field.name for field in dataclasses.fields(PlasmodInputs)}
        # Convert to list to copy the keys, as we are changing the size
        # of the dict during iteration.
        for input_name in list(inputs.keys()):
            if input_name not in fields:
                bluemira_warn(f"Ignoring unknown plasmod input '{input_name}'.")
                inputs.pop(input_name)
        return inputs

    def _write_input(self):
        """
        Write inputs to file to be read by plasmod.

        Raises
        ------
        CodesError
            Cannot write input file
        """
        try:
            with open(self.plasmod_input_file, "w") as io_stream:
                self.inputs.write(io_stream)
        except OSError as os_error:
            raise CodesError(
                f"Could not write plasmod input file: '{self.plasmod_input_file}':"
                f" {os_error}"
            ) from os_error
