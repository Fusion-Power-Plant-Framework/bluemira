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
Defines the 'Setup' stage of the plasmod solver.
"""

import copy
import dataclasses
from typing import Any, Callable, Dict, Optional, Union

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import Task
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.solver._inputs import PlasmodInputs
from bluemira.codes.utilities import get_send_mapping


class Setup(Task):
    """
    Setup task for a plasmod solver.

    On run, this task writes a plasmod input file using the input values
    defined in this class.

    Parameters
    ----------
    params: ParameterFrame
        The bluemira parameters for the task. Note that this task does
        not apply any mappings to the ParameterFrame, so they should
        already be set. Most likely by a solver.
    problem_settings: Dict[str, Any]
        Any non-bluemira parameters that should be passed to plasmod.
    input_file: str
        The path where the plasmod input file should be written.
    """

    def __init__(
        self,
        params: ParameterFrame,
        problem_settings: Dict[str, Any],
        # TODO(hsaunders1904): rename this; input_file is confusing,
        # because it's actually an output file of this task
        input_file: str,
    ) -> None:
        super().__init__(params)

        self.inputs = PlasmodInputs()
        self.input_file = input_file

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
        pass

    def read(self):
        """
        Run plasmod setup in read mode.

        No need to generate an input file as results will be read from
        file.
        """
        pass

    def update_inputs(self, new_inputs: Dict[str, Any] = None):
        """
        Update plasmod inputs using the given values.
        """
        # Create a new PlasmodInputs object so we still benefit from
        # the __post_init__ processing (converts models to enums)
        new_inputs = {} if new_inputs is None else new_inputs
        new_inputs = self._remove_non_plasmod_inputs(new_inputs)
        new = self.get_new_inputs()
        new.update(new_inputs)
        self.inputs = PlasmodInputs(**new)

    @staticmethod
    def _remove_non_plasmod_inputs(_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove non-plasmod inputs from a dictionary. Warn that the
        removed inputs will be ignored.

        This copies the original dictionary, the input dictionary is not
        modified.
        """
        inputs = copy.deepcopy(_inputs)
        fields = set(field.name for field in dataclasses.fields(PlasmodInputs))
        for input_name in list(inputs.keys()):
            if input_name not in fields:
                bluemira_warn(f"Ignoring unknown plasmod input '{input_name}'.")
                inputs.pop(input_name)
        return inputs

    def _write_input(self):
        """
        Write inputs to file to be read by plasmod.
        """
        try:
            with open(self.input_file, "w") as io_stream:
                self.inputs.write(io_stream)
        except OSError as os_error:
            raise CodesError(
                f"Could not write plasmod input file: '{self.input_file}': {os_error}"
            ) from os_error

    def get_new_inputs(self, remapper: Optional[Union[Callable, Dict]] = None):
        """
        Get new key mappings from the ParameterFrame.

        Parameters
        ----------
        remapper: Optional[Union[callable, dict]]
            a function or dictionary for remapping variable names.
            Useful for renaming old variables

        Returns
        -------
        _inputs: dict
            key value pairs of external program variable names and values

        TODO unit conversion
        """
        # TODO(hsaunders): refactor out to base class, make private?
        _inputs = {}

        if not (callable(remapper) or isinstance(remapper, (type(None), Dict))):
            raise TypeError("remapper is not callable or a dictionary")
        elif isinstance(remapper, Dict):
            orig_remap = remapper.copy()

            def remapper(x):
                return orig_remap[x]

        elif remapper is None:

            def remapper(x):
                return x

        for prog_key, bm_key in self._send_mapping.items():
            prog_key = remapper(prog_key)
            if isinstance(prog_key, list):
                for key in prog_key:
                    _inputs[key] = self._convert_units(self.params.get_param(bm_key))
                continue

            _inputs[prog_key] = self._convert_units(self.params.get_param(bm_key))

        return _inputs

    def _convert_units(self, param):
        code_unit = param.mapping[PLASMOD_NAME].unit
        if code_unit is not None:
            return raw_uc(param.value, param.unit, code_unit)
        else:
            return param.value

    @property
    def _send_mapping(self) -> Dict[str, str]:
        self.__send_mapping = get_send_mapping(self.params, PLASMOD_NAME)
        return self.__send_mapping
