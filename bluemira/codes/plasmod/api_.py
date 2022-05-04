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
API for the transport code PLASMOD and related functions
"""

from enum import auto
from typing import Any, Callable, Dict, Optional, Union

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import RunMode, Task
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.params import PlasmodInputs
from bluemira.codes.utilities import get_send_mapping


class PlasmodRunMode(RunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class Setup(Task):
    """
    Setup task for a Plasmod solver.

    On run, this task writes a plasmod input file using the input values
    defined in this class.
    """

    DEFAULT_INPUT_FILE = "plasmod_input.dat"

    def __init__(
        self,
        params: ParameterFrame,
        problem_settings: Dict[str, Any] = None,
        input_file=DEFAULT_INPUT_FILE,
    ) -> None:
        super().__init__(params)

        self.inputs = PlasmodInputs()
        self.input_file = input_file

        self.update_inputs(problem_settings)

    @property
    def params(self) -> ParameterFrame:
        """Return the parameters associated with this task."""
        return self._params

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
        self._update_inputs_from_dict(self.get_new_inputs())
        if new_inputs:
            self._update_inputs_from_dict(new_inputs)

    def _write_input(self):
        """
        Write inputs to file to be read by plasmod.
        """
        try:
            with open(self.input_file, "w") as io_stream:
                self.inputs.write(io_stream)
        except OSError as os_error:
            raise CodesError(
                f"Could not write Plasmod input file: '{self.input_file}': {os_error}"
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

    def _update_inputs_from_dict(self, new_inputs: Dict[str, Any]):
        """
        Update inputs with the values in the input dictionary.

        Warn if a given input is not known to Plasmod.
        """
        for key, value in new_inputs.items():
            if hasattr(self.inputs, key):
                setattr(self.inputs, key, value)
            else:
                bluemira_warn(f"Plasmod input '{key}' not known.")

    def _convert_units(self, param):
        code_unit = param.mapping[self.parent.NAME].unit
        if code_unit is not None:
            return raw_uc(param.value, param.unit, code_unit)
        else:
            return param.value

    @property
    def _send_mapping(self) -> Dict[str, str]:
        self.__send_mapping = get_send_mapping(self.params, PLASMOD_NAME)
        return self.__send_mapping
