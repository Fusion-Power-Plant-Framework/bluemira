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

from typing import Any, Dict

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import Task
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.solver._outputs import PlasmodOutputs
from bluemira.codes.utilities import get_recv_mapping


class Teardown(Task):
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
        super().__init__(params)
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
        self._update_params_from_outputs()

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
        self._update_params_from_outputs()

    @property
    def params(self) -> ParameterFrame:
        """Return the Bluemira parameters associated with this task."""
        return self._params

    def _update_params_from_outputs(self):
        """
        Update this object's ParameterFrame with plasmod outputs.
        """
        bm_outputs = self._map_outputs_to_bluemira()
        self._prepare_outputs(bm_outputs, source=PLASMOD_NAME)

    def _map_outputs_to_bluemira(self) -> Dict[str, Any]:
        """
        Iterate over the plasmod-bluemira parameter mappings and map the
        bluemira parameter names to plasmod output values.
        """
        bm_outputs: Dict[str, Any] = {}
        for plasmod_key, bm_key in self._recv_mapping.items():
            try:
                output_value = getattr(self.outputs, plasmod_key)
            except AttributeError as attr_error:
                raise CodesError(
                    f"No plasmod output '{plasmod_key}' in plasmod outputs list."
                ) from attr_error
            if output_value is None:
                # Catches cases where parameters may be missing from the
                # output file, in which case we get the default, which
                # can be None.
                bluemira_warn(
                    f"No value for plasmod parameter '{bm_key}' found in output."
                )
            else:
                bm_outputs[bm_key] = output_value
        return bm_outputs

    @property
    def _recv_mapping(self):
        """Return the plasmod-to-bluemira parameter mappings."""
        self.__recv_mapping = get_recv_mapping(self.params, PLASMOD_NAME)
        return self.__recv_mapping

    def _prepare_outputs(self, bm_outputs: Dict[str, Any], source: str):
        """
        Update this object's ParameterFrame with the given outputs.

        Implicitly converts to bluemira units if unit available.

        Parameters
        ----------
        outputs: Dict
            key value pair of code outputs
        source: Optional[str]
            Set the source of all outputs, by default is code name

        """
        for bm_key, value in bm_outputs.items():
            try:
                code_unit = self.params.get_param(bm_key).mapping[PLASMOD_NAME].unit
            except AttributeError as exc:
                raise CodesError(f"No mapping found for '{bm_key}'.") from exc
            if code_unit is not None:
                bm_outputs[bm_key] = {"value": value, "unit": code_unit}

        self.params.update_kw_parameters(bm_outputs, source=source)
