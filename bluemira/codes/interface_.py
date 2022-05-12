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
"""Base classes for solvers using external codes."""

import abc
from typing import Any, Dict

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import SolverABC, Task
from bluemira.codes.error import CodesError
from bluemira.codes.utilities import get_recv_mapping, run_subprocess


class CodesTask(Task):
    """
    Base task for an external codes Solver.
    """

    def __init__(self, params: ParameterFrame, codes_name: str) -> None:
        super().__init__(params)
        self._name = codes_name

    def _run_subprocess(self, command, **kwargs):
        """
        Run a subprocess command and raise a CodesError if it returns a
        non-zero exit code.
        """
        return_code = run_subprocess(command, **kwargs)
        if return_code != 0:
            raise CodesError(
                f"'{self._name} subprocess task exited with non-zero error code "
                f"'{return_code}'."
            )


class CodesTeardown(CodesTask):
    """
    Base teardown task for an external codes Solver.

    Parameters
    ----------
    params: ParameterFrame
        The parameters for this task.
    codes_name: str
        The name of the external code the task is associated with.
    """

    def _update_params_with_outputs(self, outputs: Dict[str, Any]):
        """
        Update this task's parameters with the external code's outputs.

        This performs implicitly performs any unit conversions.

        Raises
        ------
        CodesError
            If any outputs do not have a mapping to bluemira, or the
            mapping points to a parameter name that does not exist in
            this object's ParameterFrame.
        """
        mapped_outputs = self._map_external_outputs_to_bluemira_params(outputs)
        self.params.update_kw_parameters(mapped_outputs, source=self._name)

    def _map_external_outputs_to_bluemira_params(
        self, external_outputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Loop through external outputs, find the corresponding bluemira
        parameter name, and map it to the output's value and unit.

        Parameters
        ----------
        external_outputs: Dict[str, Any]
            An output produced by an external code. The keys are the
            outputs' names (not the bluemira version of the name), the
            values are the output's value (in the external code's unit).

        Returns
        -------
        mapped_outputs: Dict[str, Dict[str, Any]]
            The keys are bluemira parameter names and the values are a
            dict of form '{"value": Any, "unit": str}', where the value
            is the external code's output value, and the unit is the
            external code's unit.
        """
        mapped_outputs = {}
        recv_mappings = get_recv_mapping(self.params, self._name)
        for external_key, bluemira_key in recv_mappings.items():
            output_value = self._get_output_or_raise(external_outputs, external_key)
            if output_value is None:
                continue
            bluemira_param = self._get_parameter_or_raise(bluemira_key)
            codes_unit = bluemira_param.mapping[self._name].unit
            if codes_unit is not None:
                mapped_outputs[bluemira_key] = {
                    "value": output_value,
                    "unit": codes_unit,
                }
        return mapped_outputs

    def _get_output_or_raise(
        self, external_outputs: Dict[str, Any], parameter_name: str
    ):
        try:
            output_value = external_outputs[parameter_name]
        except KeyError as key_error:
            raise CodesError(
                f"No output value from code '{self._name}' found for parameter "
                f"'{parameter_name}'."
            ) from key_error
        if output_value is None:
            bluemira_warn(
                f"No value for output parameter '{parameter_name}' from code "
                f"'{self._name}'."
            )
        return output_value

    def _get_parameter_or_raise(self, bluemira_param_name: str):
        try:
            return self.params.get_param(bluemira_param_name)
        except AttributeError as attr_error:
            raise CodesError(
                f"No mapping defined between parameter '{bluemira_param_name}' and "
                f"code '{self._name}'."
            ) from attr_error


class CodesSolver(SolverABC):
    """
    Base class for solvers running an external code.
    """

    @abc.abstractproperty
    def name(self):
        """
        The name of the solver.

        In the base class, this is used to find mappings and specialise
        error messages for the concrete solver.
        """
        pass

    def modify_mappings(self, send_recv: Dict[str, Dict[str, bool]]):
        """
        Modify the send/receive truth values of a parameter.

        If a parameter's 'send' is set to False, its value will not be
        passed to the external code (a default will be used). Likewise,
        if a parameter's 'recv' is False, its value will not be updated
        from the external code's outputs.

        Parameters
        ----------
        mappings: dict
            A dictionary where keys are variables to change the mappings
            of, and values specify 'send', and or, 'recv' booleans.

            E.g.,

            .. code-block:: python

                {
                    "var1": {"send": False, "recv": True},
                    "var2": {"recv": False}
                }
        """
        for key, val in send_recv.items():
            try:
                p_map = getattr(self.params, key).mapping[self.name]
            except (AttributeError, KeyError):
                bluemira_warn(f"No mapping known for {key} in {self.name}")
            else:
                for sr_key, sr_val in val.items():
                    setattr(p_map, sr_key, sr_val)
