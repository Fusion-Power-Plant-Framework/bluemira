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
from typing import Any, Callable, Dict, List, Optional, Union

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.solver import SolverABC, Task
from bluemira.codes.error import CodesError
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.utilities import run_subprocess


class CodesTask(Task):
    """
    Base class for a task used by a solver for an external code.
    """

    params: MappedParameterFrame

    def __init__(self, params: MappedParameterFrame, codes_name: str) -> None:
        super().__init__(params)
        self._name = codes_name

    def _run_subprocess(self, command: List[str], **kwargs):
        """
        Run a subprocess command and raise a CodesError if it returns a
        non-zero exit code.
        """
        return_code = run_subprocess(command, **kwargs)
        if return_code != 0:
            raise CodesError(
                f"'{self._name}' subprocess task exited with non-zero error code "
                f"'{return_code}'."
            )


class CodesSetup(CodesTask):
    """
    Base class for setup tasks of a solver for an external code.
    """

    def _get_new_inputs(
        self, remapper: Optional[Union[Callable, Dict[str, str]]] = None
    ) -> Dict[str, float]:
        """
        Retrieve inputs values to the external code from this tasks'
        ParameterFrame.

        Convert the inputs' units to those used by the external code.

        Parameters
        ----------
        remapper: Optional[Union[Callable, Dict[str, str]]]
            A function or dictionary for remapping variable names.
            Useful for renaming old variables

        Returns
        -------
        _inputs: Dict[str, float]
            Keys are external code parameter names, values are the input
            values for those parameters.
        """
        _inputs = {}

        if not (callable(remapper) or isinstance(remapper, (type(None), Dict))):
            raise TypeError("remapper is not callable or a dictionary")
        if isinstance(remapper, Dict):
            orig_remap = remapper.copy()

            def remapper(x):
                return orig_remap[x]

        elif remapper is None:

            def remapper(x):
                return x

        for bm_name, mapping in self.params.mappings.items():
            if not mapping.send:
                continue
            external_name = remapper(mapping.name)
            bm_param = getattr(self.params, bm_name)
            target_unit = mapping.unit
            if isinstance(external_name, list):
                for name in external_name:
                    _inputs[name] = self._convert_units(bm_param, target_unit)
            else:
                _inputs[external_name] = self._convert_units(bm_param, target_unit)
        return _inputs

    def _convert_units(self, param, target_unit: str):
        if target_unit is not None:
            return raw_uc(param.value, param.unit, target_unit)
        else:
            return param.value


class CodesTeardown(CodesTask):
    """
    Base class for teardown tasks of a solver for an external code.

    Parameters
    ----------
    params: ParameterFrame
        The parameters for this task.
    codes_name: str
        The name of the external code the task is associated with.
    """

    def _update_params_with_outputs(
        self, outputs: Dict[str, float], recv_all: bool = False
    ):
        """
        Update this task's parameters with the external code's outputs.

        This implicitly performs any unit conversions.

        Parameters
        ----------
        outputs: Dict[str, float]
            Key are the external code's parameter names, the values are
            the values for those parameters.
        recv_all: bool
            Whether to ignore the 'recv' attribute on the parameter
            mapping, and update all output parameter values.

        Raises
        ------
        CodesError
            If any output does not have a mapping to a bluemira
            parameter, or the output maps to a bluemira parameter that
            does not exist in this object's ParameterFrame.
        """
        mapped_outputs = self._map_external_outputs_to_bluemira_params(outputs, recv_all)
        self.params.update_values(mapped_outputs, source=self._name)

    def _map_external_outputs_to_bluemira_params(
        self, external_outputs: Dict[str, Any], recv_all: bool
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
        recv_all: bool
            Whether to ignore the 'recv' attribute on the parameter
            mapping, and update all output parameter values.

        Returns
        -------
        mapped_outputs: Dict[str, float]
            The keys are bluemira parameter names and the values are the
            external codes' outputs for those parameters (with necessary
            unit conversions made).
        """
        mapped_outputs = {}
        for bm_name, mapping in self.params.mappings.items():
            if not (mapping.recv or recv_all):
                continue
            output_value = self._get_output_or_raise(external_outputs, mapping.name)
            if output_value is None or mapping.unit is None:
                continue
            value = raw_uc(
                output_value, getattr(self.params, bm_name).unit, mapping.unit
            )
            mapped_outputs[bm_name] = value
        return mapped_outputs

    def _get_output_or_raise(
        self, external_outputs: Dict[str, Any], parameter_name: str
    ):
        try:
            output_value = external_outputs.get(parameter_name, None)
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


class CodesSolver(SolverABC):
    """
    Base class for solvers running an external code.
    """

    params: MappedParameterFrame

    def __init__(self, params: MappedParameterFrame):
        super().__init__(params)

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
        param_mappings = self.params.mappings
        for key, val in send_recv.items():
            try:
                p_map = param_mappings[key]
            except (AttributeError, KeyError):
                bluemira_warn(f"No mapping known for '{key}' in '{self.name}'.")
            else:
                for sr_key, sr_val in val.items():
                    setattr(p_map, sr_key, sr_val)
