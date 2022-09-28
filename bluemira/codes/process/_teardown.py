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
PROCESS teardown functions
"""

import json
import os
from typing import Dict, List, Union

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter import ParameterMapping
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesTeardown
from bluemira.codes.process.api import PROCESS_DICT, MFile, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.params import ProcessSolverParams


class Teardown(CodesTeardown):
    """
    Teardown task for PROCESS solver.

    Parameters
    ----------
    params: ProcessSolverParams
        The parameters for this task.
    run_directory: str
        The directory in which to run PROCESS. Used in run, and runinput
        functions.
    read_directory: str
        The directory to read PROCESS output files from. Used in read,
        readall, and mock functions.
    """

    params: ProcessSolverParams

    MOCK_JSON_NAME = "mockPROCESS.json"

    def __init__(
        self, params: ProcessSolverParams, run_directory: str, read_directory: str
    ):
        super().__init__(params, PROCESS_NAME)
        self.run_directory = run_directory
        self.read_directory = read_directory
        self._mfile_wrapper: _MFileWrapper = None

    def run(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(os.path.join(self.run_directory, "MFILE.DAT"), recv_all=False)

    def runinput(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(os.path.join(self.run_directory, "MFILE.DAT"), recv_all=True)

    def read(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(os.path.join(self.read_directory, "MFILE.DAT"), recv_all=False)

    def readall(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(os.path.join(self.read_directory, "MFILE.DAT"), recv_all=True)

    def mock(self):
        """
        Mock teardown the PROCESS solver.

        This loads a mockProcess.json file from the run directory and
        loads the values into this task's params.
        """
        bluemira_print("Mocking PROCESS systems code run")
        mock_file_path = os.path.join(self.read_directory, self.MOCK_JSON_NAME)
        outputs = _read_json_file_or_raise(mock_file_path)
        self.params.update_values(outputs, source=self._name)

    def get_raw_outputs(self, params: Union[List, str]) -> List[float]:
        """
        Get raw variables from an MFILE.

        Mapped bluemira parameters will have bluemira names.

        Parameters
        ----------
        params: Union[List, str]
            Names of parameters to access.

        Returns
        -------
        values: List[float]
            The parameter values.
        """
        if self._mfile_wrapper:
            return self._mfile_wrapper.extract_outputs(params)
        raise CodesError(
            "Cannot retrieve output from PROCESS MFile. "
            "The solver has not been run, so no MFile is available to read."
        )

    def _load_mfile(self, path: str, recv_all: bool):
        """
        Load the MFile at the given path, and update this object's
        params with the MFile's values.

        If recv_all, then ignore existing mappings and update all the
        params that correspond to a PROCESS output. If recv_all is
        False, then only update a parameter if its mapping has
        ``recv == True``.
        """
        outputs = self._read_mfile(path, self.params.mappings())
        self._update_params_with_outputs(outputs, self.params.mappings(), recv_all)

    def _read_mfile(self, path: str, param_mappings: Dict[str, ParameterMapping]):
        """
        Read an MFile, applying the given mappings, and performing unit
        conversions.
        """
        bm_units = {}
        for param in self.params:
            bm_units[param.name] = param.unit
        self._mfile_wrapper = _MFileWrapper(path, param_mappings, bm_units)
        return self._mfile_wrapper.read()


class _MFileWrapper:
    """
    Utility class to wrap a PROCESS MFile, and map its data to bluemira
    parameters.

    Parameters
    ----------
    file_path: str
        Path to an MFile.
    parameter_mappings: Dict[str, ParameterMapping]
        ... TODO(hsaunders1904)
    """

    def __init__(
        self,
        file_path: str,
        parameter_mappings: Dict[str, ParameterMapping],
        bm_units: Dict[str, str],
    ):
        if not os.path.isfile(file_path):
            raise CodesError(f"Path '{file_path}' is not a file.")
        self.file_path = file_path
        self.bm_to_p_mappings = self.update_mappings(parameter_mappings)
        self.bm_units = bm_units
        self.mfile = MFile(file_path)
        _raise_on_infeasible_solution(self.mfile)
        self._data = {}  # hold reference to data that has been read and mapped

    def read(self) -> Dict:
        """
        Read the data from the PROCESS MFile.

        Return the data as a dictionary where keys are bluemira parameter
        names, and values are values read from the MFile - performing any
        necessary unit conversions.
        """
        data = {}
        for key, val in self.mfile.data.items():
            data[key] = val["scan01"]
        self._data = data
        return data

    @staticmethod
    def update_mappings(old_mappings: Dict[str, ParameterMapping]):
        """
        Convert old PROCESS mappings to new ones

        Parameters
        ----------
        old_mappings: dict
            dictionary of parameter mappings

        Returns
        -------
        new_mappings: dict
            dictionary of new parameter mappings
        """
        new_mappings = {}
        for key, val in old_mappings.items():
            new_name = update_obsolete_vars(val.name)
            if new_name != val.name:
                val = ParameterMapping(
                    new_name, send=val.send, recv=val.recv, unit=val.unit
                )
            new_mappings[key] = val
        return new_mappings

    def extract_outputs(self, outputs: Union[List, str]) -> List[float]:
        """
        Searches MFile for variable.

        Outputs defined in bluemira variable names if they are mapped
        otherwise process variable names are the default

        Parameters
        ----------
        outputs: Union[List, str]
            parameter names to access

        Returns
        -------
        List of values

        """
        if isinstance(outputs, str):
            # Handle single variable request
            outputs = [outputs]

        out: List[float] = []
        for output in outputs:
            try:
                value = self._data[output]
            except KeyError:
                if (process_var := self.bm_to_p_mappings.get(output, None)) is not None:
                    process_var = process_var.name
                else:
                    process_var = output
                try:
                    process_value = self.mfile.data[process_var]
                except KeyError:
                    outputs.append(0.0)
                    bluemira_warn(
                        f"bluemira variable '{output}' a.k.a. "
                        f"PROCESS variable '{process_var}' "
                        "not found in PROCESS output. Value set to 0.0."
                    )
                    continue
                value = process_value["scan01"]
            out.append(value)
        return out


def _read_json_file_or_raise(file_path: str) -> Dict[str, float]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except OSError as os_error:
        raise CodesError(
            f"Cannot open mock PROCESS results file '{file_path}'."
        ) from os_error


def _raise_on_infeasible_solution(m_file) -> None:
    """
    Check that PROCESS found a feasible solution.

    Parameters
    ----------
    m_file: _MFileWrapper
        The PROCESS MFILE to check for a feasible solution

    Raises
    ------
    CodesError
        If a feasible solution was not found.
    """
    error_code = int(m_file.data["ifail"]["scan01"])
    if error_code != 1:
        message = (
            f"PROCESS did not find a feasible solution. ifail = {error_code}."
            " Check PROCESS logs."
        )
        raise CodesError(message)
