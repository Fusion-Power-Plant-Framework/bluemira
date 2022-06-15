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
from typing import Dict, Iterable, List, Union

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesTeardown
from bluemira.codes.process.api import PROCESS_DICT, MFile, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.utilities import get_recv_mapping


class Teardown(CodesTeardown):
    """
    Teardown task for PROCESS solver.

    Parameters
    ----------
    params: ParameterFrame
        The parameters for this task.
    run_directory: str
        The directory in which to run PROCESS. Used in run, and runinput
        functions.
    read_directory: str
        The directory to read PROCESS output files from. Used in read,
        readall, and mock functions.
    """

    MOCK_JSON_NAME = "mockPROCESS.json"

    def __init__(self, params: ParameterFrame, run_directory: str, read_directory: str):
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
        self.params.update_kw_parameters(outputs)

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
        param_mappings = get_recv_mapping(self.params, PROCESS_NAME, recv_all)
        self._mfile_wrapper = self._read_mfile(path, param_mappings)
        _raise_on_infeasible_solution(self._mfile_wrapper)
        param_names = param_mappings.keys()
        param_values = self._mfile_wrapper.extract_outputs(param_mappings.values())
        self._update_params_with_outputs(dict(zip(param_names, param_values)), recv_all)

    def _read_mfile(self, path: str, param_mappings: Dict[str, str]):
        """
        Read an MFile, applying the given mappings, and performing unit
        conversions.
        """
        unit_mappings = _get_unit_mappings(self.params, param_mappings.values())
        return _MFileWrapper(path, param_mappings, unit_mappings)


class _MFileWrapper:
    """
    Utility class to wrap a PROCESS MFile object, and map its data to
    bluemira parameters.
    """

    def __init__(self, filename, parameter_mapping, units):
        if not os.path.isfile(filename):
            raise CodesError(f"Path '{filename}' is not a file.")

        self.filename = filename
        self.params = {}  # ParameterFrame dictionary

        self.defs = self.get_defs(PROCESS_DICT["DICT_DESCRIPTIONS"])

        # TODO read units directly from PROCESS_NAME, waiting until python api
        self.units = units

        self.ptob_mapping = parameter_mapping
        self.btop_mapping = self.new_mappings(
            {val: key for key, val in parameter_mapping.items()}
        )

        self.mfile = MFile(filename)
        self.read()

    @staticmethod
    def new_mappings(old_mappings):
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
            new_mappings[key] = update_obsolete_vars(val)
        return new_mappings

    @staticmethod
    def get_defs(dictionary):
        """
        Get value definitions
        """
        for key, val in dictionary.items():
            dictionary[key] = " ".join(val.split("\n"))
        return dictionary

    def build_parameter_frame(self, msubdict):
        """
        Build a ParameterFrame from a sub-dictionary.
        """
        param = []
        for key, val in msubdict.items():
            try:
                desc = self.defs[key]
            except KeyError:
                desc = f"{key}: PROCESS variable description not found"
            try:
                unit = self.units[key]
            except KeyError:
                unit = "dimensionless"
            param.append([key, desc, val, unit, None, PROCESS_NAME])
        return ParameterFrame(param)

    def read(self):
        """
        Read the data.
        """
        var_mod = set()
        for val in self.mfile.data.values():
            var_mod.add(val["var_mod"])

        dic = {key: {} for key in var_mod}  # Nested dictionary
        self.params = {}
        for key, val in self.mfile.data.items():
            dic[val["var_mod"]][self.ptob_mapping.get(key, key)] = val["scan01"]
        for key in dic.keys():
            self.params[key] = self.build_parameter_frame(dic[key])
        self.rebuild_RB_dict()

    def rebuild_RB_dict(self):  # noqa :N802
        """
        Takes the TF coil detailed breakdown and reconstructs the radial
        build ParameterFrame.

        Notes
        -----
        The PROCESS radial build is taken along the diagonal (maximum
        length) of the TF coil, so this must be taken into consideration
        when translating the geometry into the mid-plane.
        """
        # TODO: Handle case of interrupted run causing a half-written to be
        # re-read into memory in the next run and not having the right keys
        rb = self.params["Radial Build"]
        tf = self.params["TF coils"]
        pl = self.params["Plasma"]
        # TODO: Handle ST case (copper coil breakdown not as detailed)
        for val in ["tk_tf_wp", "tk_tf_front_ib", "tk_tf_nose"]:
            if val in tf.keys():
                rb.add_parameter(tf.get_param(val))
        # No mapping for PRECOMP, TFCTH, RMINOR
        # (given mapping not valid)
        try:
            rtfin = rb["r_cs_in"] + rb["tk_cs"] + rb["precomp"] + rb["g_cs_tf"]
            r_ts_ib_in = rtfin + rb["tk_tf_inboard"] + rb["g_ts_tf"] + rb["tk_ts"]
            r_vv_ib_in = r_ts_ib_in + rb["g_vv_ts"] + rb["tk_vv_in"] + rb["tk_sh_in"]
            r_fw_ib_in = r_vv_ib_in + rb["g_vv_bb"] + rb["tk_bb_ib"] + rb["tk_fw_in"]
            r_fw_ob_in = (
                r_fw_ib_in + rb["tk_sol_ib"] + 2 * pl["rminor"] + rb["tk_sol_ob"]
            )
            r_vv_ob_in = r_fw_ob_in + rb["tk_fw_out"] + rb["tk_bb_ob"] + rb["g_vv_bb"]
        except KeyError as key_err:
            raise CodesError(
                f"A bluemira parameter is missing in the PROCESS output: {key_err}"
            )
        radial_params = [
            ("r_tf_in", "Inboard radius of the TF coil inboard leg", rtfin),
            ("r_ts_ib_in", "Inboard TS inner radius", r_ts_ib_in),
            ("r_vv_ib_in", "Inboard vessel inner radius", r_vv_ib_in),
            ("r_fw_ib_in", "Inboard first wall inner radius", r_fw_ib_in),
            ("r_fw_ob_in", "Outboard first wall inner radius", r_fw_ob_in),
            ("r_vv_ob_in", "Outboard vessel inner radius", r_vv_ob_in),
        ]
        for radial_param in radial_params:
            rb.add_parameter(*radial_param, "m", None, PROCESS_NAME)
        self.params["Radial Build"] = rb

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
        out: List[float] = []
        if isinstance(outputs, str):
            # Handle single variable request
            outputs = [outputs]
        for var in outputs:
            if isinstance(var, list):
                for v in var:
                    if found := self._find_var_in_frame(v, out):
                        break
            else:
                found = self._find_var_in_frame(var, out)

            if not found:
                process_var = self.btop_mapping.get(var, var)
                bluemira_var = "N/A" if var == process_var else var
                out.append(0.0)
                bluemira_warn(
                    f"bluemira variable '{bluemira_var}' a.k.a. "
                    f"PROCESS variable '{process_var}' "
                    "not found in PROCESS output. Value set to 0.0."
                )
        return out

    def _find_var_in_frame(self, var, out):
        """
        Find variable value in parameter frame

        Parameters
        ----------
        var:str
            variable
        out: list
            output list

        Returns
        -------
        bool
            if variable found
        """
        found = False
        for frame in self.params.values():
            if var in frame.keys():
                out.append(frame[var])
                found = True
                break  # only keep one!
        return found


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
    error_code = m_file.params["Numerics"]["ifail"]
    if error_code != 1:
        message = (
            f"PROCESS did not find a feasible solution. ifail = {error_code}."
            " Check PROCESS logs."
        )
        raise CodesError(message)


def _get_unit_mappings(
    params: ParameterFrame, param_names: Iterable[str]
) -> Dict[str, str]:
    """
    Get the PROCESS units for the given bluemira parameters.
    """
    units = {}
    for param_name in param_names:
        param = params.get_param(param_name)
        unit = param.mapping[PROCESS_NAME].unit
        if unit:
            units[param_name] = unit
        else:
            raise CodesError(
                f"No PROCESS unit conversion defined for parameter '{param_name}'."
            )
    return units


def _read_json_file_or_raise(file_path: str) -> Dict[str, float]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except OSError as os_error:
        raise CodesError(
            f"Cannot open mock PROCESS results file '{file_path}'."
        ) from os_error
