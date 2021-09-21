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
PROCESS interface dictionaries and tools
"""
import os
import subprocess  # noqa (S404)
from collections import namedtuple
from typing import Dict
import re

from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print

from BLUEPRINT.base.file import get_PROCESS_root, get_BP_path
from BLUEPRINT.base.error import SysCodesError

from process.io.mfile import MFile
from process.io.python_fortran_dicts import get_dicts
from process.io.in_dat import InDat

try:
    from process.io.obsolete_vars import OBS_VARS
except (ModuleNotFoundError, FileNotFoundError):
    OBS_VARS = dict()
    bluemira_warn("The OBS_VAR dict is not installed in your PROCESS installed version")

# Load dicts from dicts JSON file
PROCESS_DICT = get_dicts()

PATH = None
try:
    PATH = get_PROCESS_root()
except FileNotFoundError:
    pass

DEFAULT_INDAT = os.path.join(get_BP_path("syscodes"), "PROCESS_DEFAULT_IN.DAT")

PTOBUNITS = {
    "a": "A",
    "a/m2": "A/m^2",
    "h": "H",
    "k": "K",
    "kw": "kW",
    "m": "m",
    "m2": "m^2",
    "m3": "m^3",
    "mpa": "MPa",
    "mw": "MW",
    "ohm": "Ohm",
    "pa": "Pa",
    "v": "V",
    "kv": "kV",
    "w": "W",
    "wb": "Wb",
}

BTOPUNITS = {v: k for k, v in PTOBUNITS.items()}


def update_obsolete_vars(process_map_name: str) -> str:
    """
    Check if the BLUEPRINT variable is up to date using the OBS_VAR dict.
    If the PROCESS variable name has been updated in the installed version
    this function will provide the updated variable name.

    Parameters
    ----------
    process_map_name: str
        PROCESS variable name obtained from the BLUEPRINT mapping.

    Returns
    -------
    process_name: str
        PROCESS variable names valid for the install (if OBS_VAR is updated
        correctly)
    """
    process_name = process_map_name
    while process_name in OBS_VARS:
        process_name = OBS_VARS[process_name]
    if not process_name == process_map_name:
        bluemira_print(
            f"Obsolete {process_map_name} PROCESS mapping name."
            f"The current PROCESS name is {process_name}"
        )
    return process_name


def _convert(dictionary, k):
    if k in dictionary.keys():
        return dictionary[k]
    return k


def _pconvert(dictionary, k):
    kn = _convert(dictionary, k)
    if kn is None:
        raise ValueError(f'Define a parameter conversion for "{k}"')
    return kn


def convert_unit_p_to_b(s):
    """
    Conversion from PROCESS units to BLUEPRINT units
    Handles text formatting only
    """
    return _convert(PTOBUNITS, s)


def convert_unit_b_to_p(s):
    """
    Conversion from BLUEPRINT units to PROCESS units
    """
    return _convert(BTOPUNITS, s)


def get_PROCESS_read_mapping(inputs, read_all=False) -> Dict[str, str]:
    """
    Get the read mapping for PROCESS variables from the input ParameterFrame

    Parameters
    ----------
    inputs: ParameterFrame
        The parameter frame containing the BLUEPRINT parameters and their mapping to
        PROCESS variables.
    read_all: bool, optional
        If True then read all variables with a mapping defined, even if read=False. By
        default, False.

    Returns
    -------
    read_mapping: Dict[str, str]
        The mapping between PROCESS names (key) and BLUEPRINT names (value) for
        Parameters that are to be read from PROCESS.
    """
    read_mapping = {}
    for k in inputs.keys():
        p = inputs.get_param(k)
        if p.mapping is not None and "PROCESS" in p.mapping:
            m = p.mapping["PROCESS"]
            if read_all or m.read:
                read_mapping[m.name] = k
    return read_mapping


class PROCESSInputWriter(InDat):
    """
    BLUEPRINT IN.DAT writer for PROCESS input.

    Parameters
    ----------
    template_indat: str
        Path to the IN.DAT file to use as the template for PROCESS parameters.
    """

    def __init__(self, template_indat=DEFAULT_INDAT):
        if os.path.isfile(template_indat):
            # InDat autoloads IN.DAT without checking for existence
            super().__init__(filename=template_indat)
        else:
            super().__init__(filename=None)
            self.filename = template_indat


class BMFile(MFile):
    """
    BLUEPRINT MFile reader for PROCESS output
    Sub-classed from PROCESS utilities
    Builds ParameterFrames of output in logical chunks
    """

    def __init__(self, path, parameter_mapping):
        filename = os.path.join(path, "MFILE.DAT")

        if not os.path.isfile(filename):
            raise SysCodesError(f"No MFILE.dat found in: {path}")

        super().__init__(filename=filename)
        self.defs = self.unitsplit(PROCESS_DICT["DICT_DESCRIPTIONS"])
        self.ptob_mapping = parameter_mapping
        self.btop_mapping = {v: k for k, v in parameter_mapping.items()}
        for k, v in self.btop_mapping.items():
            self.btop_mapping[k] = update_obsolete_vars(v)
        self.read()

    @staticmethod
    def linesplit(line):
        """
        Split a line in the MFILE.dat.
        """
        # TODO improve re catching and pick up [] etc
        li = line.split("\n")
        v = " ".join(li)
        try:
            u = re.search(r"\((\w+)\)", li[0]).group(1)
            v = v.replace("(" + u + ")", "")
        except AttributeError:
            u = "N/A"
        return v, u

    def unitsplit(self, dictionary):
        """
        Splits description of variable and returns dict of k, (value, unit)
        """
        p = namedtuple("PROCESSparameter", ["Descr", "Unit"])
        d = {}
        for k, v in dictionary.items():
            v, u = self.linesplit(v)
            d[k] = p(v, u)
        return d

    def build_parameter_frame(self, msubdict):
        """
        Build a ParameterFrame from a sub-dictionary.
        """
        p = []
        for k, v in msubdict.items():
            try:
                descr, unit = self.defs[k][0], convert_unit_p_to_b(self.defs[k][1])
            except KeyError:
                descr = k + ": PROCESS variable description not found"
                unit = "N/A"
            p.append([k, descr, v, unit, None, "PROCESS"])
        return ParameterFrame(p)

    def read(self):
        """
        Read the data.
        """
        var_mod = []
        for v in self.data.values():
            var_mod.append(v["var_mod"])
        var_mod = set(var_mod)
        d = {k: {} for k in var_mod}  # Nested dictionary
        self.params = {}  # ParameterFrame dictionary
        for k, v in self.data.items():
            d[v["var_mod"]][self.ptob_mapping.get(k, k)] = v["scan01"]
        for k in d.keys():
            self.params[k] = self.build_parameter_frame(d[k])
        self.rebuild_RB_dict()

    def rebuild_RB_dict(self):  # noqa (N802)
        """
        Takes the TF coil detailed breakdown and reconstructs the radial build
        ParameterFrame.

        Notes
        -----
        The PROCESS radial build is taken along the diagonal (maximum length) of the
        TF coil, so this must be taken into consideration when translating the geometry
        into the mid-plane.
        """
        # TODO: Handle case of interrupted run causing a half-written to be
        # re-read into memory in the next run and not having the right keys
        rb, tf = self.params["Radial Build"], self.params["TF coils"]
        pl = self.params["Plasma"]
        # TODO: Handle ST case (copper coil breakdown not as detailed)
        for v in [
            "tk_tf_wp",
            "tk_tf_front_ib",
            "tk_tf_nose",
        ]:
            if v in tf.keys():
                rb.add_parameter(tf.get(v))
        # No mapping for PRECOMP, TFCTH, FWITH, FWOTH, RMINOR
        # (given mapping not valid)
        rtfin = rb["r_cs_in"] + rb["tk_cs"] + rb["precomp"] + rb["g_cs_tf"]
        r_ts_ib_in = rtfin + rb["tfcth"] + rb["g_ts_tf"] + rb["tk_ts"]
        r_vv_ib_in = r_ts_ib_in + rb["g_vv_ts"] + rb["tk_vv_in"] + rb["tk_sh_in"]
        r_fw_ib_in = r_vv_ib_in + rb["g_vv_bb"] + rb["tk_bb_ib"] + rb["fwith"]
        r_fw_ob_in = r_fw_ib_in + rb["tk_sol_ib"] + 2 * pl["rminor"] + rb["tk_sol_ob"]
        r_vv_ob_in = r_fw_ob_in + rb["fwoth"] + rb["tk_bb_ob"] + rb["g_vv_bb"]
        # fmt:off
        rb.add_parameter("r_tf_in", "Inboard radius of the TF coil inboard leg", rtfin, "m", None, "PROCESS")
        rb.add_parameter("r_ts_ib_in", "Inboard TS inner radius", r_ts_ib_in, "m", None, "PROCESS")
        rb.add_parameter("r_vv_ib_in", "Inboard vessel inner radius", r_vv_ib_in, "m", None, "PROCESS")
        rb.add_parameter("r_fw_ib_in", "Inboard first wall inner radius", r_fw_ib_in, "m", None, "PROCESS")
        rb.add_parameter("r_fw_ob_in", "Outboard first wall inner radius", r_fw_ob_in, "m", None, "PROCESS")
        rb.add_parameter("r_vv_ob_in", "Outboard vessel inner radius", r_vv_ob_in, "m", None, "PROCESS")
        # fmt:on
        self.params["Radial Build"] = rb

    def extract_outputs(self, outputs):
        """
        Searches MFile for variable
        Outputs defined in BLUEPRINT variable names
        """
        out = []
        if isinstance(outputs, str):
            # Handle single variable request
            outputs = [outputs]
        for var in outputs:
            found = False
            for frame in self.params.values():
                if var in frame.keys():
                    out.append(frame[var])
                    found = True
                    break  # only keep one!
            if not found:
                out.append(0.0)
                bluemira_warn(
                    f'BLUEPRINT variable "{var}" a.k.a. '
                    f'PROCESS variable "{self.btop_mapping[var]}" '
                    "not found in PROCESS output. Value set to 0.0."
                )
        return out


class PROCESSRunner:
    """
    Helper class to manage PROCESS runs.

    Parameters
    ----------
    inputs: ParameterFrame
         The parameters for this run. Values with a mapping will be used by PROCESS.
    template_indat: str
        Path to the template IN.DAT file to be used for the run, by default
        the value specified by DEFAULT_INDAT.
    run_dir: str
        Path to the PROCESS run directory, where the main PROCESS executable is located
        and the input/output files will be written.
    run_input: bool, optional
        No PROCESS template value input is overwritten
        Default False
    read_all: bool, optional
        All PROCESS output is read
        Default False=False
    """

    output_files = [
        "OUT.DAT",
        "MFILE.DAT",
        "OPT.DAT",
        "SIG_TF.DAT",
    ]

    def __init__(
        self,
        inputs,
        tempate_indat=DEFAULT_INDAT,
        run_dir=".",
        run_input=False,
        read_all=False,
    ):
        self.run_dir = run_dir
        self.template_indat = (
            tempate_indat if tempate_indat is not None else DEFAULT_INDAT
        )
        self.read_parameter_mapping = get_PROCESS_read_mapping(inputs, read_all=read_all)
        self.write_indat(inputs, run_input=run_input)

    def write_indat(self, inputs, run_input=False):
        """
        Write the IN.DAT file and stores in the main PROCESS folder.

        Parameters
        ----------
        inputs: ParameterFrame
            parameters for the run. Values with a mapping will be used by
            PROCESS.
        run_input: bool, optional
            Option to keep the template run as it, ignoring BLUEPRINT inputs.
            Default, False
        """
        # Load defaults in BLUEPRINT folder
        writer = PROCESSInputWriter(template_indat=self.template_indat)
        if not run_input:
            for p in inputs.get_parameter_list():  # Overwrite variables
                if p.mapping is not None and "PROCESS" in p.mapping:
                    m = p.mapping["PROCESS"]
                    if m.write:
                        writer.add_parameter(update_obsolete_vars(m.name), p.value)

        filename = os.path.join(self.run_dir, "IN.DAT")
        writer.write_in_dat(output_filename=filename)

    def run(self):
        """
        Run PROCESS.
        """
        self._clear_PROCESS_output()
        self._run()
        self._check_PROCESS_output()

    def read_mfile(self):
        """
        Read the MFILE.DAT from the PROCESS run_dir.

        Returns
        -------
        mfile: BMFile
            The object representation of the output MFILE.DAT.
        """
        m_file = BMFile(self.run_dir, self.read_parameter_mapping)
        self._check_feasible_solution(m_file)
        return m_file

    def _clear_PROCESS_output(self):
        """
        Clear the output files from PROCESS run directory.
        """
        for filename in self.output_files:
            filepath = os.sep.join([self.run_dir, filename])
            if os.path.exists(filepath):
                os.remove(filepath)

    def _check_PROCESS_output(self):
        """
        Check that PROCESS has produced valid (non-zero lined) output.

        Raises
        ------
        SysCodesError
            If any resulting output files don't exist or are empty.
        """
        for filename in self.output_files:
            filepath = os.sep.join([self.run_dir, filename])
            if os.path.exists(filepath):
                with open(filepath) as fh:
                    if len(fh.readlines()) == 0:
                        message = (
                            f"PROCESS generated an empty {filename} "
                            f"file in {self.run_dir} - check PROCESS logs."
                        )
                        bluemira_warn(message)
                        raise SysCodesError(message)
            else:
                message = (
                    f"PROCESS run did not generate the {filename} "
                    f"file in {self.run_dir} - check PROCESS logs."
                )
                bluemira_warn(message)
                raise SysCodesError(message)

    @staticmethod
    def _check_feasible_solution(m_file):
        """
        Check that PROCESS found a feasible solution.

        Parameters
        ----------
        m_file: BMFile
            The PROCESS MFILE to check for a feasible solution

        Raises
        ------
        SysCodesError
            If a feasible solution was not found.
        """
        error_code = m_file.params["Numerics"]["ifail"]
        if error_code != 1:
            message = (
                f"PROCESS did not find a feasible solution. ifail = {error_code}."
                " Check PROCESS logs."
            )
            bluemira_warn(message)
            raise SysCodesError(message)

    def _run(self):
        subprocess.run("process", cwd=self.run_dir)  # noqa (S603)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
