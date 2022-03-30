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
from typing import List, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import bluemira.codes.interface as interface
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.process.api import PROCESS_DICT, MFile, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS
from bluemira.utilities.tools import is_num


class BMFile(MFile):
    """
    Bluemira MFile reader for PROCESS output
    Sub-classed from PROCESS utilities
    Builds ParameterFrames of output in logical chunks
    """

    def __init__(self, path, parameter_mapping, units):
        filename = os.path.join(path, "MFILE.DAT")

        if not os.path.isfile(filename):
            raise CodesError(f"No MFILE.dat found in: {path}")

        super().__init__(filename=filename)

        self.defs = self.get_defs(PROCESS_DICT["DICT_DESCRIPTIONS"])

        # TODO read units directly from PROCESS, waiting until python api
        self.units = units

        self.ptob_mapping = parameter_mapping
        self.btop_mapping = self.new_mappings(
            {val: key for key, val in parameter_mapping.items()}
        )

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

    def get_defs(self, dictionary):
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
            param.append([key, desc, val, unit, None, PROCESS])
        return ParameterFrame(param)

    def read(self):
        """
        Read the data.
        """
        var_mod = []
        for val in self.data.values():
            var_mod.append(val["var_mod"])
        var_mod = set(var_mod)
        dic = {key: {} for key in var_mod}  # Nested dictionary
        self.params = {}  # ParameterFrame dictionary
        for key, val in self.data.items():
            dic[val["var_mod"]][self.ptob_mapping.get(key, key)] = val["scan01"]
        for key in dic.keys():
            self.params[key] = self.build_parameter_frame(dic[key])
        self.rebuild_RB_dict()

    def rebuild_RB_dict(self):  # noqa :N802
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
        for val in [
            "tk_tf_wp",
            "tk_tf_front_ib",
            "tk_tf_nose",
        ]:
            if val in tf.keys():
                rb.add_parameter(tf.get_param(val))
        # No mapping for PRECOMP, TFCTH, RMINOR
        # (given mapping not valid)
        rtfin = rb["r_cs_in"] + rb["tk_cs"] + rb["precomp"] + rb["g_cs_tf"]
        r_ts_ib_in = rtfin + rb["tk_tf_inboard"] + rb["g_ts_tf"] + rb["tk_ts"]
        r_vv_ib_in = r_ts_ib_in + rb["g_vv_ts"] + rb["tk_vv_in"] + rb["tk_sh_in"]
        r_fw_ib_in = r_vv_ib_in + rb["g_vv_bb"] + rb["tk_bb_ib"] + rb["tk_fw_in"]
        r_fw_ob_in = r_fw_ib_in + rb["tk_sol_ib"] + 2 * pl["rminor"] + rb["tk_sol_ob"]
        r_vv_ob_in = r_fw_ob_in + rb["tk_fw_out"] + rb["tk_bb_ob"] + rb["g_vv_bb"]
        # fmt:off
        rb.add_parameter("r_tf_in", "Inboard radius of the TF coil inboard leg", rtfin, "m", None, PROCESS)
        rb.add_parameter("r_ts_ib_in", "Inboard TS inner radius", r_ts_ib_in, "m", None, PROCESS)
        rb.add_parameter("r_vv_ib_in", "Inboard vessel inner radius", r_vv_ib_in, "m", None, PROCESS)
        rb.add_parameter("r_fw_ib_in", "Inboard first wall inner radius", r_fw_ib_in, "m", None, PROCESS)
        rb.add_parameter("r_fw_ob_in", "Outboard first wall inner radius", r_fw_ob_in, "m", None, PROCESS)
        rb.add_parameter("r_vv_ob_in", "Outboard vessel inner radius", r_vv_ob_in, "m", None, PROCESS)
        # fmt:on
        self.params["Radial Build"] = rb

    def extract_outputs(self, outputs: Union[List, str]):
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
        out = []
        if isinstance(outputs, str):
            # Handle single variable request
            outputs = [outputs]
        for var in outputs:
            if isinstance(var, list):
                for v in var:
                    found = self._find_var_in_frame(var, out)
                    if found:
                        break
            else:
                found = self._find_var_in_frame(var, out)

            if not found:
                process_var = self.btop_mapping.get(var, var)
                bluemira_var = "N/A" if var == process_var else var
                out.append(0.0)
                bluemira_warn(
                    f'bluemira variable "{bluemira_var}" a.k.a. '
                    f'PROCESS variable "{process_var}" '
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


class Teardown(interface.Teardown):
    """
    Teardown Task for process
    """

    def _run(self):
        self._check_PROCESS_output_files()

        # Load PROCESS results into bluemira
        self.load_PROCESS_run(recv_all=False)

    def _runinput(self):
        self._check_PROCESS_output_files()

        # Load PROCESS results into bluemira
        self.load_PROCESS_run(recv_all=True)

    def _read(self):
        self.load_PROCESS_run(path=self.parent.read_dir, recv_all=False)

    def _readall(self):
        self.load_PROCESS_run(path=self.parent.read_dir, recv_all=True)

    def _mock(self):
        self.mock_PROCESS_run()

    def load_PROCESS_run(self, path: str = None, recv_all: bool = False):
        """
        Loads a PROCESS output file (MFILE.DAT) and extract some or all its output data

        Parameters
        ----------
            path: str, optional
                path to PROCESS output file (MFILE.DAT) to load
                uses `run_dir` if not provided
            recv_all: bool, optional
                True - Read all PROCESS output with a mapping,
                False - reads only PROCESS output with a mapping and recv = True.
                Default, False
        """
        bluemira_print("Loading PROCESS MFILE.DAT")

        # Load all PROCESS vars mapped with a bluemira input
        self.bm_file = self.read_mfile(path)
        var = (
            self.parent._parameter_mapping.values()
            if recv_all
            else self.parent._recv_mapping.values()
        )
        param = self.bm_file.extract_outputs(var)

        self.prepare_outputs(dict(zip(var, param)), source=PROCESS)

    def read_mfile(self, path: str = None):
        """
        Read the MFILE.DAT from the PROCESS run_dir.

        Parameters
        ----------
            path: str, optional
                path to PROCESS output file (MFILE.DAT) to load
                uses `run_dir` if not provided

        Returns
        -------
        mfile: BMFile
            The object representation of the output MFILE.DAT.
        """
        m_file = BMFile(
            self.parent.run_dir if path is None else path,
            self.parent._parameter_mapping,
            {
                key: self.parent.params.get_param(val).mapping[PROCESS].unit
                for key, val in self.parent._parameter_mapping.items()
            },
        )
        self._check_feasible_solution(m_file)
        return m_file

    def mock_PROCESS_run(self):
        """
        Mock PROCESS. To be used in tests and examples only!
        """
        bluemira_print("Mocking PROCESS systems code run")

        # Create mock PROCESS file.
        path = self.parent.read_dir
        filename = os.sep.join([path, "mockPROCESS.json"])
        with open(filename, "r") as fh:
            process = json.load(fh)

        self.prepare_outputs(process, source=f"{PROCESS} (Mock)")

    def _check_PROCESS_output_files(self):
        """
        Check that PROCESS has produced valid (non-zero lined) output.

        Raises
        ------
        CodesError
            If any resulting output files don't exist or are empty.
        """
        for filename in self.parent.output_files:
            filepath = os.sep.join([self.parent.run_dir, filename])
            if os.path.exists(filepath):
                with open(filepath) as fh:
                    if len(fh.readlines()) == 0:
                        message = (
                            f"PROCESS generated an empty {filename} "
                            f"file in {self.parent.run_dir} - check PROCESS logs."
                        )
                        raise CodesError(message)
            else:
                message = (
                    f"PROCESS run did not generate the {filename} "
                    f"file in {self.parent.run_dir} - check PROCESS logs."
                )
                raise CodesError(message)

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


def boxr(ri, ro, w, off=0):
    """
    Generate coordinates for an arbitrary height radial width. Used in plotting.
    """
    xc = [ri, ri, ro, ro, ri]
    yc = [-w, w, w, -w, -w]
    yc = [i + off for i in yc]
    return xc, yc


def read_rb_line(line):
    """
    Inputs: a line from the PROCESS radial/vertical build
    Outputs: the first three columns from that line
    """
    line = line.split()
    for i, v in enumerate(line):
        if is_num(v) is False:
            if i > 0:
                line[0] = " ".join([line[0], v])
        elif is_num(v) is True:
            line[1] = float(v)
            line[2] = float(line[i + 1])
            return line[:3]


def strip_num(line, typ="float", n=0):
    """
    Returns a single number in a line
    """
    numb = [float(i) for i in line.split() if is_num(i) is True][n]
    if typ == "int":
        numb = int(numb)
    return numb


def read_n_line(line):
    """
    Reads a line from the PROCESS output in the format below:
    Major radius (m)   / (rmajor)     /           9.203  ITV
    Returns the variable name [0], its value [1], and the rest [2]
    """
    line = line.split()
    out = [""] * 3
    for i, word in enumerate(line):
        if word.startswith("(") is True or word.endswith(")") is True:
            out[2] = " ".join([out[2], word]).lstrip()
        elif is_num(word) is True:
            out[1] = float(word)
        elif word.isupper() is True:
            out[2] = " ".join([out[2], word]).lstrip()
        else:
            out[0] = " ".join([out[0], word]).lstrip()
    return out


def setup_radial_build(run, width=1.0):
    """
    Plots radial and vertical build of a PROCESS run
    Input: Dictionary of PROCESS output
    Output: Plots
    """
    from bluemira.geometry._deprecated_loop import Loop

    R_0 = run["R_0"]

    col = {
        "Gap": "w",
        "blanket": "#edb120",
        "TF coil": "#7e2f8e",
        "Vacuum vessel": "k",
        "Plasma": "#f77ec7",
        "first wall": "#edb120",
        "Machine bore": "w",
        "precomp": "#0072bd",
        "scrape-off": "#a2142f",
        "solenoid": "#0072bd",
        "Thermal shield": "#77ac30",
    }

    f, ax = plt.subplots(figsize=[14, 10])

    lpatches = []
    gkeys = [
        "blanket",
        "TF coil",
        "Vacuum vessel",
        "Plasma",
        "scrape-off",
        "solenoid",
        "Thermal shield",
    ]
    glabels = {
        "blanket": "Breeding blanket",
        "TF coil": "TF coil",
        "Plasma": "Plasma",
        "Vacuum vessel": "Vacuum vessel",
        "scrape-off": "Scrape-off layer",
        "solenoid": "Central solenoid",
        "Thermal shield": "Thermal shield",
    }
    for comp in run["Radial Build"]:
        xc, yc = boxr(comp[2] - comp[1], comp[2], width)
        yc = np.array(yc)
        loop = Loop(x=xc, y=yc)
        for key, c in col.items():
            if key in comp[0]:
                ax.plot(xc, yc, color=c, linewidth=0, label=key)
                if comp[1] > 0:
                    loop.plot(ax, facecolor=c, edgecolor="k", linewidth=0)
                if key in gkeys:
                    gkeys.remove(key)
                    lpatches.append(patches.Patch(color=c, label=glabels[key]))

    ax.set_xlim([0, np.ceil(run["Radial Build"][-1][-1])])
    ax.set_ylim([-width * 0.5, width * 0.5])
    ax.set_xticks(list(ax.get_xticks()) + [R_0])

    def tick_format(value, n):
        if value == R_0:
            return "\n$R_{0}$"
        else:
            return int(value)

    def tick_formaty(value, n):
        if value == 0:
            return int(value)
        else:
            return ""

    ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formaty))
    ax.set_xlabel("$x$ [m]")
    ax.set_aspect("equal")
    ax.legend(
        handles=lpatches,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
    )


def process_RB_fromOUT(f):  # noqa :N802
    """
    Parse PROCESS radial build from an OUT.DAT file.
    """
    # If the input is a string, treat as file name, and ensure it is closed.
    if isinstance(f, str):
        with open(f) as fh:
            return process_RB_fromOUT(fh)  # Recursive call with file object
    raw = f.readlines()
    raw = raw[1:]
    if not raw:
        raise IOError("Cannot read from input file.")
    if PROCESS not in raw[1] and PROCESS not in raw[2]:
        bluemira_warn(
            "Either this ain't a PROCESS OUT.DAT file, or those hijos "
            "changed the format."
        )

    def read_radial_build(num):  # Be careful that the numbers don't change
        rb = []
        num += 1
        while "***" not in raw[num]:
            if read_rb_line(raw[num]) is None:
                pass
            else:
                rb.append(read_rb_line(raw[num]))
            num += 1
        return rb

    flag1, flag2, flag3 = False, False, False
    for num, line in enumerate(raw):
        if "* Radial Build *" in line:
            flag1 = True
            rb = read_radial_build(num)
        if "n_tf" in line:
            flag2 = True
            n_TF = strip_num(line, typ="int")
        if "Major radius" in line:
            flag3 = True
            R_0 = strip_num(line)
        if flag1 and flag2 and flag3:
            break
    return {"Radial Build": rb, "n_TF": n_TF, "R_0": R_0}


def plot_radial_build(sys_code_dir: str, width: float = 1.0, show: bool = True):
    """
    Plot PROCESS radial build.

    Parameters
    ----------
    sys_code_dir: str
        OUT.DAT directory location
    width: float
        The relative width of the plot.
    show: bool
        If True then immediately display the plot, else delay displaying the plot until
        the user shows it, by default True.
    """
    filename = os.sep.join([sys_code_dir, "OUT.DAT"])

    if not os.path.exists(filename):
        raise CodesError(f"Could not find PROCESS OUT.DAT results at {filename}")

    radial_build = process_RB_fromOUT(filename)
    setup_radial_build(radial_build, width=width)
    if show:
        plt.show()
