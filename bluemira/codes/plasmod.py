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
The bluemira Plasmod wrapper.
"""
import os
from bluemira.codes.wrapper import RunMode
from bluemira.base.parameter import Parameter, ParameterFrame, ParameterMapping
from typing import Dict
from bluemira.base.file import get_bluemira_path

DEFAULT_PLASMOD_DATA_FOLDER = (get_bluemira_path("codes", subfolder="data"),)
DEFAULT_PLASMOD_INPUT_FILE = os.path.join(
    DEFAULT_PLASMOD_DATA_FOLDER, "plasmod_input.dat"
)
DEFAULT_PLASMOD_INPUT_FILE = os.path.join(
    DEFAULT_PLASMOD_DATA_FOLDER, "plasmod_output.dat"
)

# Parameter DataFrame
# Var_name    Name    Value     Unit        Description        Source   Mapping
# Mapping -> {external_var_name, read, write}

params = [
    ["a", None, 0, None, None, None],
    ["b", None, 1, None, None, None, None],
    ["c", None, 2, None, None, None, {"Plasmod": ParameterMapping("cp", False, False)}],
    ["d", None, 3, None, None, None, {"Plasmod": ParameterMapping("dp", False, True)}],
    ["e", None, 4, None, None, None, {"Plasmod": ParameterMapping("ep", True, False)}],
    ["f", None, 5, None, None, None, {"Plasmod": ParameterMapping("fp", True, True)}],
    ["g", None, 6, None, None, None, {"FAKE_CODE": ParameterMapping("gp", True, True)}],
]


def get_Plasmod_read_mapping(inputs, read_all=False) -> Dict[str, str]:
    """
    Get the read mapping for Plasmod variables from the input ParameterFrame

    Parameters
    ----------
    inputs: ParameterFrame
        The parameter frame containing the bluemira parameters and their mapping to
        Plasmod variables.
    read_all: bool, optional
        If True then read all variables with a mapping defined, even if read=False. By
        default, False.

    Returns
    -------
    read_mapping: Dict[str, str]
        The mapping between Plasmod names (key) and bluemira names (value) for
        Parameters that are to be read from Plasmod.
    """
    read_mapping = {}
    for k in inputs.keys():
        p = inputs.get_param(k)
        if p.mapping is not None and "Plasmod" in p.mapping:
            m = p.mapping["Plasmod"]
            if read_all or m.read:
                read_mapping[m.name] = k
    return read_mapping


def plasmod_input_writer(params, input_file=DEFAULT_PLASMOD_INPUT_FILE, template=False):
    """
    Bluemira writer for Plasmod input.

    Parameters
    ----------
    params: Union[dict, ParameterFrame]
        parameter frame to
    input_file

    template_indat: str
        Path to the IN.DAT file to use as the template for Plasmod parameters.
    """

    def __init__(self, input_file=DEFAULT_PLASMOD_INPUT_FILE, template=False):

        if os.path.isfile(template_indat):
            # InDat autoloads IN.DAT without checking for existence
            super().__init__(filename=template_indat)
        else:
            super().__init__(filename=None)
            self.filename = template_indat


class PlasmodRunMode(RunMode):
    def _prominence(self):
        pass

    def _batch(self):
        pass

    def _mock(self):
        pass


class Setup:
    """
    This class set up the Plasmod environment and parameters.

    Parameters
    ----------
    runmode: str
        The running method for plasmod. See bluemira.codes.plasmod.RunMode
        for possible values.
    save_path: str
        path to save plasmod input and output data
    """

    def __init__(self, runmode, save_path="data/plasmod"):
        self.set_runmode(runmode)
        self.save_path = save_path

    def set_runmode(self, runmode):
        self.runmode = RunMode[runmode]


class Run:
    def __init__(self, setup):
        self.result = setup.runmode(self, *setup.args)
