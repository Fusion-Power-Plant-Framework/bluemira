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
PROCESS api
"""

import os

from bluemira.base.file import get_bluemira_root
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print

PROCESS_ENABLED = True


# Create dummy PROCESS objects.
class MFile:
    """
    Dummy  MFile Class. Replaced by PROCESS import if PROCESS installed.
    """

    def __init__(self, filename):
        self.filename = filename


class InDat:
    """
    Dummy InDat Class. Replaced by PROCESS import if PROCESS installed.
    """

    def __init__(self, filename):
        self.filename = filename


def get_dicts():
    """
    Dummy get_dicts function. Replaced by PROCESS import if PROCESS installed.
    """
    pass


OBS_VARS = dict()
PROCESS_DICT = dict()

# Import PROCESS objects, override the above dummy objects if PROCESS installed.
# Note: noqa used to ignore "redefinition of unused variable" errors.
try:
    from process.io.mfile import MFile  # noqa: F811,F401
    from process.io.in_dat import InDat  # noqa: F811,F40
    from process.io.python_fortran_dicts import get_dicts  # noqa: F811
except (ModuleNotFoundError, FileNotFoundError):
    PROCESS_ENABLED = False
    bluemira_warn("PROCESS not installed on this machine; cannot run PROCESS.")

# Get dict of obsolete vars from PROCESS (if installed)
if PROCESS_ENABLED:
    try:
        from process.io.obsolete_vars import OBS_VARS
    except (ModuleNotFoundError, FileNotFoundError):
        OBS_VARS = dict()
        bluemira_warn(
            "The OBS_VAR dict is not installed in your PROCESS installed version"
        )
    # Load dicts from dicts JSON file
    PROCESS_DICT = get_dicts()

DEFAULT_INDAT = os.path.join(
    get_bluemira_root(), "bluemira", "codes", "process", "PROCESS_DEFAULT_IN.DAT"
)

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

BTOPUNITS = {val: key for key, val in PTOBUNITS.items()}


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


def _convert(dictionary, key):
    if key in dictionary.keys():
        return dictionary[key]
    return key


def _pconvert(dictionary, key):
    key_name = _convert(dictionary, key)
    if key_name is None:
        raise ValueError(f'Define a parameter conversion for "{key}"')
    return key_name


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


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
