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
from enum import Enum
from pathlib import Path

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.utilities.tools import flatten_iterable


# Create dummy PROCESS objects. Required for docs to build properly and
# for testing.
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


OBS_VARS = dict()
PROCESS_DICT = dict()
imp_data = None  # placeholder for PROCESS module

try:
    import process.data.impuritydata as imp_data  # noqa: F401, F811
    from process.io.in_dat import InDat  # noqa: F401, F811
    from process.io.mfile import MFile  # noqa: F401, F811
    from process.io.python_fortran_dicts import get_dicts

    ENABLED = True
except (ModuleNotFoundError, FileNotFoundError):
    bluemira_warn("PROCESS not installed on this machine; cannot run PROCESS.")
    ENABLED = False

# Get dict of obsolete vars from PROCESS (if installed)
if ENABLED:
    try:
        from process.io.obsolete_vars import OBS_VARS
    except (ModuleNotFoundError, FileNotFoundError):
        bluemira_warn(
            "The OBS_VAR dict is not installed in your PROCESS installed version"
        )

    PROCESS_DICT = get_dicts()


DEFAULT_INDAT = os.path.join(
    get_bluemira_path("codes/process"), "PROCESS_DEFAULT_IN.DAT"
)


class Impurities(Enum):
    """
    PROCESS impurities Enum
    """

    H = 1
    He = 2
    Be = 3
    C = 4
    N = 5
    O = 6  # noqa: E741
    Ne = 7
    Si = 8
    Ar = 9
    Fe = 10
    Ni = 11
    Kr = 12
    Xe = 13
    W = 14

    def file(self):
        """
        Get PROCESS impurity data file path
        """
        try:
            return Path(Path(imp_data.__file__).parent, f"{self.name:_<2}Lzdata.dat")
        except NameError:
            raise CodesError("PROCESS impurity data directory not found")

    def id(self):
        """
        Get variable string for impurity fraction
        """
        return f"fimp({self.value:02}"


def update_obsolete_vars(process_map_name: str) -> str:
    """
    Check if the bluemira variable is up to date using the OBS_VAR dict.
    If the PROCESS variable name has been updated in the installed version
    this function will provide the updated variable name.

    Parameters
    ----------
    process_map_name: str
        PROCESS variable name obtained from the bluemira mapping.

    Returns
    -------
    process_name: str
        PROCESS variable names valid for the install (if OBS_VAR is updated
        correctly)
    """
    process_name = _nested_check(process_map_name)

    if not process_name == process_map_name:
        bluemira_print(
            f"Obsolete {process_map_name} PROCESS mapping name."
            f"The current PROCESS name is {process_name}"
        )
    return process_name


def _nested_check(process_name):
    """
    Recursively checks for obsolete variable names
    """
    while process_name in OBS_VARS:
        process_name = OBS_VARS[process_name]
        if isinstance(process_name, list):
            names = []
            for p in process_name:
                names += [_nested_check(p)]
            return list(flatten_iterable(names))
    return process_name
