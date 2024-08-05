# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS api
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.utilities.tools import flatten_iterable

if TYPE_CHECKING:
    from collections.abc import Iterable


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


OBS_VARS = {}
PROCESS_DICT = {}
imp_data = None  # placeholder for PROCESS module

try:
    from process.impurity_radiation import ImpurityDataHeader, read_impurity_file
    from process.io.in_dat import InDat  # noqa: F401
    from process.io.mfile import MFile  # noqa: F401
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


@dataclass
class _INVariable:
    """
    Process io.in_dat.INVariable replica

    Used to simulate what process does to input variables
    for the InDat input file writer

    This allows the defaults to imitate the same format as PROCESS'
    InDat even if PROCESS isn't installed.
    Therefore they will work the same in all cases and we dont always
    need to be able to read a PROCESS input file.
    """

    name: str
    _value: float | list | dict
    v_type: TypeVar("InVarValueType")
    parameter_group: str
    comment: str

    @property
    def get_value(self) -> float | list | dict:
        """Return value in correct format"""
        return self._value

    @property
    def value(self) -> str | list | dict:
        """
        Return the string of a value if not a Dict or a List
        """
        if not isinstance(self._value, list | dict):
            return f"{self._value}"
        return self._value

    @value.setter
    def value(self, new_value):
        """
        Value setter
        """
        self._value = new_value


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

    def files(self) -> dict[str, Path]:
        """
        Get PROCESS impurity data file path

        Raises
        ------
        CodesError
            Impurity directory not found
        """
        with resources.path(
            "process.data.lz_non_corona_14_elements", "Ar_lz_tau.dat"
        ) as dp:
            data_path = dp.parent

        try:
            return {
                i: Path(data_path, f"{self.name:_<3}{i}_tau.dat")
                for i in ("lz", "z", "z2")
            }
        except NameError:
            raise CodesError("PROCESS impurity data directory not found") from None

    def id(self):
        """
        Get variable string for impurity fraction
        """
        return f"fimp({self.value:02})"

    def read_impurity_files(
        self, filetype: Iterable[Literal["lz", "z2", "z"]]
    ) -> tuple[list[ImpurityDataHeader]]:
        """Get contents of impurity data files"""
        files = self.files()
        return tuple(
            read_impurity_file(files[file])
            for file in sorted(set(filetype).intersection(files), key=filetype.index)
        )


def update_obsolete_vars(process_map_name: str) -> str | list[str] | None:
    """
    Check if the bluemira variable is up to date using the OBS_VAR dict.
    If the PROCESS variable name has been updated in the installed version
    this function will provide the updated variable name.

    Parameters
    ----------
    process_map_name:
        PROCESS variable name.

    Returns
    -------
    PROCESS variable names valid for the install (if OBS_VAR is updated
    correctly). Returns a list if an obsolete variable has been
    split into more than one new variable (e.g., a thermal shield
    thickness is split into ib/ob thickness). Returns `None` if there
    is no alternative.
    """
    process_name = _nested_check(process_map_name)

    if process_name is not None and process_name != process_map_name:
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
        if process_name == "None":
            return None
        if isinstance(process_name, list):
            names = []
            for p in process_name:
                names += [_nested_check(p)]
            return list(flatten_iterable(names))
    return process_name
