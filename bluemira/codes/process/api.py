# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

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


OBS_VARS = {}
PROCESS_DICT = {}
imp_data = None  # placeholder for PROCESS module

try:
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
    _value: Union[float, List, Dict]
    v_type: TypeVar("InVarValueType")
    parameter_group: str
    comment: str

    @property
    def get_value(self) -> Union[float, List, Dict]:
        """Return value in correct format"""
        return self._value

    @property
    def value(self) -> Union[str, List, Dict]:
        """
        Return the string of a value if not a Dict or a List
        """
        if not isinstance(self._value, (List, Dict)):
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

    def file(self):
        """
        Get PROCESS impurity data file path
        """
        # TODO(je-cook) fixme process data has moved/ been removed
        with resources.path(
            "process.data.lz_non_corona_14_elements", "Ar_lz_tau.dat"
        ) as dp:
            data_path = dp.parent

        try:
            return Path(data_path, f"{self.name:_<2}lz_tau.dat")
        except NameError:
            raise CodesError("PROCESS impurity data directory not found") from None

    def id(self):  # noqa: A003
        """
        Get variable string for impurity fraction
        """
        return f"fimp({self.value:02}"


def update_obsolete_vars(process_map_name: str) -> Union[str, List[str], None]:
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


class PROCESSTemplateBuilder:
    """
    An API patch to make PROCESS a little easier to work with before
    the PROCESS team write a Python API.
    """

    def __init__(self):
        self.values: Dict[str, float] = {}
        self.bounds: Dict[str, Dict[str, str]] = {}
        self.icc: List[int] = []
        self.ixc: List[int] = []
        self.minmax: int = 0
        self.ioptimiz: int = 1

    def set_minimisation_objective(self, name: str):
        """
        Set the minimisation objective equation to use when running PROCESS
        """
        minmax = OBJECTIVE_EQ_MAPPING.get(name, None)
        if not minmax:
            raise ValueError(f"There is no objective equation: '{name}'")

        self.minmax = minmax

    def set_maximisation_objective(self, name: str):
        """
        Set the maximisation objective equation to use when running PROCESS
        """
        minmax = OBJECTIVE_EQ_MAPPING.get(name, None)
        if not minmax:
            raise ValueError(f"There is no objective equation: '{name}'")
        if minmax in OBJECTIVE_MIN_ONLY:
            raise ValueError(
                f"Equation {name} can only be used as a minimisation objective."
            )
        self.minmax = -minmax

    def add_constraint(self, name: str):
        """
        Add a constraint to the PROCESS run
        """
        constraint = CONSTRAINT_EQ_MAPPING.get(name, None)
        if not constraint:
            raise ValueError(f"There is no constraint equation: '{name}'")
        if constraint in self.icc:
            bluemira_warn(f"Constraint {name} is already in the constraint list.")

        if constraint in FV_CONSTRAINT_ITVAR_MAPPING.keys():
            # Sensible (?) defaults. bounds are standard PROCESS for f-values for _most_
            # f-value constraints.
            self.add_fvalue_constraint(name, 0.5, 1e-3, 1.0)
        else:
            self.icc.append(constraint)

    def add_fvalue_constraint(
        self,
        name: str,
        value: float,
        lower_bound: float = 1e-3,
        upper_bound: float = 1.0,
    ):
        """
        Add an f-value constraint to the PROCESS run
        """
        constraint = CONSTRAINT_EQ_MAPPING.get(name, None)
        if not constraint:
            raise ValueError(f"There is no constraint equation: '{name}'")

        if constraint not in FV_CONSTRAINT_ITVAR_MAPPING.keys():
            raise ValueError(f"Constraint '{name}' is not an f-value constraint.")

        itvar = FV_CONSTRAINT_ITVAR_MAPPING[constraint]
        if itvar not in self.ixc:
            self.add_variable(
                VAR_ITERATION_MAPPING[itvar], value, lower_bound, upper_bound
            )

    def add_variable(
        self,
        name: str,
        value: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        """
        Add an iteration variable to the PROCESS run
        """
        itvar = ITERATION_VAR_MAPPING.get(name, None)
        if not itvar:
            raise ValueError(f"There is no iteration variable: '{name}'")

        if itvar in self.ixc:
            bluemira_warn(
                "Iterable variable {name} is already in the variable list. Updating value and bounds."
            )
            self.values[name] = value
            if lower_bound:
                self.bounds[str(itvar)]["l"] = lower_bound
            if upper_bound:
                self.bounds[str(itvar)]["u"] = upper_bound

        else:
            self.ixc.append(itvar)
            self.values[name] = value

        if lower_bound or upper_bound:
            var_bounds = {}
            if lower_bound:
                var_bounds["l"] = lower_bound
            if upper_bound:
                var_bounds["u"] = upper_bound
            self.bounds[str(itvar)] = var_bounds

    def make_inputs(self) -> Dict[str, _INVariable]:
        """
        Make the ProcessInputs InVariable for the specified template
        """
        return ProcessInputs(
            bounds=self.bounds,
            icc=self.icc,
            ixc=self.ixc,
            minmax=self.minmax,
            **self.values,
        ).to_invariable()
