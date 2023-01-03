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
Optimisation variable class.
"""

from __future__ import annotations

import json
from operator import attrgetter
from typing import Dict, List, Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.palettes import BLUEMIRA_PALETTE
from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.tools import json_writer


def normalise_value(value, lower_bound, upper_bound):
    """
    Normalise a value uniformly [0 -> 1].

    Parameters
    ----------
    value: float
        Value to normalise
    lower_bound: float
        Lower bound at which to normalise
    upper_bound: float
        Upper bound at which to normalise

    Returns
    -------
    v_norm: float
        Normalised value [0 -> 1]
    """
    return (value - lower_bound) / (upper_bound - lower_bound)


def denormalise_value(v_norm, lower_bound, upper_bound):
    """
    Denormalise a value uniformly from [0 -> 1] w.r.t bounds.

    Parameters
    ----------
    v_norm: float
        Normalised value
    lower_bound: float
        Lower bound at which to denormalise
    upper_bound: float
        Upper bound at which to denormalise

    Returns
    -------
    value: float
        Denormalised value w.r.t. bounds
    """
    return lower_bound + v_norm * (upper_bound - lower_bound)


class BoundedVariableEncoder(json.JSONEncoder):
    """
    A JSONEncoder for serialising BoundedVariable instances.
    """

    def default(self, obj):
        """
        Serialises a Bounded Variable by extracting the value, lower_bound, upper_bound,
        and fixed properties.
        """
        if isinstance(obj, BoundedVariable):
            return {
                "value": obj.value,
                "lower_bound": obj.lower_bound,
                "upper_bound": obj.upper_bound,
                "fixed": obj.fixed,
            }
        return super().default(obj)


class BoundedVariable:
    """
    A bounded variable, uniformly normalised from 0 to 1 w.r.t. its bounds.

    Parameters
    ----------
    name: str
        Name of the variable
    value: float
        Value of the variable
    lower_bound: float
        Lower bound of the variable
    upper_bound: float
        Upper bound of the variable
    fixed: bool
        Whether or not the variable is to be held constant
    descr: str
        Description of the variable
    """

    __slots__ = ("name", "_value", "lower_bound", "upper_bound", "fixed", "_description")

    def __init__(self, name, value, lower_bound, upper_bound, fixed=False, descr=None):
        self.name = name
        self._validate_bounds(lower_bound, upper_bound)
        self._validate_value(value, lower_bound, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._value = None
        self.fixed = False  # Required to set value initially
        self.value = value
        self.fixed = fixed
        self._description = descr

    @property
    def value(self):
        """
        The value of the variable.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Set the value of the variable, enforcing bounds.
        """
        if self.fixed:
            raise OptVariablesError("Cannot set the value of a fixed variable.")

        self._validate_value(value, self.lower_bound, self.upper_bound)
        self._value = value

    @property
    def description(self):
        """
        The description of the variable.
        """
        return self._description

    def fix(self, value: float):
        """
        Fix the variable at a specified value. Ignores bounds.

        Parameters
        ----------
        value: float
            Value at which to fix the variable.
        """
        self.fixed = True
        if value is not None:
            self._value = value

    def adjust(
        self, value=None, lower_bound=None, upper_bound=None, *, strict_bounds=True
    ):
        """
        Adjust the BoundedVariable.

        Parameters
        ----------
        name: str
            Name of the variable to adjust
        value: Optional[float]
            Value of the variable to set
        lower_bound: Optional[float]
            Value of the lower bound to set
        upper_bound: Optional[float]
            Value of the upper to set
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        if self.fixed:
            raise OptVariablesError("Cannot adjust a fixed variable.")

        if lower_bound is not None:
            self.lower_bound = lower_bound

        if upper_bound is not None:
            self.upper_bound = upper_bound

        self._validate_bounds(self.lower_bound, self.upper_bound)

        if value is not None:
            if not strict_bounds:
                self._adjust_bounds(value, self.lower_bound, self.upper_bound)

            self.value = value

    @property
    def normalised_value(self) -> float:
        """
        The normalised value of the variable.
        """
        return normalise_value(self.value, self.lower_bound, self.upper_bound)

    def _adjust_bounds(self, value, lower_bound, upper_bound):
        """
        Adjust the bounds to the value
        """
        if value < lower_bound:
            bluemira_warn(
                f"BoundedVariable '{self.name}': value was set to below its lower bound. Adjusting bound."
            )
            self.lower_bound = value

        if value > upper_bound:
            bluemira_warn(
                f"BoundedVariable '{self.name}': value was set to above its upper bound. Adjusting bound."
            )
            self.upper_bound = value

    def _validate_bounds(self, lower_bound, upper_bound):
        if lower_bound > upper_bound:
            raise OptVariablesError(
                f"BoundedVariable '{self.name}': lower bound is higher than upper bound."
            )

    def _validate_value(self, value, lower_bound, upper_bound):
        if not lower_bound <= value <= upper_bound:
            raise OptVariablesError(
                f"BoundedVariable '{self.name}': value {value} is out of bounds."
            )

    def __repr__(self) -> str:
        """
        Representation of Bounded variable
        """
        lower_bound, upper_bound, fixed = self.lower_bound, self.upper_bound, self.fixed
        return f"{self.__class__.__name__}({self.name}, {self.value}, {lower_bound=}, {upper_bound=}, {fixed=})"

    def __str__(self) -> str:
        """
        Pretty representation of Bounded variable
        """
        bound = (
            f" Bounds: ({self.lower_bound}, {self.upper_bound})"
            if not self.fixed
            else ""
        )
        descr = f' "{self.description}"' if self.description is not None else ""

        return f"{self.name} = {self.value}{bound}{descr}"


class OptVariables:
    """
    A set of ordered variables to facilitate optimisation using normalised values.

    Parameters
    ----------
    variables: List[BoundedVariable]
        Set of variables to use
    frozen: bool
        Whether or not the OptVariables set is to be frozen upon instantiation. This
        prevents any adding or removal of variables after instantiation.
    """

    def __init__(self, variables, frozen=False):
        self._var_dict = {v.name: v for v in variables}
        self.frozen = frozen

    def add_variable(self, variable):
        """
        Add a variable to the set.

        Parameters
        ----------
        variable: BoundedVariable
            Variable to add to the set.
        """
        if self.frozen:
            raise OptVariablesError(
                "This OptVariables instance is frozen, no variables can be added or removed."
            )
        if variable.name in self._var_dict:
            raise OptVariablesError(f"Variable {variable.name} already in OptVariables.")

        self._var_dict[variable.name] = variable

    def remove_variable(self, name):
        """
        Remove a variable from the set.

        Parameters
        ----------
        name: str
            Name of the variable to remove
        """
        if self.frozen:
            raise OptVariablesError(
                "This OptVariables instance is frozen, no variables can be added or removed."
            )
        self._check_presence(name)

        del self._var_dict[name]

    def adjust_variable(
        self,
        name,
        value=None,
        lower_bound=None,
        upper_bound=None,
        fixed=False,
        *,
        strict_bounds=True,
    ):
        """
        Adjust a variable in the set.

        Parameters
        ----------
        name: str
            Name of the variable to adjust
        value: Optional[float]
            Value of the variable to set
        lower_bound: Optional[float]
            Value of the lower bound to set
        upper_bound: Optional[float]
            Value of the upper to set
        fixed: bool
            Whether or not the variable is to be held constant
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        self._check_presence(name)

        if fixed:
            self._var_dict[name].adjust(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                strict_bounds=strict_bounds,
            )
            self.fix_variable(name, value)

        else:
            self._var_dict[name].adjust(
                value, lower_bound, upper_bound, strict_bounds=strict_bounds
            )

    def adjust_variables(self, var_dict=None, *, strict_bounds=True):
        """
        Adjust multiple variables in the set.

        Parameters
        ----------
        var_dict: Optional[dict]
            Dictionary with which to update the set, of the form
            {"var_name": {"value": v, "lower_bound": lb, "upper_bound": ub}, ...}
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        if var_dict:
            for k, v in var_dict.items():

                args = [
                    v.get("value", None),
                    v.get("lower_bound", None),
                    v.get("upper_bound", None),
                    v.get("fixed", None),
                ]
                if all([i is None for i in args]):
                    raise OptVariablesError(
                        "When adjusting variables in a OptVariables instance, the dictionary"
                        " must be of the form: {'var_name': {'value': v, 'lower_bound': lb, 'upper_bound': ub}, ...}"
                    )

                self.adjust_variable(k, *args, strict_bounds=strict_bounds)

    def fix_variable(self, name, value=None):
        """
        Fix a variable in the set, removing it from optimisation but preserving a
        constant value.

        Parameters
        ----------
        name: str
            Name of the variable to fix
        value: Optional[float]
            Value at which to fix the variable (will default to present value)
        """
        self._check_presence(name)

        self._var_dict[name].fix(value=value)

    def get_normalised_values(self):
        """
        Get the normalised values of all free variables.

        Returns
        -------
        x_norm: np.ndarray
            Array of normalised values
        """
        return np.array(
            [v.normalised_value for v in self._var_dict.values() if not v.fixed]
        )

    def set_values_from_norm(self, x_norm):
        """
        Set values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values
        """
        true_values = self.get_values_from_norm(x_norm)
        for name, value in zip(self._opt_vars, true_values):
            variable = self._var_dict[name]
            variable.value = value

    def get_values_from_norm(self, x_norm):
        """
        Get actual values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values

        Returns
        -------
        x_true: np.ndarray
            Array of actual values in units
        """
        if len(x_norm) != self.n_free_variables:
            raise OptVariablesError(
                f"Number of normalised variables {len(x_norm)} != {self.n_free_variables}."
            )

        true_values = []
        for name, v_norm in zip(self._opt_vars, x_norm):
            variable = self._var_dict[name]
            value = denormalise_value(v_norm, variable.lower_bound, variable.upper_bound)
            true_values.append(value)
        return true_values

    @property
    def names(self):
        """
        All variable names of the variable set.
        """
        return [v for v in self._var_dict.keys()]

    @property
    def values(self):
        """
        All un-normalised values of the variable set (including fixed variable values).
        """
        return np.array([v.value for v in self._var_dict.values()])

    @property
    def n_free_variables(self) -> int:
        """
        Number of free variables in the set.
        """
        return len(self._opt_vars)

    @property
    def _fixed_variable_indices(self) -> list:
        """
        Indices of fixed variables in the set.
        """
        indices = []
        for i, v in enumerate(self._var_dict.values()):
            if v.fixed:
                indices.append(i)
        return indices

    @property
    def _opt_vars(self):
        return [v.name for v in self._var_dict.values() if not v.fixed]

    def _check_presence(self, name):
        if name not in self._var_dict.keys():
            raise OptVariablesError(f"Variable {name} not in OptVariables instance.")

    def __getitem__(self, name) -> BoundedVariable:
        """
        Dictionary-like access to variables.

        Parameters
        ----------
        name: str
            Name of the variable to get

        Returns
        -------
        variable: BoundedVariable
            Variable with the name
        """
        self._check_presence(name)
        return self._var_dict[name]

    def __getattribute__(self, attr):
        """
        Attribute access for variable values

        Parameters
        ----------
        attr: str
            attribute to access

        Returns
        -------
        attribute value

        """
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            try:
                return self[attr].value
            except KeyError:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{attr}'"
                ) from None

    def as_dict(self) -> Dict:
        """
        Dictionary Representation of OptVariables
        """
        return {
            key: {
                k: v
                for k, v in zip(
                    BoundedVariable.__slots__,
                    attrgetter(*BoundedVariable.__slots__)(self._var_dict[key]),
                )
            }
            for key in self._var_dict.keys()
        }

    def tabulate(self, keys: Optional[List] = None, tablefmt: str = "fancy_grid") -> str:
        """
        Tabulate OptVariables

        Parameters
        ----------
        keys: Optional[List]
            table column keys
        tablefmt: str (default="fancy_grid")
            The format of the table - see
            https://github.com/astanin/python-tabulate#table-format

        Returns
        -------
        tabulated: str
            The tabulated data
        """
        columns = [
            "Name",
            "Value",
            "Lower Bound",
            "Upper Bound",
            "Fixed",
            "Description",
        ]
        if keys is not None:
            columns = keys

        records = sorted([tuple(val) for val in self.as_dict().values()])

        return tabulate(
            records,
            headers=columns,
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
        )

    def __str__(self) -> str:
        """
        Pretty prints a representation of the OptVariables inside the console
        """
        return self.tabulate()

    def __repr__(self) -> str:
        """
        Prints a representation of the OptVariables inside the console
        """
        return (
            f"{self.__class__.__name__}(\n    "
            + "\n    ".join([repr(var) for var in self._var_dict.values()])
            + "\n)"
        )

    def to_json(self, file: str, **kwargs):
        """
        Write the json representation of these OptVariables to a file.

        Parameters
        ----------
        file: str
            The path to the file.
        """
        cls = kwargs.pop("cls", None)
        if cls is None:
            cls = BoundedVariableEncoder
        json_writer(self._var_dict, file, cls=cls, **kwargs)

    @classmethod
    def from_json(cls, file: Union[str, TextIO], frozen=False) -> OptVariables:
        """
        Create an OptVariables instance from a json file.

        Parameters
        ----------
        file: Union[str, TextIO]
            The path to the file, or an open file handle that supports reading.
        """
        if isinstance(file, str):
            with open(file, "r") as fh:
                return cls.from_json(fh)

        var_dict: Dict = json.load(file)
        new_vars = [BoundedVariable(key, **val) for key, val in var_dict.items()]
        return cls(new_vars, frozen=frozen)

    def plot(self):
        """
        Plot the OptVariables.
        """
        _, ax = plt.subplots()
        left_labels = [
            f"{v.name}: {v.lower_bound:.2f} " for v in self._var_dict.values()
        ]
        right_labels = [f"{v.upper_bound:.2f}" for v in self._var_dict.values()]
        y_pos = np.arange(len(left_labels))

        x_norm = [
            v.normalised_value if not v.fixed else 0.5 for v in self._var_dict.values()
        ]
        colors = [
            BLUEMIRA_PALETTE["red"] if v.fixed else BLUEMIRA_PALETTE["blue"]
            for v in self._var_dict.values()
        ]

        values = [f"{v:.2f}" for v in self.values]
        ax2 = ax.twinx()
        ax.barh(y_pos, x_norm, color="w")
        ax2.barh(y_pos, x_norm, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(left_labels)
        ax.invert_yaxis()

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(right_labels)
        ax2.invert_yaxis()
        ax.set_xlim([-0.1, 1.1])
        ax.set_xlabel("$x_{norm}$")

        for xi, yi, vi in zip(x_norm, y_pos, values):
            ax.text(xi, yi, vi)
