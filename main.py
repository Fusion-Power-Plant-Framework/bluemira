"""
OptVariablesV2 class
"""
import json
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    TextIO,
    TypedDict,
    Union,
    get_type_hints,
)

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.palettes import BLUEMIRA_PALETTE
from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.tools import json_writer


class OptVarVarDictT(TypedDict, total=False):
    """Typed dictionary for an OptVariable from a var_dict."""

    value: float
    lower_bound: float
    upper_bound: float
    fixed: bool
    description: str


class OptVarDictT(TypedDict):
    """Typed dictionary for an OptVariable."""

    value: float
    lower_bound: float
    upper_bound: float
    fixed: bool
    description: str


class OptVariable:
    """
    A bounded variable, uniformly normalised from 0 to 1 w.r.t. its bounds.

    Parameters
    ----------
    name:
        Name of the variable
    value: float
        Value of the variable
    lower_bound:
        Lower bound of the variable
    upper_bound:
        Upper bound of the variable
    fixed:
        Whether or not the variable is to be held constant
    description:
        Description of the variable
    """

    __slots__ = ("name", "_value", "lower_bound", "upper_bound", "fixed", "description")

    def __init__(
        self,
        name: str,
        value: float,
        lower_bound: float,
        upper_bound: float,
        fixed: bool = False,
        description: Optional[str] = None,
    ):
        self.name = name

        self._value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fixed = fixed
        self.description = description

        self._validate_bounds()
        self._validate_value(value)

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

        self._validate_value(value)
        self._value = value

    @property
    def normalised_value(self) -> float:
        """
        The value uniformly normalised between 0 and 1 w.r.t. its bounds
        """
        return (self.value - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def from_normalised(self, norm: float) -> float:
        """
        The value from a normalised value between [0 -> 1], w.r.t its bounds
        """
        return self.lower_bound + norm * (self.upper_bound - self.lower_bound)

    def fix(self, value: Optional[float] = None):
        """
        Fix the variable at a specified value. Ignores bounds.

        Parameters
        ----------
        value:
            Value at which to fix the variable.
        """
        self.fixed = True
        if value:
            self._value = value

    def adjust(
        self,
        value: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        strict_bounds: bool = True,
    ):
        """
        Adjust the OptVariable.

        Parameters
        ----------
        value:
            Value of the variable to set
        lower_bound:
            Value of the lower bound to set
        upper_bound:
            Value of the upper to set
        strict_bounds:
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        if self.fixed:
            raise OptVariablesError(f"'{self.name}' is fixed and cannot be adjusted.")

        if lower_bound:
            self.lower_bound = lower_bound

        if upper_bound:
            self.upper_bound = upper_bound

        if value:
            if not strict_bounds:
                self._adjust_bounds_to(value)
            self.value = value

        self._validate_bounds()

    def as_dict(self, with_name: bool = True) -> Dict[str, Any]:
        """Dictionary representation of OptVariable"""
        n = {"name": self.name}
        d = {
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "fixed": self.fixed,
            "description": self.description,
        }
        if with_name:
            return {**n, **d}
        return d

    @classmethod
    def from_dict(cls, name: str, var_dict: OptVarDictT):
        """Create an OptVariable from a dictionary"""
        return cls(name=name, **var_dict)

    def _adjust_bounds_to(self, value):
        """
        Adjust the bounds to the value
        """
        if value < self.lower_bound:
            bluemira_warn(
                f"OptVariable '{self.name}': value was set to below its lower bound. Adjusting bound."
            )
            self.lower_bound = value

        if value > self.upper_bound:
            bluemira_warn(
                f"OptVariable '{self.name}': value was set to above its upper bound. Adjusting bound."
            )
            self.upper_bound = value

    def _validate_bounds(self):
        if self.lower_bound > self.upper_bound:
            raise OptVariablesError(
                f"OptVariable '{self.name}': lower bound is higher than upper bound."
            )

    def _validate_value(self, value):
        if not self.lower_bound <= value <= self.upper_bound:
            raise OptVariablesError(
                f"OptVariable '{self.name}': value {value} is out of bounds."
            )

    def __repr__(self) -> str:
        """
        Representation of OptVariable
        """
        lower_bound, upper_bound, fixed = (
            self.lower_bound,
            self.upper_bound,
            self.fixed,
        )
        return f"{self.__class__.__name__}({self.name}, {self.value}, {lower_bound=}, {upper_bound=}, {fixed=})"

    def __str__(self) -> str:
        """
        Pretty representation of OptVariable
        """
        bound = (
            f" Bounds: ({self.lower_bound}, {self.upper_bound})"
            if not self.fixed
            else ""
        )
        descr = f' "{self.description}"' if self.description is not None else ""

        return f"{self.name} = {self.value}{bound}{descr}"


@dataclass
class OptVariablesFrame:
    """
    Class to model the variables for an optimisation
    """

    def __new__(cls, *args, **kwargs):
        """
        Prevent instantiation of this class.
        """
        if cls == OptVariablesFrame:
            raise TypeError(
                "Cannot instantiate an OptVariablesFrame directly. It must be subclassed."
            )
        field_types = get_type_hints(cls)
        for field_name in field_types:
            if field_types[field_name] != OptVariable:
                raise TypeError(
                    f"OptVariablesFrame contains non-OptVariable object '{field_name}: {field_types[field_name]}'"
                )
        return super().__new__(cls)

    def __iter__(self) -> Generator[OptVariable, None, None]:
        """
        Iterate over this frame's parameters.

        The order is based on the order in which the parameters were
        declared.
        """
        field_types = get_type_hints(self)
        for field_name in field_types:
            yield getattr(self, field_name)

    def __getitem__(self, name: str) -> OptVariable:
        """
        Dictionary-like access to variables.

        Parameters
        ----------
        name:
            Name of the variable to get
        """
        return getattr(self, name)

    def adjust_variable(
        self,
        name: str,
        value: Optional[float] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        fixed: bool = False,
        strict_bounds: bool = True,
    ):
        """
        Adjust a variable in the frame.

        Parameters
        ----------
        name:
            Name of the variable to adjust
        value:
            Value of the variable to set
        lower_bound:
            Value of the lower bound to set
        upper_bound:
            Value of the upper to set
        fixed:
            Whether or not the variable is to be held constant
        strict_bounds: bool
            If True, will raise errors if values are outside the bounds. If False, the
            bounds are dynamically adjusted to match the value.
        """
        # todo: once a var is fixed, it cannot be unfixed is that what we want?
        opt_var = self[name]
        if fixed:
            # sets the bounds and fixes the var ignoring them
            opt_var.adjust(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                strict_bounds=strict_bounds,
            )
            opt_var.fix(value)
        else:
            opt_var.adjust(value, lower_bound, upper_bound, strict_bounds)

    def adjust_variables(
        self, var_dict: Optional[Dict[str, OptVarVarDictT]] = None, strict_bounds=True
    ):
        """
        Adjust multiple variables in the frame.

        Parameters
        ----------
        var_dict:
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

    def fix_variable(self, name: str, value: Optional[float] = None):
        """
        Fix a variable in the frame, removing it from optimisation but preserving a
        constant value.

        Parameters
        ----------
        name:
            Name of the variable to fix
        value:
            Value at which to fix the variable (will default to present value)
        """
        self[name].fix(value)

    def get_normalised_values(self):
        """
        Get the normalised values of all free variables.

        Returns
        -------
        x_norm: np.ndarray
            Array of normalised values
        """
        return np.array([opv.normalised_value for opv in self._opt_vars])

    def set_values_from_norm(self, x_norm):
        """
        Set values from a normalised vector.

        Parameters
        ----------
        x_norm: np.ndarray
            Array of normalised values
        """
        true_values = self.get_values_from_norm(x_norm)
        for opv, value in zip(self._opt_vars, true_values):
            opv.value = value

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
        return [
            opv.from_normalised(v_norm) for opv, v_norm in zip(self._opt_vars, x_norm)
        ]

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
        # todo: does this need to be an np.array?
        return np.array([v.value for v in self])

    @property
    def n_free_variables(self) -> int:
        """
        Number of free variables in the set.
        """
        return len(self._opt_vars)

    @property
    def _opt_vars(self):
        return [v for v in self if not v.fixed]

    @property
    def _fixed_vars(self):
        return [v for v in self if v.fixed]

    @property
    def _fixed_variable_indices(self) -> list:
        """
        Indices of fixed variables in the set.
        """
        return [i for i, _ in enumerate(self._fixed_vars)]

    def as_dict(self) -> Dict[str, OptVarDictT]:
        """
        Dictionary Representation of the frame
        """
        return {opv.name: opv.as_dict(with_name=False) for opv in self}

    @classmethod
    def from_dict(cls, data: Dict[str, OptVarDictT]):
        """"""
        opt_vars = {name: OptVariable.from_dict(name, val) for name, val in data.items()}
        return cls(**opt_vars)

    def to_json(self, file: str, **kwargs):
        """
        Save the OptVariables to a json file.

        Parameters
        ----------
        path: str
            Path to save the json file to.
        """
        json_writer(self.as_dict(), file, **kwargs)

    @classmethod
    def from_json(cls, file: Union[str, TextIO], frozen=False):
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

        d = json.load(file)
        return cls.from_dict(d)

    def tabulate(self, tablefmt: str = "fancy_grid") -> str:
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
        records = sorted([val.as_dict() for val in self], key=lambda x: x["name"])

        return tabulate(
            records,
            headers="keys",
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
        )

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
            + "\n    ".join([repr(var) for var in self])
            + "\n)"
        )


class TryFrame(OptVariablesFrame):
    b: OptVariable
    a: OptVariable = OptVariable("a", 1, 0, 10)


trying = TryFrame()
# for f in trying:
#     print(f)
# trying.adjust_variables({"a": {"value": 4}, "b": {"value": 4}})
# trying.to_json("test.json")
TryFrame.from_json("test.json")
print(repr(trying))
