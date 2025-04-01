# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, TypedDict

import numpy as np
import pint
from typeguard import config, typechecked

from bluemira.base.constants import raw_uc, units_compatible


def type_fail(exc, memo):  # noqa: ARG001
    """
    Raises
    ------
    TypeError
        Wrong type

    Notes
    -----
    typeguard by default raises a TypeCheckError
    may want to have a custom checker in future

    """
    raise TypeError(f"{exc._path[0]} {exc.args[0]}")


config.typecheck_fail_callback = type_fail

ParameterValueType = TypeVar("ParameterValueType")


class ParamDictT(TypedDict):
    """Typed dictionary for a Parameter."""

    name: str
    value: ParameterValueType
    unit: str
    source: str
    description: str
    long_name: str


@dataclass
class ParameterValue(Generic[ParameterValueType]):
    """Holds parameter value information."""

    value: ParameterValueType
    source: str


class Parameter(Generic[ParameterValueType]):
    """
    Represents a parameter with physical units.

    Parameters
    ----------
    name
        The name of the parameter.
    value
        The parameter's value.
    unit
        The parameter's unit.
    source
        The origin of the parameter's value.
    description
        A description of the parameter.
    long_name
        A longer name for the parameter.
    """

    @typechecked
    def __init__(
        self,
        name: str,
        value: ParameterValueType,
        unit: str = "",
        source: str = "",
        description: str = "",
        long_name: str = "",
        _value_types: tuple[type, ...] | None = None,
    ):
        value = self._type_check(name, value, _value_types)
        self._name = name
        self._value = value
        self._unit = pint.Unit(unit)
        self._source = source
        self._description = description
        self._long_name = long_name

        self._history: list[ParameterValue[ParameterValueType]] = []
        self._add_history_record()

    @staticmethod
    def _type_check(
        name: str, value: ParameterValueType, value_types: tuple[type, ...] | None
    ) -> ParameterValueType:
        if value_types and value is not None:
            if float in value_types and isinstance(value, int):
                value = float(value)
            elif (
                int in value_types
                and isinstance(value, float)
                and np.isclose(value, int(value), rtol=0)
            ):
                value = int(value)
            elif not isinstance(value, value_types):
                raise TypeError(
                    f'{name}: type of "value" must be one of {value_types}; '
                    f"got {type(value)} (value: {value}) instead"
                )
        return value

    def __repr__(self) -> str:
        """String repr of class instance.

        Returns
        -------
        :
            The string representation of the class instance.
        """
        return f"<{type(self).__name__}({self.name}={self.value} {self.unit})>"

    def __eq__(self, o: object, /) -> bool:
        """
        Check if this parameter is equal to another.

        Parameters are equal if their names and values (with matching
        units) are equal.

        Returns
        -------
        :
            True if the parameters are equal, False otherwise.
        """
        if not isinstance(o, Parameter):
            return NotImplemented
        try:
            o_value_with_correct_unit = raw_uc(o.value, o.unit, self.unit)
        except pint.DimensionalityError:
            # incompatible units
            return False
        return (self.name == o.name) and (self.value == o_value_with_correct_unit)

    def __hash__(self) -> int:
        return hash((self._name, self._description, self._long_name))

    def history(self) -> list[ParameterValue[ParameterValueType]]:
        """Return the history of this parameter's value.

        Returns
        -------
        :
            A list of ParameterValue objects, the history of this parameter's value.
        """
        return copy.deepcopy(self._history)

    @typechecked
    def set_value(self, new_value: ParameterValueType, source: str = ""):
        """Set the parameter's value and update the source."""
        self._value = new_value
        self._source = source
        self._add_history_record()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the parameter to a dictionary.

        Returns
        -------
        :
            A dictionary representation of the parameter.
        """
        out = {
            "name": self.name,
            "value": self.value,
            "unit": "dimensionless" if not self.unit else self.unit,
        }
        for field in ["source", "description", "long_name"]:
            if value := getattr(self, field):
                out[field] = value
        return out

    @property
    def name(self) -> str:
        """Return the name of the parameter."""
        return self._name

    @property
    def value(self) -> ParameterValueType:
        """Return the current value of the parameter."""
        return self._value

    @value.setter
    def value(self, new_value: ParameterValueType):
        self.set_value(new_value, source="")

    def value_as(self, unit: str | pint.Unit) -> ParameterValueType | None:
        """
        Return the current value in a given unit

        Parameters
        ----------
        unit:
            The unit to convert the value to

        Returns
        -------
        :
            The value in the new unit

        Raises
        ------
        ValueError
            Unit conversion failed
        TypeError
            If the wrong unit type is passed in

        Notes
        -----
        If the current value of the parameter is None the function checks for
        a valid unit conversion
        """
        try:
            return raw_uc(self.value, self.unit, unit)
        except pint.errors.PintError as pe:
            raise ValueError("Unit conversion failed") from pe
        except TypeError as te:
            if self.value is None:
                if units_compatible(self.unit, unit):
                    return None
                raise ValueError("Unit conversion failed") from te
            raise

    @property
    def unit(self) -> str:
        """Return the physical unit of the parameter."""
        return f"{self._unit:~P}"

    @property
    def source(self) -> str:
        """Return the source that last set the value of this parameter."""
        return self._source

    @property
    def long_name(self) -> str:
        """Return a long name for this parameter."""
        return self._long_name

    @property
    def description(self) -> str:
        """Return a description for the parameter."""
        return self._description

    def _add_history_record(self):
        history_entry = ParameterValue(self.value, self.source)
        self._history.append(history_entry)
