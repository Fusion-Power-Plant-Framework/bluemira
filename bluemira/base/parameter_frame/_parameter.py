from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, Type, TypedDict, TypeVar, Union

import pint
from typeguard import typechecked

from bluemira.base.constants import raw_uc

ParameterValueType = TypeVar("ParameterValueType")

base_unit_defaults = {
    "[time]": "second",
    "[length]": "metre",
    "[mass]": "kilogram",
    "[current]": "ampere",
    "[temperature]": "kelvin",
    "[substance]": "mol",
    "": "degree",  # dimensionality == {}
}

combined_unit_defaults = {
    "kg/m^3": {"[length]": -3, "[mass]": 1},
    "1/m^3": {"[length]": -3},
    "1/m^2/s": {"[length]": -2, "[time]": -1},
}


class ParamDictT(TypedDict, Generic[ParameterValueType]):
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


class NewParameter(Generic[ParameterValueType]):
    """
    Represents a parameter with physical units.

    Parameters
    ----------
    name: str
        The name of the parameter.
    value: ParameterValueType
        The parameter's value.
    unit: str
        The parameter's unit.
    source: str
        The origin of the parameter's value.
    description: str
        A description of the parameter.
    long_name: str
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
        _value_types: Tuple[Type, ...] = None,
    ):
        if _value_types:
            if float in _value_types and isinstance(value, int):
                value = float(value)
            elif not isinstance(value, _value_types):
                raise TypeError(
                    f'type of argument "value" must be one of {_value_types}; '
                    f"got {type(value)} instead."
                )
        self._name = name
        self._value, self._unit = self.check_unit(value, unit)
        self._source = source
        self._description = description
        self._long_name = long_name

        self._history: List[ParameterValue] = []
        self._add_history_record()

    def __repr__(self) -> str:
        """String repr of class instance."""
        return f"<{type(self).__name__}({self.name}={self.value} {self.unit})>"

    def __eq__(self, __o: object) -> bool:
        """
        Check if this parameter is equal to another.

        Parameters are equal if their names and values (with matching
        units) are equal.
        """
        if not isinstance(__o, NewParameter):
            return NotImplemented
        try:
            o_value_with_correct_unit = raw_uc(__o.value, __o.unit, self.unit)
        except pint.DimensionalityError:
            # incompatible units
            return False
        return (self.name == __o.name) and (self.value == o_value_with_correct_unit)

    def check_unit(
        self, value: ParameterValueType, unit: Union[str, pint.Unit]
    ) -> Tuple[ParameterValueType, pint.Unit]:
        quantity = pint.Quantity(value, unit)
        dimensionality = quantity.units.dimensionality
        dim_list = list(map(base_unit_defaults.get, dimensionality.keys()))
        dim_pow = list(dimensionality.values())
        if not dim_list:
            raise NotImplementedError("dimensionless units need work")
        else:
            unit = pint.Unit("".join([f"{j[0]}^{j[1]}" for j in zip(dim_list, dim_pow)]))

        if unit != quantity.units:
            value = raw_uc(quantity.magnitude, quantity.units, unit)
        return value, unit

    def history(self) -> List[ParameterValue]:
        """Return the history of this parameter's value."""
        return copy.deepcopy(self._history)

    @typechecked
    def set_value(self, new_value: ParameterValueType, source: str = ""):
        """Set the parameter's value and update the source."""
        self._value = new_value
        self._source = source
        self._add_history_record()

    def to_dict(self) -> Dict:
        """Serialize the parameter to a dictionary."""
        out = {"name": self.name, "value": self.value}
        for field in ["unit", "source", "description", "long_name"]:
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

    @property
    def unit(self) -> str:
        """Return the physical unit of the parameter."""
        return self._unit

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
