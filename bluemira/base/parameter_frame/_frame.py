from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Type, Union, get_args

from bluemira.base.parameter_frame._parameter import NewParameter, ParameterValueType


@dataclass
class NewParameterFrame:
    """
    A data class to hold a collection of `NewParameter` objects.

    The class should be declared using on of the following forms:

    .. code-block:: python

        @parameter_frame
        class MyFrame:
            param_1: Parameter[float]
            param_2: Parameter[int]


        @dataclass
        class AnotherFrame(NewParameterFrame):
            param_1: Parameter[float]
            param_2: Parameter[int]

    """

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Mapping[str, Union[str, ParameterValueType]]],
        allow_unknown=False,
    ):
        """Initialize an instance from a dictionary."""
        data = copy.deepcopy(data)
        kwargs: Dict[str, NewParameter] = {}
        for member in cls.__dataclass_fields__:
            try:
                param_data = data.pop(member)
            except KeyError:
                raise ValueError(f"Data for parameter '{member}' not found.")

            value_type = cls._validate_parameter_field(member)
            kwargs[member] = NewParameter(**param_data, _value_types=value_type)

        if not allow_unknown and len(data) > 0:
            raise ValueError(f"Unknown parameter(s) '{list(data)}' in dict.")
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize this NewParameterFrame to a dictionary."""
        out = {}
        for member in self.__dataclass_fields__:
            out[member] = getattr(self, member).to_dict()
        return out

    @classmethod
    def _validate_parameter_field(cls, field: str) -> Tuple[Type]:
        member_type = cls.__dataclass_fields__[field].type
        if (member_type is not NewParameter) and (
            not hasattr(member_type, "__origin__")
            or member_type.__origin__ is not NewParameter
        ):
            raise TypeError(f"Field '{field}' does not have type NewParameter.")
        value_types = get_args(member_type)
        return value_types

    @classmethod
    def from_frame(cls, frame: NewParameterFrame) -> NewParameterFrame:
        """Initialise an instance from another NewParameterFrame."""
        kwargs = {}
        for field in cls.__dataclass_fields__:
            try:
                kwargs[field] = getattr(frame, field)
            except AttributeError:
                raise ValueError(
                    f"Cannot create NewParameterFrame from other. "
                    f"Other frame does not contain field '{field}'."
                )
        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_in: Union[str, json.SupportsRead]) -> NewParameterFrame:
        """Initialise an instance from a JSON file, string, or reader."""
        if hasattr(json_in, "read"):
            # load from file stream
            return cls.from_dict(json.load(json_in))
        elif not json_in.startswith("{"):
            # load from file
            with open(json_in, "r") as f:
                return cls.from_dict(json.load(f))
        # load from a JSON string
        return cls.from_dict(json.loads(json_in))
