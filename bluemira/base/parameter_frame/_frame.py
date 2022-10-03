from __future__ import annotations

import copy
import json
from contextlib import suppress
from dataclasses import dataclass, fields
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
)

import pint
from tabulate import tabulate

from bluemira.base.constants import raw_uc
from bluemira.base.parameter_frame._parameter import NewParameter as Parameter
from bluemira.base.parameter_frame._parameter import ParamDictT, ParameterValueType

_PfT = TypeVar("_PfT", bound="NewParameterFrame")


base_unit_defaults = {
    "[time]": "second",
    "[length]": "metre",
    "[mass]": "kilogram",
    "[current]": "ampere",
    "[temperature]": "kelvin",
    "[substance]": "mol",
    "[luminosity]": "candela",
    "[angle]": "degree",  # dimensionality == {}
}

combined_unit_defaults = {
    "[energy]": "joules",
    "[pressure]": "pascal",
    "[magnetic_field]": "tesla",
    "[electric_potential]": "volt",
    "[power]": "watt",
    "[force]": "newton",
    "[resistance]": "ohm",
}

combined_unit_dimensions = {
    "[energy]": {"[length]": 2, "[mass]": 1, "[time]": -2},
    "[pressure]": {"[length]": -1, "[mass]": 1, "[time]": -2},
    "[magnetic_field]": {"[current]": -1, "[mass]": 1, "[time]": -2},
    "[electric_potential]": {"[length]": 2, "[mass]": 1, "[time]": -2},
    "[power]": {"[length]": 2, "[mass]": 1, "[time]": -3},
    "[force]": {"[length]": 2, "[mass]": 1, "[time]": -3},
    "[resistance]": {"[current]": -2, "[length]": 2, "[mass]": 1, "[time]": -3},
}

ANGLE = [
    "radian",
    "turn",
    "degree",
    "arcminute",
    "arcsecond",
    "milliarcsecond",
    "grade",
    "mil",
]

# day
# gigajoule
# kiloelectron_volt
# megaampere
# megaampere * meter / megawatt
# megaampere / meter ** 2
# meganewton
# megawatt
# megawatt / meter ** 2
# meter
# meter ** 3
# meter ** 3 * pascal / second
# ohm
# pascal
# second
# tesla
# unified_atomic_mass_unit
# volt
# watt


# degree
# displacements_per_atom
# full_power_year


@dataclass
class NewParameterFrame:
    """
    A data class to hold a collection of `Parameter` objects.

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

    def __iter__(self) -> Generator[Parameter, None, None]:
        """
        Iterate over this frame's parameters.

        The order is based on the order in which the parameters were
        declared.
        """
        for field in fields(self):
            yield getattr(self, field.name)

    def update_values(self, new_values: Dict[str, ParameterValueType], source: str = ""):
        """Update the given parameter values."""
        for key, value in new_values.items():
            param: Parameter = getattr(self, key)
            param.set_value(value, source)

    @classmethod
    def from_dict(
        cls: Type[_PfT],
        data: Dict[str, ParamDictT],
        allow_unknown=False,
    ) -> _PfT:
        """Initialize an instance from a dictionary."""
        data = copy.deepcopy(data)
        kwargs: Dict[str, Parameter] = {}
        for member in cls.__dataclass_fields__:
            try:
                param_data = data.pop(member)
            except KeyError as e:
                raise ValueError(f"Data for parameter '{member}' not found.") from e

            value_type = cls._validate_parameter_field(member)
            cls._validate_units(param_data, value_type)
            kwargs[member] = Parameter(
                name=member, **param_data, _value_types=value_type
            )

        if not allow_unknown and len(data) > 0:
            raise ValueError(f"Unknown parameter(s) {str(list(data))[1:-1]} in dict.")
        return cls(**kwargs)

    @classmethod
    def from_frame(cls: Type[_PfT], frame: NewParameterFrame) -> _PfT:
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
    def from_json(cls: Type[_PfT], json_in: Union[str, json.SupportsRead]) -> _PfT:
        """Initialise an instance from a JSON file, string, or reader."""
        if hasattr(json_in, "read"):
            # load from file stream
            return cls.from_dict(json.load(json_in))
        elif not isinstance(json_in, str):
            raise ValueError(f"Cannot read JSON from type '{type(json_in).__name__}'.")
        elif not json_in.startswith("{"):
            # load from file
            with open(json_in, "r") as f:
                return cls.from_dict(json.load(f))
        # load from a JSON string
        return cls.from_dict(json.loads(json_in))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize this NewParameterFrame to a dictionary."""
        out = {}
        for param_name in self.__dataclass_fields__:
            param_data = getattr(self, param_name).to_dict()
            # We already have the name of the param, and use it as a
            # key. No need to repeat the name in the data, so pop it.
            param_data.pop("name")
            out[param_name] = param_data
        return out

    @classmethod
    def _validate_parameter_field(cls, field: str) -> Tuple[Type, ...]:
        member_type = cls.__dataclass_fields__[field].type
        if (member_type is not Parameter) and (
            not hasattr(member_type, "__origin__")
            or member_type.__origin__ is not Parameter
        ):
            raise TypeError(f"Field '{field}' does not have type Parameter.")
        value_types = get_args(member_type)
        return value_types

    @classmethod
    def _validate_units(cls, param_data: Dict, value_type):

        quantity = pint.Quantity(param_data["value"], param_data["unit"])

        if dimensionality := quantity.units.dimensionality:
            dim_list = list(map(base_unit_defaults.get, dimensionality.keys()))
            dim_pow = list(dimensionality.values())
            unit = pint.Unit(
                ".".join([f"{j[0]}^{j[1]}" for j in zip(dim_list, dim_pow)])
            )
            unit = cls._fix_combined_units(unit)
        else:
            unit = quantity.units

        unit = cls._fix_weird_units(unit, quantity.units)

        if unit != quantity.units:
            val = raw_uc(quantity.magnitude, quantity.units, unit)
            if isinstance(param_data["value"], int) and int in value_type:
                val = int(val)
            param_data["value"] = val
        param_data["unit"] = unit

    @classmethod
    def _fix_weird_units(
        cls, modified_unit: pint.Unit, orig_unit: pint.Unit
    ) -> pint.Unit:
        unit_str = f"{orig_unit:D}"

        for ang in ANGLE:
            if ang in unit_str:
                ang_unit = ang
                break
        else:
            ang_unit = None

        fpy = "full_power_year" in unit_str
        dpa = "displacements_per_atom" in unit_str

        if not (fpy or dpa) and ang_unit is None:
            return pint.Unit(modified_unit)

        if modified_unit == pint.Unit("dimensionless"):
            if dpa and not ang_unit:
                return pint.Unit(
                    f"displacements_per_atom^{-1 if len(unit_str.split('/')) > 1 else 1}"
                )
            else:
                return cls._fix_angle_units(
                    modified_unit, orig_unit, base_unit_defaults["[angle]"]
                )
        elif fpy:
            if not modified_unit.dimensionality.keys() - ["[time]"]:
                expon = list(modified_unit.dimensionality.values())[0]
                if dpa:
                    dpa_expon = -1 if len(unit_str.split("/")) > 1 and expon == 1 else 1
                    return pint.Unit(f"dpa^{dpa_expon}.fpy^{expon}")
                return pint.Unit(f"fpy^{expon}")
            elif dpa:
                # More complex
                raise NotImplementedError()
            else:
                # More complex
                raise NotImplementedError()
        else:
            # More complex
            raise NotImplementedError()
            return modified_unit
            # import ipdb
            # ipdb.set_trace()
            # thing/angle
            # thing/angle **2
            # angle ** 2 / thing
            # angle / thing ** 2

    @staticmethod
    def _fix_combined_units(unit: pint.Unit) -> pint.Unit:
        dim_keys = list(combined_unit_dimensions.keys())
        dim_val = list(combined_unit_dimensions.values())
        with suppress(ValueError):
            return pint.Unit(
                combined_unit_defaults[dim_keys[dim_val.index(unit.dimensionality)]]
            )
        return pint.Unit(unit)

    @staticmethod
    def _fix_angle_units(modified_unit, orig_unit, default_unit):
        raise NotImplementedError()

    def tabulate(self, keys: Optional[List] = None, tablefmt: str = "fancy_grid") -> str:
        """
        Tabulate the ParameterFrame

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
            The tabulated DataFrame
        """
        columns = list(ParamDictT.__annotations__.keys()) if keys is None else keys
        rec_col = copy.deepcopy(columns)
        rec_col.pop(columns.index("name"))
        records = sorted(
            [
                [key, *[param.get(col, "N/A") for col in rec_col]]
                for key, param in self.to_dict().items()
            ]
        )
        return tabulate(
            records,
            headers=columns,
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
        )

    def __str__(self) -> str:
        """
        Pretty print ParameterFrame
        """
        return self.tabulate()


def make_parameter_frame(
    params: Union[Dict[str, ParamDictT], NewParameterFrame, str, None],
    param_cls: Type[_PfT],
) -> Union[_PfT, None]:
    """
    Factory function to generate a `ParameterFrame` of a specific type.

    Parameters
    ----------
    params: Union[Dict[str, ParamDictT], NewParameterFrame, str, None]
        The parameters to initialise the class with.
        This parameter can be several types:

            * Dict[str, ParamDictT]:
                A dict where the keys are parameter names, and the
                values are the data associated with the corresponding
                name.
            * ParameterFrame:
                A reference to the parameters on this frame will be
                assigned to the new ParameterFrame's parameters. Note
                that this makes no copies, so updates to parameters in
                the new frame will propagate to the old, and vice versa.
            * str:
                The path to a JSON file, or, if the string starts with
                '{', a JSON string.
            * None:
                For the case where no parameters are actually required.
                This is intended for internal use, to aid in validation
                of parameters in `Builder`\\s and `Designer`\\s.

    param_cls: Type[NewParameterFrame]
        The `ParameterFrame` class to create a new instance of.

    Returns
    -------
    Union[ParameterFrame, None]
        A frame of the type `param_cls`, or `None` if `params` and
        `param_cls` are both `None`.
    """
    if param_cls is None:
        if params is None:
            # Case for where there are no parameters associated with the object
            return params
        raise ValueError("Cannot process parameters, 'param_cls' is None.")
    elif isinstance(params, dict):
        return param_cls.from_dict(params)
    elif isinstance(params, param_cls):
        return params
    elif isinstance(params, str):
        return param_cls.from_json(params)
    elif isinstance(params, NewParameterFrame):
        return param_cls.from_frame(params)
    raise TypeError(f"Cannot interpret type '{type(params)}' as {param_cls.__name__}.")
