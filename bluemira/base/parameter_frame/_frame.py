# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import copy
import json
from contextlib import suppress
from dataclasses import dataclass, fields
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    get_args,
    get_type_hints,
)

import pint
from tabulate import tabulate

from bluemira.base.constants import (
    ANGLE_UNITS,
    base_unit_defaults,
    combined_unit_defaults,
    combined_unit_dimensions,
    raw_uc,
    units_compatible,
)
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame._parameter import (
    ParamDictT,
    Parameter,
    ParameterValueType,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import GenericAlias

    from bluemira.base.parameter_frame.typed import ParameterFrameLike, ParameterFrameT
    from bluemira.base.reactor_config import ConfigParams


@dataclass
class ParameterFrame:
    """
    A data class to hold a collection of `Parameter` objects.

    The class should be declared using the following form:

    .. code-block:: python

        @dataclass
        class AnotherFrame(ParameterFrame):
            param_1: Parameter[float]
            param_2: Parameter[int]

    """

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        """
        Prevent instantiation of this class.

        Returns
        -------
        ParameterFrame
            The new instance of the class

        Raises
        ------
        TypeError
            Initialising ParameterFrame directly
        """
        if cls == ParameterFrame:
            raise TypeError(
                "Cannot instantiate a ParameterFrame directly. It must be subclassed."
            )

        if cls != EmptyFrame and not cls.__dataclass_fields__:
            bluemira_warn(f"{cls} is empty, @dataclass is possibly missing")

        return super().__new__(cls)

    def __post_init__(self):
        """Get types from frame

        Raises
        ------
        TypeError
            Inconsistent Parameter name or wrong type
        """
        self._types = self._get_types()

        for field, field_name, value_type in zip(
            self, self.__dataclass_fields__, self._types.values(), strict=False
        ):
            if not isinstance(field, Parameter):
                raise TypeError(
                    f"ParameterFrame contains non-Parameter object '{field_name}:"
                    f" {type(field)}'"
                )
            if field_name != field.name:
                raise TypeError(
                    "ParameterFrame contains Parameter with incorrect name"
                    f" '{field.name}', defined as '{field_name}'"
                )
            vt = _validate_parameter_field(field, value_type)

            val_unit = {
                "value": Parameter._type_check(field.name, field.value, vt),
                "unit": field.unit,
            }
            _validate_units(val_unit, vt)
            if field.unit != val_unit["unit"]:
                field.set_value(val_unit["value"], "unit enforcement")
                field._unit = pint.Unit(val_unit["unit"])
            else:
                # ensure int-> float conversion
                field._value = val_unit["value"]

    @classmethod
    def _get_types(cls) -> dict[str, GenericAlias]:
        """Gets types for the frame even with annotations imported.

        Returns
        -------
        :
            The field name to type mapping of the frame
        """
        frame_type_hints = get_type_hints(cls)
        return {f.name: frame_type_hints[f.name] for f in fields(cls)}

    def __iter__(self) -> Iterator[Parameter]:
        """
        Iterate over this frame's parameters.

        The order is based on the order in which the parameters were
        declared.

        Yields
        ------
        :
            Each parameter in the frame
        """
        for field in fields(self):
            yield getattr(self, field.name)

    def update(
        self,
        new_values: dict[str, ParameterValueType] | ParamDictT | ParameterFrame,
    ):
        """Update the given frame"""
        if isinstance(new_values, ParameterFrame):
            self.update_from_frame(new_values)
        else:
            try:
                self.update_from_dict(new_values)
            except TypeError:
                self.update_values(new_values)

    def get_values(self, *names: str) -> tuple[ParameterValueType, ...]:
        """Get values of a set of Parameters.

        Parameters
        ----------
        names
            The names of the Parameters to get the values of.

        Returns
        -------
        :
            The values of the Parameters in the order they were requested.

        Raises
        ------
        AttributeError
            Unknown Parameter name
        """
        try:
            return tuple(getattr(self, n).value for n in names)
        except AttributeError as ae:
            raise AttributeError(
                f"Parameters {[n for n in names if not hasattr(self, n)]} not in"
                " ParameterFrame"
            ) from ae

    def update_values(self, new_values: dict[str, ParameterValueType], source: str = ""):
        """Update the given parameter values."""
        for key, value in new_values.items():
            param: Parameter = getattr(self, key)
            param.set_value(value, source)

    def update_from_dict(self, new_values: dict[str, ParamDictT], source: str = ""):
        """Update from a dictionary representation of a ``ParameterFrame``"""
        for key, value in new_values.items():
            if "name" in value:
                del value["name"]
            if source:
                value["source"] = source
            self._set_param(
                key,
                Parameter(
                    name=key,
                    value=value.pop("value"),
                    **value,
                    _value_types=_validate_parameter_field(key, self._types[key]),
                ),
            )

    def update_from_frame(self, frame: ParameterFrameT):
        """Update the frame with the values of another frame"""
        for o_param in frame:
            if hasattr(self, o_param.name):
                self._set_param(o_param.name, o_param)

    def _set_param(self, name: str, o_param: Parameter):
        """
        Sets the information from a Parameter to an existing Parameter in this frame.

        Raises
        ------
        ValueError
            if the units are mismatched
        """
        param = getattr(self, name)

        if not units_compatible(param.unit, o_param.unit):
            expected_unit_str = f"{param.unit}" if param.unit else "None"
            raise ValueError(
                f"Incompatible unit for parameter {name}.\n"
                f"Expected unit: {expected_unit_str}, Received: {o_param.unit}"
            )

        param.set_value(
            (
                o_param.value
                if not param.unit or o_param.value is None
                else o_param.value_as(param.unit)
            ),
            o_param.source,
        )
        if o_param.long_name:
            param._long_name = o_param.long_name
        if o_param.description:
            param._description = o_param.description

    @classmethod
    def from_dict(
        cls: type[ParameterFrameT],
        data: dict[str, ParamDictT],
        *,
        allow_unknown: bool = False,
    ) -> ParameterFrameT:
        """Initialise an instance from a dictionary.

        Returns
        -------
        :
            A new ParameterFrame instance

        Raises
        ------
        ValueError
            Parameter data not found or unknown parameter
        """
        data = copy.deepcopy(data)
        kwargs: dict[str, Parameter] = {}
        for member in cls.__dataclass_fields__:
            try:
                param_data = data.pop(member)
            except KeyError as e:
                if cls.__dataclass_fields__[member].type == ClassVar:
                    continue
                raise ValueError(f"Data for parameter '{member}' not found.") from e

            kwargs[member] = cls._member_data_to_parameter(
                member,
                param_data,
            )

        if not allow_unknown and len(data) > 0:
            raise ValueError(f"Unknown parameter(s) {str(list(data))[1:-1]} in dict.")
        return cls(**kwargs)

    @classmethod
    def from_frame(
        cls: type[ParameterFrameT], frame: ParameterFrameT
    ) -> ParameterFrameT:
        """Initialise an instance from another ParameterFrame.

        Returns
        -------
        :
            A new ParameterFrame instance

        Raises
        ------
        ValueError
            Cannot find Parameter in provided frame
        """
        kwargs = {}
        for field in cls.__dataclass_fields__:
            try:
                kwargs[field] = getattr(frame, field)
            except AttributeError:  # noqa: PERF203
                raise ValueError(
                    "Cannot create ParameterFrame from other. "
                    f"Other frame does not contain field '{field}'."
                ) from None
        return cls(**kwargs)

    @classmethod
    def from_json(
        cls: type[ParameterFrameT],
        json_in: str | json.SupportsRead,
        *,
        allow_unknown: bool = False,
    ) -> ParameterFrameT:
        """Initialise an instance from a JSON file, string, or reader.

        Returns
        -------
        :
            A new ParameterFrame instance

        Raises
        ------
        TypeError
            Cannot read json data
        """
        if hasattr(json_in, "read"):
            # load from file stream
            return cls.from_dict(json.load(json_in), allow_unknown=allow_unknown)
        if not isinstance(json_in, str):
            raise TypeError(f"Cannot read JSON from type '{type(json_in).__name__}'.")
        if not json_in.startswith("{"):
            # load from file
            with open(json_in) as f:
                return cls.from_dict(json.load(f), allow_unknown=allow_unknown)
        # load from a JSON string
        return cls.from_dict(json.loads(json_in), allow_unknown=allow_unknown)

    @classmethod
    def from_config_params(
        cls: type[ParameterFrameT], config_params: ConfigParams
    ) -> ParameterFrameT:
        """
        Initialise an instance from a
        :class:`~bluemira.base.reactor_config.ConfigParams` object.

        A ConfigParams objects holds a ParameterFrame of global_params
        and a dict of local_params. This function merges the two together
        to form a unified ParameterFrame.

        Parameters in global_params will overwrite those in
        local_params, when defined in both.
        All references to Parameters in global_params are maintained
        (i.e. there's no copying).

        Returns
        -------
        :
            A new ParameterFrame instance

        Raises
        ------
        ValueError
            Parameter data not found
        """
        kwargs = {}

        lp = config_params.local_params
        for member in cls.__dataclass_fields__:
            if member not in lp:
                continue
            kwargs[member] = cls._member_data_to_parameter(
                member,
                lp[member],
            )

        gp = config_params.global_params
        for member in cls.__dataclass_fields__:
            if member not in gp.__dataclass_fields__:
                continue
            kwargs[member] = getattr(gp, member)

        # now validate all dataclass_fields are in kwargs
        # (which could be super set)
        for member in cls.__dataclass_fields__:
            try:
                kwargs[member]
            except KeyError as e:  # noqa: PERF203
                raise ValueError(f"Data for parameter '{member}' not found.") from e

        return cls(**kwargs)

    @classmethod
    def _member_data_to_parameter(
        cls,
        member: str,
        member_param_data: ParamDictT,
    ) -> Parameter:
        """Convert a member's data to a Parameter object.

        Returns
        -------
        :
            The Parameter object

        Raises
        ------
        ValueError
            Unit conversion failed
        """
        value_type = _validate_parameter_field(member, cls._get_types()[member])
        try:
            _validate_units(member_param_data, value_type)
        except pint.errors.PintError as pe:
            raise ValueError("Unit conversion failed") from pe
        return Parameter(
            name=member,
            **member_param_data,
            _value_types=value_type,
        )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialise this ParameterFrame to a dictionary.

        Returns
        -------
        :
            The serialised data
        """
        out = {}
        for param_name in self.__dataclass_fields__:
            if self.__dataclass_fields__[param_name].type == ClassVar:
                continue
            param_data = getattr(self, param_name).to_dict()
            # We already have the name of the param, and use it as a
            # key. No need to repeat the name in the data, so pop it.
            param_data.pop("name")
            out[param_name] = param_data
        return out

    def tabulation_data(
        self,
        keys: list[str] | None = None,
        floatfmt: str = ".5g",
        value_label: str | None = "value",
    ) -> tuple[list[str], list[list[str]]]:
        """
        Create the tabulated data for use with tabulate.
        Useful for combining frames for comparison

        Parameters
        ----------
        keys:
            table column keys
        tablefmt:
            The format of the table (default="fancy_grid") - see
            https://github.com/astanin/python-tabulate#table-format
        floatfmt:
            Format floats to this precision
        value_label:
            The header title for the 'value' column

        Returns
        -------
        :
            The tabulated data as column headers and a list of rows
        """
        try:
            pkey = keys.index("Parameter")
            keys.pop(pkey)
            if "name" in keys:
                keys.pop(keys.index("name"))
            if "unit" in keys:
                keys.pop(keys.index("unit"))
            keys.insert(pkey, "unit")
            keys.insert(pkey, "name")
        except (ValueError, AttributeError):
            pkey = None

        columns = list(ParamDictT.__annotations__.keys()) if keys is None else keys
        rec_col = copy.deepcopy(columns)
        nindex = columns.index("name")
        rec_col.pop(nindex)

        try:
            vindex = rec_col.index("value")
            columns[columns.index("value")] = value_label
        except ValueError:
            vindex = None

        records = sorted([
            [key, *[param.get(col, "N/A") for col in rec_col]]
            for key, param in self.to_dict().items()
        ])
        if pkey is not None:
            columns.pop(pkey)
            columns.pop(pkey)
            columns.insert(pkey, "Parameter")

        for r in records:
            if vindex is not None and isinstance(r[vindex], float):
                # tabulate's floatfmt only works if the whole column is a float
                r[vindex] = f"{r[vindex]: {floatfmt}}"

            if pkey is not None:
                unit = r.pop(pkey + 1)
                r[0] += f" [{'' if unit == 'dimensionless' else unit}]"
            if nindex != 0:
                r.insert(nindex, r.pop(0))

        return columns, records

    def tabulate(
        self,
        keys: list[str] | None = None,
        tablefmt: str = "fancy_grid",
        floatfmt: str = ".5g",
        value_label: str | None = "value",
    ) -> str:
        """
        Tabulate the ParameterFrame

        Parameters
        ----------
        keys:
            table column keys
        tablefmt:
            The format of the table (default="fancy_grid") - see
            https://github.com/astanin/python-tabulate#table-format
        floatfmt:
            Format floats to this precision
        value_label:
            The header title for the 'value' column

        Returns
        -------
        :
            The tabulated data
        """
        column_widths = dict(
            zip(
                [*list(ParamDictT.__annotations__.keys()), "Parameter"],
                [20, None, 20, 20, 20, 20, 20],
                strict=False,
            )
        )

        columns, records = self.tabulation_data(keys, floatfmt, value_label)
        column_widths[value_label] = column_widths["value"]

        return tabulate(
            records,
            headers=columns,
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
            maxcolwidths=list(itemgetter(*columns)(column_widths)),
        )

    def __str__(self) -> str:
        """
        Pretty print ParameterFrame.

        Returns
        -------
        :
            The formatted ParameterFrame
        """
        return self.tabulate()


def _validate_parameter_field(field, member_type: type) -> tuple[type, ...]:
    if (member_type is not Parameter) and (
        not hasattr(member_type, "__origin__") or member_type.__origin__ is not Parameter
    ):
        raise TypeError(f"Field '{field}' does not have type Parameter.")
    return get_args(member_type)


def _validate_units(param_data: ParamDictT, value_type: Iterable[type]):
    try:
        quantity = pint.Quantity(param_data["value"], param_data["unit"])
    except ValueError:
        try:
            quantity = pint.Quantity(f"{param_data['value']}*{param_data['unit']}")
        except pint.errors.PintError as pe:
            if param_data["value"] is None:
                quantity = pint.Quantity(
                    1 if param_data["unit"] in {None, ""} else param_data["unit"]
                )
                param_data["source"] = f"{param_data.get('source', '')}\nMAD UNIT 🤯 😭:"
            else:
                raise ValueError("Unit conversion failed") from pe
        else:
            param_data["value"] = quantity.magnitude
        param_data["unit"] = quantity.units
    except KeyError as ke:
        raise ValueError("Parameters need a value and a unit") from ke
    except TypeError:
        if param_data["value"] is None:
            # dummy for None values
            quantity = pint.Quantity(
                1 if param_data["unit"] in {None, ""} else param_data["unit"]
            )
        elif isinstance(param_data["value"], bool | str):
            param_data["unit"] = "dimensionless"
            return
        else:
            raise

    if dimensionality := quantity.units.dimensionality:
        unit = _fix_combined_units(_remake_units(dimensionality))
    else:
        unit = quantity.units

    unit = _fix_weird_units(unit, quantity.units)

    if unit != quantity.units and param_data["value"] is not None:
        val = raw_uc(quantity.magnitude, quantity.units, unit)
        if isinstance(param_data["value"], int) and int in value_type:
            val = int(val)
        param_data["value"] = val

    param_data["unit"] = f"{unit:~P}"

    if "MAD UNIT" in param_data.get("source", ""):
        param_data["source"] += f"{quantity.magnitude}{param_data['unit']}"


def _remake_units(dimensionality: dict | pint.util.UnitsContainer) -> pint.Unit:
    """
    Reconstruct unit from its dimensionality.

    Parameters
    ----------
    dimensionality:
        The dimensionality of the unit

    Returns
    -------
    :
        The reconstructed unit
    """
    dim_list = list(map(base_unit_defaults.get, dimensionality.keys()))
    dim_pow = list(dimensionality.values())
    return pint.Unit(
        ".".join([f"{j[0]}^{j[1]}" for j in zip(dim_list, dim_pow, strict=False)])
    )


def _fix_combined_units(unit: pint.Unit) -> pint.Unit:
    """Converts base unit to a composite unit if they exist in the defaults.

    Parameters
    ----------
    unit:
        The unit to convert

    Returns
    -------
    :
        The converted unit
    """
    dim_keys = list(combined_unit_dimensions.keys())
    dim_val = list(combined_unit_dimensions.values())
    with suppress(ValueError):
        return pint.Unit(
            combined_unit_defaults[dim_keys[dim_val.index(unit.dimensionality)]]
        )
    return pint.Unit(unit)


def _convert_angle_units(
    modified_unit: pint.Unit, orig_unit_str: str, angle_unit: str
) -> pint.Unit:
    """
    Converts angle units to the base unit default for angles.

    Angles are dimensionless therefore dimensionality conversion
    from pint doesn't work. Conversions between angle units is also not
    very robust.

    Parameters
    ----------
    modified_unit
        reconstructed unit without the angle
    orig_unit_str
        the user supplied unit (without spaces)
    angle_unit
        the angle unit in `orig_unit`

    Returns
    -------
    :
        The converted unit
    """
    breaking_units = ["steradian", "square_degree"]
    new_angle_unit = base_unit_defaults["[angle]"]
    for b in breaking_units:
        if b in orig_unit_str:
            raise NotImplementedError(f"{breaking_units} not supported for conversion")
    if f"{angle_unit}**" in orig_unit_str:
        raise NotImplementedError("Exponent angles >1, <-1 are not supported")
    unit_list = orig_unit_str.split("/", 1)
    exp = "." if angle_unit in unit_list[0] else "/"
    modified_unit = "".join(str(modified_unit).split(angle_unit))
    return pint.Unit(f"{modified_unit}{exp}{new_angle_unit}")


def _fix_weird_units(modified_unit: pint.Unit, orig_unit: pint.Unit) -> pint.Unit:
    """
    Essentially a crude unit parser for when we have no dimensions
    or non-commutative dimensions.

    Full power years (dimension [time]) and displacements per atom (dimensionless)
    need to be readded to units as they will be removed by the dimensionality conversion.

    Angle units are dimensionless and conversions between them are not robust

    Returns
    -------
    :
        The fixed unit

    Raises
    ------
    ValueError
        Multiple angle units provided
    """
    unit_str = f"{orig_unit:C}"

    ang_unit = [ang for ang in ANGLE_UNITS if ang in unit_str]
    if len(ang_unit) > 1:
        raise ValueError(f"More than one angle unit not supported...🤯 {orig_unit}")
    ang_unit = ang_unit[0] if len(ang_unit) == 1 else None

    fpy = "full_power_year" in unit_str
    dpa = "displacements_per_atom" in unit_str

    if not (fpy or dpa) and ang_unit is None:
        return pint.Unit(modified_unit)

    new_unit = _non_comutative_unit_conversion(
        dict(modified_unit.dimensionality), unit_str.split("/")[0], dpa, fpy
    )

    # Deal with angles
    return _convert_angle_units(new_unit, unit_str, ang_unit) if ang_unit else new_unit


def _non_comutative_unit_conversion(dimensionality, numerator, dpa, fpy):
    """
    Full power years (dimension [time]) and displacements per atom (dimensionless)
    need to be readded to units as they will be removed by the dimensionality conversion.

    Full power years even though time based is not the same as straight 'time' and
    is therefore dealt with after other standard unit conversions.

    Only first order of both of these units is dealt with.

    Returns
    -------
    :
        The converted unit
    """
    dpa_str = (
        ("dpa." if "displacements_per_atom" in numerator else "dpa^-1.") if dpa else ""
    )
    if fpy:
        if "full_power_year" in numerator:
            dimensionality["[time]"] += -1
            fpy_str = "fpy."
        else:
            dimensionality["[time]"] += 1
            fpy_str = "fpy^-1."

        if dimensionality["[time]"] == 0:
            del dimensionality["[time]"]
    else:
        fpy_str = ""

    return pint.Unit(
        f"{dpa_str}{fpy_str}{_fix_combined_units(_remake_units(dimensionality))}"
    )


@dataclass
class EmptyFrame(ParameterFrame):
    """
    Class to represent an empty `ParameterFrame` (one with no Parameters).

    Can be used when initialising a
    :class:`~bluemira.base.reactor_config.ConfigParams` object with no global params.
    """

    def __init__(self) -> None:
        super().__init__()


def make_parameter_frame(
    params: ParameterFrameLike,
    param_cls: type[ParameterFrameT] | None,
    *,
    allow_unknown: bool = False,
) -> ParameterFrameT | None:
    """
    Factory function to generate a `ParameterFrame` of a specific type.

    Parameters
    ----------
    params:
        The parameters to initialise the class with.
        This parameter can be several types:

            * dict[str, ParamDictT]:
                A dict where the keys are parameter names, and the
                values are the data associated with the corresponding
                name.
            * ParameterFrame:
                A reference to the parameters on this frame will be
                assigned to the new ParameterFrame's parameters. Note
                that this makes no copies, so updates to parameters in
                the new frame will propagate to the old, and vice versa.
            * :class:`~bluemira.base.reactor_config.ConfigParams`:
                An object that holds a `global_params` ParameterFrame
                and a `local_params` dict, which are merged to create
                a new ParameterFrame. Values defined in `local_params`
                will be overwritten by those in `global_params` when
                defined in both.
            * str:
                The path to a JSON file, or, if the string starts with
                '{', a JSON string.
            * None:
                For the case where no parameters are actually required.
                This is intended for internal use, to aid in validation
                of parameters in `Builder`\\s and `Designer`\\s.

    param_cls:
        The `ParameterFrame` class to create a new instance of.
    allow_unknown:
        Dictionary and json input checks if unknown parameters are passed
        in. By default this will error unless this flag is set to true

    Returns
    -------
        A frame of the type `param_cls`, or `None` if `params` and
        `param_cls` are both `None`.

    Raises
    ------
    ValueError
        No params or param_cls provided
    TypeError
        Cannot interpret params type
    """
    from bluemira.base.reactor_config import ConfigParams  # noqa: PLC0415

    if param_cls is None:
        if params is None:
            # Case for where there are no parameters associated with the object
            return params
        raise ValueError("Cannot process parameters, 'param_cls' is None.")
    if isinstance(params, dict):
        return param_cls.from_dict(params, allow_unknown=allow_unknown)
    if isinstance(params, param_cls):
        return params
    if isinstance(params, str):
        return param_cls.from_json(params, allow_unknown=allow_unknown)
    if isinstance(params, ParameterFrame):
        return param_cls.from_frame(params)
    if isinstance(params, ConfigParams):
        return param_cls.from_config_params(params)
    raise TypeError(f"Cannot interpret type '{type(params)}' as {param_cls.__name__}.")


def tabulate_values_from_multiple_frames(
    frames: Iterable[ParameterFrameT],
    value_labels: Iterable[str],
    tablefmt: str = "fancy_grid",
    floatfmt: str = ".5g",
) -> str:
    """
    Tabulate the contents of parameter frames of the same type.

    Parameters
    ----------
    frames:
        ParameterFrames to compare
    value_labels:
        The header title for each 'value' column
    tablefmt:
        The format of the table (default="fancy_grid") - see
        https://github.com/astanin/python-tabulate#table-format
    floatfmt:
        Format floats to this precision

    Returns
    -------
    :
        The tabulated data

    Raises
    ------
    TypeError
        The ParameterFrames must all be the same type

    Notes
    -----
    This function creates a table with a single "Parameter" column and
    multiple value columns
    """
    names = iter(value_labels)
    columns, records = frames[0].tabulation_data(
        ["Parameter", "value"], floatfmt=floatfmt, value_label=next(names)
    )
    for frame in frames[1:]:
        if not isinstance(frame, type(frames[0])):
            raise TypeError("All ParameterFrames must be of the same type")
        column, record = frame.tabulation_data(
            ["value"], floatfmt=floatfmt, value_labe=next(names)
        )
        columns.append(column)
        for r_ind in range(len(records)):
            records[r_ind].extend(record[r_ind])

    return tabulate(
        records,
        headers=columns,
        tablefmt=tablefmt,
        showindex=False,
        numalign="right",
        maxcolwidths=[20, *(None for _ in columns[1:])],
    )
