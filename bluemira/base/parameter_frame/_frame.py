# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, fields
from operator import itemgetter
from typing import TYPE_CHECKING, Any, ClassVar, get_args, get_type_hints

import pint
from tabulate import tabulate

from bluemira.base.constants import units_compatible, ureg
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame._parameter import (
    ParamDictT,
    Parameter,
    ParameterValueType,
)
from bluemira.base.parameter_frame._units import _validate_units

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
                field._unit = ureg.Unit(val_unit["unit"])
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
        self, new_values: dict[str, ParameterValueType] | ParamDictT | ParameterFrame
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

            kwargs[member] = cls._member_data_to_parameter(member, param_data)

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
            except AttributeError:
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
            kwargs[member] = cls._member_data_to_parameter(member, lp[member])

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
            except KeyError as e:
                raise ValueError(f"Data for parameter '{member}' not found.") from e

        return cls(**kwargs)

    @classmethod
    def _member_data_to_parameter(
        cls, member: str, member_param_data: ParamDictT
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
            member_param_data = _validate_units(member_param_data, value_type)
        except pint.errors.PintError as pe:
            raise ValueError(
                f"Unit conversion failed for {member} from {member_param_data['unit']}"
            ) from pe
        return Parameter(name=member, **member_param_data, _value_types=value_type)

    def to_dict(self, *, use_last: bool = False) -> dict[str, dict[str, Any]]:
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
            param_data = getattr(self, param_name).to_dict(use_last=use_last)
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

        try:
            nindex = columns.index("name")
            rec_col.pop(nindex)
            records = sorted([
                [key, *[param.get(col, "N/A") for col in rec_col]]
                for key, param in self.to_dict().items()
            ])
        except ValueError:
            nindex = 0
            records = sorted([
                [*[param.get(col, "N/A") for col in rec_col]]
                for param in self.to_dict().values()
            ])

        try:
            vindex = rec_col.index("value")
            columns[columns.index("value")] = value_label
        except ValueError:
            vindex = None

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
            ["value"], floatfmt=floatfmt, value_label=next(names)
        )
        columns.extend([*column, *([""] * (len(record[0]) - len(column)))])
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
