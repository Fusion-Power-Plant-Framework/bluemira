# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
The home of Parameter and ParameterFrame objects

These objects contain the definitions for the configuration of physical parameters in a
bluemira analysis.
"""
import copy
from dataclasses import dataclass
import json
import os
import gc
from pandas import DataFrame
from tabulate import tabulate
from typing import Any, Dict, List, Union
import wrapt
import numpy as np
from functools import wraps

from bluemira.base.error import ParameterError
from bluemira.base.look_and_feel import bluemira_warn


__all__ = ["Parameter", "ParameterFrame", "ParameterMapping"]


@dataclass
class ParameterMapping:
    """
    Simple class containing information on mapping of a bluemira parameter to one in
    external software.

    Parameters
    ----------
    name: str
       name of mapped parameter
    recv: bool
        recieve data from mapped parameter (to overwrite bluemira parameter)
    send: bool
        send data to mapped parameter (from bluemira parameter)

    """

    name: str
    recv: bool = True
    send: bool = True

    def to_dict(self):
        """
        Convert this object to a dictionary with attributes as values.
        """
        return {"name": self.name, "recv": self.recv, "send": self.send}

    def __str__(self):
        """
        Create a string representation of of this object which is more compact than that
        provided by the default `__repr__` method.
        """
        return repr(self.to_dict())


def inplace_wrapper(method):
    """
    Decorator to update history for inplace operators
    """

    @wraps(method)
    def wrapped(*args, **kwargs):
        ret = method(*args, **kwargs)
        args[0]._source = None
        args[0]._update_history()
        return ret

    return wrapped


def inplace_wrapt(cls):
    """
    Wrap wrapt.ObjectProxy's inplace operations.
    """
    for attrname in dir(cls):
        if (
            attrname.startswith("__i")
            and attrname.endswith("__")
            and attrname != "__init__"
        ):
            setattr(cls, attrname, inplace_wrapper(getattr(cls, attrname)))
    return cls


@inplace_wrapt
class Parameter(wrapt.ObjectProxy):
    """
    The Parameter base class.

    This provides meta-data for implementations of new parameter objects.

    Once a Parameter has been created it will act like the type of value.

    All operations you would normally do with for instance an 'int' will work the same
    """

    __slots__ = (
        "var",
        "name",
        "unit",
        "description",
        "_source",
        "mapping",
        "_value_history",
        "_source_history",
    )

    var: str
    name: Union[str, None]
    unit: Union[str, None]
    description: Union[str, None]
    _source: Union[str, None]
    mapping: Union[Dict[str, ParameterMapping], None]
    _value_history: Union[list, None]
    _source_history: Union[list, None]

    def __init__(
        self,
        var: str,
        name: Union[str, None] = None,
        value: Union[str, float, int, None] = None,
        unit: Union[str, None] = None,
        description: Union[str, None] = None,
        source: Union[str, bool] = None,
        mapping: Union[Dict[str, ParameterMapping], None] = None,
    ):
        """
        Parameters
        ----------
        var: str
            The parameter variable name, as referenced by the ParameterFrame.
        name: Union[str, None]
            The parameter name.
        value: Union[str, float, int, None]
            The value of the parameter, by default None.
        unit: Union[str, None]
            The unit of the parameter, by default None.
        description: Union[str, None]
            The description of the parameter.
        source: Union[str, None]
            The source (reference and/or code) that the parameter was obtained from,
            by default None.
        mapping: Union[Dict[str, ParameterMapping], None]
            The names used for this parameter in external software, and whether
            that parameter should be written to and/or read from the external tool,
            by default, None.
        """
        super().__init__(value)
        self.var = var
        self.name = name
        self.unit = unit
        self.description = description
        self.mapping = mapping

        self._source = source
        if value is not None:
            self._value_history = [value.copy() if hasattr(value, "copy") else value]
            self._source_history = [source]
        else:
            self._value_history = []
            self._source_history = []

    def __deepcopy__(self, memo):
        """
        Get a deep copy of the Parameter
        """
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        _dict = {k: getattr(self, k) for k in self.__slots__}
        for k, v in _dict.items():
            setattr(result, k, copy.deepcopy(v, memo))
        setattr(result, "__wrapped__", copy.deepcopy(self.__wrapped__, memo))
        try:
            setattr(result, "__qualname__", copy.deepcopy(self.__qualname__, memo))
        except AttributeError:
            pass
        return result

    def __repr__(self):
        """
        Get representation.
        """
        return self.__wrapped__.__repr__()

    def __reduce_ex__(self, protocol):
        """
        Make Parameter pickleable

        Parameters
        ----------
        protocol: int
            pickle protocol version

        """
        ty = type(self).from_json
        return ty, (
            self.var,
            self.name,
            self.__wrapped__,
            self.unit,
            self.description,
            self._source,
            self.mapping,
            self.value_history,
            self.source_history,
        )

    def __array__(self, dtype=None):
        """
        Allow usage in numpy arrays

        Parameters
        ----------
        dtype: type
            Set the type of the array contents

        Notes
        -----
        specifically fixes np.array([Parameter(...)])

        In order numpy calls
        __array_struct__
        __array_interface__
        __array__

        So one of the other two methods may be a better option to implement
        but this was the easiest

        """
        return np.array(self.value, dtype=dtype)

    @property
    def _history_keys(self):
        return ["_value_history", "_source_history"]

    @property
    def value_history(self):
        """
        Get value_history
        """
        return self._value_history

    @property
    def source_history(self):
        """
        Get source_history
        """
        return self._source_history

    @property
    def value(self):
        """
        Get value.
        """
        return self.__wrapped__

    @value.setter
    def value(self, val):
        """
        Set value as an alternative to the built in method.

        Useful for a single Parameter not part of a ParameterFrame.

        This will make source==None as the source should be updated
        immediately after the value of value is updated.

        Parameters
        ----------
        val:
           new value for value

        """
        self.__wrapped__ = val
        self._source = None

        if val is None and len(self.value_history) == 0:
            pass
        else:
            self._update_history()

    @property
    def source(self):
        """
        Get source.
        """
        return self._source

    @source.setter
    def source(self, val):
        """
        Set the origin of the Parameter.

        Parameters
        ----------
        val: str
            new value for source

        """
        self._source = val

        if self.source_history[-1] is None:
            self._source_history[-1] = self._source
        else:
            self._update_history()

    def _update_history(self):
        if (
            len(self.source_history) > 0 and self.source_history[-1] is None
        ):  # Should I be more strict and error out here?
            bluemira_warn(
                f"The source of the value of {self.var} not consistently known"
            )

        self._value_history += [
            self.__wrapped__.copy()
            if hasattr(self.__wrapped__, "copy")
            else self.__wrapped__
        ]
        self._source_history += [self._source]

    @classmethod
    def from_json(cls, *args):
        """
        Regenerate Parameter from json
        """
        if len(args) > 7:
            _dict = {"value_history": args[-2], "source_history": args[-1]}
            args = args[:-2]
        else:
            _dict = {}

        value_history = _dict.pop("value_history", None)
        source_history = _dict.pop("source_history", None)

        new_cls = cls(*args)

        if value_history is not None:
            new_cls._value_history = value_history
        if source_history is not None:
            new_cls._source_history = source_history
        return new_cls

    def _full_slots(self):
        """
        Return a list of slots including 'value'.

        Returns
        -------
        list
            list of variables

        """
        wrp = "__wrapped__"
        slots_copy = list(self.__slots__)
        slots_copy.insert(2, wrp)
        return slots_copy

    @staticmethod
    def _get_k(name):
        """
        Get the correct name of keys for reinitialisation of Parameter

        Parameters
        ----------
        name: str
            key name to check

        Returns
        -------
        str
            correct name for key
        """
        if name == "__wrapped__":
            return "value"
        elif name.startswith("_"):
            return name[1:]
        else:
            return name

    def to_dict(self, ignore_var=False, history=False) -> dict:
        """
        Get the Parameter as a dictionary, optionally ignoring the var value.

        Parameters
        ----------
        ignore_var: bool
            If True then does not include the var attribute in the dictionary,
            by default False.

        Returns
        -------
        the_dict: dict
            The dictionary representation of the Parameter.

        """
        ign = ["var"] + self._history_keys
        if not ignore_var:
            ign.pop(ign.index("var"))
        if history:
            [ign.pop(ign.index(i)) for i in self._history_keys]
        return {
            self._get_k(k): copy.deepcopy(getattr(self, k))
            for k in self._full_slots()
            if k not in ign
        }

    def to_list(self, history=False) -> list:
        """
        Get the Parameter as a list.

        Returns
        -------
        the_list: List[str]
            The field values as a list [var, name, value, unit, description, source].

        """
        hst_lst = [] if history else self._history_keys
        return [
            copy.deepcopy(getattr(self, k))
            for k in self._full_slots()
            if k not in hst_lst
        ]

    def history(self, string=False) -> Union[list, str]:
        """
        Collates the history of the parameter

        Parameters
        ----------
        string: bool
            return a string instead of a list

        Returns
        -------
        list or str
            history list in the specified form

        """
        hst = list(zip(self.value_history, self.source_history))
        if string:
            vs_frame = "{}   {}\n"
            hst_str = ""
            for v, s in hst:
                hst_str += vs_frame.format(v, s)
            return hst_str
        return hst

    def __str__(self) -> str:
        """
        Return a string representation of the Parameter

        Returns
        -------
        the_string: str
            The string representation of the Parameter.

        """
        mapping_str = (
            " {"
            + ", ".join([repr(k) + ": " + str(v) for k, v in self.mapping.items()])
            + "}"
            if self.mapping is not None
            else ""
        )
        return (
            f"{self.var}"
            f"{' = ' + str(self.value) if self.value is not None else ''}"
            f"{' ' + self.unit if self.unit is not None else ''}"
            f"{' (' + self.name + ')' if self.name is not None else ''}"
            f"{' : ' + self.description if self.description is not None else ''}"
            f"{mapping_str}"
        )


class ParameterFrame:
    """
    The ParameterFrame class; for storing collections of Parameters and their
    meta-data

    Parameters
    ----------
    record_list: List[List[Any]]
        The list of records from which to build a ParameterFrame of Parameters
    with_defaults: bool
        initialise with the default parameters as a base, values will be
        overwritten by anything in the record list. default = False

    """

    __default_params = {}
    __defaults_setting = False
    __defaults_set = False
    __template_params = {}

    def __init__(self, record_list=None, *, with_defaults=False):
        if with_defaults:
            self._reinit()
        if record_list is not None:
            self.add_parameters(record_list)

    @classmethod
    def set_default_parameters(cls, params):
        """
        Set the default parameters for all reactor objects.

        TODO: A cleaner way of doing this

        Parameters
        ----------
        params: list
            default parameters

        """
        if not cls.__defaults_set:
            cls.__defaults_setting = True

            sv = cls.add_parameter
            sv_set = cls.__setattr__

            cls.__setattr__ = cls.__setattr
            cls.__setattr = sv_set
            cls.add_parameter = cls._add_parameter
            cls._add_parameter = sv

            cls.add_parameters(cls, params)

            cls._add_parameter = cls.add_parameter
            cls.add_parameter = sv
            cls.__setattr = cls.__setattr__
            cls.__setattr__ = sv_set

            cls.__defaults_setting = False
        else:
            raise ParameterError(
                "Default parameters already set please use"
                "'_force_update_defaults' if you really want to do this."
            )
        cls.__defaults_set = True

    @classmethod
    def set_template_parameters(cls, params: List[List[Any]]):
        """
        Fills the template parameters from the minimal content of the provided parameter
        records list.

        Parameters
        ----------
        params: List[List[Any]]
            The parameter record list to use to populate the template.
        """
        for param in params:
            cls.__template_params[param[0]] = {
                "name": param[1],
                "unit": param[3],
            }

    @classmethod
    def __setattr(cls, *args, **kwargs):
        return cls.__setattr(cls, *args, **kwargs)

    @classmethod
    def from_template(cls, param_vars: List[str]) -> "ParameterFrame":
        """
        Generate a minimal ParameterFrame from the provided parameter variable names.

        Parameters
        ----------
        param_vars: List[str]
            The parameter variable names to include in the resulting ParameterFrame.

        Returns
        -------
        params: ParameterFrame
            The ParameterFrame including the minimal content for the requested parameter
            variable names.
        """
        if not hasattr(cls, "__template_params") or cls.__template_params != {}:
            from BLUEPRINT.systems.config import Configuration

            cls.set_template_parameters(Configuration.params)

        params = ParameterFrame()
        for var in param_vars:
            if var not in cls.__template_params:
                raise ParameterError(
                    f"Parameter with short name {var} is not known as a template "
                    f"parameter for class {cls.__name__}."
                )
            name = cls.__template_params[var]["name"]
            unit = cls.__template_params[var]["unit"]
            params.add_parameter(var=var, name=name, unit=unit)

        return params

    @staticmethod
    def modify_source(source, param):
        """
        Modify the source term of a Parameter

        Parameters
        ----------
        source: str
            New source
        param: Union[Parameter, list, dict]
            Parameter that will have its source modified

        Returns
        -------
        param
            Modified Parameter
        """
        if source is not None:
            if isinstance(param, list):
                # + 1 because 'value' isn't in __slots__
                param[Parameter.__slots__.index("_source") + 1] = source
            elif isinstance(param, Parameter):
                param.source = source
            elif isinstance(param, dict):
                param["source"] = source
            else:
                param = (param, source)
        return param

    def _force_update_default(self, attr, value, source=None):
        """
        Force update a default Parameter.

        Should be used sparingly as it will overwrite values for all current
        instances of reactors.

        Parameters
        ----------
        attr: str
            Parameter short name
        value: Union[Parameter, list, dict, float, int]
            The Parameter like object that can either overwrite just the
            current value of the parameter or change the whole Parameter
            instance.
        source: str
            overwrite source on Parameter

        Notes
        -----
        Works by looking in the garbage collector for instances of ParameterFrame
        and modifying all of them along with the class default variable.

        """
        if isinstance(value, Parameter) or not isinstance(value, (list, dict, tuple)):
            value = self.modify_source(source, value)
        else:
            value = self.modify_source(
                source,
                Parameter(**value) if isinstance(value, dict) else Parameter(*value),
            )

        if attr not in self.__class__.__default_params and not isinstance(
            value, Parameter
        ):
            raise ValueError(f"No default Parameter {attr} found")

        self.__class__.__setattr__(
            self.__class__, attr, value, allow_new=True, defaults=True
        )

        instances = [
            obj
            for obj in gc.get_referrers(self.__class__)
            if isinstance(obj, self.__class__)
        ]
        for instance in instances:
            instance.__setattr__(attr, value, allow_new=True)

    @classmethod
    def _clean(cls):
        """
        Clean ParameterFrame to remove all defaults
        from the internal state saving.
        """
        cls.__default_params = {}
        cls.__defaults_set = False

    def _reinit(self):
        """
        Reinitialise class with defaults.
        """
        self.__dict__ = self.__default_params.copy()

    def to_records(self):
        """
        Convert the ParameterFrame to a record of lists
        """
        return sorted(
            [list(self.__dict__[key].to_list()) for key in self.__dict__.keys()]
        )

    @classmethod
    def _add_parameter(cls, *args, **kwargs):
        """
        Add parameter as a class method for defaults.
        """
        return cls._add_parameter(cls, *args, **kwargs)

    def add_parameter(
        self,
        var: str,
        name: str = None,
        value=None,
        unit: Union[str, None] = None,
        description: Union[str, None] = None,
        source: Union[str, None] = None,
        mapping=None,
        value_history: Union[list, None] = None,
        source_history: Union[list, None] = None,
    ):
        """
        Takes a list or Parameter object and adds it to the ParameterFrame
        Handles updates if existing parameter (Var_name sorted).

        Parameters
        ----------
        var: str
            The short parameter name
        name: Union[str, None]
            The long parameter name, by default None.
        value: Union[str, float, int, None]
            The value of the parameter, by default None.
        unit: Union[str, None]
            The unit of the parameter, by default None.
        description: Union[str, None]
            The long description of the parameter, by default None.
        source: Union[str, None]
            The source (reference and/or code) of the parameter, by default None.
        mapping: Union[Dict[str, ParameterMapping], None]
            The names used for this parameter in external software, and whether
            that parameter should be written to and/or read from the external tool,
            by default, None.
        value_history: Union[list, None]
            History of the value
        source_history: Union[list, None]
            History of the source of the value
        """
        if isinstance(var, Parameter):
            (
                var,
                name,
                value,
                unit,
                description,
                source,
                mapping,
                value_history,
                source_history,
            ) = var.to_list(history=True)
        self.__setattr__(
            var,
            # Should we make a copy of `mapping` here, as it is mutable?
            Parameter.from_json(
                var,
                name,
                value,
                unit,
                description,
                source,
                copy.deepcopy(mapping),
                value_history,
                source_history,
            ),
            allow_new=True,
            defaults=self.__defaults_setting,
        )

    def add_parameters(self, record_list, source=None):
        """
        Handles a record_list for ParameterFrames and updates accordingly.
        Items in record_list may be Parameter objects or lists in the following format:

        [var, name, value, unit, description, source]

        If a record_list is a dict, it is passed to update_kw_parameters
        with the specified source.

        Parameters
        ----------
        record_list: Union[dict, list, Parameter]
            Container of individual Parameters
        source: str
            Updates the source parameter for each item in record_list with the
            specified value, by default None (i.e. the value is left unchanged).
        """
        if isinstance(record_list, dict):
            self.update_kw_parameters(record_list, source=source)
        else:
            for param in record_list:
                if isinstance(param, Parameter):
                    self.add_parameter(self.modify_source(source, param))
                else:
                    self.add_parameter(*self.modify_source(source, param))

    def set_parameter(self, var, value, source=None):
        """
        Updates only the value of a parameter in the ParameterFrame

        Parameters
        ----------
        var: str
            variable name
        value: Union[Parameter, int, float, str ...]
            new value of parameter
        source: str
            override value for source

        """
        self.__setattr__(var, self.modify_source(source, value))

    def update_kw_parameters(self, kwargs, source=None):
        """
        Handles dictionary keys like update

        Parameters
        ----------
        kwargs: dict
            dictionary of updates value of key can be:

                * A Parameter
                * A dictionary with keys value and source
                    where value is a base type not a Parameter
                * A base type value

        source: str
            override value for source

        Notes
        -----
        if the value in the dictionary is a Parameter only the
        source and the value are taken (unless the source is overridden)

        """
        for key, var in kwargs.items():
            if key not in self.__dict__:
                # Skip keys that aren't parameters, note this could mask typos!
                continue
            if isinstance(var, dict):
                src = var.get("source") if source is None else source
                desc = var.get("description")
                mapping = var.get("mapping")
                var = var.get("value")
            elif isinstance(var, Parameter):
                src = var.source if source is None else source
                desc = var.description
                mapping = var.mapping
                var = var.value
            else:
                src = source
                desc = None
                mapping = None

            self.__setattr__(key, self.modify_source(src, var))
            if desc is not None:
                self.__dict__[key].description = desc
            if mapping is not None:
                self.__dict__[key].mapping = mapping

    def items(self):
        """
        Returns dictionary-like behaviour of .items()
        """
        return [(key, parameter.value) for (key, parameter) in self.__dict__.items()]

    def keys(self):
        """
        Returns dictionary-like behaviour of .keys()
        """
        return self.__dict__.keys()

    def get_parameter_list(self):
        """
        Get a list of Parameters for the ParameterFrame

        Returns
        -------
        p_list: List[Parameter]
            The list of Parameters in the ParameterFrame
        """
        return list(self.__dict__.values())

    def get_param(self, var) -> Parameter:
        """
        Returns a Parameter object using the short var_name
        """
        if isinstance(var, str):  # Handle single variable request
            return self.__getattribute__(var)
        elif isinstance(var, list):  # Handle multiple variable request
            return ParameterFrame([self.__dict__[v].to_list() for v in var])

    def get(self, var, default=None):
        """
        Returns a Parameter object's value using the short var_name
        """
        try:
            return self.__getitem__(var)
        except KeyError:
            return default

    @staticmethod
    def float_format(num):
        """
        Format a float
        """
        if type(num) is float:
            return f"{num:.4g}"
        else:
            return num

    def format_values(self):
        """
        Format values in the underlying DataFrame
        """
        db = self._get_db()
        return db["Value"].apply(self.float_format)

    def _get_db(self):
        columns = [
            "Var_name",
            "Name",
            "Value",
            "Unit",
            "Description",
            "Source",
            "Mapping",
        ]
        db = DataFrame.from_records(self.to_records(), columns=columns)
        return db

    def tabulator(self, keys=None, db=None, tablefmt="fancy_grid"):
        """
        Tabulate the underlying DataFrame of the ParameterFrame

        Parameters
        ----------
        keys: list
            database column keys
        db: DataFrame
            database to tabulate
        tablefmt: str (default="fancy_grid")
            The format of the table - see
            https://github.com/astanin/python-tabulate#table-format

        Returns
        -------
        tabulated: str
            The tabulated DataFrame
        """
        db = self._get_db() if db is None else db
        if keys is None:
            columns = list(db.columns)
        else:
            db = db[keys]
            columns = keys
        return tabulate(
            db,
            headers=columns,
            tablefmt=tablefmt,
            showindex=False,
            numalign="right",
        )

    def copy(self):
        """
        Get a deep copy of the ParameterFrame
        """
        return copy.deepcopy(self)

    def __eq__(self, other):
        """
        Check Parameter for equality

        Parameters
        ----------
        other: ParameterFrame
            The other ParameterFrame to compare with

        Returns
        -------
        equal: bool
            Whether or not the ParameterFrames are identical
        """
        if type(self) != type(other):
            return False
        return self.to_records() == other.to_records()

    def __repr__(self):
        """
        Prints a representation of the ParameterFrame inside the console
        """
        fdb = self._get_db()
        fdb["Value"] = self.format_values()
        return self.tabulator(["Name", "Value", "Unit", "Source"], fdb)

    def __getitem__(self, var):
        """
        Enables the ParameterFrame to be used like a dictionary, returning
        the value when keyed with the short var_name
        """
        # TODO: remove and use attribute access only?
        # NOTE: useful for attributes like H* where attribute access can't be used

        try:
            return self.__dict__[var].value
        except KeyError:
            raise KeyError(f"Var name {var} not present in ParameterFrame")

    def __setattr__(self, attr, value, allow_new=False, defaults=False):
        """
        Set an attribute on the ParameterFrame

        If the attribute is a Parameter then the value is set on the underlying Parameter
        within the ParameterFrame.

        If the value is provided as a Parameter the the underlying Parameter is set
        within the ParameterFrame.

        Parameters
        ----------
        attr: str
            The var name of the Parameter in the ParameterFrame
        value: Union[Parameter, tuple, list, str, float, int]
            The value of the Parameter to set
        allow_new: bool
            Whether or not to allow new Parameters (previously unset) in the
            ParameterFrame

        Notes
        -----
        If value is a two element list and the second element is a string
        the second element is assumed to be the source and the first is the value
        of the parameter.

        """
        # Move as much of this as possible into Parameter
        if defaults:
            _dict = self.__default_params
        else:
            _dict = self.__dict__

        if not allow_new:
            if attr != "__dict__" and attr not in _dict:
                raise ValueError(f"Attribute {attr} not defined in ParameterFrame.")

        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[1], (str, type(None)))
        ):
            source = value[1]
            value = value[0]
        elif isinstance(value, dict) and list(value.keys()) == ["value", "source"]:
            source = value["source"]
            value = value["value"]
        else:
            source = None

        if isinstance(value, Parameter):
            if attr != value.var:
                raise ValueError(
                    f"Mismatch between parameter var {value.var} and attribute to be set {attr}."
                )
            if source is not None:
                value.source = source
            if allow_new:
                _dict[attr] = value
            else:
                _dict[attr].value = value.value
                _dict[attr].source = value.source
        elif isinstance(_dict.get(attr, None), Parameter):
            _dict[attr].value = value
            _dict[attr].source = source
        else:
            # what other attributes need to be set?
            super().__setattr__(attr, value)

    def to_dict(self, verbose=False) -> dict:
        """
        Get the ParameterFrame as a dictionary

        Parameters
        ----------
        verbose: bool
            If True then the full parameter details will be output, by default False.

        Returns
        -------
        the_dict: dict
            The dictionary representation of the ParameterFrame.
        """
        if verbose:
            return {
                key: parameter.to_dict(ignore_var=True)
                for (key, parameter) in self.__dict__.items()
            }
        else:
            return {
                key: {"value": parameter.value, "source": parameter.source}
                for (key, parameter) in self.__dict__.items()
            }

    @classmethod
    def from_dict(cls, the_dict: dict):
        """
        Create a new ParameterFrame from the dictionary.

        Note that the full parameter details are needed to run this method. To set the
        values from a concise dictionary then create an instance and set the values on
        that instance using update_kw_parameters.

        Parameters
        ----------
        the_dict: dict
            The dictionary to create the ParameterFrame from.

        Returns
        -------
        the_parameter_frame: ParameterFrame
            The ParameterFrame created from the dictionary.
        """
        records = [
            [key]
            + [
                val.get("name"),
                val.get("value", None),
                val.get("unit", None),
                val.get("description", None),
                val.get("source", None),
                val.get("mapping", None),
                val.get("value_history", None),
                val.get("source_history", None),
            ]
            for (key, val) in the_dict.items()
        ]
        return cls(records)

    def to_list(self) -> list:
        """
        Get the ParameterFrame as a list of records.

        Returns
        -------
        the_list: List[List[6]]
            The list of field values, each as a list:
                [name, value, unit, description, source].
        """
        return self.to_records()

    @classmethod
    def from_list(cls, the_list: list):
        """
        Create a new ParameterFrame from the list.

        Parameters
        ----------
        the_list: List[List[6]]
            The dictionary to create the ParameterFrame from.

        Returns
        -------
        the_parameter_frame: ParameterFrame
            The ParameterFrame created from the list.
        """
        return cls(the_list)

    class ParameterMappingEncoder(json.JSONEncoder):
        """
        Class to handle serialisation of ParameterMapping objects to JSON.
        """

        def default(self, obj):
            """Overridden JSON serialisation method which will be called if an
            object is not an instance of one of the classes with
            built-in serialisations (e.g., list, dict, etc.).

            """
            if isinstance(obj, ParameterMapping):
                return obj.to_dict()
            return json.JSONEncoder.default(self, obj)

    def to_json(self, output_path=None, verbose=False, return_output=False) -> str:
        """
        Convert the ParameterFrame to a JSON representation.

        Parameters
        ----------
        output_path: Union[str, None]
            The optional path to the file containing the JSON, by default None.
        verbose: bool
            If True then the full parameter details will be output, by default False.
        return_output: bool
            If an output path is specified, then if True returns the JSON output,
            by default False.

        Returns
        -------
        the_json: Union[str, None]
            The JSON representation of the Parameter.
        """
        the_json = json.dumps(
            self.to_dict(verbose), indent=2, cls=self.ParameterMappingEncoder
        )
        if output_path is not None:
            with open(output_path, "w") as fh:
                fh.write(the_json)
            if return_output:
                return the_json
        else:
            return the_json

    @staticmethod
    def parameter_mapping_hook(dct: Dict) -> ParameterMapping:
        """
        Callback to convert suitable JSON objects (dictionaries) into
        ParameterMapping objects.
        """
        if {"name", "send", "recv"} == set(dct.keys()):
            return ParameterMapping(**dct)
        return dct

    @classmethod
    def from_json(cls, data: str):
        """
        Create a new ParameterFrame from the json data.

        Note that the full parameter details are needed to run this method. To set the
        values from a concise dictionary then create an instance and set the values on
        that instance using the set_values_from_json method.

        Parameters
        ----------
        data: str
            The JSON string, or file path to the JSON data.

        Returns
        -------
        the_parameter_frame: ParameterFame
            The ParameterFrame created from the json data.
        """
        if os.path.isfile(data):
            with open(data, "r") as fh:
                return cls.from_json(fh.read())
        else:
            the_data = json.loads(data, object_hook=cls.parameter_mapping_hook)
            if any(
                not isinstance(v, dict) or list(v.keys()) == ["value", "source"]
                for v in the_data.values()
            ):
                raise ValueError(
                    f"Creating a {cls.__name__} using from_json requires a verbose json format."
                )
            return cls.from_dict(the_data)

    def set_values_from_json(self, data: str, source="Input"):
        """
        Set the parameter values from the JSON data.

        Note that this method accepts data created in the concise JSON format
        (key-value pairs).

        Parameters
        ----------
        data: str
            The JSON string, or file path to the JSON data.
        """
        if os.path.isfile(data):
            with open(data, "r") as fh:
                self.set_values_from_json(fh.read(), source=source)
                return self
        else:
            the_data = json.loads(data)
            if any(
                isinstance(v, dict) and list(v.keys()) != ["value", "source"]
                for v in the_data.values()
            ):
                raise ValueError(
                    f"Setting the values on a {self.__class__.__name__} using set_values_from_json requires a concise json format."
                )
            self.update_kw_parameters(the_data, source=source)

    def diff_params(self, other: "ParameterFrame", include_new=False):
        """
        TODO test this

        Diff this ParameterFrame with another ParameterFrame.
        """
        diff = ParameterFrame()
        for key in self.keys():
            if key in other.keys() and self[key] != other[key]:
                diff.add_parameter(*other.get_param(key).to_list())

        if include_new:
            for key in other.keys():
                if key not in self.keys():
                    diff.add_parameter(*other.get_param(key).to_list())
            for key in self.keys():
                if key not in other.keys():
                    diff.add_parameter(*self.get_param(key).to_list())

        return diff
