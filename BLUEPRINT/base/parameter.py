# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
"""
import copy
from dataclasses import dataclass
import json
import os
from pandas import DataFrame
from tabulate import tabulate
from typing import Dict
from typing import Union


@dataclass
class ParameterMapping:
    """
    Simple class containing information on mapping of a BLUEPRINT
    parameter to one in external software.
    """

    name: str
    read: bool = True
    write: bool = True

    def todict(self):
        """
        Convert this object to a dictionary with attributes as values.
        """
        return {"name": self.name, "read": self.read, "write": self.write}

    def __str__(self):
        """
        Create a string representation of of this object which is more
        compact than that provided by the default `__repr__` method.
        """
        return repr(self.todict())


class Parameter:
    """
    The Parameter base class.

    This provides meta-data for implementations of new parameter objects.
    """

    __slots__ = ("var", "name", "value", "unit", "description", "source", "mapping")

    var: str
    name: Union[str, None]
    value: Union[str, float, int, None]
    unit: Union[str, None]
    description: Union[str, None]
    source: Union[str, None]
    mapping: Union[Dict[str, ParameterMapping], None]

    def __init__(
        self,
        var: str,
        name: Union[str, None] = None,
        value: Union[str, float, int, None] = None,
        unit: Union[str, None] = None,
        description: Union[str, None] = None,
        source: Union[str, None] = None,
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
        self.var = var
        self.name = name
        self.value = value
        self.unit = unit
        self.description = description
        self.source = source
        self.mapping = mapping

    def to_dict(self, ignore_var=False) -> dict:
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
        return {
            k: copy.deepcopy(getattr(self, k))
            for k in self.__slots__
            if not ignore_var or k != "var"
        }

    def to_list(self) -> list:
        """
        Get the Parameter as a list

        Returns
        -------
        the_list: List[str]
            The field values as a list [var, name, value, unit, description, source].
        """
        return [copy.deepcopy(getattr(self, k)) for k in self.__slots__]

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
    """

    def __init__(self, record_list):
        for entry in record_list:
            self.add_parameter(*entry)

    def to_records(self):
        """
        Convert the ParameterFrame to a record of lists
        """
        return [list(self.__dict__[k].to_list()) for k in self.__dict__.keys()]

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

    def add_parameter(
        self,
        var,
        name=None,
        value=None,
        unit=None,
        description=None,
        source=None,
        mapping=None,
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
        """
        if isinstance(var, Parameter):
            (var, name, value, unit, description, source, mapping) = var.to_list()
        self.__setattr__(
            var,
            # Should we make a copy of `mapping` here, as it is mutable?
            Parameter(
                var, name, value, unit, description, source, copy.deepcopy(mapping)
            ),
            allow_new=True,
        )

    def add_parameters(self, record_list):
        """
        Handles a record_list for ParameterFrames and updates accordingly
        if a dict is used, passes to update_kw_parameters
        """
        if isinstance(record_list, dict):
            self.update_kw_parameters(record_list)
        else:
            for p in record_list:
                if isinstance(p, Parameter):
                    self.add_parameter(p)
                else:
                    self.add_parameter(*p)

    def set_parameter(self, var, value, source=None):
        """
        Updates only the value of a parameter in the ParameterFrame
        """
        self.__dict__[var].value = value
        self.__dict__[var].source = source

    def update_kw_parameters(self, kwargs):
        """
        Handles dictionary keys like update
        """
        # TODO: remove me ?
        for k, v in kwargs.items():
            if k not in self.__dict__:
                # Skip keys that aren't parameters, note this could mask typos!
                continue
            if isinstance(v, Parameter):
                self.__dict__[k].value = v.value
            else:
                self.__dict__[k].value = v

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

    def get(self, var):
        """
        Returns a Parameter object of the short var_name
        """
        if isinstance(var, str):  # Handle single variable request
            return self.__dict__[var]
        elif isinstance(var, list):  # Handle multiple variable request
            return ParameterFrame([self.__dict__[v].to_list() for v in var])

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

    def full_table(self):
        """
        Tabulate the underlying DataFrame of the ParameterFrame
        """
        db = self._get_db()
        return tabulate(
            db,
            headers=list(db.columns),
            tablefmt="fancy_grid",
            showindex=False,
            numalign="right",
        )

    def copy(self):
        """
        Get a deep copy of the ParameterFrame
        """
        return copy.deepcopy(self)

    def __repr__(self):
        """
        Prints a representation of the ParameterFrame inside the console
        """
        fdb = self._get_db()
        fdb["Value"] = self.format_values()
        return tabulate(
            fdb[["Name", "Value", "Unit"]],
            headers=["Name", "Value", "Unit"],
            tablefmt="fancy_grid",
            showindex=False,
            numalign="right",
        )

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

    def __getattribute__(self, attr):
        """
        Get an attribute from the ParameterFrame

        If the attribute is a Parameter then the parameter's value is elevated to the
        attribute on the ParameterFrame.

        Parameters
        ----------
        attr: str
            The var name of the Parameter in the ParameterFrame

        Returns
        -------
        value: Union[float, str]
            The value of the Parameter in the ParameterFrame
        """
        if attr == "__dict__":
            return super().__getattribute__(attr)
        elif isinstance(self.__dict__.get(attr, None), Parameter):
            return self.__dict__[attr].value
        else:
            return super().__getattribute__(attr)

    def __setattr__(self, attr, value, allow_new=False):
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
        value: Union[Parameter, str, float, int]
            The value of the Parameter to set
        allow_new: bool
            Whether or not to allow new Parameters (previously unset) in the
            ParameterFrame
        """
        if not allow_new:
            if attr not in self.__dict__:
                raise ValueError(f"Attribute {attr} not defined in ParameterFrame.")

        if isinstance(value, Parameter):
            if attr != value.var:
                raise ValueError(
                    f"Mismatch between parameter var {value.var} and attribute to be set {attr}."
                )

            self.__dict__[attr] = value
        elif isinstance(self.__dict__.get(attr, None), Parameter):
            self.__dict__[attr].value = value
        else:
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
            return {key: parameter.value for (key, parameter) in self.__dict__.items()}

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
            [k]
            + [
                v.get("name"),
                v.get("value", None),
                v.get("unit", None),
                v.get("description", None),
                v.get("source", None),
                v.get("mapping", None),
            ]
            for (k, v) in the_dict.items()
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
                return obj.todict()
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
        if {"name", "read", "write"} == set(dct.keys()):
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
            if any(not isinstance(v, dict) for v in the_data.values()):
                raise ValueError(
                    f"Creating a {cls.__name__} using from_json requires a verbose json format."
                )
            return cls.from_dict(the_data)

    def set_values_from_json(self, data: str):
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
                self.set_values_from_json(fh.read())
                return self
        else:
            the_data = json.loads(data)
            if any(isinstance(v, dict) for v in the_data.values()):
                raise ValueError(
                    f"Setting the values on a {self.__class__.__name__} using set_values_from_json requires a concise json format."
                )
            self.update_kw_parameters(the_data)

    def diff_params(self, other: "ParameterFrame", include_new=False):
        """
        Diff this ParameterFrame with another ParameterFrame.
        """
        diff = ParameterFrame([])
        for key in self.keys():
            if key in other.keys() and self[key] != other[key]:
                diff.add_parameter(*other.get(key).to_list())

        if include_new:
            for key in other.keys():
                if key not in self.keys():
                    diff.add_parameter(*other.get(key).to_list())
            for key in self.keys():
                if key not in other.keys():
                    diff.add_parameter(*self.get(key).to_list())

        return diff


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
