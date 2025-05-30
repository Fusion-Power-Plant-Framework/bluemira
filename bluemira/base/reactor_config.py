# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Class to hold parameters and config values."""

from __future__ import annotations

import json
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bluemira.base.error import ReactorConfigError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.base.parameter_frame import make_parameter_frame

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.base.parameter_frame._parameter import ParamDictT
    from bluemira.base.parameter_frame.typed import ParameterFrameT


@dataclass
class ConfigParams:
    """Container for the global and local parameters of a `ReactorConfig`."""

    global_params: ParameterFrameT
    local_params: dict[str, ParamDictT]


_PARAMETERS_KEY = "params"
_FILEPATH_PREFIX = "$path:"
_FILEPATH_EXPANSION_PREFIX = "$path_expand:"


class ReactorConfig:
    """
    Class that provides a simple interface over config JSON files and
    handles overwriting multiply defined attributes.

    If an attribute is defined more than once in a component,
    the more globally scoped value is used (global overwrites local).

    Parameters
    ----------
    config_path:
        The path to the config JSON file or a dict of the data.
    global_params_type:
        The ParameterFrame type for the global params.
    warn_on_duplicate_keys:
        Print a warning when duplicate keys are found,
        whose value will be overwritten.
    warn_on_empty_local_params:
        Print a warning when the local params for some args are empty,
        when calling params_for(args)
    warn_on_empty_config:
        Print a warning when the config for some args are empty,
        when calling config_for(args)

    Example
    -------

    .. code-block:: python

        from bluemira.base.parameter_frame import Parameter, ParameterFrame

        @dataclass
        class GlobalParams(ParameterFrame):
            a: Parameter[int]


        reactor_config = ReactorConfig(
            {
                "params": {"a": {"value": 10, "unit": 'm'}},
                "comp A": {
                    "params": {
                        "a": {"value": 5, "unit": 'm'},
                        "b": {"value": 5, "unit": 'm'},
                    },
                    "designer": {
                        "params": {"a": {"value": 1, "unit": 'm'}},
                        "some_config": "some_value",
                    },
                    "builder": {
                        "params": {
                            "b": {"value": 1, "unit": 'm'},
                            "c": {"value": 1, "unit": 'm'},
                        },
                        "another_config": "another_value",
                    },
                },
                "comp B": {
                    "params": {"b": {"value": 1, "unit": 'm'}},
                    "builder": {
                        "third_config": "third_value",
                    },
                },
            },
            GlobalParams,
        )

    """

    def __init__(
        self,
        config_path: str | Path | dict,
        global_params_type: type[ParameterFrameT],
        *,
        warn_on_duplicate_keys: bool = False,
        warn_on_empty_local_params: bool = False,
        warn_on_empty_config: bool = False,
    ):
        self.warn_on_duplicate_keys = warn_on_duplicate_keys
        self.warn_on_empty_local_params = warn_on_empty_local_params
        self.warn_on_empty_config = warn_on_empty_config

        config_data = self._read_or_return(config_path)
        if isinstance(config_path, Path | str):
            self._expand_paths_in_dict(config_data, Path(config_path).parent)

        self.config_data = config_data
        self.global_params = make_parameter_frame(
            self.config_data.get(
                _PARAMETERS_KEY, None if global_params_type is None else {}
            ),
            global_params_type,
        )

        if not self.global_params:
            bluemira_warn("Empty global params")

    def __str__(self) -> str:
        """Returns config_data as a nicely pretty formatted string.

        Returns
        -------
        :
            The pretty formatted string of the config_data.
        """
        return self._pprint_dict(self.config_data)

    @staticmethod
    def _warn_or_debug_log(msg: str, *, warn: bool = False) -> None:
        if warn:
            bluemira_warn(msg)
        else:
            bluemira_debug(msg)

    def params_for(self, component_name: str, *args: str) -> ConfigParams:
        """
        Gets the params for the `component_name` from the config file.

        These are all the values defined by a "params"
        key in the config file.

        This will merge all multiply defined params,
        with global overwriting local.

        Parameters
        ----------
        component_name:
            The component name, must match a key in the config
        *args:
            Optionally, specify the keys of nested attributes.

            This will hoist the values defined in the nested attributes
            to the top level of the `local_params` dict
            in the returned `ConfigParams` object.

            The args must be in the order that they appear in the config.

        Returns
        -------
        Holds the global_params (from `self.global_params`)
        and the extracted local_params.

        Use the
        :func:`bluemira.base.parameter_frame._frame.make_parameter_frame`
        helper function to convert it into a typed ParameterFrame.
        """
        args = (component_name, *args)
        self._check_args_are_strings(args)

        local_params = self._extract(args, is_config=False)
        if not local_params:
            self._warn_or_debug_log(
                f"Empty local params for args: {args}",
                warn=self.warn_on_empty_local_params,
            )

        return ConfigParams(
            global_params=self.global_params,
            local_params=local_params,
        )

    def config_for(self, component_name: str, *args: str) -> dict:
        """
        Gets the config for the `component_name` from the config file.

        These are all the values other than
        those defined by a "params" key in the config file.

        This will merge all multiply defined values,
        with global overwriting local.

        Parameters
        ----------
        component_name:
            The component name, must match a key in the config
        *args:
            Optionally, specify the keys of nested attributes.

            This will hoist the values defined in the nested attributes
            to the top level of the returned dict.

            The args must be in the order that they appear in the config.

        Returns
        -------
        The extracted config.
        """
        args = (component_name, *args)
        self._check_args_are_strings(args)

        _return = self._extract(args, is_config=True)
        if not _return:
            self._warn_or_debug_log(
                f"Empty config for args: {args}", warn=self.warn_on_empty_config
            )

        return _return

    @staticmethod
    def _read_or_return(config_path: str | Path | dict) -> dict:
        if isinstance(config_path, str | Path):
            return ReactorConfig._read_json_file(config_path)
        if isinstance(config_path, dict):
            return config_path
        raise ReactorConfigError(
            "config_path must be either a dict, a Path object, or a string, found"
            f" {type(config_path)}."
        )

    @staticmethod
    def _read_json_file(path: Path | str) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _pprint_dict(d: dict) -> str:
        return pprint.pformat(d, sort_dicts=False, indent=1)

    def _warn_on_duplicate_keys(
        self,
        shared_key: str,
        arg: str,
        existing_value,
    ):
        self._warn_or_debug_log(
            "duplicate config key: "
            f"'{shared_key}' in {arg} wil be overwritten with {existing_value}",
            warn=self.warn_on_duplicate_keys,
        )

    @staticmethod
    def _check_args_are_strings(args: Iterable[str]):
        for a in args:
            if not isinstance(a, str):
                raise ReactorConfigError("args must be strings")

    def _expand_paths_in_dict(self, d: dict[str, Any], rel_path: Path):
        """
        Expand all file paths by replacing their values with the json file's contents.

        Notes
        -----
            This mutates the passed in dict.
        """
        for k in d:  # noqa: PLC0206
            d[k], rel_path_from = self._extract_and_expand_file_data_if_needed(
                d[k], rel_path
            )
            if isinstance(d[k], str):
                d[k] = self._expand_filepath_if_needed(d[k], rel_path_from)

            if isinstance(d[k], dict):
                self._expand_paths_in_dict(d[k], rel_path_from)

    def _extract_and_expand_file_data_if_needed(
        self, value: Any, rel_path: Path
    ) -> tuple[Any | dict, Path]:
        """
        Returns the file data and the path to the file if value is a path.

        Otherwise, returns value and rel_path that was passed in.

        rel_path is the path to the file that the value is in.

        Returns
        -------
        :
            Tuple of the file data and the path to the file if value is a path.

        Raises
        ------
        FileNotFoundError
            Cannot find provided path

        Notes
        -----
        If the value is not a path, returns the value and the passed in rel_path.
        """
        if not isinstance(value, str):
            return value, rel_path
        if not value.startswith(_FILEPATH_PREFIX):
            return value, rel_path

        # remove _FILEPATH_PREFIX
        f_path = value[len(_FILEPATH_PREFIX) :]

        # if the path does not start with a /, it is considered a relative path,
        # relative to the file the path is in (i.e. rel_path)
        f_path = rel_path / f_path if not f_path.startswith("/") else Path(f_path)

        # check if file exists
        if not f_path.is_file():
            raise FileNotFoundError(f"Cannot find file {f_path}")

        f_data = self._read_json_file(f_path)
        return f_data, f_path.parent

    def _extract(self, arg_keys: tuple[str], *, is_config: bool = True) -> dict:
        extracted = {}

        # this routine is designed not to copy any dict's while parsing

        current_layer = self.config_data

        for next_idx, current_arg_key in enumerate(arg_keys, start=1):
            current_layer = current_layer.get(current_arg_key, {})
            next_arg_key = arg_keys[next_idx] if next_idx < len(arg_keys) else None

            to_extract = current_layer
            if not is_config:
                # if doing a params extraction,
                # get the values from the _PARAMETERS_KEY
                to_extract = current_layer.get(_PARAMETERS_KEY, {})

            if not isinstance(to_extract, dict):
                raise ReactorConfigError(
                    f"Arg {current_arg_key} is too specific, "
                    "it must either be another JSON object "
                    "or a path to a JSON file."
                )

            # add all keys not in extracted already
            # if doing a config, ignore the "params" (_PARAMETERS_KEY)
            # and don't add the next arg key
            for k, v in to_extract.items():
                if k in extracted:
                    self._warn_on_duplicate_keys(k, current_arg_key, extracted[k])
                    continue
                if is_config:
                    if k == _PARAMETERS_KEY:
                        continue
                    if next_arg_key and k == next_arg_key:
                        continue
                extracted[k] = v

        return extracted

    @staticmethod
    def _expand_filepath_if_needed(value: str, config_dir: Path) -> str:
        """
        Checks if the value has a path expansion directive and expands it to the absolute
        path with respect to the config directory.

        Returns
        -------
        Absolute path as str
        """
        if not value.startswith(_FILEPATH_EXPANSION_PREFIX):
            return value

        # remove _FILEPATH_PREFIX
        path_str = value[len(_FILEPATH_EXPANSION_PREFIX) :]

        # if the path does not start with a /, it is considered a relative path,
        # relative to the file the path is in (i.e. rel_path)
        path_value = (
            config_dir / path_str if not path_str.startswith("/") else Path(path_str)
        )
        path_value = path_value.resolve()

        return path_value.as_posix()
