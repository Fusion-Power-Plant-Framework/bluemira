"""Class to hold parameters and config values."""

import json
import pprint
from dataclasses import dataclass
from typing import Dict, Tuple, Type, TypeVar, Union

from bluemira.base.error import ReactorConfigError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame


@dataclass
class ConfigParams:
    """Container for the global and local parameters of a `ReactorConfig`."""

    global_params: ParameterFrame
    local_params: Dict


_PfT = TypeVar("_PfT", bound=ParameterFrame)
_PARAMETERS_KEY = "params"


class ReactorConfig:
    """
    Class that provides a simple interface over config JSON files and
    handles overwriting multiply defined attributes.

    If an attribute is defined more than once in a component,
    the more globally scoped value is used (global overwrites local).

    Parameters
    ----------
    config_path: str | dict
        The path to the config JSON file or a dict of the data.

    Example
    -------

    .. code-block:: python

        @dataclass
        class GlobalParams(ParameterFrame):
            a: Parameter[int]


        reactor_config = ReactorConfig(
            {
                "params": {"a": 10},
                "comp A": {
                    "params": {"a": 5, "b": 5},
                    "designer": {
                        "params": {"a": 1},
                        "some_config": "some_value",
                    },
                    "builder": {
                        "params": {"b": 1, "c": 1},
                        "another_config": "another_value",
                    },
                },
                "comp B": {
                    "params": {"b": 5},
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
        config_path: Union[str, dict],
        global_params_type: Type[_PfT],
    ):
        if isinstance(config_path, str):
            self.config_data = ReactorConfig._read_json_file(config_path)
        elif isinstance(config_path, dict):
            self.config_data = config_path
        else:
            raise ReactorConfigError("Invalid config_path")

        self.global_params: _PfT = make_parameter_frame(
            self.config_data.get(_PARAMETERS_KEY, {}),
            global_params_type,
        )

    def __str__(self) -> str:
        """Returns config_data as a nicely pretty formatted string"""
        return self._pprint_dict(self.config_data)

    def params_for(self, component_name: str, *args: str) -> ConfigParams:
        """
        Gets the params for the `component_name` from the config file.

        These are all the values defined by a "params"
        key in the config file.

        This will merge all multiply defined params,
        with global overwriting local.

        Parameters
        ----------
        component_name: str
            The component name, must match a key in the config
        *args
            Optionally, specify the keys of nested attributes.

            This will hoist the values defined in the nested attributes
            to the top level of the `local_params` dict
            in the returned `ConfigParams` object.

            The args must be in the order that they appear in the config.

        Returns
        -------
        ConfigParams
            Holds the global_params (from `self.global_params`)
            and the extracted local_params.

            Use the
            :func:`~bluemira.base.parameter_frame._frame.make_parameter_frame`
            helper function to convert it into a typed ParameterFrame.
        """
        args = (component_name,) + args
        self._check_args_are_strings(args)

        return ConfigParams(
            global_params=self.global_params,
            local_params=self._extract(args, is_config=False),
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
        component_name: str
            The component name, must match a key in the config
        *args
            Optionally, specify the keys of nested attributes.

            This will hoist the values defined in the nested attributes
            to the top level of the returned dict.

            The args must be in the order that they appear in the config.

        Returns
        -------
        dict
            The extracted config.
        """
        args = (component_name,) + args
        self._check_args_are_strings(args)

        _return = self._extract(args, is_config=True)
        if not _return:
            bluemira_warn(f"No config for '{component_name}'")

        return _return

    @staticmethod
    def _read_json_file(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def _pprint_dict(self, d: dict) -> str:
        return pprint.pformat(d, sort_dicts=False, indent=1)

    def _warn_on_duplicate_keys(
        self,
        shared_key: str,
        arg: str,
        existing_value,
    ):
        bluemira_warn(
            "duplicate config key: "
            f"'{shared_key}' in {arg} wil be overwritten with {existing_value}"
        )

    def _check_key_in(self, key: str, config_layer: dict):
        if key not in config_layer:
            raise KeyError(
                f"'{key}' not present in config:\n{self._pprint_dict(config_layer)}",
            )

    def _check_args_are_strings(self, args):
        for a in args:
            if not isinstance(a, str):
                raise ReactorConfigError("args must strings")

    def _extract(self, arg_keys: Tuple[str], is_config: bool) -> dict:
        extracted = {}

        # this routine is designed not to copy any dict's while parsing

        current_layer = self.config_data
        for next_idx, current_arg_key in enumerate(arg_keys, start=1):
            self._check_key_in(current_arg_key, current_layer)
            current_layer: dict = current_layer[current_arg_key]

            next_arg_key = arg_keys[next_idx] if next_idx < len(arg_keys) else None

            to_extract = current_layer
            if not is_config:
                # if doing a params extraction,
                # get the values from the _PARAMETERS_KEY
                try:
                    to_extract = current_layer[_PARAMETERS_KEY]
                except KeyError:
                    if next_arg_key is None:
                        # only warn if it's the final arg
                        bluemira_warn(f"'{_PARAMETERS_KEY}' not in {current_arg_key}")
                    continue

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
