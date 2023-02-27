"""Class to hold parameters and config values."""


import json
import pprint
from dataclasses import dataclass
from typing import Dict, List, Type, TypeVar, Union

from bluemira.base.error import ReactorConfigError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame


@dataclass
class ConfigParams:
    global_params: ParameterFrame
    local_params: Dict


_PfT = TypeVar("_PfT", bound=ParameterFrame)
_PARAMETERS_KEY = "params"


class ReactorConfig:
    """
    Provide a simple interface over config JSON files
    to define some structure that the config files must
    comply to and handles globally vs locally scoped params.

    Global (or more globally scoped) params always overwrite
    locally scoped params.
    Params can be defined at the global level
    (i.e. for every component, for every builder and designer)
    and at the component (i.e. defined for both builder and designer).

    Parameters
    ----------
    config_path: str | dict
        The path to the config JSON file or a dict of the data.

    Example
    -------

    .. code-block:: python

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
            }
        )

        print(reactor_config.designer_params("comp A"))
        # {'a': 10, 'b': 5}
        print(reactor_config.designer_config("comp A"))
        # {'some_config': 'a_value', 'a': 10, 'b': 5}
        print(reactor_config.builder_params("comp A"))
        # {'a': 10, 'b': 5, 'c': 1}
        print(reactor_config.builder_config("comp A"))
        # {'another_config': 'b_value', 'a': 10, 'b': 5}
        print(reactor_config.builder_config("comp B"))
        # {'third_config': 'third_value', 'a': 10, 'b': 5}

    """

    @staticmethod
    def _read_json_file(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

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

        self.global_params = make_parameter_frame(
            self.config_data.get(_PARAMETERS_KEY, {}),
            global_params_type,
        )

    def __str__(self) -> str:
        """Returns config_data as a nicely pretty formatted string"""
        return self._pprint_dict(self.config_data)

    def _pprint_dict(self, d: dict) -> str:
        return pprint.pformat(d, sort_dicts=False, indent=1)

    def _warn_on_duplicate_keys(
        self,
        component_name: str,
        sub_name: str,
        global_params: dict,
        local_params: dict,
        local_sub: dict,
    ):
        keyset_global_params = set(global_params.keys())
        keyset_local_params = set(local_params.keys())
        keyset_local_sub = set(local_sub.keys())

        # warnings for duplicate keys

        for shared_key in keyset_local_sub.intersection(keyset_global_params):
            bluemira_warn(
                f"'{shared_key}' is defined in the global {_PARAMETERS_KEY} as well as in {component_name}'s {sub_name} {_PARAMETERS_KEY}"
            )
        for shared_key in keyset_local_params.intersection(keyset_global_params):
            bluemira_warn(
                f"'{shared_key}' is defined in the global {_PARAMETERS_KEY} as well as in {component_name}'s {_PARAMETERS_KEY}"
            )
        for shared_key in keyset_local_sub.intersection(keyset_local_params):
            bluemira_warn(
                f"'{shared_key}' is defined in {component_name}'s {_PARAMETERS_KEY} as well as in {component_name}'s {sub_name} {_PARAMETERS_KEY}"
            )

    def _check_key_in(self, key: str, config_layer: dict):
        if key not in config_layer:
            raise ReactorConfigError(
                f"'{key}' not present in config:\n{self._pprint_dict(config_layer)}",
            )

    def _check_args_are_strings(self, args):
        for a in args:
            if not isinstance(a, str):
                raise ReactorConfigError("args must strings")

    def _extract(self, arg_keys: List[str], is_config: bool) -> dict:
        extracted = {}

        prev_layer = self.config_data
        for next_idx, current_arg_key in enumerate(arg_keys, start=1):
            self._check_key_in(current_arg_key, prev_layer)
            current_layer = prev_layer[current_arg_key]

            next_arg_key = arg_keys[next_idx] if next_idx < len(arg_keys) else None

            # add all keys not in extracted already
            # and don't add the next key
            # and if doing a config, don't add any "params" (_PARAMETERS_KEY)
            for k, v in current_layer.items():
                if k in extracted:
                    continue
                if next_arg_key and k == next_arg_key:
                    continue
                if is_config and k == _PARAMETERS_KEY:
                    continue
                extracted[k] = v

        # higher up the tree overwrites lower
        return extracted

    def params_for(self, component_name: str, *args: str, file_name="") -> ConfigParams:
        args = component_name + args
        self._check_args_are_strings(args)

        return ConfigParams(
            global_params=self.global_params,
            local_params=self._extract(args, is_config=False),
        )

    def config_for(self, component_name: str, *args: str, file_name="") -> dict:
        args = component_name + args
        self._check_args_are_strings(args)

        return self._extract(args, is_config=True)
