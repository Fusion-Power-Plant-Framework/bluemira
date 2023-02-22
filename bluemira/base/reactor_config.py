"""Class to hold parameters and config values."""


import json
from typing import Union

from bluemira.base.error import ReactorConfigError
from bluemira.base.look_and_feel import bluemira_warn

_PARAMETERS_KEY = "params"


class ReactorConfig:
    """TODO"""

    @staticmethod
    def _read_json_file(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def __init__(self, config_path: Union[str, dict]):
        if isinstance(config_path, str):
            self.config_data = ReactorConfig._read_json_file(config_path)
        elif isinstance(config_path, dict):
            self.config_data = config_path
        else:
            raise ReactorConfigError("Invalid config_path")

    @property
    def global_params(self) -> dict:
        """
        Gets the global parameters (top level "params" in the config file).
        If no global params were defined in the config file,
        returns an empty dict.
        """
        return self.config_data.get(_PARAMETERS_KEY, {})

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
        for shared_key in keyset_local_sub.intersection(keyset_local_params):
            bluemira_warn(
                f"{shared_key} is defined in the {sub_name} '{_PARAMETERS_KEY}' as well as in {component_name}'s '{_PARAMETERS_KEY}'"
            )
        for shared_key in keyset_local_sub.intersection(keyset_global_params):
            bluemira_warn(
                f"{shared_key} is defined in the {component_name}'s {sub_name} '{_PARAMETERS_KEY}' as well as global '{_PARAMETERS_KEY}'"
            )
        for shared_key in keyset_local_params.intersection(keyset_global_params):
            bluemira_warn(
                f"{shared_key} is defined in the {component_name}'s '{_PARAMETERS_KEY}' as well as global '{_PARAMETERS_KEY}'"
            )

    def _get_params(self, component_name: str, sub_name: str) -> dict:
        global_params = self.global_params
        local_params = self.config_data[component_name].get(_PARAMETERS_KEY, {})
        local_sub_params = self.config_data[component_name][sub_name][_PARAMETERS_KEY]

        self._warn_on_duplicate_keys(
            component_name,
            sub_name,
            global_params,
            local_params,
            local_sub_params,
        )

        # higher up the tree overwrites lower
        return {
            **local_sub_params,
            **local_params,
            **global_params,
        }

    def _get_config(self, component_name: str, sub_name: str) -> dict:
        global_params = self.global_params
        local_params = self.config_data[component_name].get(_PARAMETERS_KEY, {})
        local_sub_config: dict = self.config_data[component_name][sub_name].copy()
        local_sub_config.pop(_PARAMETERS_KEY, {})

        self._warn_on_duplicate_keys(
            component_name,
            sub_name,
            global_params,
            local_params,
            local_sub_config,
        )

        # higher up the tree overwrites lower
        return {
            **local_sub_config,
            **local_params,
            **global_params,
        }

    def designer_params(self, component_name: str) -> dict:
        """TODO"""
        return self._get_params(
            component_name=component_name,
            sub_name="designer",
        )

    def builder_params(self, component_name: str) -> dict:
        """TODO"""
        return self._get_params(
            component_name=component_name,
            sub_name="builder",
        )

    def designer_config(self, component_name: str) -> dict:
        """TODO"""
        return self._get_config(
            component_name=component_name,
            sub_name="designer",
        )

    def builder_config(self, component_name: str) -> dict:
        """TODO"""
        return self._get_config(
            component_name=component_name,
            sub_name="designer",
        )
