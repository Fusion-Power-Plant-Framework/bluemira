# COPYRIGHT PLACEHOLDER

import copy
import os

import pytest

from bluemira.power_cycle.base import PowerCycleABC
from bluemira.power_cycle.errors import (
    PowerCycleABCError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
from bluemira.power_cycle.net.loads import PowerLoad
from bluemira.power_cycle.net.manager import PowerCycleManager, PowerCycleSystem
from tests.power_cycle.kits_for_tests import (  # NetLoadsTestKit,; TimeTestKit,; ToolsTestKit,
    NetManagerTestKit,
)

# import matplotlib.pyplot as plt

# from bluemira.power_cycle.tools import adjust_2d_graph_ranges

# tools_testkit = ToolsTestKit()
# time_testkit = TimeTestKit()
# netloads_testkit = NetLoadsTestKit()
manager_testkit = NetManagerTestKit()


class TestPowerCycleSystem:
    tested_class_super = PowerCycleABC
    tested_class_super_error = PowerCycleABCError
    tested_class = PowerCycleSystem
    tested_class_error = PowerCycleSystemError

    def setup_method(self):
        all_system_inputs = manager_testkit.inputs_for_systems()
        self.all_system_inputs = all_system_inputs

        all_load_inputs = manager_testkit.inputs_for_loads()
        self.all_load_inputs = all_load_inputs

        all_class_attr = [
            "_system_format",
            "_load_format",
        ]
        self.all_class_attr = all_class_attr

        highest_level_json_keys = [
            "name",
            "production",
            "reactive",
            "active",
        ]
        self.highest_level_json_keys = highest_level_json_keys

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_constructor(self):
        tested_class = self.tested_class
        all_system_inputs = self.all_system_inputs

        all_samples = []
        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            system_inputs = all_system_inputs[system_label]
            sample = tested_class(system_inputs)
            all_samples.append(sample)

    def test_validate_config(self):
        tested_class = self.tested_class
        all_system_inputs = self.all_system_inputs
        highest_level_json_keys = self.highest_level_json_keys

        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            system_inputs = all_system_inputs[system_label]
            outputs = tested_class._validate_config(system_inputs)

            n_valid_keys = len(highest_level_json_keys)
            for k in range(n_valid_keys):
                valid_key = highest_level_json_keys[k]

                k_out = outputs[k]
                key_config = system_inputs[valid_key]
                assert k_out == key_config

    def test_build_loads_from_config(self):
        tested_class = self.tested_class
        all_load_inputs = self.all_load_inputs

        load_list = tested_class._build_loads_from_config(all_load_inputs)
        for load in load_list:
            load_is_correct_class = type(load) == PowerLoad
            assert load_is_correct_class

        import pprint

        assert 0

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------


class TestPowerCycleGroup:
    def setup_method(self):
        pass


class TestPowerCycleManager:
    def setup_method(self):
        pass
