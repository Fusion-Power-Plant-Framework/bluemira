# COPYRIGHT PLACEHOLDER

import copy

import pytest

from bluemira.power_cycle.base import PowerCycleABC
from bluemira.power_cycle.errors import (
    PowerCycleABCError,
    PowerCycleGroupError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
from bluemira.power_cycle.net.loads import PhaseLoad, PowerLoad
from bluemira.power_cycle.net.manager import (
    PowerCycleGroup,
    PowerCycleManager,
    PowerCycleSystem,
)
from bluemira.power_cycle.time import PowerCycleScenario
from bluemira.power_cycle.tools import validate_dict
from tests.power_cycle.kits_for_tests import (  # NetLoadsTestKit,; TimeTestKit,; ToolsTestKit,
    NetManagerTestKit,
)

#
# import os
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
        self.scenario = manager_testkit.scenario

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

        all_instance_attr = [
            "scenario",
            "_system_config",
            "_active_config",
            "_reactive_config",
            "_production_config",
        ]
        self.all_instance_attr = all_instance_attr

        all_instance_properties = [
            "active_loads",
            "reactive_loads",
            "production_loads",
        ]
        self.all_instance_properties = all_instance_properties

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_multiple_samples(self):
        tested_class = self.tested_class

        scenario = self.scenario
        all_system_inputs = self.all_system_inputs

        all_samples = []
        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            system_config = all_system_inputs[system_label]
            sample = tested_class(scenario, system_config)
            all_samples.append(sample)
        return all_samples

    def test_constructor(self):
        all_instance_attr = self.all_instance_attr
        all_instance_properties = self.all_instance_properties

        all_samples = self.construct_multiple_samples()
        for sample in all_samples:

            for instance_attr in all_instance_attr:
                attr_was_created = hasattr(sample, instance_attr)
                assert attr_was_created

            for instance_property in all_instance_properties:
                property_is_defined = hasattr(sample, instance_property)
                assert property_is_defined

    def test_validate_scenario(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        scenario = self.scenario

        validated_scenario = tested_class._validate_scenario(scenario)
        assert validated_scenario == scenario
        assert type(scenario) == PowerCycleScenario

        not_a_scenario = "not_a_scenario"
        with pytest.raises(tested_class_error):
            validated_scenario = tested_class._validate_scenario(not_a_scenario)

    def test_unpack_system_config(self):
        """
        No new functionality to be tested.
        """
        tested_class = self.tested_class
        assert callable(tested_class._unpack_system_config)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def list_all_load_configs(self):
        all_system_inputs = self.all_system_inputs
        highest_level_json_keys = self.highest_level_json_keys

        all_load_types = copy.deepcopy(highest_level_json_keys)
        all_load_types.remove("name")

        all_load_configs = []
        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            system_config = all_system_inputs[system_label]
            for load_type in all_load_types:
                load_config = system_config[load_type]
                load_config = load_config.values()
                all_load_configs += load_config

        return all_load_configs

    def list_all_phaseload_inputs(self):
        tested_class = self.tested_class

        all_load_configs = self.list_all_load_configs()
        all_phaseload_inputs = []
        for load_config in all_load_configs:
            load_name = load_config["name"]

            module = load_config["module"]
            variables_map = load_config["variables_map"]

            phaseload_inputs = tested_class.import_phaseload_inputs(
                module,
                variables_map,
            )
            all_phaseload_inputs.append(phaseload_inputs)
        return all_phaseload_inputs

    def test_import_phaseload_inputs(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        phaseload_inputs_format = {
            "phase_list": list,
            "normalize_list": list,
            "powerload_list": list,
        }

        all_phaseload_inputs = self.list_all_phaseload_inputs()
        for phaseload_inputs in all_phaseload_inputs:

            assert validate_dict(phaseload_inputs, phaseload_inputs_format)

            valid_keys = phaseload_inputs_format.keys()
            for key in valid_keys:
                list_in_key = phaseload_inputs[key]

                if key == "phase_list":
                    valid_type = str
                elif key == "normalize_list":
                    valid_type = bool
                elif key == "powerload_list":
                    valid_type = PowerLoad

                types_are_right = [type(e) == valid_type for e in list_in_key]
                assert all(types_are_right)

        inexistent_module = "inexistent_module"
        example_variables_map = dict()
        with pytest.raises(tested_class_error):
            phaseload_inputs = tested_class.import_phaseload_inputs(
                inexistent_module,
                example_variables_map,
            )

    def test_build_phaseloads(self):
        all_samples = self.construct_multiple_samples()
        all_phaseload_inputs = self.list_all_phaseload_inputs()
        for sample in all_samples:
            sample_scenario = sample.scenario
            pulse_set = sample_scenario.pulse_set

            all_phases = [pulse.phase_set for pulse in pulse_set]
            valid_phases = list(set(all_phases))

            for phaseload_inputs in all_phaseload_inputs:

                example_load_name = sample.name + " load"
                phaseload_list = sample._build_phaseloads(
                    example_load_name, phaseload_inputs
                )

                for phaseload in phaseload_list:
                    assert type(phaseload) == PhaseLoad

                    phase = phaseload.phase
                    assert phase in valid_phases

    def test_make_phaseloads_from_config(self):
        pass

    """
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
    """

    """
    def test_make_phaseloads_from_config(self):
        tested_class = self.tested_class
        all_load_inputs = self.all_load_inputs

        load_list = tested_class._build_loads_from_config(all_load_inputs)
        for load in load_list:
            load_is_correct_class = type(load) == PowerLoad
            assert load_is_correct_class

        # import pprint
        # assert 0
    """

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------


class TestPowerCycleGroup:
    tested_class_super = PowerCycleABC
    tested_class_super_error = PowerCycleABCError
    tested_class = PowerCycleGroup
    tested_class_error = PowerCycleGroupError

    def setup_method(self):
        pass


class TestPowerCycleManager:
    tested_class_super = None
    tested_class_super_error = None
    tested_class = PowerCycleManager
    tested_class_error = PowerCycleManagerError

    def setup_method(self):
        pass
