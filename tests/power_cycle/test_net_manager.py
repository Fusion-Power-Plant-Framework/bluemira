# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt
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
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, unnest_list, validate_dict
from tests.power_cycle.kits_for_tests import NetManagerTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
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
            all_phases = unnest_list(all_phases)
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
        all_instance_properties = self.all_instance_properties

        all_system_inputs = self.all_system_inputs
        all_sample_inputs = list(all_system_inputs.values())

        all_samples = self.construct_multiple_samples()

        n_samples = len(all_samples)
        for s in range(n_samples):

            sample = all_samples[s]
            sample_inputs = all_sample_inputs[s]
            for load_type in all_instance_properties:
                system_loads = getattr(sample, load_type)
                number_of_loads = len(system_loads)

                type_label = load_type.split("_")[0]
                type_config = sample_inputs[type_label]
                assert number_of_loads == len(type_config)


class TestPowerCycleGroup:
    tested_class_super = PowerCycleABC
    tested_class_super_error = PowerCycleABCError
    tested_class = PowerCycleGroup
    tested_class_error = PowerCycleGroupError

    def setup_method(self):
        scenario = manager_testkit.scenario
        self.scenario = scenario

        all_group_inputs = manager_testkit.inputs_for_groups()
        self.all_group_inputs = all_group_inputs

        all_instance_attr = [
            "group_config",
            "system_library",
        ]
        self.all_instance_attr = all_instance_attr

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_multiple_samples(self):
        tested_class = self.tested_class

        scenario = self.scenario
        all_group_inputs = self.all_group_inputs

        all_samples = []
        all_group_labels = all_group_inputs.keys()
        for group_label in all_group_labels:
            group_inputs = all_group_inputs[group_label]

            group_name = group_inputs["name"]
            group_config = group_inputs["systems_config"]

            sample = tested_class(
                group_name,
                scenario,
                group_config,
            )
            all_samples.append(sample)
        return all_samples

    def test_constructor(self):
        all_instance_attr = self.all_instance_attr

        all_samples = self.construct_multiple_samples()
        for sample in all_samples:
            for instance_attr in all_instance_attr:
                attr_was_created = hasattr(sample, instance_attr)
                assert attr_was_created

    def test_build_system_library(self):
        """
        No new functionality to be tested.
        """
        tested_class = self.tested_class
        assert callable(tested_class._build_system_library)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------


class TestPowerCycleManager:
    tested_class_super = None
    tested_class_super_error = None
    tested_class = PowerCycleManager
    tested_class_error = PowerCycleManagerError

    def setup_method(self):
        self.scenario_json_path = manager_testkit.scenario_json_path
        self.manager_json_path = manager_testkit.manager_json_path

        scenario = manager_testkit.scenario
        self.scenario = scenario

        manager_json_contents = manager_testkit.inputs_for_manager()
        self.manager_json_contents = manager_json_contents

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_sample(self):
        tested_class = self.tested_class

        scenario_json_path = self.scenario_json_path
        manager_json_path = self.manager_json_path

        sample = tested_class(scenario_json_path, manager_json_path)
        return sample

    def test_constructor(self):
        sample = self.construct_sample()
        assert isinstance(sample, PowerCycleManager)

    def test_build_group_library(self):
        tested_class = self.tested_class

        scenario = self.scenario
        manager_json_contents = self.manager_json_contents

        all_group_labels = manager_json_contents.keys()

        contents = manager_json_contents
        all_keys = all_group_labels
        all_systems = [contents[key]["systems"] for key in all_keys]

        all_system_labels = unnest_list(all_systems)

        group_library = tested_class._build_group_library(
            scenario,
            manager_json_contents,
        )

        for group_label in all_group_labels:
            group = group_library[group_label]
            assert type(group) == PowerCycleGroup

            system_library = group.system_library
            keys_in_library = system_library.keys()
            for key in keys_in_library:
                assert key in all_system_labels

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "load_type",
        ["active"],  # ["active", "reactive", "production"],
    )
    def test_build_pulseload(self, load_type):
        ax = tools_testkit.prepare_figure(load_type)

        sample = self.construct_sample()
        pulseload = sample._build_pulseload(load_type)

        # all_phaseloads = pulseload.phaseload_set
        # import pprint
        # assert 0

        ax, _ = pulseload.plot(
            ax=ax,
            detailed=False,
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
