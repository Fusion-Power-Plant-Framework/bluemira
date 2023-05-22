# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.base import LoadType, PowerCycleABC
from bluemira.power_cycle.errors import (
    PowerCycleGroupError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
from bluemira.power_cycle.net.loads import PhaseLoad, PowerLoad
from bluemira.power_cycle.net.manager import (
    PowerCycleGroup,
    PowerCycleLoadConfig,
    PowerCycleManager,
    PowerCycleSystem,
    PowerCycleSystemConfig,
)
from bluemira.power_cycle.time import PowerCycleScenario
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, unnest_list
from tests.power_cycle.kits_for_tests import NetManagerTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
# time_testkit = TimeTestKit()
# netloads_testkit = NetLoadsTestKit()
manager_testkit = NetManagerTestKit()


class TestPowerCycleSystem:
    def setup_method(self):
        self.scenario = manager_testkit.scenario
        self.all_system_inputs = manager_testkit.inputs_for_systems()
        self.all_load_inputs = manager_testkit.inputs_for_loads()
        self.all_class_attr = ["_system_format", "_load_format"]
        self.highest_level_json_keys = ["name", "reactive", "active"]

        self.all_instance_attr = [
            "scenario",
            "_system_config",
            "_active_config",
            "_reactive_config",
        ]

        self.all_instance_properties = ["active_loads", "reactive_loads"]

    def construct_multiple_samples(self):
        all_system_inputs = self.all_system_inputs

        all_samples = []
        all_system_labels = all_system_inputs.keys()
        for system_label in all_system_labels:
            all_samples.append(
                PowerCycleSystem(
                    self.scenario,
                    PowerCycleSystemConfig(**all_system_inputs[system_label]),
                )
            )
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
                # getattr(sample, instance_property)
                property_is_defined = hasattr(sample, instance_property)
                assert property_is_defined

    def list_all_load_configs(self):
        return [
            PowerCycleSystemConfig(**data) for data in self.all_system_inputs.values()
        ]

    def list_all_phaseload_inputs(self):
        phaseloads = []
        for sys_config in self.list_all_load_configs():
            for load_type in (LoadType.ACTIVE, LoadType.REACTIVE):
                load = getattr(sys_config, load_type.name.lower())
                phaseloads.append(
                    PowerCycleSystem.import_phaseload_inputs(
                        load.module, load.variable_map
                    )
                )
        return phaseloads

    def test_build_phaseloads(self):
        all_samples = self.construct_multiple_samples()
        all_phaseload_inputs = self.list_all_phaseload_inputs()
        for sample in all_samples:
            sample_scenario = sample.scenario
            pulse_set = sample_scenario.pulse_set

            all_phases = [pulse.phase_set for pulse in pulse_set]
            all_phases = unnest_list(all_phases)

            valid_phases = []
            for phase in all_phases:
                add_check = True
                for valid_phase in valid_phases:
                    if phase == valid_phase:
                        add_check = False
                if add_check:
                    valid_phases.append(phase)
            # valid_phases = list(set(all_phases))

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

    def construct_multiple_samples(self):
        scenario = self.scenario
        all_group_inputs = self.all_group_inputs

        all_samples = []
        all_group_labels = all_group_inputs.keys()
        for group_label in all_group_labels:
            group_inputs = all_group_inputs[group_label]

            group_name = group_inputs["name"]
            group_config = group_inputs["systems_config"]

            sample = PowerCycleGroup(
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


class TestPowerCycleManager:
    def setup_method(self):
        self.scenario_json_path = manager_testkit.scenario_json_path
        self.manager_json_path = manager_testkit.manager_json_path

        scenario = manager_testkit.scenario
        self.scenario = scenario

        manager_json_contents = manager_testkit.inputs_for_manager()
        self.manager_json_contents = manager_json_contents

    def construct_sample(self):
        return PowerCycleManager(self.scenario_json_path, self.manager_json_path)

    def test_build_group_library(self):
        manager_json_contents = self.manager_json_contents

        all_group_labels = manager_json_contents.keys()

        contents = manager_json_contents
        all_keys = all_group_labels
        all_systems = [contents[key]["systems"] for key in all_keys]

        all_system_labels = unnest_list(all_systems)

        group_library = PowerCycleManager(
            self.scenario_json_path, self.manager_json_path
        ).group_library

        for group_label in all_group_labels:
            group = group_library[group_label]
            assert type(group) == PowerCycleGroup

            system_library = group.system_library
            keys_in_library = system_library.keys()
            for key in keys_in_library:
                assert key in all_system_labels

    @pytest.mark.parametrize("load_type", ["active", "reactive"])
    def test_build_pulseload_of_type(self, load_type):
        """
        ax = tools_testkit.prepare_figure(load_type)

        sample = self.construct_sample()
        pulseload = sample._build_pulseload_of_type(load_type)

        # all_phaseloads = pulseload.phaseload_set
        # import pprint
        # assert 0

        pulseload_name = f"PulseLoad for {load_type!r} loads"
        pulseload.name = pulseload_name
        ax, _ = pulseload.plot(
            ax=ax,
            detailed=True,
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()
        """
        pass

    # def test_net_active(self):
    #     """
    #     No new functionality to be tested.
    #     """
    #     sample = self.construct_sample()
    #     assert hasattr(sample, "net_active")

    # def test_net_reactive(self):
    #     """
    #     No new functionality to be tested.
    #     """
    #     sample = self.construct_sample()
    #     assert hasattr(sample, "net_reactive")

    def test_plot(self):
        sample = self.construct_sample()

        figure_title = "PowerCycleManager"
        ax = tools_testkit.prepare_figure(figure_title)

        ax, _ = sample.plot(ax=ax)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()
