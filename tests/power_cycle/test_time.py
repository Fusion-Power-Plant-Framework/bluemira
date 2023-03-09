# COPYRIGHT PLACEHOLDER

import copy

import pytest

from bluemira.power_cycle.errors import PowerCyclePhaseError, ScenarioBuilderError
from bluemira.power_cycle.time import (
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleScenario,
    ScenarioBuilder,
)
from tests.power_cycle.kits_for_tests import TimeTestKit, assert_value_is_nonnegative

time_testkit = TimeTestKit()


class TestPowerCyclePhase:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_breakdowns,
        ) = time_testkit.inputs_for_phase()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            breakdown = sample_breakdowns[s]
            sample = PowerCyclePhase(name, breakdown)
            all_samples.append(sample)
        self.sample_breakdowns = sample_breakdowns
        self.all_samples = all_samples

    breakdown_arguments = [
        [None, None, None, None],
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [1.1, 2.2, 3.3, 4.4],
        ["1", "2", "3", "4"],
    ]

    @pytest.mark.parametrize("test_keys", breakdown_arguments)
    @pytest.mark.parametrize("test_values", breakdown_arguments)
    def test_validate_breakdown(self, test_keys, test_values):
        name = "Name for dummy sample"
        breakdown = dict(zip(test_keys, test_values))
        possible_errors = (TypeError, ValueError, PowerCyclePhaseError)
        try:
            sample = PowerCyclePhase(name, breakdown)
        except possible_errors:

            str_keys = [isinstance(k, str) for k in test_keys]
            all_keys_are_str = all(str_keys)
            nonnegative_errors = (TypeError, ValueError)
            nonstr_keys_errors = PowerCyclePhaseError
            if all_keys_are_str:
                with pytest.raises(nonnegative_errors):
                    sample = PowerCyclePhase(name, breakdown)
            else:
                with pytest.raises(nonstr_keys_errors):
                    sample = PowerCyclePhase(name, breakdown)


class TestPowerCyclePulse:
    def setup_method(self):
        (
            _,
            sample_phases,
        ) = time_testkit.inputs_for_pulse()

        name = "Pulse example"
        phase_set = sample_phases
        pulse = PowerCyclePulse(name, phase_set)
        self.sample_phases = sample_phases
        self.sample = pulse

    def test_validate_phase_set(self):
        sample_phases = self.sample_phases
        for phase in sample_phases:
            phase_set = PowerCyclePulse._validate_phase_set(phase)
            individual_phase_becomes_list = isinstance(phase_set, list)
            assert individual_phase_becomes_list
        phase_set = PowerCyclePulse._validate_phase_set(sample_phases)
        phase_set_becomes_list = isinstance(phase_set, list)
        assert phase_set_becomes_list


class TestPowerCycleScenario:
    def setup_method(self):
        (
            _,
            sample_pulses,
        ) = time_testkit.inputs_for_scenario()

        name = "Scenario example"
        pulse_set = sample_pulses
        timeline = PowerCycleScenario(name, pulse_set)
        self.sample_pulses = pulse_set
        self.sample = timeline

    def test_validate_pulse_set(self):
        sample_pulses = self.sample_pulses
        for pulse in sample_pulses:
            pulse_set = PowerCycleScenario._validate_pulse_set(pulse)
            individual_pulse_becomes_list = isinstance(pulse_set, list)
            assert individual_pulse_becomes_list
        pulse_set = PowerCycleScenario._validate_pulse_set(sample_pulses)
        pulse_set_becomes_list = isinstance(pulse_set, list)
        assert pulse_set_becomes_list


class TestScenarioBuilder:
    def setup_method(self):
        tested_class = ScenarioBuilder
        self.tested_class = tested_class

        scenario_json_path = time_testkit.scenario_json_path
        self.scenario_json_path = scenario_json_path

        scenario_json_contents = time_testkit.inputs_for_builder()
        self.scenario_json_contents = scenario_json_contents

        all_class_attr = [
            "_config_dict",
            "_scenario_dict",
            "_pulse_dict",
            "_phase_dict",
            "_breakdown_dict",
        ]
        self.all_class_attr = all_class_attr

        highest_level_json_keys = [
            "scenario",
            "pulse-library",
            "phase-library",
            "breakdown-library",
        ]
        self.highest_level_json_keys = highest_level_json_keys

    def run_validate_config(self):
        tested_class = self.tested_class
        scenario_json_contents = self.scenario_json_contents
        (
            scenario_config,
            pulse_config,
            phase_config,
            breakdown_config,
        ) = tested_class._validate_config(scenario_json_contents)

        return (
            scenario_config,
            pulse_config,
            phase_config,
            breakdown_config,
        )

    def run_build_breakdown_library(self):
        tested_class = self.tested_class
        (
            scenario_config,
            pulse_config,
            phase_config,
            breakdown_config,
        ) = self.run_validate_config()

        breakdown_library = tested_class._build_breakdown_library(
            breakdown_config,
        )
        return (
            breakdown_library,
            phase_config,
            pulse_config,
            scenario_config,
        )

    def run_build_phase_library(self):
        tested_class = self.tested_class
        (
            breakdown_library,
            phase_config,
            pulse_config,
            scenario_config,
        ) = self.run_build_breakdown_library()

        phase_library = tested_class._build_phase_library(
            phase_config,
            breakdown_library,
        )
        return (
            phase_library,
            pulse_config,
            scenario_config,
        )

    def run_build_pulse_library(self):
        tested_class = self.tested_class
        (
            phase_library,
            pulse_config,
            scenario_config,
        ) = self.run_build_phase_library()

        pulse_library = tested_class._build_pulse_library(
            pulse_config,
            phase_library,
        )
        return (
            pulse_library,
            scenario_config,
        )

    def run_build_scenario(self):
        tested_class = self.tested_class
        (
            pulse_library,
            scenario_config,
        ) = self.run_build_pulse_library()

        scenario = tested_class._build_scenario(
            scenario_config,
            pulse_library,
        )
        return scenario

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_class_attributes(self):
        all_class_attr = self.all_class_attr
        valid_config_json_values = [
            dict,
            str,
            list,
        ]
        tested_class = self.tested_class
        for attr in all_class_attr:
            assert hasattr(tested_class, attr)

            attr_in_tested_class = getattr(tested_class, attr)
            attr_is_dict = isinstance(attr_in_tested_class, dict)
            assert attr_is_dict

            keys_in_attr = attr_in_tested_class.keys()
            for key in keys_in_attr:
                value = attr_in_tested_class[key]
                assert value in valid_config_json_values

    def test_constructor(self):
        scenario_json_path = self.scenario_json_path
        sample = ScenarioBuilder(scenario_json_path)
        assert isinstance(sample, ScenarioBuilder)

        scenario = sample.scenario
        scenario_duration = scenario.duration
        pulse_set = scenario.pulse_set
        for pulse in pulse_set:
            pulse_duration = pulse.duration
            phase_set = pulse.phase_set
            for phase in phase_set:
                phase_duration = phase.duration
                assert phase_duration == sum(phase.durations_list)
            assert pulse_duration == sum(pulse.durations_list)
        assert scenario_duration == sum(scenario.durations_list)

    @staticmethod
    def copy_dict_with_wrong_key(right_dict, key_to_substitute):
        """
        Make deep copy of dictionary, but substitute one key
        by the 'wrong_key' string.
        """
        wrong_dict = copy.deepcopy(right_dict)
        wrong_dict["wrong_key"] = wrong_dict.pop(key_to_substitute)
        return wrong_dict

    @staticmethod
    def copy_dict_with_wrong_value(right_dict, key, value_to_substitute):
        """
        Make deep copy of dictionary, but substitute the value in 'key'
        by 'value_to_substitute'.
        """
        wrong_dict = copy.deepcopy(right_dict)
        wrong_dict[key] = value_to_substitute
        return wrong_dict

    def test_validate_dict(self):
        all_class_attr = self.all_class_attr
        scenario_json_contents = self.scenario_json_contents
        valid_highest_level_json_keys = self.highest_level_json_keys

        tested_class = self.tested_class
        for valid_key in valid_highest_level_json_keys:
            key_contents = scenario_json_contents[valid_key]

            necessary_dict = None
            text_before_hiphen = valid_key.split("-", 1)[0]
            for class_attr in all_class_attr:
                if text_before_hiphen in class_attr:
                    necessary_dict = getattr(tested_class, class_attr)

            wrong_necessary_dict = self.copy_dict_with_wrong_key(
                necessary_dict,
                "name",
            )

            recursion_is_needed = text_before_hiphen != "scenario"
            if recursion_is_needed:

                validated_contents = tested_class._validate_subdict(
                    key_contents,
                    necessary_dict,
                )
                with pytest.raises(ScenarioBuilderError):
                    validated_contents = tested_class._validate_subdict(
                        key_contents,
                        wrong_necessary_dict,
                    )

            else:

                validated_contents = tested_class._validate_dict(
                    key_contents,
                    necessary_dict,
                )
                with pytest.raises(ScenarioBuilderError):
                    validated_contents = tested_class._validate_dict(
                        key_contents,
                        wrong_necessary_dict,
                    )

                wrong_contents = self.copy_dict_with_wrong_value(
                    key_contents,
                    "name",
                    ["not", "a", "string"],
                )
                with pytest.raises(ScenarioBuilderError):
                    validated_contents = tested_class._validate_dict(
                        wrong_contents,
                        necessary_dict,
                    )

            validated_contents_is_dict = isinstance(validated_contents, dict)
            assert validated_contents_is_dict

    def test_validate_subdict(self):
        """
        No new functionality to be tested.
        """
        tested_class = self.tested_class
        assert callable(tested_class._validate_subdict)

    def test_validate_config(self):
        tested_class = self.tested_class
        scenario_json_contents = self.scenario_json_contents
        valid_highest_level_json_keys = self.highest_level_json_keys
        all_configs = self.run_validate_config()

        for config in all_configs:
            config_is_dict = isinstance(config, dict)
            assert config_is_dict

        for valid_key in valid_highest_level_json_keys:
            wrong_contents = self.copy_dict_with_wrong_key(
                scenario_json_contents,
                valid_key,
            )
            with pytest.raises(ScenarioBuilderError):
                wrong_configs = tested_class._validate_config(wrong_contents)

    @pytest.mark.parametrize(
        "test_module",
        [
            "None",
            "equilibria",
            "pumping",
            "not-implemented_importer",
        ],
    )
    def test_import_duration(self, test_module):
        tested_class = self.tested_class
        (_, _, _, breakdown_config) = self.run_validate_config()

        element_with_module_none = "plb"
        if test_module == "None":
            element_config = breakdown_config[element_with_module_none]
            variables_map = element_config["variables_map"]
            duration = tested_class.import_duration(
                test_module,
                variables_map,
            )
            assert_value_is_nonnegative(duration)

        elif test_module == "not-implemented_importer":
            variable_map = None
            with pytest.raises(ScenarioBuilderError):
                duration = tested_class.import_duration(
                    test_module,
                    variable_map,
                )

        else:
            duration_comes_from_importer_class = True
            assert duration_comes_from_importer_class

    def test_build_breakdown_library(self):
        (breakdown_library, _, _, _) = self.run_build_breakdown_library()
        assert type(breakdown_library) is dict

        library_items = breakdown_library.items()
        for (element, value) in library_items:
            assert type(element) is str

            assert len(value) == 1
            assert type(value) is dict

            value_items = value.items()
            for (key, duration) in value_items:
                assert type(key) is str
                assert_value_is_nonnegative(duration)

    def test_build_phase_breakdown(self):
        tested_class = self.tested_class
        (breakdown_library, _, _, _) = self.run_build_breakdown_library()
        breakdown_list = breakdown_library.keys()

        test_operators = ["&", "|"]
        for operator in test_operators:
            phase_breakdown = tested_class._build_phase_breakdown(
                breakdown_library, breakdown_list, operator
            )
            assert type(phase_breakdown) == dict

            if operator == "&":
                assert len(phase_breakdown) != 1
            elif operator == "|":
                assert len(phase_breakdown) == 1
            else:
                operator_test_implemented = False
                assert operator_test_implemented

    def test_build_phase_library(self):
        (phase_library, _, _) = self.run_build_phase_library()
        assert type(phase_library) is dict

        library_items = phase_library.items()
        for (key, value) in library_items:
            assert type(key) is str
            assert type(value) is PowerCyclePhase

    def test_build_time_set(self):
        """
        No new functionality to be tested.
        """
        tested_class = self.tested_class
        assert callable(tested_class._build_time_set)

    def test_build_pulse_library(self):
        (pulse_library, _) = self.run_build_pulse_library()
        assert type(pulse_library) is dict

        library_items = pulse_library.items()
        for (key, value) in library_items:
            assert type(key) is str
            assert type(value) is PowerCyclePulse

    def test_build_scenario(self):
        scenario = self.run_build_scenario()
        assert type(scenario) is PowerCycleScenario
