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
from tests.power_cycle.kits_for_tests import TimeTestKit

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

        scenario_json_path = time_testkit.scenario_json_path
        sample = ScenarioBuilder(scenario_json_path)
        self.sample = sample

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
        tested_class = ScenarioBuilder
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
        sample = self.sample
        assert isinstance(sample, ScenarioBuilder)

    def test_validate_dict(self):
        pass

    def test_validate_subdict(self):
        pass

    def test_validate_config(self):
        sample = self.sample
        scenario_json_contents = self.scenario_json_contents

        valid_highest_level_json_keys = [
            "scenario",
            "pulse-library",
            "phase-library",
            "breakdown-library",
        ]

        all_configs = sample._validate_config(scenario_json_contents)
        for config in all_configs:
            config_is_dict = isinstance(config, dict)
            assert config_is_dict

        for valid_key in valid_highest_level_json_keys:
            wrong_contents = copy.deepcopy(scenario_json_contents)
            wrong_contents["wrong_key"] = wrong_contents.pop(valid_key)
            with pytest.raises(ScenarioBuilderError):
                wrong_configs = sample._validate_config(wrong_contents)

    def test_import_duration(self):
        """
        importer_class = self.importer_class
        import_parameters = self.import_parameters
        available_modules = import_parameters.keys()

        for module in available_modules:
            variable_map = import_parameters[module]
            duration = importer_class.duration_from_module(module, variable_map)
            self.assert_value_is_nonnegative(duration)

        unavailable_module = "not-implemented"
        example_variable_map = dict()
        with pytest.raises(PowerCycleImporterError):
            duration = importer_class.duration_from_module(
                unavailable_module,
                example_variable_map,
            )
        """
        pass

    def test_build_breakdown_library(self):
        """
        def test_split_pulse_config(self):
        all_libraries = self.config_libraries
        for library in all_libraries:
            library_is_dict = type(library) == dict
            assert library_is_dict

        pulse_config = self.pulse_config
        wrong_config = copy.deepcopy(pulse_config)
        wrong_config.pop("pulse-library")
        with pytest.raises(PowerCycleManagerError):
            (
                pulse_library,
                phase_library,
                breakdown_library,
            ) = PowerCycleManager._split_pulse_config(wrong_config)
        """

    pass

    def test_build_phase_breakdown(self):
        pass

    def test_build_phase_library(self):
        pass

    def test_build_phase_set(self):
        pass

    def test_build_pulse_library(self):
        pass

    def test_build_scenario(self):
        pass
        # every element of libraries must have correct class
