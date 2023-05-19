# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.base import PowerCycleTimeABC
from bluemira.power_cycle.errors import (
    PowerCycleError,
    PowerCyclePhaseError,
    PowerCyclePulseError,
    PowerCycleScenarioError,
    ScenarioBuilderError,
)
from bluemira.power_cycle.time import (
    Breakdown,
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleScenario,
    ScenarioBuilder,
    ScenarioBuilderConfig,
)
from bluemira.power_cycle.tools import validate_list
from tests.power_cycle.kits_for_tests import (
    TimeTestKit,
    assert_value_is_nonnegative,
    copy_dict_with_wrong_key,
)

time_testkit = TimeTestKit()


class TestPowerCyclePhase:
    tested_class_super = PowerCycleTimeABC
    tested_class = PowerCyclePhase
    tested_class_error = PowerCyclePhaseError

    def setup_method(self):
        tested_class = self.tested_class

        (
            n_samples,
            sample_names,
            sample_breakdowns,
            sample_labels,
        ) = time_testkit.inputs_for_phase()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            breakdown = sample_breakdowns[s]
            label = sample_labels[s]
            sample = tested_class(name, breakdown.values(), label=label)
            all_samples.append(sample)
        self.sample_breakdowns = sample_breakdowns
        self.all_samples = all_samples

    breakdown_argument_examples = [
        [None, None, None, None],
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [1.1, 2.2, 3.3, 4.4],
        ["1", "2", "3", "4"],
    ]

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("test_keys", breakdown_argument_examples)
    @pytest.mark.parametrize("test_values", breakdown_argument_examples)
    def test_validate_breakdown(self, test_keys, test_values):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        name = "Name for dummy sample"
        breakdown = dict(zip(test_keys, test_values))
        possible_errors = (TypeError, ValueError, tested_class_error)
        try:
            print(breakdown)
            sample = tested_class(name, test_values)
        except possible_errors:
            str_keys = [isinstance(k, str) for k in test_keys]
            all_keys_are_str = all(str_keys)
            nonnegative_errors = (TypeError, ValueError)
            nonstr_keys_errors = tested_class_error
            if all_keys_are_str:
                with pytest.raises(nonnegative_errors):
                    sample = tested_class(name, breakdown)
            else:
                with pytest.raises(nonstr_keys_errors):
                    sample = tested_class(name, breakdown)

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------


class TestPowerCyclePulse:
    tested_class_super = PowerCycleTimeABC
    tested_class = PowerCyclePulse
    tested_class_error = PowerCyclePulseError

    def setup_method(self):
        tested_class = self.tested_class

        (
            _,
            sample_phases,
        ) = time_testkit.inputs_for_pulse()

        name = "Pulse example"
        phase_set = sample_phases
        pulse = tested_class(name, phase_set)
        self.sample_phases = sample_phases
        self.sample = pulse

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_validate_phase_set(self):
        tested_class = self.tested_class

        sample_phases = self.sample_phases
        for phase in sample_phases:
            phase_set = tested_class._validate_phase_set(phase)
            individual_phase_becomes_list = isinstance(phase_set, list)
            assert individual_phase_becomes_list
        phase_set = tested_class._validate_phase_set(sample_phases)
        phase_set_becomes_list = isinstance(phase_set, list)
        assert phase_set_becomes_list

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------

    def test_build_phase_library(self):
        """
        No new functionality to be tested.
        """
        sample = self.sample
        assert callable(sample.build_phase_library)


class TestPowerCycleScenario:
    tested_class_super = PowerCycleTimeABC
    tested_class = PowerCycleScenario
    tested_class_error = PowerCycleScenarioError

    def setup_method(self):
        tested_class = self.tested_class

        (
            _,
            sample_pulses,
        ) = time_testkit.inputs_for_scenario()

        name = "Scenario example"
        pulse_set = sample_pulses
        scenario = tested_class(name, pulse_set)
        self.sample_pulses = pulse_set
        self.sample = scenario

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_validate_pulse_set(self):
        tested_class = self.tested_class

        sample_pulses = self.sample_pulses
        for pulse in sample_pulses:
            pulse_set = tested_class._validate_pulse_set(pulse)
            individual_pulse_becomes_list = isinstance(pulse_set, list)
            assert individual_pulse_becomes_list
        pulse_set = tested_class._validate_pulse_set(sample_pulses)
        pulse_set_becomes_list = isinstance(pulse_set, list)
        assert pulse_set_becomes_list

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------

    def test_build_phase_library(self):
        """
        No new functionality to be tested.
        """
        sample = self.sample
        assert callable(sample.build_phase_library)

    def test_build_pulse_library(self):
        """
        No new functionality to be tested.
        """
        sample = self.sample
        assert callable(sample.build_pulse_library)


class TestScenarioBuilder:
    def setup_method(self):
        self.scenario_json_path = time_testkit.scenario_json_path
        self.scenario_json_contents = time_testkit.inputs_for_builder()

        highest_level_json_keys = [
            "scenario",
            "pulse_library",
            "phase_library",
            "breakdown_library",
        ]
        self.highest_level_json_keys = highest_level_json_keys
        self.sample = ScenarioBuilder(self.scenario_json_path)

    def run_validate_config(self):
        return ScenarioBuilderConfig(**self.scenario_json_contents)

    def test_constructor(self):
        for pulse in self.sample.scenario.pulse_set:
            pulse_duration = pulse.duration
            phase_set = pulse.phase_set
            for phase in phase_set:
                phase_duration = phase.duration
                assert phase_duration == sum(phase.durations_list)
            assert pulse_duration == sum(pulse.durations_list)
        assert self.sample.scenario.duration == sum(self.sample.scenario.durations_list)

    def test_validate_config(self):
        for valid_key in self.highest_level_json_keys:
            wrong_contents = copy_dict_with_wrong_key(
                self.scenario_json_contents,
                valid_key,
            )
            with pytest.raises((KeyError, TypeError)):
                ScenarioBuilderConfig(**wrong_contents)

    @pytest.mark.parametrize(
        "test_module",
        [
            None,
            "equilibria",
            "pumping",
            "not-implemented_importer",
        ],
    )
    def test_import_duration(self, test_module):
        breakdown_config = self.sample.scenario_config._breakdown_library

        if test_module is None:
            duration = self.sample.import_duration(
                test_module,
                breakdown_config["plb"].variables_map,
            )
            assert_value_is_nonnegative(duration)

        elif test_module == "not-implemented_importer":
            with pytest.raises(PowerCycleError):
                duration = self.sample.import_duration(
                    test_module,
                    {},
                )

        else:
            raise NotImplementedError("TODO")

    def test_build_breakdown_library(self):
        breakdown_library = self.sample._breakdown_library
        assert isinstance(breakdown_library, dict)

        for element, value in breakdown_library.items():
            assert isinstance(element, str)
            assert isinstance(value, Breakdown)

            assert isinstance(value.name, str)
            assert_value_is_nonnegative(value.duration)

    # def test_build_phase_breakdown(self):
    #     phase_breakdown = self.sample._phase_library
    #     assert isinstance(phase_breakdown, dict)

    #     test_operators = ["&", "|"]
    #     for operator in test_operators:
    #         if operator == "&":
    #             assert len(phase_breakdown) != 1
    #         elif operator == "|":
    #             assert len(phase_breakdown) == 1
    #         else:
    #             raise NotImplementedError("TODO")

    def test_build_phase_library(self):
        phase_library = self.sample._phase_library
        assert isinstance(phase_library, dict)

        library_items = phase_library.items()
        for key, value in library_items:
            assert isinstance(key, str)
            assert isinstance(value, PowerCyclePhase)

    def test_build_pulse_library(self):
        pulse_library = self.sample._pulse_library
        assert isinstance(pulse_library, dict)

        library_items = pulse_library.items()
        for key, value in library_items:
            assert isinstance(key, str)
            assert isinstance(value, PowerCyclePulse)

    def test_build_scenario(self):
        assert isinstance(self.sample.scenario, PowerCycleScenario)
