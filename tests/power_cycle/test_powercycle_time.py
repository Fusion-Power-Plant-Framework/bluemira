# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.errors import PowerCycleABCError, PowerCyclePhaseError
from bluemira.power_cycle.time import (
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleTimeline,
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

        try:
            sample = PowerCyclePhase(name, breakdown)

        except (PowerCyclePhaseError, PowerCycleABCError):

            str_keys = [isinstance(k, str) for k in test_keys]
            all_keys_are_str = all(str_keys)
            if all_keys_are_str:
                # Error informs requirement of nonnegative values
                with pytest.raises(PowerCycleABCError):
                    sample = PowerCyclePhase(name, breakdown)
            else:
                # Error informs requirement of non-string dict keys
                with pytest.raises(PowerCyclePhaseError):
                    sample = PowerCyclePhase(name, breakdown)

    def test_duration(self):
        all_samples = self.all_samples
        for sample in all_samples:
            breakdown = sample.duration_breakdown
            durations_in_breakdown = list(breakdown.values())
            total_duration = sum(durations_in_breakdown)
            assert sample.duration == total_duration


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


class TestPowerCycleTimeline:
    def setup_method(self):
        (
            _,
            sample_pulses,
        ) = time_testkit.inputs_for_timeline()

        name = "Timeline example"
        pulse_set = sample_pulses
        timeline = PowerCycleTimeline(name, pulse_set)
        self.sample_pulses = pulse_set
        self.sample = timeline

    def test_validate_pulse_set(self):
        sample_pulses = self.sample_pulses
        for pulse in sample_pulses:
            pulse_set = PowerCycleTimeline._validate_pulse_set(pulse)
            individual_pulse_becomes_list = isinstance(pulse_set, list)
            assert individual_pulse_becomes_list
        pulse_set = PowerCycleTimeline._validate_pulse_set(sample_pulses)
        pulse_set_becomes_list = isinstance(pulse_set, list)
        assert pulse_set_becomes_list
