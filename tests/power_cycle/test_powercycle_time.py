# COPYRIGHT PLACEHOLDER

import functools

import pytest

import bluemira.base.constants as constants
from bluemira.power_cycle.errors import PowerCycleABCError, PowerCyclePhaseError
from bluemira.power_cycle.time import (
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleTimeline,
)


@functools.lru_cache(maxsize=1)
def inputs_for_phase():
    """
    Function to create inputs for PowerCyclePhase testing.
    The lists 'input_names' and 'input_breakdowns' must have the same
    length.
    """
    input_names = [
        "Dwell",
        "Transition between dwell and flat-top",
        "Flat-top",
        "Transition between flat-top and dwell",
    ]
    input_breakdowns = [
        {
            "CS-recharge + pumping": constants.raw_uc(10, "minute", "second"),
        },
        {
            "ramp-up": 157,
            "heating": 19,
        },
        {"burn": constants.raw_uc(2, "hour", "second")},
        {
            "cooling": 123,
            "ramp-down": 157,
        },
    ]
    assert len(input_names) == len(input_breakdowns)
    n_inputs = len(input_names)

    return (
        n_inputs,
        input_names,
        input_breakdowns,
    )


class TestPowerCyclePhase:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_breakdowns,
        ) = inputs_for_phase()

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


def inputs_for_pulse():
    """
    Function to create inputs for PowerCyclePulse testing.
    """
    (
        n_inputs,
        input_names,
        input_breakdowns,
    ) = inputs_for_phase()

    input_phases = []
    for i in range(n_inputs):
        name = input_names[i]
        breakdown = input_breakdowns[i]
        phase = PowerCyclePhase(name, breakdown)
        input_phases.append(phase)

    return (
        n_inputs,
        input_phases,
    )


class TestPowerCyclePulse:
    def setup_method(self):
        (
            n_samples,
            sample_phases,
        ) = inputs_for_pulse()

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


def inputs_for_timeline():
    """
    Function to create inputs for PowerCycleTimeline testing.
    """
    (
        _,
        input_phases,
    ) = inputs_for_pulse()

    n_pulses = 10

    input_pulses = []
    for p in range(n_pulses):
        name = "Pulse " + str(p)
        phase = input_phases
        pulse = PowerCyclePulse(name, phase)
        input_pulses.append(pulse)

    return (
        n_pulses,
        input_pulses,
    )


class TestPowerCycleTimeline:
    def setup_method(self):
        (
            n_samples,
            sample_pulses,
        ) = inputs_for_timeline()

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
