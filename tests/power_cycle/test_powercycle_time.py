# COPYRIGHT PLACEHOLDER

import pytest

import bluemira.base.constants as constants
from bluemira.power_cycle.errors import (  # PowerCycleTimeABCError,; PowerCyclePulseError,; PowerCycleTimelineError,
    BOPPhaseError,
    PowerCycleABCError,
    PowerCyclePhaseError,
)
from bluemira.power_cycle.time import (  # BOPPulse,; BOPTimeline,
    BOPPhase,
    BOPPhaseDependency,
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleTimeline,
)

# from bluemira.power_cycle.base import PowerCycleTimeABC


# from bluemira.power_cycle.tools import (
#    adjust_2d_graph_ranges,
#    validate_axes,
# )


def example_phase_inputs():
    phase_name = "Transition between dwell and flat-top"
    phase_breakdown = {
        "pump-down": constants.raw_uc(10, "minute", "second"),
        "ramp-up": constants.raw_uc(5.2, "minute", "second"),
        "heating": constants.raw_uc(1.4, "minute", "second"),
    }
    return phase_name, phase_breakdown


class TestPowerCyclePhase:
    def setup_method(self):
        phase_name, phase_breakdown = example_phase_inputs()
        self.sample_name = phase_name
        self.sample_breakdown = phase_breakdown

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
        name = self.sample_name
        breakdown = dict(zip(test_keys, test_values))

        try:
            sample = PowerCyclePhase(name, breakdown)
        except (PowerCyclePhaseError, PowerCycleABCError):

            str_keys = [isinstance(k, str) for k in test_keys]
            all_keys_are_str = all(str_keys)
            if all_keys_are_str:
                # Error must be of nonnegative values
                with pytest.raises(PowerCycleABCError):
                    sample = PowerCyclePhase(name, breakdown)
            else:
                # Error must be of non-string dictionary keys
                with pytest.raises(PowerCyclePhaseError):
                    sample = PowerCyclePhase(name, breakdown)

    def test_constructor(self):
        sample_name = self.sample_name
        sample_breakdown = self.sample_breakdown
        sample = PowerCyclePhase(sample_name, sample_breakdown)

        all_durations_in_breakdown = list(sample_breakdown.values())
        total_duration = sum(all_durations_in_breakdown)
        assert sample.duration == total_duration


class TestPowerCyclePulse:
    def setup_method(self):
        phase_names = [
            "phase 1",
            "phase 2",
            "phase 3",
        ]
        phase_breakdowns = [
            {
                "phase 1 - subphase 1": 1,
                "phase 1 - subphase 2": 2,
                "phase 1 - subphase 3": 4,
                "phase 1 - subphase 4": 8,
            },
            {
                "phase 2 - subphase 1": 10,
            },
            {
                "phase 3 - subphase 1": 100,
                "phase 3 - subphase 2": 100,
                "phase 3 - subphase 3": 100,
            },
        ]

        test_phases = []
        n_phases = len(phase_names)
        for p in range(n_phases):

            name = phase_names[p]
            breakdown = phase_breakdowns[p]
            phase = PowerCyclePhase(name, breakdown)
            test_phases.append(phase)

        self.test_phases = test_phases

    def test_validate_phase_set(self):
        all_phases = self.test_phases
        for phase in all_phases:
            phase_set = PowerCyclePulse._validate_phase_set(phase)
            individual_phase_becomes_list = isinstance(phase_set, list)
            assert individual_phase_becomes_list
        phase_set = PowerCyclePulse._validate_phase_set(all_phases)
        phase_set_becomes_list = isinstance(phase_set, list)
        assert phase_set_becomes_list

    def test_constructor(self):
        sample_phases = self.test_phases
        sample_name = "Test instance of PowerCyclePulse"
        sample = PowerCyclePulse(sample_name, sample_phases)
        self.sample = sample


class TestPowerCycleTimeline:
    def setup_method(self):
        pulse_test_class = TestPowerCyclePulse()
        pulse_test_class.setup_method()
        pulse_test_class.test_constructor()
        pulse_sample = pulse_test_class.sample
        number_of_pulses = 10
        test_pulses = [pulse_sample] * number_of_pulses
        self.test_pulses = test_pulses

    def test_validate_pulse_set(self):
        all_pulses = self.test_pulses
        for pulse in all_pulses:
            pulse_set = PowerCycleTimeline._validate_pulse_set(pulse)
            individual_pulse_becomes_list = isinstance(pulse_set, list)
            assert individual_pulse_becomes_list
        pulse_set = PowerCycleTimeline._validate_pulse_set(all_pulses)
        pulse_set_becomes_list = isinstance(pulse_set, list)
        assert pulse_set_becomes_list

    def test_constructor(self):
        sample_pulses = self.test_pulses
        sample_name = "Test instance of PowerCycleTimeline"
        sample = PowerCycleTimeline(sample_name, sample_pulses)
        self.sample = sample


class TestBOPPhaseDependency:
    def test_members(self):
        all_names = [member.name for member in BOPPhaseDependency]
        all_values = [member.value for member in BOPPhaseDependency]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


class TestBOPPhase:
    def setup_method(self):
        phase_name, phase_breakdown = example_phase_inputs()
        self.sample_name = phase_name
        self.sample_breakdown = phase_breakdown

    dependency_arguments = [
        None,
        154,
        "ss",
        "tt",
        BOPPhaseDependency("ss"),
        BOPPhaseDependency("tt"),
    ]

    @pytest.mark.parametrize("test_dependency", dependency_arguments)
    def test_constructor(self, test_dependency):
        name = self.sample_name
        breakdown = self.sample_breakdown
        try:
            sample = BOPPhase(name, breakdown, test_dependency)
        except (BOPPhaseError):
            dependency_class = type(test_dependency)
            dependency_is_valid = dependency_class == BOPPhaseDependency
            assert not dependency_is_valid


class TestBOPPulse:
    pass


class TestBOPTimeline:
    pass
