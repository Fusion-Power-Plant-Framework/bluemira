from pprint import pformat

import pytest

import bluemira.base.constants as constants
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.base import PowerCycleABCError
from bluemira.power_cycle.time import (  # BOPPhase,
    BOPPhaseDependency,
    PowerCyclePhase,
    PowerCyclePhaseError,
    PowerCyclePulse,
    PowerCyclePulseError,
    PowerCycleTimeABC,
    PowerCycleTimeABCError,
)

# from bluemira.power_cycle.tools import (
#    adjust_2d_graph_ranges,
#    validate_axes,
# )


def script_title():
    return "Test Power Cycle 'time'"


def test_PowerCycleTimeABCError():
    with pytest.raises(PowerCycleTimeABCError):
        raise PowerCycleTimeABCError(
            None,
            "Some error in the 'PowerCycleTimeABC' class.",
        )


class TestPowerCycleTimeABC:
    class SampleConcreteClass(PowerCycleTimeABC):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        pass

    def setup_method(self):
        name = "A sample instance name"
        durations_list = [0, 1, 5, 10]
        sample = self.SampleConcreteClass(name, durations_list)
        self.sample = sample

    def test_constructor(self):
        sample = self.sample
        name = "instance being created in constructor test"
        test_arguments = [
            None,
            1.2,
            -1.2,
            70,
            -70,
            [0, 1, 2, 3, 4],
            [0, -1, -2, -3, -4],
            "some string",
            (0, 1, 2, 3, 4),
            (0, -1, -2, -3, -4),
            sample,
        ]

        for argument in test_arguments:
            bluemira_debug(
                f"""
                {script_title()} (PowerCycleTimeABC constructor)

                Argument:
                {pformat(argument)}
                """
            )

            # If not already, insert argument in a list, for e.g. 'sum'
            argument_in_list = sample.validate_list(argument)
            try:
                test_instance = self.SampleConcreteClass(name, argument)
                assert test_instance.duration == sum(argument_in_list)

            except (PowerCycleABCError):
                with pytest.raises(PowerCycleABCError):
                    if argument:
                        for value in argument_in_list:
                            sample.validate_nonnegative(value)
                    else:
                        sample.validate_nonnegative(argument)


def test_PowerCyclePhaseError():
    with pytest.raises(PowerCyclePhaseError):
        raise PowerCyclePhaseError(
            None,
            "Some error in the 'PowerCyclePhase' class.",
        )


class TestPowerCyclePhase:
    def setup_method(self):
        sample_name = "Transition between dwell and flat-top"
        sample_breakdown = {
            "pump-down": constants.raw_uc(10, "minute", "second"),
            "ramp-up": constants.raw_uc(5.2, "minute", "second"),
            "heating": constants.raw_uc(1.4, "minute", "second"),
        }
        self.sample_name = sample_name
        self.sample_breakdown = sample_breakdown

    def test_validate_breakdown(self):
        name = self.sample_name
        test_arguments = [
            [None, None, None, None],
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [1.1, 2.2, 3.3, 4.4],
            ["1", "2", "3", "4"],
        ]
        for test_keys in test_arguments:
            for test_values in test_arguments:

                breakdown = dict(zip(test_keys, test_values))
                bluemira_debug(
                    f"""
                    {script_title()} (PowerCyclePhase._validate_breakdown)

                    Test keys:
                    {pformat(test_keys)}

                    Test values:
                    {pformat(test_keys)}

                    Test breakdown dictionary:
                    {pformat(breakdown)}
                    """
                )

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
        bluemira_debug(
            f"""
            {script_title()} (PowerCyclePhase constructor)

            Breakdown dictionary:
            {pformat(sample_breakdown)}

            Total duration:
            {pformat(total_duration)}

            Duration calculated by instance:
            {pformat(sample.duration)}
            """
        )
        assert sample.duration == total_duration


def test_PowerCyclePulseError():
    with pytest.raises(PowerCyclePulseError):
        raise PowerCyclePulseError(
            None,
            "Some error in the 'PowerCyclePulse' class.",
        )


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
        bluemira_debug(
            f"""
            {script_title()} (PowerCyclePhase._validate_phase_set)

            All phases:
            {pformat(all_phases)}
            """
        )
        for phase in all_phases:
            phase_set = PowerCyclePulse._validate_phase_set(phase)
            assert isinstance(phase_set, list)
        phase_set = PowerCyclePulse._validate_phase_set(all_phases)
        assert isinstance(phase_set, list)

    def test_constructor(self):
        sample_phases = self.test_phases
        sample_name = "Test instance of PowerCyclePulse"
        sample = PowerCyclePulse(sample_name, sample_phases)
        bluemira_debug(
            f"""
            {script_title()} (PowerCyclePhase._validate_phase_set)

            Sample phases:
            {pformat(sample_phases)}

            Phase duration breakdowns:
            {pformat([p.duration_breakdown for p in sample_phases])}

            Test sample:
            {pformat(sample)}

            Sample durations list (total duration of each phase):
            {pformat(sample.durations_list)}

            Sample duration:
            {pformat(sample.duration)}
            """
        )


class TestBOPPhaseDependency:
    def test_members(self):

        all_names = [member.name for member in BOPPhaseDependency]
        all_values = [member.value for member in BOPPhaseDependency]
        bluemira_debug(
            f"""
            {script_title()} (BOPPhaseDependency)

            All member names:
            {pformat(all_names)}

            All member values:
            {pformat(all_values)}
            """
        )


class TestBOPPhase:
    pass


class TestBOPPulse:
    pass
