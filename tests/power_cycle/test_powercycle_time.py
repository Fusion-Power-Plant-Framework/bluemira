from pprint import pformat

import pytest

import bluemira.base.constants as constants
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.base import PowerCycleABCError
from bluemira.power_cycle.time import (  # BOPPhaseDependency,; BOPPhase,
    PowerCyclePhase,
    PowerCyclePhaseError,
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
        sample = PowerCyclePhase(sample_name, sample_breakdown)
        self.sample = sample

    def test_validate_breakdown(self):
        sample = self.sample
        name = sample.name
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
