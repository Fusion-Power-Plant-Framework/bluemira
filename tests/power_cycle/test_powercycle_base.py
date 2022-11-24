from pprint import pformat

import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.base import PowerCycleABC, PowerCycleABCError

# import matplotlib.pyplot as plt


def script_title():
    return "Test Power Cycle 'base'"


class TestPowerCycleABCError:
    pass


def test_PowerCycleABCError():
    with pytest.raises(PowerCycleABCError):
        raise PowerCycleABCError(
            None,
            "Some error in the 'PowerCycleABC' class.",
        )


class TestPowerCycleABC:
    class SampleConcreteClass(PowerCycleABC):  # Inner Class
        pass

    def setup_method(self):
        sample = self.SampleConcreteClass("A sample instance name")
        another_sample = self.SampleConcreteClass("Another name")

        test_arguments = [
            None,
            1.2,
            70,
            "some string",
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            sample,
            another_sample,
        ]

        self.sample = sample
        self.test_arguments = test_arguments

    # ------------------------------------------------------------------
    #  TESTS
    # ------------------------------------------------------------------

    def test_constructor(self):
        all_arguments = self.test_arguments
        for argument in all_arguments:
            if isinstance(argument, str):
                right_sample = self.SampleConcreteClass(argument)
                assert isinstance(right_sample, PowerCycleABC)
            else:
                with pytest.raises(PowerCycleABCError):
                    wrong_sample = self.SampleConcreteClass(argument)
                    bluemira_debug(
                        f"""
                        {script_title()} (PowerCycleABC constructor)

                        Name given to sample:
                        {pformat(wrong_sample.name)}

                        Class of name given to sample:
                        {pformat(wrong_sample.name.__class__)}
                        """
                    )

    def test_validate_list(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            validated_argument = sample.validate_list(argument)
            assert isinstance(validated_argument, list)

    def test_validate_class(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            if isinstance(argument, self.SampleConcreteClass):
                validated_argument = sample.validate_class(argument)
                assert validated_argument == argument
            else:
                with pytest.raises(PowerCycleABCError):
                    validated_argument = sample.validate_class(argument)
