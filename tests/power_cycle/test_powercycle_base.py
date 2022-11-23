from pprint import pformat

import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.base import PowerCycleABC, PowerCycleError  # NetPowerABC,

# import matplotlib.pyplot as plt


def script_title():
    return "Test Power Cycle Base"


def test_PowerCycleError():
    with pytest.raises(PowerCycleError):
        raise PowerCycleError("Some error in the Power Cycle module.")


class Test_PowerCycleABC:
    class SampleConcreteClass(PowerCycleABC):  # Inner Class
        pass

    def _create_sample(self):
        str_name = "Sample instance name"
        right_sample = self.SampleConcreteClass(str_name)
        return right_sample

    def test_constructor(self):

        right_sample = self._create_sample()
        assert isinstance(right_sample, PowerCycleABC)

        int_name = 17
        with pytest.raises(PowerCycleError):
            wrong_sample = self.SampleConcreteClass(int_name)
            wrong_name = wrong_sample.name
            bluemira_debug(
                f"""
                {script_title()} (class constructor)

                Name given to sample:
                {pformat(wrong_name)}

                Class of name given to sample:
                {pformat(wrong_name.__class__)}
                """
            )

    def test_validate_list(self):
        right_sample = self._create_sample()
        test_arguments = [
            None,
            1.2,
            "cat, mouse",
            [1, 2, 3, 4],
        ]
        for argument in test_arguments:
            validated_argument = right_sample.validate_list(argument)
            assert isinstance(validated_argument, list)

    # def test_validate():
