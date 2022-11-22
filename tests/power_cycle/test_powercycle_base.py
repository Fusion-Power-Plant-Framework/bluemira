# from pprint import pprint

# import matplotlib.pyplot as plt
import pytest

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.base import PowerCycleABC  # PowerCycleError,; NetPowerABC,

bluemira_print("Test Power Cycle Base Classes")


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
        with pytest.raises(TypeError):
            wrong_sample = self.SampleConcreteClass(int_name)
            name = wrong_sample.name
            print(name.__class__)

    def test_validate_list(self):
        right_sample = self._create_sample()
        test_arguments = [
            None,
            1.2,
            "cat, mouse",
            [1, 2, 3, 4],
        ]
        for argument in test_arguments:
            validated_argument = right_sample._validate_list(argument)
            assert isinstance(validated_argument, list)

    # def test_validate():
