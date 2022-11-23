from pprint import pformat

import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.net import NetPowerABC, NetPowerABCError


def script_title():
    return "Test Power Cycle 'net'"


def test_NetPowerABCError():
    with pytest.raises(NetPowerABCError):
        raise NetPowerABCError(
            None,
            "Some error in the Power Cycle module.",
        )


class Test_NetPowerABC:
    class SampleConcreteClass(NetPowerABC):  # Inner Class
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
        self.another_sample = another_sample
        self.test_arguments = test_arguments

    def test_validate_n_points(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:

            if not argument:
                default_n_points = sample._n_points
                validated_arg = sample._validate_n_points(argument)
                assert validated_arg == default_n_points

            elif isinstance(argument, int) or isinstance(argument, float):
                validated_arg = sample._validate_n_points(argument)
                assert isinstance(validated_arg, int)

            else:
                with pytest.raises(NetPowerABCError):
                    validated_arg = sample._validate_n_points(argument)
                    bluemira_debug(
                        f"""
                        {script_title()} ('_validate_n_points')

                        Argument:
                        {pformat(argument)}
                        """
                    )

    def test_make_secondary_in_plot(self):
        one_sample = self.sample
        another_sample = self.another_sample

        another_sample._make_secondary_in_plot()
        attr_to_compare = [
            "_text_index",
            "_plot_kwargs",
        ]

        for attribute in attr_to_compare:
            one_attr = getattr(one_sample, attribute)
            another_attr = getattr(another_sample, attribute)
            assert one_attr != another_attr
