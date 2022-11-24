from pprint import pformat

import matplotlib.pyplot as plt
import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.net import (
    NetPowerABC,
    NetPowerABCError,
    PowerData,
    PowerDataError,
)


def script_title():
    return "Test Power Cycle 'net'"


def test_NetPowerABCError():
    with pytest.raises(NetPowerABCError):
        raise NetPowerABCError(
            None,
            "Some error in the 'NetPowerABC' class.",
        )


class TestNetPowerABC:
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
            bluemira_debug(
                f"""
                {script_title()} (NetPowerABC._validate_n_points)

                Argument:
                {pformat(argument)}
                """
            )
            if not argument:
                default_n_points = sample._n_points
                validated_arg = sample._validate_n_points(argument)
                assert validated_arg == default_n_points

            elif (type(argument) is int) or (type(argument) is float):
                validated_arg = sample._validate_n_points(argument)
                assert isinstance(validated_arg, int)

            else:
                with pytest.raises(NetPowerABCError):
                    validated_arg = sample._validate_n_points(argument)

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


def test_PowerDataError():
    with pytest.raises(PowerDataError):
        raise PowerDataError(
            None,
            "Some error in the 'PowerData' class.",
        )


class TestPowerData:
    def setup_method(self):
        sample_name = "Sample PowerData Instance"
        sample_time = [0, 4, 7, 8]
        sample_data = [6, 9, 7, 8]
        sample = PowerData(sample_name, sample_time, sample_data)
        self.sample = sample

    def test_is_increasing(self):
        sample = self.sample
        increasing_list_example = sample.time
        non_increasing_list_example = sample.data
        bluemira_debug(
            f"""
            {script_title()} (PowerData._is_increasing)

            Example of increasing list:
            {pformat(increasing_list_example)}

            Example of non-increasing list:
            {pformat(non_increasing_list_example)}
            """
        )
        assert sample._is_increasing(increasing_list_example)
        with pytest.raises(PowerDataError):
            sample._is_increasing(non_increasing_list_example)

    def test_sanity(self):
        sample = self.sample
        name = sample.name
        list_of_given_length = sample.time
        list_shorter = list_of_given_length[:-1]
        list_longer = list_of_given_length + [10]
        all_lists = [
            list_of_given_length,
            list_shorter,
            list_longer,
        ]
        for time in all_lists:
            for data in all_lists:

                length_time = len(time)
                length_data = len(data)
                bluemira_debug(
                    f"""
                    {script_title()} (PowerData._sanity)

                    List used as time vector:
                    {pformat(time)}

                    List used as data vector:
                    {pformat(data)}
                    """
                )

                if length_time == length_data:
                    test_instance = PowerData(name, time, data)
                    assert isinstance(test_instance, PowerData)

                else:
                    with pytest.raises(PowerDataError):
                        test_instance = PowerData(name, time, data)

    def test_plot(self):
        fig = plt.figure()
        plt.grid()
        sample = self.sample
        all_axes = sample.plot()
        bluemira_debug(
            f"""
            {script_title()} (PowerData._plot)

            All plotted objects:
            {pformat(all_axes)}
            """
        )
        plt.show()  # Run with `pytest --plotting-on` to visualize
