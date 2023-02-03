# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.base import NetPowerABC, PowerCycleABC, PowerCycleTimeABC
from bluemira.power_cycle.errors import (  # PowerCycleTimeABCError,
    NetPowerABCError,
    PowerCycleABCError,
)


class TestPowerCycleABC:
    class SampleConcreteClass(PowerCycleABC):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        pass

    def setup_method(self):
        sample = self.SampleConcreteClass("A sample instance name")
        another_sample = self.SampleConcreteClass("Another name")

        test_arguments = [
            None,
            1.2,
            -1.2,
            70,
            -70,
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

    def test_validate_list(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            validated_argument = sample.validate_list(argument)
            assert isinstance(validated_argument, list)

    def test_validate_nonnegative(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            is_integer = isinstance(argument, int)
            is_float = isinstance(argument, float)
            is_numerical = is_integer or is_float
            if is_numerical:
                is_nonnegative = argument >= 0
                if is_nonnegative:
                    out = sample.validate_nonnegative(argument)
                    assert out == argument
                else:
                    with pytest.raises(PowerCycleABCError):
                        out = sample.validate_nonnegative(argument)
            else:
                with pytest.raises(PowerCycleABCError):
                    out = sample.validate_nonnegative(argument)

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


class TestNetPowerABC:
    class SampleConcreteClass(NetPowerABC):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

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

            elif (type(argument) is int) or (type(argument) is float):
                validated_arg = sample._validate_n_points(argument)
                assert isinstance(validated_arg, int)

            else:
                with pytest.raises(NetPowerABCError):
                    validated_arg = sample._validate_n_points(argument)

    @pytest.mark.parametrize(
        "attribute",
        [
            "_text_index",
            "_plot_kwargs",
        ],
    )
    def test_make_secondary_in_plot(self, attribute):
        one_sample = self.sample
        another_sample = self.another_sample

        another_sample._make_secondary_in_plot()

        one_attr = getattr(one_sample, attribute)
        another_attr = getattr(another_sample, attribute)
        assert one_attr != another_attr
