# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.base import (
    PowerCycleABC,
    PowerCycleImporterABC,
    PowerCycleLoadABC,
    PowerCycleTimeABC,
)
from bluemira.power_cycle.tools import (
    validate_list,
    validate_nonnegative,
    validate_vector,
)
from tests.power_cycle.kits_for_tests import ToolsTestKit

tools_testkit = ToolsTestKit()


class TestPowerCycleABC:
    tested_class_super = None
    tested_class_super_error = None
    tested_class = PowerCycleABC

    class SampleConcreteClass(tested_class):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        class_str = "a string"
        class_num = 10
        class_dict = {"key_0": None, "key_1": 1, "key_2": "2"}

        def __init__(self, name: str, label=None):
            super().__init__(name, label=label)
            self.instance_str = "another string"
            self.instance_num = 11
            self.instance_dict = self.class_dict["key_0"]

    def setup_method(self):
        sample = self.SampleConcreteClass("A sample instance name")
        another_sample = self.SampleConcreteClass("Another name")

        test_arguments = tools_testkit.build_list_of_example_arguments()
        test_arguments.append(sample)
        test_arguments.append(another_sample)

        self.sample = sample
        self.another_sample = another_sample
        self.test_arguments = test_arguments

    def test_validate_class(self):
        for argument in self.test_arguments:
            if isinstance(argument, self.SampleConcreteClass):
                validated_argument = self.sample.validate_class(argument)
                assert validated_argument == argument
            else:
                with pytest.raises(TypeError):
                    validated_argument = self.sample.validate_class(argument)

    def test_equality(self):
        sample = self.sample
        another_sample = self.another_sample

        equal_even_with_different_names_and_labels = sample == another_sample
        assert equal_even_with_different_names_and_labels

        sample.instance_str = 100000
        assert sample != another_sample


class TestPowerCycleTimeABC:
    tested_class_super = PowerCycleABC
    tested_class = PowerCycleTimeABC

    class SampleConcreteClass(tested_class):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        pass

    def setup_method(self):
        name = "A sample instance name"
        durations_list = [0, 1, 5, 10]
        sample = self.SampleConcreteClass(name, durations_list)

        test_arguments = tools_testkit.build_list_of_example_arguments()
        test_arguments.append(sample)

        self.sample = sample
        self.test_arguments = test_arguments

    def test_constructor(self):
        test_arguments = self.test_arguments
        name = "instance being created in constructor test"
        possible_errors = (TypeError, ValueError)
        for argument in test_arguments:
            argument_in_list = validate_list(argument)
            try:
                test_instance = self.SampleConcreteClass(name, argument)
                assert test_instance.duration == sum(argument_in_list)

            except possible_errors:
                with pytest.raises(possible_errors):
                    if argument:
                        for value in argument_in_list:
                            validate_nonnegative(value)
                    else:
                        validate_nonnegative(argument)

    def test_duration(self):
        sample = self.sample
        durations_list = sample.durations_list
        total_duration = sample.duration
        assert total_duration == sum(durations_list)


class TestPowerCycleLoadABC:
    tested_class_super = PowerCycleABC
    tested_class = PowerCycleLoadABC

    class SampleConcreteClass(tested_class):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        @property
        def intrinsic_time(self):
            """
            Define concrete version of abstract property.
            """
            pass

    def setup_method(self):
        sample = self.SampleConcreteClass("A sample instance name")
        another_sample = self.SampleConcreteClass("Another name")

        test_arguments = tools_testkit.build_list_of_example_arguments()
        test_arguments.append(sample)
        test_arguments.append(another_sample)

        self.sample = sample
        self.another_sample = another_sample
        self.test_arguments = test_arguments

    def test_validate_n_points(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            arg_is_bool = type(argument) is bool

            arg_type = type(argument)
            arg_is_numeric = (arg_type is int) or (arg_type is float)
            if arg_is_numeric:
                arg_is_nonnegative = argument > 0
            else:
                arg_is_nonnegative = False

            if not argument:
                default_n_points = sample._n_points
                validated_arg = sample._validate_n_points(argument)
                assert validated_arg == default_n_points
            elif arg_is_bool or arg_is_nonnegative:
                validated_arg = sample._validate_n_points(argument)
                assert isinstance(validated_arg, int)
            else:
                with pytest.raises((ValueError, TypeError)):
                    validated_arg = sample._validate_n_points(argument)

    @pytest.mark.parametrize("refinement_order", range(10))
    def test_refine_vector(self, refinement_order):
        sample = self.sample
        test_arguments = self.test_arguments

        possible_errors = (TypeError, ValueError)
        for argument in test_arguments:
            try:
                argument = validate_vector(argument)
            except possible_errors:
                return

            numeric_list = argument
            refined_list = sample._refine_vector(
                numeric_list,
                refinement_order,
            )
            numeric_list_set = set(numeric_list)
            refined_time_set = set(refined_list)
            assert numeric_list_set.issubset(refined_time_set)

            numeric_list_length = len(numeric_list)
            number_of_segments = numeric_list_length - 1
            points_in_refined = number_of_segments * refinement_order + 1

            refined_list_length = len(refined_list)
            assert points_in_refined == refined_list_length

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
