# COPYRIGHT PLACEHOLDER

import os
from itertools import combinations

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.tools import (
    FormattedABC,
    FormattedDict,
    FormattedLibrary,
    adjust_2d_graph_ranges,
    convert_string_into_numeric_list,
    copy_dict_without_key,
    read_json,
    remove_characters,
    unique_and_sorted_vector,
    unnest_list,
    validate_axes,
    validate_file,
    validate_list,
    validate_lists_to_have_same_length,
    validate_nonnegative,
    validate_numerical,
    validate_vector,
)
from tests.power_cycle.kits_for_tests import NetManagerTestKit, TimeTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()
manager_testkit = NetManagerTestKit()


class TestValidationTools:
    def setup_method(self):
        test_arguments = tools_testkit.build_list_of_example_arguments()
        self.test_arguments = test_arguments

        (
            format_example,
            dictionary_example,
            subdictionaries_example,
        ) = tools_testkit.build_dictionary_examples()
        self.format_example = format_example
        self.dictionary_example = dictionary_example
        self.subdictionaries_example = subdictionaries_example

        test_file_path = tools_testkit.test_file_path
        self.test_file_path = test_file_path

    def test_validate_list(self):
        all_arguments = self.test_arguments
        for argument in all_arguments:
            validated_argument = validate_list(argument)
            assert isinstance(validated_argument, list)

    def test_validate_numerical(self):
        all_arguments = self.test_arguments
        for argument in all_arguments:
            check_int = isinstance(argument, int)
            check_float = isinstance(argument, float)
            if not (check_int or check_float):
                with pytest.raises(TypeError):
                    argument = validate_numerical(argument)

    def test_validate_nonnegative(self):
        all_arguments = self.test_arguments
        for argument in all_arguments:
            is_integer = isinstance(argument, int)
            is_float = isinstance(argument, float)
            is_numerical = is_integer or is_float
            if is_numerical:
                is_nonnegative = argument >= 0
                if is_nonnegative:
                    out = validate_nonnegative(argument)
                    assert out == argument
                else:
                    with pytest.raises(ValueError):
                        out = validate_nonnegative(argument)
            else:
                with pytest.raises(TypeError):
                    out = validate_nonnegative(argument)

    def test_validate_vector(self):
        """
        No new functionality to be tested.
        """
        assert callable(validate_vector)

    def test_validate_file(self):
        relative_path = self.test_file_path
        absolute_path = os.path.abspath(relative_path)

        return_path = validate_file(relative_path)
        assert return_path == absolute_path

        wrong_path = absolute_path.replace("txt", "doc")
        with pytest.raises(FileNotFoundError):
            return_path = validate_file(wrong_path)

    @pytest.mark.parametrize("max_n_arguments", [10])
    def test_validate_lists_to_have_same_length(self, max_n_arguments):
        test_arguments = self.test_arguments
        test_arguments = test_arguments[0:max_n_arguments]
        n_arguments = len(test_arguments)
        for n in range(n_arguments):
            all_subsets_of_arguments = combinations(test_arguments, n + 1)
            for subset in all_subsets_of_arguments:
                subset_as_lists = []
                for argument in subset:
                    if type(argument) == list:
                        subset_as_lists.append(argument)
                    else:
                        subset_as_lists.append([argument])

                all_lengths = [len(a) for a in subset_as_lists]
                first_length = all_lengths[0]

                if all(length == first_length for length in all_lengths):
                    out = validate_lists_to_have_same_length(*subset)
                    assert out == first_length
                else:
                    with pytest.raises(ValueError):
                        out = validate_lists_to_have_same_length(*subset)


class TestManipulationTools:
    def setup_method(self):
        test_arguments = tools_testkit.build_list_of_example_arguments()
        self.test_arguments = test_arguments

        test_file_path = tools_testkit.test_file_path
        self.test_file_path = test_file_path

        scenario_json_path = time_testkit.scenario_json_path
        self.scenario_json_path = scenario_json_path

        (
            format_example,
            dictionary_example,
            _,
        ) = tools_testkit.build_dictionary_examples()
        self.format_example = format_example
        self.dictionary_example = dictionary_example

    @pytest.mark.parametrize("index_to_remove", [1, 2, 5, 10])
    def test_copy_dict_without_key(self, index_to_remove):
        all_values = self.test_arguments

        dictionary_1 = dict()
        dictionary_2 = dict()

        count = 0
        for value in all_values:
            count += 1
            key = f"key {str(count)}"
            dictionary_1[key] = value
            dictionary_2[key] = value
        assert dictionary_1 == dictionary_2

        key_to_remove = f"key {str(index_to_remove)}"

        new_dictionary_1 = copy_dict_without_key(dictionary_1, key_to_remove)
        new_dictionary_2 = copy_dict_without_key(dictionary_2, key_to_remove)
        assert new_dictionary_1 != dictionary_1
        assert new_dictionary_2 != dictionary_2
        assert new_dictionary_1 == new_dictionary_2

        new_keys_1 = list(new_dictionary_1.keys())
        new_keys_2 = list(new_dictionary_2.keys())
        assert key_to_remove not in new_keys_1
        assert key_to_remove not in new_keys_2

    def test_unnest_list(self):
        list_of_lists = [
            [1, 2, 3, 4],
            [5, 6, 7],
            [8, 9],
        ]
        simple_list = unnest_list(list_of_lists)
        assert len(simple_list) == 9

    def test_unique_and_sorted_vector(self):
        """
        No new functionality to be tested.
        """
        assert callable(unique_and_sorted_vector)

    def test_remove_characters(self):
        example_string = "[item_1, item_2, item_3]"
        character_list = ["[", "]", "_", ","]
        desired_string = "item1 item2 item3"

        result_string = remove_characters(example_string, character_list)
        assert result_string == desired_string

    def test_convert_string_into_numeric_list(self):
        example_string = "[1, 2.5, 3]"
        desired_list = [1, 2.5, 3]

        result_list = convert_string_into_numeric_list(example_string)
        assert result_list == desired_list

    def test_read_json(self):
        right_path = self.scenario_json_path
        contents = read_json(right_path)
        contents_are_dict = type(contents) == dict
        assert contents_are_dict

        wrong_path = self.test_file_path
        with pytest.raises(TypeError):
            contents = read_json(wrong_path)


class TestPlottingTools:
    def setup_method(self):
        self.sample_x = [1, 2]
        self.sample_y = [3, 4]

    def teardown_method(self):
        plt.close("all")

    @staticmethod
    def _query_axes_limits(ax):
        all_limits = []
        all_axes = ["x", "y"]
        for axis in all_axes:
            if axis == "x":
                axis_lim = ax.get_xlim()
            elif axis == "y":
                axis_lim = ax.get_ylim()
            all_limits.append(axis_lim)
        return all_limits

    # ------------------------------------------------------------------
    #  TESTS
    # ------------------------------------------------------------------
    def test_validate_axes(self):
        _, sample_ax = plt.subplots()
        sample_ax = sample_ax.plot(self.sample_x, self.sample_y)
        not_an_ax = self.sample_x

        all_axes = [
            None,
            sample_ax,
            not_an_ax,
        ]
        for axes in all_axes:
            if (axes is None) or isinstance(axes, plt.Axes):
                checked_axes = validate_axes(axes)
                assert isinstance(checked_axes, plt.Axes)
            else:
                with pytest.raises(TypeError):
                    checked_axes = validate_axes(axes)

    @pytest.mark.parametrize("scale", ("linear", "log"))
    def test_adjust_2d_graph_ranges(self, scale):
        ax_title = "Test 2D Graph Ranges Adjustment"
        test_ax = tools_testkit.prepare_figure(ax_title)

        test_ax.plot(self.sample_x, self.sample_y)
        test_ax.set_xscale(scale)
        test_ax.set_yscale(scale)
        old_limits = self._query_axes_limits(test_ax)
        adjust_2d_graph_ranges(ax=test_ax)
        new_limits = self._query_axes_limits(test_ax)
        adjusted_ax_is_still_plt_axes_object = isinstance(test_ax, plt.Axes)
        assert adjusted_ax_is_still_plt_axes_object

        n_axes = len(old_limits)
        for axis_index in range(n_axes):
            old_axis_scale_limits = old_limits[axis_index]
            new_axis_scale_limits = new_limits[axis_index]

            lower_limit_of_old_scale = old_axis_scale_limits[0]
            lower_limit_of_new_scale = new_axis_scale_limits[0]
            check_lower = lower_limit_of_new_scale < lower_limit_of_old_scale

            upper_limit_of_old_scale = old_axis_scale_limits[1]
            upper_limit_of_new_scale = new_axis_scale_limits[1]
            check_upper = upper_limit_of_new_scale > upper_limit_of_old_scale

            axis_ranges_have_increased = check_lower and check_upper
            assert axis_ranges_have_increased

        del test_ax
        plt.close("all")


class TestFormattedABC:
    tested_class = FormattedABC

    def setup_method(self):
        (
            self.right_format_input,
            self.wrong_format_input,
        ) = tools_testkit.inputs_for_format()

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_Format_validate_list_of_types(self):
        tested_method = self.tested_class.Format._validate_list_of_types
        input_zip = zip(
            self.right_format_input.values(),
            self.wrong_format_input.values(),
        )
        for right_value, wrong_value in input_zip:
            validated_value = tested_method(right_value)
            elements_are_types = [isinstance(e, type) for e in validated_value]
            assert all(elements_are_types)

            with pytest.raises(TypeError):
                validated_value = tested_method(wrong_value)

    def test_Format_constructor(self):
        right_format = self.tested_class.Format(self.right_format_input)
        assert isinstance(right_format, self.tested_class.Format)

        for key, value in right_format.items():
            assert key in self.right_format_input.keys()

            original_value = self.right_format_input[key]
            original_in_list = validate_list(original_value)
            for o, v in zip(original_in_list, value):
                if o is None:
                    assert v == type(o)
                else:
                    assert v == o

        with pytest.raises(TypeError):
            wrong_format = self.tested_class.Format(self.wrong_format_input)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_display(self):
        assert callable(self.tested_class.display)


class TestFormattedDict:
    tested_class = FormattedDict

    def setup_method(self):
        (
            self.example_format,
            self.example_dict,
            self.right_index_input,
            self.wrong_index_input,
        ) = tools_testkit.inputs_for_formatteddict()

    def build_format(self):
        return self.tested_class.Format(self.right_format_input)

    def build_default_empty_instance(self):
        """
        Build empty 'FormattedDict' without specifying 'format_index'.
        """
        return self.tested_class(self.example_format)

    def build_indexed_empty_instance(self):
        """
        Build empty 'FormattedDict' and specify 'format_index'.
        """
        return self.tested_class(
            self.example_format,
            format_index=self.right_index_input,
        )

    def build_full_instance(self):
        """
        Build 'FormattedDict' by converting an existing 'dict'.
        """
        return self.tested_class(
            self.example_format,
            dictionary=self.example_dict,
        )

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_validate_index(self):
        default_sample = self.build_default_empty_instance()
        right_index = default_sample._validate_index(self.right_index_input)
        assert right_index == self.right_index_input
        assert all(isinstance(e, int) for e in right_index)

        short_index = self.right_index_input[:-1]
        long_index = self.right_index_input + [self.right_index_input[-1]]
        with pytest.raises((TypeError, ValueError)):
            short_index = default_sample._validate_index(short_index)
            long_index = default_sample._validate_index(long_index)
            wrong_index = default_sample._validate_index(self.wrong_index_input)

    def test_empty(self):
        default_sample = self.build_default_empty_instance()
        indexed_sample = self.build_indexed_empty_instance()
        assert isinstance(default_sample, self.tested_class)
        assert isinstance(indexed_sample, self.tested_class)

        allowed_keys = default_sample.allowed_format.keys()

        for key, value in default_sample.items():
            assert key in allowed_keys

            valid_types = default_sample.allowed_format[key]
            assert isinstance(value, valid_types[0])

        for n, (key, value) in enumerate(indexed_sample.items()):
            assert key in allowed_keys

            valid_types = indexed_sample.allowed_format[key]
            chosen_index = self.right_index_input[n]
            assert isinstance(value, valid_types[chosen_index])

    def test_constructor(self):
        sample = self.build_full_instance()
        assert isinstance(sample, self.tested_class)

        for key, value in sample.items():
            assert key in sample.allowed_format.keys()

            valid_types = tuple(sample.allowed_format[key])
            assert isinstance(value, valid_types)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_setitem(self):
        sample = self.build_full_instance()
        assert callable(sample.__setitem__)


class TestFormattedLibrary:
    tested_class = FormattedLibrary

    def setup_method(self):
        (
            self.example_format,
            self.example_types,
            self.example_dict,
            self.example_keys_for_library,
        ) = tools_testkit.inputs_for_formattedlibrary()

        all_valid_formats = self.example_types
        all_valid_formats.append(self.example_format)
        self.all_valid_formats = all_valid_formats

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_empty(self):
        for valid_format in self.all_valid_formats:
            empty_sample = self.tested_class(valid_format)
            assert isinstance(empty_sample, self.tested_class)
            assert empty_sample.allowed_format == valid_format
            assert len(empty_sample) == 0

    def test_constructor(self):
        for valid_format in self.all_valid_formats:
            valid_dictionary = dict()
            for key in self.example_keys_for_library:
                if isinstance(valid_format, FormattedABC.Format):
                    valid_dictionary[key] = self.example_dict
                else:
                    valid_dictionary[key] = valid_format()

            full_sample = self.tested_class(
                valid_format,
                dictionary=valid_dictionary,
            )
            assert isinstance(full_sample, self.tested_class)
            assert full_sample.allowed_format == valid_format
            assert len(full_sample) == len(self.example_keys_for_library)

            for key, value in full_sample.items():
                assert key in self.example_keys_for_library

                if isinstance(valid_format, FormattedABC.Format):
                    assert isinstance(value, FormattedDict)
                else:
                    assert isinstance(value, valid_format)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_setitem(self):
        assert callable(self.tested_class.__setitem__)
