# COPYRIGHT PLACEHOLDER

import os

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.tools import (
    adjust_2d_graph_ranges,
    read_json,
    remove_characters,
    unique_and_sorted_vector,
    unnest_list,
    validate_axes,
    validate_dict,
    validate_file,
    validate_list,
    validate_nonnegative,
    validate_numerical,
    validate_subdict,
    validate_vector,
)
from tests.power_cycle.kits_for_tests import (
    NetManagerTestKit,
    TimeTestKit,
    ToolsTestKit,
    copy_dict_with_wrong_key,
    copy_dict_with_wrong_value,
)

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

    def test_validate_dict(self):
        format_example = self.format_example
        dictionary_example = self.dictionary_example

        returned_dictionary = validate_dict(
            dictionary_example,
            format_example,
        )
        assert returned_dictionary == dictionary_example

        all_allowed_keys = format_example.keys()
        all_dict_keys = returned_dictionary.keys()
        for dict_key in all_dict_keys:
            dict_key_is_allowed = dict_key in all_allowed_keys
            assert dict_key_is_allowed

            dict_value = returned_dictionary[dict_key]
            dict_value_type = type(dict_value)
            necessary_type = format_example[dict_key]
            value_is_the_necessary_type = dict_value_type == necessary_type
            assert value_is_the_necessary_type

        first_dict_key = list(all_allowed_keys)[0]
        wrong_format = copy_dict_with_wrong_key(
            format_example,
            first_dict_key,
        )
        with pytest.raises(KeyError):
            returned_dictionary = validate_dict(
                dictionary_example,
                wrong_format,
            )

        first_dict_key = list(all_dict_keys)[0]
        wrong_dictionary = copy_dict_with_wrong_value(
            dictionary_example,
            first_dict_key,
            dict(),
        )
        with pytest.raises(TypeError):
            returned_dictionary = validate_dict(
                wrong_dictionary,
                format_example,
            )

        """
        all_class_attr = self.all_class_attr
        scenario_json_contents = self.scenario_json_contents
        valid_highest_level_json_keys = self.highest_level_json_keys

        possible_errors = (TypeError, KeyError)

        tested_class = self.tested_class
        for valid_key in valid_highest_level_json_keys:
            key_contents = scenario_json_contents[valid_key]

            necessary_dict = None
            text_before_hiphen = valid_key.split("-", 1)[0]
            for class_attr in all_class_attr:
                if text_before_hiphen in class_attr:
                    necessary_dict = getattr(tested_class, class_attr)

            wrong_necessary_dict = copy_dict_with_wrong_key(
                necessary_dict,
                "name",
            )

            recursion_is_needed = text_before_hiphen != "scenario"
            if recursion_is_needed:

                validated_contents = tested_class._validate_subdict(
                    key_contents,
                    necessary_dict,
                )
                with pytest.raises(possible_errors):
                    validated_contents = tested_class._validate_subdict(
                        key_contents,
                        wrong_necessary_dict,
                    )

            else:

                validated_contents = tested_class._validate_dict(
                    key_contents,
                    necessary_dict,
                )
                with pytest.raises(possible_errors):
                    validated_contents = tested_class._validate_dict(
                        key_contents,
                        wrong_necessary_dict,
                    )

                wrong_contents = copy_dict_with_wrong_value(
                    key_contents,
                    "name",
                    ["not", "a", "string"],
                )
                with pytest.raises(possible_errors):
                    validated_contents = tested_class._validate_dict(
                        wrong_contents,
                        necessary_dict,
                    )

            validated_contents_is_dict = isinstance(validated_contents, dict)
            assert validated_contents_is_dict
    """

    def test_validate_subdict(self):
        """
        No new functionality to be tested.
        """
        assert callable(validate_subdict)


class TestManipulationTools:
    def setup_method(self):
        test_file_path = tools_testkit.test_file_path
        self.test_file_path = test_file_path

        scenario_json_path = time_testkit.scenario_json_path
        self.scenario_json_path = scenario_json_path

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
        adjusted_ax_is_still_Axes_object = isinstance(test_ax, plt.Axes)
        assert adjusted_ax_is_still_Axes_object

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
