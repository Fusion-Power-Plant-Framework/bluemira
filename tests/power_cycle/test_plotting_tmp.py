# COPYRIGHT PLACEHOLDER

# COPYRIGHT PLACEHOLDER
import copy
import os

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.tools import adjust_2d_graph_ranges

test_data_folder_path = (
    "tests",
    "power_cycle",
    "test_data",
)


class ToolsTestKit:
    def __init__(self):
        test_file_name = ("test_file.txt",)
        test_file_path = test_data_folder_path + test_file_name
        self.test_file_path = os.path.join(*test_file_path)

    @staticmethod
    def build_list_of_example_arguments():
        return [
            None,
            True,  # bool
            "some string",  # string
            70,  # int
            -70,  # negative int
            1.2,  # float
            -1.2,  # negative float
            [True, False],  # bool list
            (True, False),  # bool tuple
            ["1", "2", "3", "4"],  # str list
            ("1", "2", "3", "4"),  # str tuple
            [1, 2, 3, 4],  # int list
            [-1, -2, -3, -4],  # negative int list
            (1, 2, 3, 4),  # int tuple
            (-1, -2, -3, -4),  # negative int tuple
            [1.2, 2.2, 3.2, 4.2],  # float list
            [-1.2, -2.2, -3.2, -4.2],  # negative float list
            (1.2, 2.2, 3.2, 4.2),  # float tuple
            (-1.2, -2.2, -3.2, -4.2),  # negative float tuple
        ]

    @staticmethod
    def prepare_figure(figure_title, humor=False):
        """
        Create figure for plot testing. Use 'plt.show()' to display it.
        Run test file with with `pytest --plotting-on` to visualize it.
        Switch 'humor' on for 'xkcd' style plots.
        """
        if humor:
            with plt.xkcd():
                ax = validate_axes()
        else:
            ax = validate_axes()
        plt.grid()
        plt.title(figure_title)
        return ax

    def build_dictionary_examples(self):
        argument_examples = self.build_list_of_example_arguments()

        count = 0
        format_example = {}
        dictionary_example = {}
        for argument in argument_examples:
            argument_type = type(argument)

            count += 1
            current_key = "key " + str(count)

            format_example[current_key] = argument_type
            dictionary_example[current_key] = argument

        subdictionaries_example = {}
        for c in range(count):
            current_key = "key " + str(c)
            subdictionaries_example[current_key] = dictionary_example

        return format_example, dictionary_example, subdictionaries_example

    def inputs_for_format(self):
        """
        Function to create inputs for Format testing.
        """
        right_format_input = {
            "key_1": str,
            "key_2": list,
            "key_3": [int, float],
            "key_4": [None, bool],
            "key_5": dict,
        }
        wrong_format_input = {
            "key_1": "str",
            "key_2": [[int, float], [str, bool]],
            "key_3": 1.2,
            "key_4": True,
            "key_5": copy.deepcopy(right_format_input),
        }
        return (
            right_format_input,
            wrong_format_input,
        )

    def _format_index_inputs(self):
        right_index_input = [
            0,
            0,
            1,
            1,
            0,
        ]

        non_int = 1.2
        out_of_range = 10
        wrong_index_input = [
            0,
            0,
            non_int,
            out_of_range,
            0,
        ]

        return (
            right_index_input,
            wrong_index_input,
        )


tools_testkit = ToolsTestKit()


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
