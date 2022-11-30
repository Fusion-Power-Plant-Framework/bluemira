from pprint import pformat

import matplotlib.pyplot as plt
import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.tools import (
    _add_dict_entries,
    _join_delimited_values,
    adjust_2d_graph_ranges,
    validate_axes,
)


def script_title():
    return "Test Power Cycle 'tools'"


def test_add_dict_entries():
    old_dict = {
        "value_1": 1,
        "value_2": 2,
        "value_3": 3,
    }
    new_entries = {
        "value_1": 10,
        "value_4": 40,
    }
    new_dict = _add_dict_entries(old_dict, new_entries)
    bluemira_debug(
        f"""
        {script_title()} (_add_dict_entries)

        Old dictionary:
        {pformat(old_dict)}

        New dictionary entries:
        {pformat(new_entries)}

        New dictionary:
        {pformat(new_dict)}
        """
    )
    assert isinstance(new_dict, dict)
    assert new_dict == {**old_dict, **new_entries}
    assert 0


def test_join_delimited_values():
    all_arguments = [
        ["value 1", "value 2", "value 3"],
        {"value_1": 1, "value_2": 2, "value_3": 3},
    ]
    for test_argument in all_arguments:
        string_from_argument = _join_delimited_values(test_argument)
        bluemira_debug(
            f"""
            {script_title()} (_join_delimited_values)

            Generated string:
            {string_from_argument}
            """
        )
    assert isinstance(string_from_argument, str)


class TestPlottingTools:
    def setup_method(self):
        self.sample_x = [1, 2]
        self.sample_y = [3, 4]

    def teardown_method(self):
        self._close_all_plots()

    @staticmethod
    def _close_all_plots():
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

        sample_plot = plt.gca()
        sample_plot = sample_plot.plot(self.sample_x, self.sample_y)
        not_a_plot = self.sample_x

        all_axes = [
            None,
            sample_plot,
            not_a_plot,
        ]
        for axes in all_axes:
            if (axes is None) or isinstance(axes, plt.Axes):
                checked_axes = validate_axes(axes)
                assert isinstance(checked_axes, plt.Axes)
            else:
                with pytest.raises(TypeError):
                    checked_axes = validate_axes(axes)

    def test_adjust_2d_graph_ranges(self):
        all_scales = ["linear", "log"]
        for scale in all_scales:

            # Assert that output are still axes
            test_plot = validate_axes()
            test_plot.plot(self.sample_x, self.sample_y)
            test_plot.set_xscale(scale)
            test_plot.set_yscale(scale)
            old_limits = self._query_axes_limits(test_plot)
            adjust_2d_graph_ranges(ax=test_plot)
            new_limits = self._query_axes_limits(test_plot)
            assert isinstance(test_plot, plt.Axes)

            # Assert that range of each axes is increased
            n_axes = len(old_limits)
            bluemira_debug(
                f"""
                {script_title()} (adjust_2d_graph_ranges)

                Old axes limits:
                {pformat(old_limits)}

                New axes limits:
                {pformat(new_limits)}
                """
            )
            for axis_index in range(n_axes):
                old_axis = old_limits[axis_index]
                new_axis = new_limits[axis_index]

                old_lower = old_axis[0]
                new_lower = new_axis[0]
                assert new_lower < old_lower

                old_upper = old_axis[1]
                new_upper = new_axis[1]
                assert old_upper < new_upper

            del test_plot
            self._close_all_plots()
