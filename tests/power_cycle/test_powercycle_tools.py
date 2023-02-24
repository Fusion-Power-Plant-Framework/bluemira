# COPYRIGHT PLACEHOLDER

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.tools import adjust_2d_graph_ranges, unnest_list, validate_axes
from tests.power_cycle.kits_for_tests import TimeTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()


class TestManipulationTools:
    def test_unnest_list(self):
        list_of_lists = [
            [1, 2, 3, 4],
            [5, 6, 7],
            [8, 9],
        ]
        simple_list = unnest_list(list_of_lists)
        assert len(simple_list) == 9


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
