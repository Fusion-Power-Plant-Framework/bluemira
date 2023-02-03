# COPYRIGHT PLACEHOLDER

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.tools import adjust_2d_graph_ranges, validate_axes


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
        plt.close("all")
