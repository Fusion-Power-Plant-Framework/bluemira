from pprint import pprint

import matplotlib.pyplot as plt
import pytest

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.tools import (
    _add_dict_entries,
    _join_delimited_values,
    adjust_2d_graph_ranges,
    validate_axes,
)

# Simple vectors for sample plotting
sample_x = [1, 2]
sample_y = [3, 4]

bluemira_print("Test Power Cycle Tools")


def test_add_dict_entries(visualize=None):
    old_dict = {
        "value_1": 1,
        "value_2": 2,
        "value_3": 3,
    }
    new_entries = {
        "value_1": 1,
        "value_4": 4,
    }
    new_dict = _add_dict_entries(old_dict, new_entries)
    if visualize:
        bluemira_print("Test Power Cycle Tools ('_add_dict_entries')")
        pprint(old_dict)
        pprint(new_entries)
        pprint(new_dict)
    assert isinstance(new_dict, dict)
    assert new_dict == {**old_dict, **new_entries}


def test_join_delimited_values(visualize=None):
    all_arguments = [
        ["value 1", "value 2", "value 3"],
        {"value_1": 1, "value_2": 2, "value_3": 3},
    ]
    for test_argument in all_arguments:
        string_from_argument = _join_delimited_values(test_argument)
        if visualize:
            bluemira_print("Test Power Cycle Tools " "('_join_delimited_values')")
            print(string_from_argument)
    assert isinstance(string_from_argument, str)


def test_validate_axes(visualize=None):

    sample_plot = plt.gca()
    sample_plot = sample_plot.plot(sample_x, sample_y)
    not_a_plot = sample_x

    all_axes = [
        None,
        sample_plot,
        not_a_plot,
    ]
    for axes in all_axes:

        print(str(axes))
        if (axes is None) or isinstance(axes, plt.Axes):
            checked_axes = validate_axes(axes)
            assert isinstance(checked_axes, plt.Axes)
        else:
            with pytest.raises(TypeError):
                checked_axes = validate_axes(axes)

    if not visualize:
        plt.close("all")
    else:
        bluemira_print("Test Power Cycle Tools ('validate_axes')")


def test_adjust_2d_graph_ranges(visualize=None):

    all_scales = ["linear", "log"]
    for scale in all_scales:
        test_plot = validate_axes()
        test_plot.plot(sample_x, sample_y)
        test_plot.set_xscale(scale)
        test_plot.set_yscale(scale)
        adjust_2d_graph_ranges(ax=test_plot)
        assert isinstance(test_plot, plt.Axes)
        del test_plot

    if not visualize:
        plt.close("all")
    else:
        bluemira_print("Test Power Cycle Tools " "('adjust_2d_graph_ranges')")
