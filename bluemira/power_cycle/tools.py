# COPYRIGHT PLACEHOLDER

"""
Utility functions for the power cycle model.
"""
import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def read_json(file_path) -> Dict[str, Any]:
    """
    Returns the contents of a 'json' file.
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def create_axes(ax=None):
    """
    Create axes object.

    If 'None', creates a new 'axes' instance.
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax


def adjust_2d_graph_ranges(x_frac=0.1, y_frac=0.1, ax=None):
    """
    Adjust x-axis and y-axis limits of a plot given an input 'axes'
    instance (from 'matplotlib.axes') and the chosen fractional
    proportions.
    New lower limits will be shifted negatively by current range
    multiplied by the input proportion. New upper limits will be
    similarly shifted, but positevely.

    Parameters
    ----------
    x_frac: float
        Fractional number by which x-scale will be enlarged. By
        default, this fraction is set to 10%.
    x_frac: float
        Fractional number by which y-scale will be enlarged. By
        default, this fraction is set to 10%.
    ax: Axes
        Instance of the 'matplotlib.axes.Axes' class. By default,
        the currently selected axes are used.
    """
    ax = validate_axes(ax)
    ax.axis("tight")

    all_axes = ["x", "y"]
    for axis in all_axes:
        if axis == "x":
            axis_type = ax.get_xscale()
            axis_lim = ax.get_xlim()
            fraction = x_frac
        elif axis == "y":
            axis_type = ax.get_yscale()
            axis_lim = ax.get_ylim()
            fraction = y_frac
        lim_lower = axis_lim[0]
        lim_upper = axis_lim[1]

        if axis_type == "linear":
            lin_range = lim_upper - lim_lower
            lin_shift = fraction * lin_range

            lim_lower = lim_lower - lin_shift
            lim_upper = lim_upper + lin_shift

        elif axis_type == "log":
            log_range = np.log10(lim_upper / lim_lower)
            log_shift = 10 ** (fraction * log_range)

            lim_lower = lim_lower / log_shift
            lim_upper = lim_upper * log_shift

        else:
            raise ValueError(
                "The 'adjust_graph_ranges' method has not yet been "
                "implemented for this type of scale."
            )

        if axis == "x":
            x_lim = (lim_lower, lim_upper)
        elif axis == "y":
            y_lim = (lim_lower, lim_upper)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
