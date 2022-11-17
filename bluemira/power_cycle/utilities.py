"""
Utility functions for the power cycle model.
"""

# Import general packages
# import matplotlib.pyplot as plt
import numpy as np


def print_header(header=None):
    """
    Print a set of header lines to separate different script runs
    in the terminal.
    """
    # Validate header
    if not header:
        header = "NEW RUN"

    # Build header
    header = " " + header + " "
    header = header.center(72, "=")

    # Print Header
    print("\n\n")
    print(header)
    print("\n")


def adjust_2d_graph_ranges(cls, x_frac=0.1, y_frac=0.1, ax=None):
    """
    Adjust x-axis and y-axis limits of a plot given an input `axes`
    instance (from `matplotlib.axes`) and the chosen fractional
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
        Instance of the `matplotlib.axes.Axes` class. By default,
        the currently selected axes are used.
    """

    # Validate axes
    ax = cls.validate_axes(ax)

    # Tighten axes scales
    ax.axis("tight")

    # Axes to adjust
    all_axes = ["x", "y"]

    # For each axis
    for axis in all_axes:

        # Data for current axis (type, limits, fraction input)
        if axis == "x":
            type = ax.get_xscale()
            lim = ax.get_xlim()
            fraction = x_frac
        elif axis == "y":
            type = ax.get_yscale()
            lim = ax.get_ylim()
            fraction = y_frac

        # Retrieve explicit limits
        lim_lower = lim[0]
        lim_upper = lim[1]

        # Validate axis type
        if type == "linear":

            # Compute linear range
            range = lim_upper - lim_lower

            # Compute new limits
            lim_lower = lim_lower - fraction * range
            lim_upper = lim_upper + fraction * range

        elif type == "log":

            # Compute logarithmic range
            range = np.log10(lim_upper / lim_lower)

            # Compute new limits
            lim_lower = lim_lower / 10 ** (fraction * range)
            lim_upper = lim_upper * 10 ** (fraction * range)

        else:
            raise ValueError(
                """
                The "adjust_graph_ranges" method has not yet been
                implemented for this scale type.
                """
            )

        # Store new limits for current axis
        if axis == "x":
            x_lim = (lim_lower, lim_upper)
        elif axis == "y":
            y_lim = (lim_lower, lim_upper)

    # Apply new limits
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
