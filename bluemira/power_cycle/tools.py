"""
Utility functions for the power cycle model.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

# ######################################################################
# MANIPULATION
# ######################################################################


def _add_dict_entries(dictionary, new_entries):
    """
    Add new (key,value) pairs to a dictionary. If a key for that entry
    already exists, it is substituted. If 'dictionary' is empty, returns
    only 'new_entries'.

    Parameters
    ----------
    dictionary: dict
        Dictionary to be modified.
    new_entries: dict
        Second dictionary, which entries will be added to
        'dictionary', unless they already exist.

    Returns
    -------
    dictionary: dict
        Modified dictionary.
    """
    python_version = sys.version_info
    if python_version >= (3, 9):
        new_dictionary = dictionary | new_entries
    elif python_version >= (3, 8):
        new_dictionary = {**dictionary, **new_entries}
    elif python_version >= (3, 4):
        new_dictionary = dictionary.copy()
        new_dictionary.update(new_entries)
    else:
        if dictionary:
            new_dictionary = dictionary
            new_entries_keys = new_entries.keys()
            for key in new_entries_keys:
                value = new_entries[key]
                # Add entry to dictionary, if not yet there
                new_dictionary.setdefault(key, value)
        else:
            new_dictionary = new_entries
    return new_dictionary


def _join_delimited_values(multiple_values):
    """
    Given a collection of values, join them by creating a string with
    quotation marks around each value and separating them with commas as
    delimiters.
    (Useful when printing valid values as part of an error message.)

    Parameters
    ----------
    multiple_values: list | dict
        Values to be joined. If the input is a `list`, elements of the
        list are considered the values to be joined. If the input is a
        `dict`, the dictionary keys are considered instead.
    """
    try:

        # Convert every value to string
        string_values = [str(element) for element in multiple_values]

        # Create string message with delimiters
        values_msg = "', '".join(string_values)
        values_msg = "'" + values_msg + "'"

    except (TypeError):
        TypeError(
            "The argument to be transformed into a delimited string of "
            "values is not a `list` or `dict`."
        )
    return values_msg


# ######################################################################
# PLOTTING
# ######################################################################


def validate_axes(ax=None):
    """
    Validate axes argument for plotting method. If `None`, create
    new `axes` instance.
    """
    if ax is None:
        ax = plt.gca()
    elif not isinstance(ax, plt.Axes):
        raise TypeError(
            "The argument 'ax' used to create a plot is not an "
            "instance of the `Axes` class, but an instance of the "
            f"'{ax.__class__}' class instead."
        )
    return ax


def adjust_2d_graph_ranges(x_frac=0.1, y_frac=0.1, ax=None):
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

    ax = validate_axes(ax)
    ax.axis("tight")

    all_axes = ["x", "y"]
    for axis in all_axes:

        # Data for current axis
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

        # Validate current axis type
        if axis_type == "linear":

            # Compute linear range
            lin_range = lim_upper - lim_lower

            # Compute new limits
            lim_lower = lim_lower - fraction * lin_range
            lim_upper = lim_upper + fraction * lin_range

        elif axis_type == "log":

            # Compute logarithmic range
            log_range = np.log10(lim_upper / lim_lower)

            # Compute new limits
            lim_lower = lim_lower / 10 ** (fraction * log_range)
            lim_upper = lim_upper * 10 ** (fraction * log_range)

        else:
            raise ValueError(
                "The 'adjust_graph_ranges' method has not yet been "
                "implemented for this type of scale."
            )

        # Store new limits for current axis
        if axis == "x":
            x_lim = (lim_lower, lim_upper)
        elif axis == "y":
            y_lim = (lim_lower, lim_upper)

    # Apply new limits
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
