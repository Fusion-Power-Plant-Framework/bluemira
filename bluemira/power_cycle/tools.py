# COPYRIGHT PLACEHOLDER

"""
Utility functions for the power cycle model.
"""
import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_root


def validate_list(argument):
    """
    Validate an argument to be a list. If the argument is just a
    single value, insert it in a list.
    """
    if not isinstance(argument, list):
        argument = [argument]
    return argument


def validate_numerical(argument):
    """
    Validate an argument to be a numerical value (i.e. an instance
    of either the 'int' or the 'float' classes).
    """
    if isinstance(argument, (int, float)):
        return argument
    else:
        argument_class = type(argument)
        raise TypeError(
            "This argument must be either an 'int' or 'float', but "
            f"it is a {argument_class!r} instead.",
        )


def validate_nonnegative(argument):
    """
    Validate an argument to be a nonnegative numerical value.
    """
    argument = validate_numerical(argument)
    if argument >= 0:
        return argument
    else:
        raise ValueError(
            "This argument must be a non-negative numerical value.",
        )


def validate_vector(argument):
    """
    Validate an argument to be a numerical list.
    """
    argument = validate_list(argument)
    for element in argument:
        element = validate_numerical(element)
    return argument


def validate_file(file_path):
    """
    Validate 'str' to be the path to a file. If the file exists, the
    function returns its absolute path. If not, 'FileNotFoundError'
    is issued.
    """
    path_is_relative = not os.path.isabs(file_path)
    if path_is_relative:
        project_path = get_bluemira_root()
        absolute_path = os.path.join(project_path, file_path)
    else:
        absolute_path = file_path

    file_exists = os.path.isfile(absolute_path)
    if file_exists:
        return absolute_path
    else:
        raise FileNotFoundError(
            "The file does not exist in the specified path.",
        )


def validate_lists_to_have_same_length(*args):
    """
    Validate an arbitrary number of arguments to be lists of the same
    length. Arguments that are not lists are considered lists of one
    element.
    If all equal, returns the length of those lists.
    """
    all_lengths = []
    for argument in args:
        argument = validate_list(argument)
        argument_length = len(argument)
        all_lengths.append(argument_length)
    unique_lengths = unique_and_sorted_vector(all_lengths)

    all_lenghts_are_not_equal = len(unique_lengths) != 1
    if all_lenghts_are_not_equal:
        raise ValueError(
            "At least one of the lists passed as argument does not "
            "have the same length as the others."
        )
    else:
        return argument_length


def copy_dict_without_key(dictionary, key_to_remove):
    """
    Returns a dictionary that is a copy of the parameter 'dictionary',
    but without the 'key_to_remove' key.
    """
    d = dictionary
    dictionary_without_key = {k: d[k] for k in d if k != key_to_remove}
    return dictionary_without_key


def unnest_list(list_of_lists):
    """
    Un-nest a list of lists into a simple list, maintaining order.
    """
    return [item for sublist in list_of_lists for item in sublist]


def unique_and_sorted_vector(vector):
    """
    Returns a sorted list, in ascending order, created from the set
    created from a vector, as a way to eliminate redundant entries.
    """
    vector = validate_vector(vector)
    unique_vector = list(set(vector))
    sorted_vector = sorted(unique_vector)
    return sorted_vector


def remove_characters(string, character_list):
    """
    Remove all 'str' in a list from a main 'string' parameter.
    """
    for character in character_list:
        string = string.replace(character, "")
    return string


def read_json(file_path):
    """
    Returns the contents of a 'json' file in 'dict' format.
    """
    try:
        with open(file_path) as json_file:
            contents_dict = json.load(json_file)
    except json.decoder.JSONDecodeError:
        raise TypeError(
            "The file could not be read as a 'json' file.",
        )
    return contents_dict


def validate_axes(ax=None):
    """
    Validate axes argument for plotting method. If 'None', create
    new 'axes' instance.
    """
    if ax is None:
        _, ax = plt.subplots()
    elif not isinstance(ax, plt.Axes):
        raise TypeError(
            "The argument 'ax' used to create a plot is not an "
            "instance of the 'Axes' class, but an instance of the "
            f"'{ax.__class__}' class instead."
        )
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
