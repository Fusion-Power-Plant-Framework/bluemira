# COPYRIGHT PLACEHOLDER

"""
Utility functions for the power cycle model.
"""
import copy
import json
import os
import pprint as pp
from abc import ABCMeta

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


# ######################################################################
# CLASSES
# ######################################################################


class FormattedABC(metaclass=ABCMeta):
    """
    Abstract base class for Formatted classes used to structure data,
    that includes the 'Format' inner class.

    Parameters
    ----------
    allowed_format: FormattedABC.Format
        Allowed format for values of the instance.
    """

    class Format(dict):
        """
        A class that allows for the creation of a dictionary that
        specifies the allowed format for objects of child classes of
        the 'FormattedABC' class. The 'Format' instance acts as a
        "blueprint" for these objects.
        """

        def __init__(self, dictionary: dict):
            if isinstance(dictionary, dict):
                for allowed_key, value in dictionary.items():
                    allowed_types = self._validate_list_of_types(value)
                    super().__setitem__(allowed_key, allowed_types)
            else:
                raise TypeError(
                    "A 'Format' instance must be created "
                    "with an instance of the 'dict' class.",
                )

        @staticmethod
        def _validate_list_of_types(argument):
            argument = validate_list(argument)
            argument = [type(t) if t is None else t for t in argument]
            elements_are_types = [isinstance(t, type) for t in argument]
            if not all(elements_are_types):
                raise TypeError(
                    "Values of a 'Format' dictionary instance must "
                    "be a 'type' object, or a list of them.",
                )
            return argument

    def __init__(
        self,
        allowed_format: Format,
    ):
        if isinstance(allowed_format, self.Format):
            self.allowed_format = allowed_format
        else:
            raise TypeError(
                "The 'allowed_format' parameter must be an "
                "instance of the inner class 'Format'.",
            )

    def display(self) -> str:
        """
        Print representation of the instance using the 'pprint' module.
        """
        pp.pp(self)


class FormattedDict(dict, FormattedABC):
    """
    Formatted dictionary.

    A child class of the 'dict' class that allows for the creation of
    dictionaries with a specific format. Formats can be specified with
    objects of the inner class 'Format'.

    Parameters
    ----------
    allowed_format: FormattedABC.Format
        Allowed format for 'FormattedDict' instance.
    dictionary: dict
        Dictionary that will be converted to a 'FormattedDict' instance.
        If 'None', an empty 'FormattedDict' instance is created using
        the format specified by the 'allowed_format' parameter.
    format_index: list[int]
        List of integers that specify the indexes applied to search
        the types used to create values in an empty instance of
        'FormattedDict'. Each type is specified in the values of
        same keys in the 'allowed_format' dictionary.
    """

    def __init__(
        self,
        allowed_format: FormattedABC.Format,
        dictionary=None,
        format_index=None,
    ):
        FormattedABC.__init__(self, allowed_format)
        self._build_formatted_dict(format_index)
        self._fill_formatted_dict(dictionary)

    def _build_formatted_dict(self, format_index):
        """
        Build a 'dict' based on the 'Format' object stored in the
        'allowed_format' attribute. Keys are the same as in the format,
        and values are empty instances of one of the elements of the
        value for that key. The element is chosen with the associated
        integer of the 'format_index' parameter.
        """
        format_index = self._validate_index(format_index)

        loop_elements = enumerate(self.allowed_format.items())
        for n, (allowed_key, allowed_types) in loop_elements:
            chosen_index = format_index[n]
            try:
                chosen_type = allowed_types[chosen_index]
            except IndexError:
                raise IndexError(
                    f"Element number {n!r} of the 'format_index' "
                    f"parameter is {chosen_index!r} but this integer "
                    "is out of range for the associated list in the "
                    "allowed_format attribute, that has length "
                    f"{len(allowed_types)!r}.",
                )
            super().__setitem__(allowed_key, chosen_type())

    def _validate_index(self, format_index):
        """
        Validate the 'format_index' parameter to be a 'list' with
        the same length as the 'allowed_format' attribute,  in which
        each element is an integer. If 'None' is given, a list of zeroes
        with the same lenght of 'allowed_format' is returned.
        """
        required_length = len(self.allowed_format)
        if format_index is None:
            format_index = [0] * required_length
        elif isinstance(format_index, list):
            if len(format_index) == required_length:
                for e in format_index:
                    if not isinstance(e, int):
                        raise TypeError(
                            "The 'format_index' parameter must "
                            "be a list of integers.",
                        )
            else:
                raise ValueError(
                    "The 'format_index' parameter must be a list "
                    "with the same length as the 'allowed_format' "
                    "parameter.",
                )
        else:
            raise TypeError(
                "The 'format_index' parameter must be a list.",
            )
        return format_index

    def _fill_formatted_dict(self, dictionary):
        if dictionary is None:
            pass
        elif isinstance(dictionary, dict):
            for key, value in dictionary.items():
                self.__setitem__(key, value)
        else:
            raise TypeError(
                "The 'dictionary' parameter must be a 'dict' or 'None'.",
            )

    def __setitem__(self, key, value):
        """
        Set the ('key', 'value') item in the instance if 'key' is valid
        and 'value' is of one of the allowed types.
        """
        if key in self.allowed_format.keys():
            valid_types = tuple(self.allowed_format[key])
            if isinstance(value, valid_types):
                super().__setitem__(key, value)
            else:
                raise TypeError(
                    f"The value in key {key!r} is not of "
                    "one of the following classes: "
                    f"{valid_types!r}.",
                )
        else:
            raise KeyError(
                f"The string {key!r} is not a valid key.",
            )


class Library(dict):
    """
    Library-style dictionary.

    A child class of the 'dict' class that allows for the creation of
    dictionaries in which each value must be of the same type.

    Parameters
    ----------
    allowed_type: type
        Allowed type for all values in the library.
    dictionary: dict
        Dictionary that will be converted to a 'Library' instance.
        If 'None', an empty 'Library' instance is created.
    """

    def __init__(
        self,
        allowed_type,
        dictionary=None,
    ):
        super().__init__()
        self._validate_allowed_type(allowed_type)
        self._fill_library(dictionary)

    def _validate_allowed_type(self, allowed_type):
        if isinstance(allowed_type, type):
            self.allowed_type = allowed_type
        else:
            raise TypeError(
                "The 'allowed_type' parameter must be a 'type' object.",
            )

    def _fill_library(self, dictionary):
        if dictionary is not None:
            for key, value in dictionary.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        """
        Set the ('key', 'value') item in the instance if 'value' is
        an object of type allowed by 'allowed_type'.
        """
        if isinstance(value, self.allowed_type):
            super().__setitem__(key, value)
        else:
            raise TypeError(
                f"The value to be set in key {key!r} is not of "
                f"the type {self.allowed_type!r}.",
            )


class FormattedLibrary(Library, FormattedABC):
    """
    Formatted library.

    A child of the 'Library' and 'FormattedABC' classes that allows for
    the creation of library-style dictionaries in which each value
    follows a specific format, specified with the 'allowed_format'
    parameter.

    Parameters
    ----------
    allowed_format: FormattedABC.Format
        Allowed format for all values in the library. All values in
        library must be the of the 'FormattedDict' class with the
        format specified by 'allowed_format'.
    dictionary: dict
        Dictionary that will be converted to a 'FormattedLibrary'
        instance. If 'None', an empty 'FormattedLibrary' instance is
        created.
    """

    def __init__(
        self,
        allowed_format: FormattedABC.Format,
        dictionary=None,
    ):
        allowed_type = FormattedDict
        Library.__init__(self, allowed_type)
        FormattedABC.__init__(self, allowed_format)
        self._fill_formatted_library(dictionary)

    def _fill_formatted_library(self, dictionary):
        if dictionary is not None:
            for key, value in dictionary.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        """
        Set the ('key', 'value') item in the instance if 'value' is a
        'dict' that can be converted into a 'FormattedDict' instance
        with the format specified by the 'allowed_format' attribute.
        """
        formatted_value = FormattedDict(self.allowed_format, dictionary=value)
        super().__setitem__(key, formatted_value)
