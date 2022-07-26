"""
Base classes for the power cycle model.
"""

# Import
import abc

import matplotlib.pyplot as plt
import numpy as np

# from typing import Union


# ######################################################################
# CLASS PROPERTY DECORATOR
# ######################################################################
class classproperty(object):
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


# ######################################################################
# POWER CYCLE ABSTRACT BASE CLASS
# ######################################################################


class PowerCycleABC(abc.ABC):
    """
    Abstract base class for classes in the Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the instance.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Plot defaults (arguments for `matplotlib.pyplot.plot`)
    _plot_kwargs = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }
    _scatter_kwargs = {
        "c": "k",  # Marker color
        "s": 100,  # Marker size
        "marker": "x",  # Marker style
    }

    # Plot text settings (for `matplotlib.pyplot.text`)
    _text_angle = 45  # rotation angle
    _text_index = -1  # index of (time,data) point used for location

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(self, name: str):

        # Store name
        self.name = self._validate_name(name)

        # Dynamically add validation classmethod
        # setattr(a, 'func', classmethod(func))

    @classmethod
    def _validate_name(cls, name):
        """
        Validate a name for class instance creation to be a string.
        """
        class_name = cls.__name__
        if not isinstance(name, (str)):
            raise TypeError(
                f"""
                The 'name' used to create an instance of the
                {class_name} class must be a `str`.
                """
            )

        return name

    @classmethod
    def _issue_error(cls, label):
        """
        Issue error associated with...
        """

        # Retrieve errors of that class
        error_dict = cls._errors

        # Validate error label
        all_labels = error_dict.keys()
        if label in all_labels:

            # Retrieve particular error
            the_error = error_dict[label]

            # Retrieve error attributes
            error_type = the_error.err_type
            error_msg = the_error.err_msg

            # Build raising function
            raise_function = error_type + "Error"
            raise_function = f"""
                raise {raise_function}('''{error_msg}''')
                """

            exec(raise_function)  # Issue error
        else:

            # Child class name
            class_name = cls.__name__

            # Issue error
            raise ValueError(
                f"""
                Unknown error label for error in class {class_name}.
                """
            )

    @staticmethod
    def _validate_list(input) -> list:
        """
        Validate a subclass input to be a list. If the input is just a
        single value, insert it in a list.
        """
        if not isinstance(input, (list)):
            input = [input]
        return input

    @classmethod
    def _validate(cls, object):
        """
        Validate `object` to be an instance of the class that calls
        this method.
        """
        class_name = cls.__name__
        if not type(object) == cls:
            raise TypeError(
                f"""
                The tested object is not an instance of the
                {class_name} class.
                """
            )
        return object

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    @classmethod
    def _validate_n_points(cls, n_points):
        """
        Validate an 'n_points' input. If `None`, retrieves the default
        of the class; else must be non-negative integer.
        """
        if not n_points:
            n_points = cls._n_points
        else:
            n_points = int(n_points)
            if n_points < 0:
                cls._issue_error("n_points")
        return n_points

    def _make_secondary_in_plot(self):
        """
        Alters the `_plot_kwargs` and `_text_index` attributes of an
        instance of this class, to enforce:

        - more subtle plotting characteristics for lines
        - a different location for texts

        that are displayed on a plot, as to not coincide with the
        primary plot.
        """
        self._text_index = 0
        self._plot_kwargs = {
            "c": "k",  # Line color
            "lw": 1,  # Line width
            "ls": "--",  # Line style
        }


# ######################################################################
# POWER CYCLE ERROR MESSAGE
# ######################################################################


class PowerCycleError(abc.ABC):
    """
    Abstract base class for handling errors in the Power Cycle module.

    Parameters
    ----------
    err_type: str
        Which type of error is raised when the error instance is called.
    err_msg: str
        Error message displayed by the `raise` command.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Valid error types
    _valid_types = [
        "Value",
        "Type",
    ]

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, err_type: str, err_msg: str):

        # Validate inputs
        self.err_type = self._validate_type(err_type)
        self.err_msg = self._validate_msg(err_msg)

    @classmethod
    def _validate_type(cls, err_type):
        """
        Validate `err_type` to be one of the valid types.
        """
        valid_types = cls._valid_types
        if err_type not in valid_types:
            msg_types = PowerCycleUtilities._join_valid_values(valid_types)
            raise ValueError(
                f"""
                The argument given as `err_type` input does not have a
                valid value. Valid values include: {msg_types}.
                """
            )
        return err_type

    @classmethod
    def _validate_msg(cls, err_msg):
        """
        Validate `err_msg` to be a valid message.
        """
        return err_msg

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _search_keywords(self):
        """
        Find strings preceeded by percent symbols ("%") in the error
        message.
        """

        # Current error message
        err_msg = self.err_msg

        # Split message in whitespaces
        words = err_msg.split()

        # Preallocate output
        keywords = []

        # For each word
        for word in words:

            # If "%" is present, make lowercase and store
            if "%" in word:
                word.lower()
                keywords.append(word)

        # Output keywords
        return keywords


# ######################################################################
# POWER CYCLE UTILITIES
# ######################################################################
class PowerCycleUtilities:
    """
    Useful functions for multiple classes in the Power Cycle module.
    """

    # ------------------------------------------------------------------
    # DATA MANIPULATION
    # ------------------------------------------------------------------
    @staticmethod
    def _join_valid_values(valid_values):
        """
        Given a collection of values, creates a string listing them as
        valid values to be printed as part of an error message by
        putting quotation marks around them and joining them with comma
        delimiters.
        If the input is a `list`, elements of the list are considered
        the valid values. If the input is a `dict`, the dictionary keys
        are considered the valid values.
        """
        if isinstance(valid_values, dict):
            values_list = list(valid_values.keys())
        elif isinstance(valid_values, list):
            values_list = valid_values
        else:
            raise TypeError(
                """
                The argument to be transformed into a message of valid
                values is not a `list`.
                """
            )

        # Convert every value to string
        string_values = [str(element) for element in values_list]

        # Create string message
        values_msg = "', '".join(string_values)
        values_msg = "'" + values_msg + "'"

        # Output message
        return values_msg

    @staticmethod
    def add_dict_entries(dictionary, new_entries):
        """
        Add (key,value) pairs to a dictionary, only if they are not
        already specified (i.e. no substitutions). If dictionary is
        empty, returns only `new_entries`.

        Parameters
        ----------
        dictionary: dict
            Dictionary to be modified.
        new_entries: dict
            Second dictionary, which entries will be added to
            `dictionary`, unless they already exist.

        Returns
        -------
        dictionary: dict
            Modified dictionary.
        """

        # Validate whether `dictionary` exists (i.e. not empty)
        if dictionary:

            # Keys of new entries
            new_entries_keys = new_entries.keys()

            # For each key
            for key in new_entries_keys:

                # Current entry value
                value = new_entries[key]

                # Add entry to dictionary, if not yet there
                dictionary.setdefault(key, value)
        else:

            # For empty `dictionary`, output only `new_entries`
            dictionary = new_entries

        # Output extended dictionary
        return dictionary

    # ------------------------------------------------------------------
    # PLOT MANIPULATION
    # ------------------------------------------------------------------
    @staticmethod
    def validate_axes(ax):
        """
        Validate axes argument for plotting method. If `None`, create
        new `axes` instance.
        """
        if ax is None:
            ax = plt.gca()
        elif not isinstance(ax, plt.Axes):
            raise TypeError(
                """
                The argument 'ax' used to create a plot is not
                an instance of the `Axes` class.
                """
            )
        return ax

    @classmethod
    def adjust_2d_graph_ranges(cls, x_frac=0.1, y_frac=0.1, ax=None):
        """ "
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
