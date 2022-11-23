"""
Classes for the calculation of net power in the power cycle model.
"""
from abc import ABCMeta
from typing import List, Union

from bluemira.power_cycle.base import PowerCycleABC, PowerCycleError
from bluemira.power_cycle.tools import _add_dict_entries, validate_axes


class NetPowerABCError(PowerCycleError):
    def _errors(self):
        errors = {
            "n_points": [
                "The argument given for 'n_points' is not a valid "
                f"value for plotting an instance of the {self._source} "
                "class. Only non-negative integers are accepted."
            ],
        }
        return errors


class NetPowerABC(PowerCycleABC, metaclass=ABCMeta):
    """
    Abstract base class for classes in the Power Cycle module that are
    used to account, sum and manage power loads.

    Parameters
    ----------
    name: str
        Description of the instance.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Default number of points (for any plotting method)
    _n_points = 50

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
    # METHODS
    # ------------------------------------------------------------------

    def _validate_n_points(self, n_points: Union[int, None]):
        """
        Validate an 'n_points' argument that specifies a "number of
        points". If the argument is 'None', retrieves the default of
        the class; else it must be a non-negative integer.
        """
        if not n_points:
            n_points = self._n_points
        else:
            try:
                n_points = int(n_points)
                if n_points < 0:
                    raise NetPowerABCError(
                        "n_points",
                        f"The value '{n_points}' is negative.",
                    )
            except (TypeError, ValueError):
                raise NetPowerABCError(
                    "n_points",
                    f"The value '{n_points}' is non-numeric.",
                )
        return n_points

    def _make_secondary_in_plot(self):
        """
        Alters the '_plot_kwargs' and '_text_index' attributes of an
        instance of this class to enforce:
            - more subtle plotting characteristics for lines; and
            - a different location for texts;
        that are displayed on a plot, as to not coincide with the
        primary plot.
        """
        self._text_index = 0
        self._plot_kwargs = {
            "c": "k",  # Line color
            "lw": 1,  # Line width
            "ls": "--",  # Line style
        }


class PowerDataError(PowerCycleError):
    def _errors(self):
        errors = {
            "increasing": [
                "The 'time' parameter used to create an instance of "
                f"the {self._source} class must be an increasing list.",
            ],
            "sanity": [
                "The attributes 'data' and 'time' of an instance of "
                f"the {self._source} class must have the same length."
            ],
        }
        return errors


class PowerData(PowerCycleABC):
    """
    Data class to store a set of time and load vectors.

    Takes a pair of (time,data) vectors and creates a 'PowerData' object
    used to build power load objects to represent the time evolution
    of a given power in the plant.
    Instances of this class do not specify any dependence between the
    data points it stores, so no method is defined for calculating
    values (e.g. interpolation). Instead, this class should be called
    by specialized classes such as 'PowerLoad'.

    Parameters
    ----------
    name: str
        Description of the 'PowerData' instance.
    time: int | float | list[int | float]
        List of time values that define the PowerData. [s]
    data: int | float | list[int | float]
        List of power values that define the PowerData. [W]
    """

    def __init__(
        self,
        name,
        time: Union[int, float, List[Union[int, float]]],
        data: Union[int, float, List[Union[int, float]]],
    ):
        super().__init__(name)

        self.data = super().validate_list(data)
        self.time = super().validate_list(time)
        self._is_increasing(self.time)

        self._sanity()

    @staticmethod
    def _is_increasing(parameter):
        """
        Validate a parameter for creation of a class instance to be an
        increasing list.
        """
        check_increasing = []
        for i in range(len(parameter) - 1):
            check_increasing.append(parameter[i] <= parameter[i + 1])

        if not all(check_increasing):
            raise PowerDataError("increasing")
        return parameter

    def _sanity(self):
        """
        Validate 'data' and 'time' attributes to have both the same
        length, so that they univocally represent power values in time.
        """
        length_data = len(self.data)
        length_time = len(self.time)
        if length_data != length_time:
            raise PowerDataError("sanity")

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def plot(self, ax=None, **kwargs):
        """
        Plot the points that define the 'PowerData' instance.

        This method applies the 'matplotlib.pyplot.scatter' imported
        method to the vectors that define the 'PowerData' instance. The
        default options for this plot are defined as class attributes,
        but can be overridden.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class. By default,
            the currently selected axes are used.
        **kwargs: dict
            Options for the 'scatter' method.

        Returns
        -------
        plot_list: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.scatter', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
        """

        ax = validate_axes(ax)

        # Set each default options in kwargs, if not specified
        default_scatter_settings = self._scatter_kwargs
        kwargs = _add_dict_entries(kwargs, default_scatter_settings)

        name = self.name
        time = self.time
        data = self.data
        plot_list = []

        label = name + " (data)"
        plot_obj = ax.scatter(time, data, label=label, **kwargs)
        plot_list.append(plot_obj)

        # Add text to plot to describe points
        index = self._text_index
        text = f"{name} (PowerData)"
        label = name + " (name)"
        angle = self._text_angle
        plot_obj = ax.text(
            time[index],
            data[index],
            text,
            label=label,
            rotation=angle,
        )
        plot_list.append(plot_obj)

        return plot_list
