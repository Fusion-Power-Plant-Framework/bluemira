"""
Classes for the calculation of net power in the power cycle model.
"""
from abc import ABCMeta
from enum import Enum
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.power_cycle.base import PowerCycleABC, PowerCycleError
from bluemira.power_cycle.tools import _add_dict_entries, validate_axes


class NetPowerABCError(PowerCycleError):
    """
    Exception class for 'NetPowerABC' class of the Power Cycle module.
    """

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
    """
    Exception class for 'PowerData' class of the Power Cycle module.
    """

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


class PowerData(NetPowerABC):
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


class PowerLoadModel(Enum):
    """
    Members define possible models used by the methods defined in the
    'PowerLoad' class to compute values between load definition points.

    The 'name' of a member roughly describes the interpolation behavior,
    while its associated 'value' specifies which kind of interpolation
    is applied when calling the imported `scipy.interpolate.interp1d`
    method.
    """

    RAMP = "linear"  # 'interp1d' linear interpolation
    STEP = "previous"  # 'interp1d' previous-value interpolation


class PowerLoadError(PowerCycleError):
    """
    Exception class for 'PowerLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "model": [
                "The argument given for the attribute 'model' is not "
                "a valid value. A model must be specified with an "
                "instance of the 'PowerLoadModel' 'Enum' class."
            ],
            "sanity": [
                "The attributes 'load' and 'model' of an instance of "
                f"the {self._source} class must have the same length."
            ],
            "time": [
                "The 'time' input used to create a curve with an "
                f"instance of the {self._source} class must be numeric "
                "or a list of numeric values.",
            ],
        }
        return errors


class PowerLoad(NetPowerABC):
    """
    Generic representation of a power load.

    Defines a power load with a set of `PowerData` instances. Each
    instance must be accompanied by a `model` specification, used to
    compute additional values between data points. This enables the
    instance to compute time-dependent curves.

    Parameters
    ----------
    name: str
        Description of the 'PowerLoad' instance.
    data_set: PowerData | list[PowerData]
        Collection of instances of the 'PowerData' class that define
        the 'PowerLoad' object.
    model: PowerLoadModel | list[PowerLoadModel]
        Mathematical model used to compute values between 'data_set'
        definition points.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------
    _n_points = 100

    def __init__(self, name, data_set, model):

        super().__init__(name)

        self.data_set = self._validate_data_set(data_set)
        self.model = self._validate_model(model)

        self._sanity()

    @staticmethod
    def _validate_data_set(data_set):
        """
        Validate 'data_set' input to be a list of 'PowerData' instances.
        """
        data_set = super(PowerLoad, PowerLoad).validate_list(data_set)
        for element in data_set:
            PowerData.validate_class(element)
        return data_set

    @staticmethod
    def _validate_model(model):
        """
        Validate 'model' input to be a list of valid models options.
        """
        model = super(PowerLoad, PowerLoad).validate_list(model)
        for element in model:
            if type(element) != PowerLoadModel:
                element_class = type(element)
                raise PowerLoadError(
                    "model",
                    "One of the arguments provided is an instance of "
                    f"the '{element_class}' class instead.",
                )
        return model

    def _sanity(self):
        """
        Validate instance to have 'data_set' and 'model' attributes of
        same length.
        """
        if len(self.data_set) != len(self.model):
            raise PowerLoadError("sanity")

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @staticmethod
    def _single_curve(powerdata, model, time):
        """
        This method applies the `scipy.interpolate.interp1d` imported
        method to a single instance of the `PowerData` class. The kind
        of interpolation is determined by the `model` input. Values are
        returned at the times specified in the `time` input, with any
        out-of-bound values set to zero.
        """

        try:
            interpolation_kind = model.value
        except (AttributeError):
            raise PowerLoadError("model")

        x = powerdata.time
        y = powerdata.data
        out_of_bounds_raises_error = False  # no error for out-of-bound
        out_of_bounds_fill_value = (0, 0)  # below-bounds/above-bounds
        interpolation_operator = interp1d(
            x,
            y,
            kind=interpolation_kind,
            fill_value=out_of_bounds_fill_value,  # substitute values
            bounds_error=out_of_bounds_raises_error,  # turn-off error
        )

        interpolated_curve = list(interpolation_operator(time))
        return interpolated_curve

    @staticmethod
    def _validate_time(time):
        """
        Validate 'time' input to be a list of numeric values.
        """
        time = super(PowerLoad, PowerLoad).validate_list(time)
        for element in time:
            if not isinstance(element, (int, float)):
                raise PowerLoadError("time")
        return time

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'scipy.interpolate.interp1d' imported
        method to each 'PowerData' object stored in the 'data_set'
        attribute and sums the results. The kind of interpolation is
        determined by each respective value in the 'model' attribute.
        Any out-of-bound values are set to zero.

        Parameters
        ----------
        time: float | list[float]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """

        time = self._validate_time(time)
        time_length = len(time)
        preallocated_curve = np.array([0] * time_length)

        data_set = self.data_set
        data_set_length = len(data_set)

        model = self.model

        for d in range(data_set_length):

            current_powerdata = data_set[d]
            current_model = model[d]

            current_curve = self._single_curve(
                current_powerdata,
                current_model,
                time,
            )
            current_curve = np.array(current_curve)

            preallocated_curve = preallocated_curve + current_curve

        curve = preallocated_curve.tolist()
        return curve

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    '''
    @staticmethod
    def _refine_vector(vector, n_points):
        """
        Add `n_point` equidistant points between each pair of points in
        the input `vector`.
        """

        # Number of vector segments
        n_segments = len(vector) - 1

        # Preallocate output
        refined_vector = []

        # Validate `n_points`
        n = n_points
        if n == 0:
            refined_vector = vector  # No alterations to vector
        else:

            # For each curve segment (i.e. pair of points)
            for s in range(n_segments):
                first = vector[s]
                last = vector[s + 1]
                refined_segment = np.linspace(first, last, n + 1, endpoint=False)
                refined_segment = refined_segment.tolist()
                refined_vector = refined_vector + refined_segment
            refined_vector.append(vector[-1])

        # Output refined vector
        return refined_vector

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a `PowerLoad` curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the `matplotlib.pyplot.plot` imported
        method to a list of values built using the `curve` method.
        The default options for this plot are defined as class
        attributes, but can be overridden.

        This method can also plot the individual `PowerData` objects
        stored in the `data_set` attribute that define the `PowerLoad`
        instance.

        Parameters
        ----------
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is `None`, which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            `PowerData` instances (computed with their respective
            `model` entries), that summed result in the normal plotted
            curve. Plotted as secondary plots, as defined in
            `PowerCycleABC` class. By default this input is set to
            `False`.
        **kwargs: dict
            Options for the `plot` method.

        Returns
        -------
        plot_list: list
            List of plot objects created by the `matplotlib` package.
            The first element of the list is the plot object created
            using the `pyplot.plot`, while the second element of the
            list is the plot object created using the `pyplot.text`
            method.
            If the `detailed` argument is set to `True`, the list
            continues to include the lists of plot objects created by
            the `PowerData` class, with the addition of plotted curves
            for the visualization of the model selected for each load.
        """

        # Validate axes
        ax = validate_axes(ax)

        # Retrieve default plot options (main curve)
        default = self._plot_kwargs

        # Set each default options in kwargs, if not specified
        kwargs = add_dict_entries(kwargs, default)

        # Validate `n_points`
        n_points = self._validate_n_points(n_points)

        # Retrieve instance attributes
        name = self.name
        data_set = self.data_set
        model = self.model

        # Number of elements in `load`
        n_elements = len(data_set)

        # Preallocate time vector for plotting
        time = []

        # For each element
        for e in range(n_elements):

            # Current PowerData time vector
            current_powerdata = data_set[e]
            current_time = current_powerdata.time

            # Refine current time vector
            current_time = self._refine_vector(current_time, n_points)

            # Append current time in time vector for plotting
            time = time + current_time

        # Sort and unique of complete time vector
        sorted_time = list(set(time))

        # Compute complete curve
        curve = self.curve(sorted_time)

        # Preallocate output
        plot_list = []

        # Plot curve as line
        label = name + " (curve)"
        plot_obj = ax.plot(sorted_time, curve, label=label, **kwargs)
        plot_list.append(plot_obj)

        # Add descriptive label to curve
        index = self._text_index
        text = f"{name} (PowerLoad)"
        label = name + " (name)"
        angle = self._text_angle
        plot_obj = ax.text(
            sorted_time[index],
            curve[index],
            text,
            label=label,
            rotation=angle,
        )
        plot_list.append(plot_obj)

        # Validate `detailed` option
        if detailed:

            # For each element
            for e in range(n_elements):

                # Current PowerData
                current_powerdata = data_set[e]

                # Current model
                current_model = model[e]

                # Compute current curve
                current_curve = self._single_curve(
                    current_powerdata, current_model, sorted_time
                )

                # Change PowerData to secondary in plot
                current_powerdata._make_secondary_in_plot()

                # Plot PowerData with same plot options
                current_plot_list = current_powerdata.plot()

                # Plot current curve as line with secondary options
                kwargs.update(current_powerdata._plot_kwargs)
                plot_obj = ax.plot(sorted_time, current_curve, **kwargs)

                # Append current plot list with current curve
                current_plot_list.append(plot_obj)

                # Store current plot list in output
                plot_list.append(current_plot_list)

        # Return list of plot objects
        return plot_list

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------

    def __add__(self, other):
        """
        Addition of `PowerLoad` instances is a new `PowerLoad` instance
        with joined `load` and `model` attributes.
        """

        this_set = self.data_set
        this_model = self.model

        other_set = other.data_set
        other_model = other.model

        another_set = this_set + other_set
        another_model = this_model + other_model
        another_name = "Resulting PowerLoad"
        another = PowerLoad(another_name, another_set, another_model)
        return another
    '''
