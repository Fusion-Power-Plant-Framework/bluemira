"""
Classes to create loads in the power cycle model.
"""
# Import general packages
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d

# Import Power Cycle packages
from bluemira.power_cycle.base import (
    PowerCycleABC,
    PowerCycleError,
    PowerCycleUtilities,
    classproperty,
)
from bluemira.power_cycle.timeline import PowerCyclePhase

# ######################################################################
# POWER DATA
# ######################################################################


class PowerData(PowerCycleABC):
    """
    Data class to store a set of time and load vectors.

    Takes a pair of (time,data) vectors and creates a `PowerData` object
    used to build power load objects to represent the time evolution
    of a given power in the plant.
    Instances of this class do not specify any dependence between the
    data points it stores, so no method is defined for calculating
    values (e.g. interpolation). Instead, this class should be called
    by specialized classes such as `PowerLoad`.

    Parameters
    ----------
    name: str
        Description of the `PowerData` instance.
    time: int | float | list[int | float]
        List of time values that define the PowerData. [s]
    data: int | float | list[int | float]
        List of power values that define the PowerData. [W]
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Error messages
    @classproperty
    def _errors(cls):
        class_name = cls.__name__
        e = {
            "increasing": PowerCycleError(
                "Value",
                f"""
                The `time` input used to create an instance of the
                {class_name} class must be an increasing list.
                """,
            ),
            "sanity": PowerCycleError(
                "Value",
                f"""
                The attributes `data` and `time` of an instance of the
                {class_name} class must have the same length.
                """,
            ),
        }
        return e

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(
        self,
        name,
        time: Union[int, float, List[Union[int, float]]],
        data: Union[int, float, List[Union[int, float]]],
    ):

        # Call superclass constructor
        super().__init__(name)

        # Validate inputs to be lists
        self.data = super()._validate_list(data)
        self.time = super()._validate_list(time)

        # Verify time is an increasing vector
        self._is_increasing(self.time)

        # Validate created instance
        self._sanity()

    @classmethod
    def _is_increasing(cls, input):
        """
        Validate an input for class instance creation to be an
        increasing list.
        """
        check_increasing = []
        for i in range(len(input) - 1):
            check_increasing.append(input[i] <= input[i + 1])

        if not all(check_increasing):
            cls._issue_error("increasing")
        return input

    def _sanity(self):
        """
        Validate that `data` and `time` attributes both have the same
        length, so that they univocally represent power values in time.
        """
        length_data = len(self.data)
        length_time = len(self.time)
        if length_data != length_time:
            self._issue_error("sanity")

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def plot(self, ax=None, **kwargs):
        """
        Plot the points that define the `PowerData` instance.

        This method applies the `matplotlib.pyplot.scatter` imported
        method to the vectors that define the `PowerData` instance. The
        default options for this plot are defined as class attributes,
        but can be overridden.

        Parameters
        ----------
        ax: Axes
            Instance of the `matplotlib.axes.Axes` class. By default,
            the currently selected axes are used.
        **kwargs: dict
            Options for the `scatter` method.

        Returns
        -------
        plot_list: list
            List of plot objects created by the `matplotlib` package.
            The first element of the list is the plot object created
            using the `pyplot.scatter`, while the second element of the
            list is the plot object created using the `pyplot.text`
            method.
        """

        # Validate axes
        ax = PowerCycleUtilities.validate_axes(ax)

        # Retrieve default plot options
        default = self._scatter_kwargs

        # Set each default options in kwargs, if not specified
        kwargs = PowerCycleUtilities.add_dict_entries(kwargs, default)

        # Retrieve instance characteristics
        name = self.name
        time = self.time
        data = self.data

        # Preallocate output
        plot_list = []

        # Plot
        label = name + " (data)"
        plot_obj = ax.scatter(time, data, label=label, **kwargs)
        plot_list.append(plot_obj)

        # Add text to plot
        index = self._text_index
        text = f"{name} (PowerData)"
        label = name + " (name)"
        angle = self._text_angle
        plot_obj = ax.text(time[index], data[index], text, label=label, rotation=angle)
        plot_list.append(plot_obj)

        # Return list of plot objects
        return plot_list


# ######################################################################
# GENERIC POWER LOAD
# ######################################################################
class PowerLoad(PowerCycleABC):
    """
    Generic representation of a power load.

    Defines a power load with a set of `PowerData` instances. Each
    instance must be accompanied by a `model` specification, used to
    compute additional values between data points. This enables the
    instance to compute time-dependent curves.

    Parameters
    ----------
    name: str
        Description of the `PowerLoad` instance.
    data_set: PowerData | list[PowerData]]
        Collection of instances of the `PowerData` class that define
        the `PowerLoad` object.
    model: str | list[str]
        List of types of model that defines values between points
        defined in the `load` Attribute. Valid models include:

        - 'ramp';
        - 'step';

        that are defined in the `_single_curve` method.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Default number of points in each curve segment
    _n_points = 100

    # Implemented models (add model name here after implementation)
    _valid_models = ["ramp", "step"]

    # Error messages
    @classproperty
    def _errors(cls):
        class_name = cls.__name__
        models = PowerCycleUtilities._join_valid_values(cls._valid_models)
        e = {
            "model": PowerCycleError(
                "Value",
                f"""
                The argument given for the attribute `model` is not a
                valid value. Only the following models are currently
                implemented in class {class_name}: {models}.
                """,
            ),
            "n_points": PowerCycleError(
                "Value",
                f"""
                    The argument given for `n_points` is not a valid
                    value for plotting an instance of the {class_name}
                    class. Only non-negative integers are accepted.
                    """,
            ),
            "sanity": PowerCycleError(
                "Value",
                f"""
                    The attributes `load` and `model` of an instance of
                    the {class_name} class must have the same length.
                    """,
            ),
            "time": PowerCycleError(
                "Type",
                f"""
                    The `time` input used to create a curve with an
                    instance of the {class_name} class must be numeric
                    or a list of numeric values.
                    """,
            ),
        }
        return e

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, name, data_set, model):

        # Call superclass constructor
        super().__init__(name)

        # Validate inputs
        self.data_set = self._validate_data_set(data_set)
        self.model = self._validate_model(model)

        # Validate created instance
        self._sanity()

    @classmethod
    def _validate_data_set(cls, data_set):
        """
        Validate 'data_set' input to be a list of `PowerData` instances.
        """
        data_set = super()._validate_list(data_set)
        for element in data_set:
            PowerData._validate(element)
        return data_set

    @classmethod
    def _validate_model(cls, model):
        """
        Validate 'model' input to be a list of valid models options.
        """
        model = super()._validate_list(model)
        for element in model:
            if element not in cls._valid_models:
                cls._issue_error("model")
        return model

    def _sanity(self):
        """
        Validate instance to have `data_set` and `model` attributes of
        same length.
        """
        if not len(self.data_set) == len(self.model):
            self._issue_error("sanity")

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def __add__(self, other):
        """
        Addition of `PowerLoad` instances is a new `PowerLoad` instance
        with joined `load` and `model` attributes.
        """

        # Retrieve `load` attributes
        this_set = self.data_set
        other_set = other.data_set

        # Retrieve `model` attributes
        this_model = self.model
        other_model = other.model

        # Create and output `another`
        another_set = this_set + other_set
        another_model = this_model + other_model
        another_name = "Resulting PowerLoad"
        another = PowerLoad(another_name, another_set, another_model)
        return another

    @classmethod
    def _validate_time(cls, time):
        """
        Validate 'time' input to be a list of numeric values.
        """
        time = super()._validate_list(time)
        for element in time:
            if not isinstance(element, (int, float)):
                cls._issue_error("time")
        return time

    @classmethod
    def _single_curve(cls, powerdata, model, time):
        """
        This method applies the `scipy.interpolate.interp1d` imported
        method to a single instance of the `PowerData` class. The kind
        of interpolation is determined by the `model` input. Values are
        returned at the times specified in the `time` input, with any
        out-of-bound values set to zero.
        """

        # Validate `model`
        if model == "ramp":
            k = "linear"  # Linear interpolation
        elif model == "step":
            k = "previous"  # Previous-value interpolation
        else:
            cls._issue_error("model")

        # Define interpolation function
        x = powerdata.time
        y = powerdata.data
        b = False  # out-of-bound values do not raise error
        f = (0, 0)  # below-bounds/above-bounds values set to 0
        lookup = interp1d(x, y, kind=k, fill_value=f, bounds_error=b)

        # Output interpolated curve
        curve = list(lookup(time))
        return curve

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the `scipy.interpolate.interp1d` imported
        method to each `PowerData` object stored in the `data_set`
        attribute and sums the results. The kind of interpolation is
        determined by each respective value in the `model` attribute.
        Any out-of-bound values are set to zero.

        Parameters
        ----------
        time: list[float]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """

        # Validate `time`
        time = self._validate_time(time)
        n_time = len(time)

        # Retrieve instance attributes
        data_set = self.data_set
        model = self.model

        # Number of elements in `data_set`
        n_elements = len(data_set)

        # Preallocate curve (with length of `time` input)
        curve = np.array([0] * n_time)

        # For each element
        for e in range(n_elements):

            # Current PowerData
            current_powerdata = data_set[e]

            # Current model
            current_model = model[e]

            # Compute current curve
            current_curve = self._single_curve(current_powerdata, current_model, time)

            # Add current curve to total curve
            current_curve = np.array(current_curve)
            curve = curve + current_curve

        # Output curve converted into list
        curve = curve.tolist()
        return curve

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

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
        ax = PowerCycleUtilities.validate_axes(ax)

        # Retrieve default plot options (main curve)
        default = self._plot_kwargs

        # Set each default options in kwargs, if not specified
        kwargs = PowerCycleUtilities.add_dict_entries(kwargs, default)

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
        time = list(set(time))
        time.sort()

        # Compute complete curve
        curve = self.curve(time)

        # Preallocate output
        plot_list = []

        # Plot curve as line
        label = name + " (curve)"
        plot_obj = ax.plot(time, curve, label=label, **kwargs)
        plot_list.append(plot_obj)

        # Add descriptive label to curve
        index = self._text_index
        text = f"{name} (PowerLoad)"
        label = name + " (name)"
        angle = self._text_angle
        plot_obj = ax.text(
            time[index],
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
                    current_powerdata, current_model, time
                )

                # Change PowerData to secondary in plot
                current_powerdata._make_secondary_in_plot()

                # Plot PowerData with same plot options
                current_plot_list = current_powerdata.plot()

                # Plot current curve as line with secondary options
                kwargs.update(current_powerdata._plot_kwargs)
                plot_obj = ax.plot(time, current_curve, **kwargs)

                # Append current plot list with current curve
                current_plot_list.append(plot_obj)

                # Store current plot list in output
                plot_list.append(current_plot_list)

        # Return list of plot objects
        return plot_list


# ######################################################################
# PHASE LOAD
# ######################################################################
class PhaseLoad(PowerCycleABC):
    """
    Representation of the total power load during a pulse phase.

    Defines the phase load with a set of `PowerLoad` instances.

    Parameters
    ----------
    name: str
        Description of the `PhaseLoad` instance.
    phase: PowerCyclePhase
        Pulse phase specification, that determines in which phase the
        load happens.
    load_set: PowerLoad | list[PowerLoad]
        Collection of instances of the `PowerLoad` class that define
        the `PhaseLoad` object.
    normalize: bool | list[bool]
        List of boolean values that defines which elements of `load_set`
        have their time-dependence normalized in respect to the phase
        duration. A value of `True` forces a normalization, while a
        value of `False` does not and time values beyond the phase
        duration are ignored.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Default number of points in visualization methods
    _n_points = 100

    # Plot defaults (arguments for `matplotlib.pyplot.plot`)
    _plot_defaults = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }

    # Detailed plot defaults (arguments for `matplotlib.pyplot.plot`)
    _detailed_defaults = {
        "c": "k",  # Line color
        "lw": 1,  # Line width
        "ls": "--",  # Line style
    }

    # Error messages
    @classproperty
    def _errors(cls):
        class_name = cls.__name__
        e = {
            "normalize": PowerCycleError(
                "Value",
                f"""
                    The argument given for `normalize` is not a valid
                    value for an instance of the {class_name} class.
                    Each element of `normalize` must be a boolean.
                    """,
            ),
            "sanity": PowerCycleError(
                "Value",
                f"""
                    The attributes `load_set` and `normalize` of an
                    instance of the {class_name} class must have the
                    same length.
                    """,
            ),
            "display_data": PowerCycleError(
                "Value",
                f"""
                    The argument passed to the `display_data` method of
                    the {class_name} class for the input `option` is not
                    valid. Only the strings 'load' and 'normal' are
                    accepted.
                    """,
            ),
        }
        return e

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, name, phase, load_set, normalize):

        # Call superclass constructor
        super().__init__(name)

        # Validate `phase`
        self.phase = self._validate_phase(phase)

        # Validate `load_set`
        self.load_set = self._validate_load_set(load_set)

        # Validate `normalize`
        self.normalize = self._validate_normalize(normalize)

        # Validate created instance
        self._sanity()

    @classmethod
    def _validate_phase(cls, phase):
        """
        Validate 'phase' input to be a valid PowerCycleTimeline phase.
        """
        PowerCyclePhase._validate(phase)
        return phase

    @classmethod
    def _validate_load_set(cls, load_set):
        """
        Validate 'load_set' input to be a list of `PowerLoad` instances.
        """
        load_set = super()._validate_list(load_set)
        for element in load_set:
            PowerLoad._validate(element)
        return load_set

    @classmethod
    def _validate_normalize(cls, normalize):
        """
        Validate 'normalize' input to be a list of boolean values.
        """
        normalize = super()._validate_list(normalize)
        for element in normalize:
            if not isinstance(element, (bool)):
                cls._issue_error("normalize")
        return normalize

    def _sanity(self):
        """
        Validate instance to have `load_set` and `normalize` attributes
        of same length.
        """
        if not len(self.load_set) == len(self.normalize):
            self._issue_error("sanity")

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @property
    def _normal_set(self):
        """
        Create a modified version of the `load_set` attribute, in which
        the time vectors of each load is normalized in respect to the
        duration of the phase instance stored in the `phase` attribute.
        """

        # Retrieve instance attributes
        phase = self.phase
        load_set = self.load_set
        normalize = self.normalize

        # Phase duration
        duration = phase.duration

        # Number of loads
        n_loads = len(load_set)

        # Preallocate output
        normal_set = []

        # For each load index
        for l_ind in range(n_loads):

            # Current load & normalization flag
            load = load_set[l_ind]
            flag = normalize[l_ind]

            # If normalization flag is True
            if flag:

                # Data set of current load
                data_set = load.data_set

                # Number of data instances in data set
                n_data = len(data_set)

                # For each data index
                for d_ind in range(n_data):

                    # Current data instance
                    data = data_set[d_ind]

                    # Current time vector
                    time = data.time

                    # Normalize time in respect to duration
                    time = [t * duration / max(time) for t in time]

                    # Store new time in current load
                    load.data_set[d_ind].time = time

            # Store current load in output
            normal_set.insert(l_ind, load)

        # Output new list
        return normal_set

    def _cut_time(self, time):
        """
        Cut a list of time values based on the duration of the phase
        stored in the `phase` attribute.
        """

        # Retrieve instance attributes
        phase = self.phase

        # Retrieve phase duration
        duration = phase.duration

        # Cut values above duration
        cut_time = [t for t in time if t <= duration]

        # If time has been cut, add phase duration as last element
        if not len(time) == len(cut_time):
            cut_time.append(duration)

        # Output new list
        return cut_time

    def curve(self, time):
        """
        Create a curve by summing all the power load contributions at
        the specified times.

        This method applies the `curve` method from the `PowerLoad`
        class to each load stored in the `load_set` attribute and sums
        the results.

        For each element of `load_set`, its respective element in the
        `normalize` attribute is also read. When `True`, the curve is
        duration in time in normalized in respect to the the duration of
        the phase stored in the `phase` attribute. When `False`, no
        normalization is performed and the curve contribution is added
        as computed by the `PowerLoad` instance; the curve is cut at the
        time equal to the phase duration.

        Parameters
        ----------
        time: list[float]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """

        # Retrieve instance attributes
        normal_set = self._normal_set

        # Cut time up to phase duration
        time = self._cut_time(time)
        n_time = len(time)

        # Preallocate curve
        curve = np.array([0] * n_time)

        # For each load
        for load in normal_set:

            # Compute current curve
            current_curve = load.curve(time)

            # Add current curve to total curve
            current_curve = np.array(current_curve)
            curve = curve + current_curve

        # Output curve converted into list
        curve = curve.tolist()
        return curve

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def display_data(self, option="load"):
        """
        Displays all data that make up the loads in the instance of
        `PhaseLoad`. Mostly used for comparing and verifying the results
        of time normalization. Two options are available:

        - 'load', the contents in `load_set` are listed;
        - 'normal', the contents in `_normal_set` are listed.

        """

        # Validate option
        if option == "load":
            base_set = self.load_set
        elif option == "normal":
            base_set = self._normal_set
        else:
            self._issue_error("display_data")

        # Preallocate output
        all_data = []

        # For each load
        for load in base_set:

            # For each data
            for element in load.data_set:

                all_data.append([element.time, element.data])

        return all_data

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a `PhaseLoad` curve, built using the attributes that define
        the instance. The curve is ploted up to the duration of the
        phase stored in the `phase` attribute.

        This method can also plot the individual `PowerLoad` objects
        stored in the `load_set` attribute that define the `PhaseLoad`
        instance. If requested by setting the `detailed` argument to
        `True`, this method calls the `plot` method of each `PowerLoad`
        object. In such a case, the number of points interpolated in
        the `PowerLoad.plot` calls can be specified.

        Parameters
        ----------
        n_points: int
            Number of points used to create time vector for plotting
            the `PhaseLoad`. The default value is `None`, which uses
            the default value set as a class attribute.
            If `detailed` is set to True, this value is also used to
            define the interpolation in each `PowerLoad` curve segment.
            In such a case, using the default value of `None` forces
            each `PowerLoad.plot` call to use the default value for
            `n_points` of that class.
        detailed: bool
            Determines whether the plot will include all individual
            `PowerLoad` instances, that summed result in the normal
            plotted curve. Plotted as secondary plots, as defined in
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

        # Retrieve instance attributes
        name = self.name

        # Retrieve phase duration
        phase = self.phase
        duration = phase.duration

        # Retrieve default plot options (main curve)
        default = self._plot_kwargs

        # Set each default options in kwargs, if not specified
        kwargs = PowerCycleUtilities.add_dict_entries(kwargs, default)

        # Validate axes
        ax = PowerCycleUtilities.validate_axes(ax)

        # Validate `n_points`
        n_points = self._validate_n_points(n_points)

        # Create generic time vector with duration
        time = np.linspace(0, duration, n_points)

        # Compute curve
        curve = self.curve(time)

        # Preallocate output
        plot_list = []

        # Plot curve as line
        label = name + " (curve)"
        plot_obj = ax.plot(time, curve, label=label, **kwargs)
        plot_list.append(plot_obj)

        # Add descriptive label to curve
        index = self._text_index
        text = f"{name} (PhaseLoad)"
        label = name + " (name)"
        angle = self._text_angle
        plot_obj = ax.text(
            time[index],
            curve[index],
            text,
            label=label,
            rotation=angle,
        )
        plot_list.append(plot_obj)

        # Validate `detailed` option
        if detailed:

            # Retrieve load set
            normal_set = self._normal_set

            # For each element
            for load in normal_set:

                # Change PowerLoad to secondary in plot
                load._make_secondary_in_plot()

                # Plot current load with default plot options
                current_plot_list = load.plot(
                    ax=ax,
                    n_points=n_points,
                )

                # Store current plot list in output
                for plot_obj in current_plot_list:
                    plot_list.append(plot_obj)

        # Return list of plot objects
        return plot_list


# ######################################################################
# PULSE LOAD
# ######################################################################


class PulseLoad(PowerCycleABC):
    """
    Representation of the total power load during a complete pulse.

    Defines the pulse load with a set of `PhaseLoad` instances.

    Parameters
    ----------
    name: str
        Description of the `PhaseLoad` instance.
    pulse: `PowerCyclePulse`
        Pulse specification, that determines in which order the pulse
        phases happen.
    """

    pass
