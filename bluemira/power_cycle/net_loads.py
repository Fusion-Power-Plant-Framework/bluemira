# COPYRIGHT PLACEHOLDER

"""
Classes for the definition of power loads in the power cycle model.
"""
from enum import Enum
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.power_cycle.base import NetPowerABC
from bluemira.power_cycle.errors import PhaseLoadError, PowerDataError, PowerLoadError
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import validate_axes


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
        (not necessarily strictly) increasing list.
        """
        check = [i <= j for i, j in zip(parameter, parameter[1:])]
        if not all(check):
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
    # OPERATIONS
    # ------------------------------------------------------------------
    def _normalize_time(self, new_end_time):
        """
        Normalize values stored in the 'time' attribute, so that the
        last time value coincides with 'new_end_time'.
        """
        old_time = self.time
        old_end_time = old_time[-1]
        norm = new_end_time / old_end_time
        new_time = [norm * t for t in old_time]
        self.time = new_time
        self._is_increasing(self.time)

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
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.scatter', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
        """
        ax = validate_axes(ax)

        # Set each default options in kwargs, if not specified
        default_scatter_kwargs = self._scatter_kwargs
        final_kwargs = {**default_scatter_kwargs, **kwargs}

        name = self.name
        time = self.time
        data = self.data
        list_of_plot_objects = []

        label = name + " (data)"
        plot_object = ax.scatter(
            time,
            data,
            label=label,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        plot_object = self._add_text_to_point_in_plot(
            ax,
            name,
            time,
            data,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        return list_of_plot_objects


class PowerLoadModel(Enum):
    """
    Members define possible models used by the methods defined in the
    'PowerLoad' class to compute values between load definition points.

    The 'name' of a member is a 'str' that roughly describes the
    interpolation behavior, while its associated 'value' is 'str' that
    specifies which kind of interpolation is applied when calling the
    imported 'scipy.interpolate.interp1d' method.
    """

    RAMP = "linear"  # 'interp1d' linear interpolation
    STEP = "previous"  # 'interp1d' previous-value interpolation


class PowerLoad(NetPowerABC):
    """
    Generic representation of a power load.

    Defines a power load with a set of 'PowerData' instances. Each
    instance must be accompanied by a 'model' specification, used to
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

    def __init__(
        self,
        name,
        data_set,
        model: Union[PowerLoadModel, List[PowerLoadModel]],
    ):

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
        This method applies the 'scipy.interpolate.interp1d' imported
        method to a single instance of the 'PowerData' class. The kind
        of interpolation is determined by the 'model' input. Values are
        returned at the times specified in the 'time' input, with any
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
        time: int | float | list[ int | float ]
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

    def _normalize_time(self, new_end_time):
        """
        Normalize the time of all 'PowerData' objects stored in the
        'data_set' attribute, so that their last time values coincide
        with 'new_end_time'.
        """
        old_data_set = self.data_set
        new_data_set = []
        for powerdata in old_data_set:
            powerdata._normalize_time(new_end_time)
            new_data_set.append(powerdata)
        self.data_set = new_data_set

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def _refine_intrinsic_time(self, n_points):
        data_set = self.data_set
        preallocated_time = []
        for powerdata in data_set:
            original_time = powerdata.time
            refined_time = self._refine_vector(original_time, n_points)
            preallocated_time = preallocated_time + refined_time
        complete_time = self._unique_and_sorted_vector(preallocated_time)
        return complete_time

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PowerLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'matplotlib.pyplot.plot' imported
        method to a list of values built using the 'curve' method.
        The default options for this plot are defined as class
        attributes, but can be overridden.

        This method can also plot the individual 'PowerData' objects
        stored in the 'data_set' attribute that define the 'PowerLoad'
        instance.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Axes on which to plot curve. If 'None' is given, a new
            instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PowerData' instances (computed with their respective
            'model' entries), that summed result in the normal plotted
            curve. Plotted as secondary plots, as defined in
            'PowerCycleABC' class. By default this input is set to
            'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'PowerData' class, with the addition of plotted curves
            for the visualization of the model selected for each load.
        """
        ax = validate_axes(ax)
        n_points = self._validate_n_points(n_points)

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        name = self.name

        computed_time = self._refine_intrinsic_time(n_points)
        computed_curve = self.curve(computed_time)

        list_of_plot_objects = []

        # Plot curve as line
        label = name + " (curve)"
        plot_object = ax.plot(
            computed_time,
            computed_curve,
            label=label,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        # Add descriptive text next to curve
        plot_object = self._add_text_to_point_in_plot(
            ax,
            name,
            computed_time,
            computed_curve,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        if detailed:

            data_set = self.data_set
            model = self.model
            number_of_data_elements = len(data_set)

            for e in range(number_of_data_elements):
                current_powerdata = data_set[e]
                current_model = model[e]

                current_curve = self._single_curve(
                    current_powerdata,
                    current_model,
                    computed_time,
                )

                # Plot current PowerData with seconday kwargs
                current_powerdata._make_secondary_in_plot()
                current_plot_list = current_powerdata.plot(ax=ax)

                # Plot current curve as line with secondary kwargs
                kwargs.update(current_powerdata._plot_kwargs)
                plot_object = ax.plot(
                    computed_time,
                    current_curve,
                    **kwargs,
                )
                current_plot_list.append(plot_object)

                list_of_plot_objects.append(current_plot_list)

        return list_of_plot_objects

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def __add__(self, other):
        """
        The addition of 'PowerLoad' instances creates a new 'PowerLoad'
        instance with joined 'load' and 'model' attributes.
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

    def __radd__(self, other):
        """
        The reverse addition operator, to enable the 'sum' method for
        'PowerLoad' instances.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)


class PhaseLoad(NetPowerABC):
    """
    Representation of the total power load during a pulse phase.

    Defines the phase load with a set of 'PowerLoad' instances.

    Parameters
    ----------
    name: str
        Description of the 'PhaseLoad' instance.
    phase: PowerCyclePhase
        Pulse phase specification, that determines in which phase the
        load happens.
    load_set: PowerLoad | list[PowerLoad]
        Collection of instances of the 'PowerLoad' class that define
        the 'PhaseLoad' object.
    normalize: bool | list[bool]
        List of boolean values that defines which elements of 'load_set'
        have their time-dependence normalized in respect to the phase
        duration. A value of 'True' forces a normalization, while a
        value of 'False' does not and time values beyond the phase
        duration are ignored.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }

    # Defaults for detailed plots
    _detailed_defaults = {
        "c": "k",  # Line color
        "lw": 1,  # Line width
        "ls": "--",  # Line style
    }

    def __init__(self, name, phase, load_set, normalize):

        super().__init__(name)

        self.phase = self._validate_phase(phase)
        self.load_set = self._validate_load_set(load_set)
        self.normalize = self._validate_normalize(normalize)

        self._sanity()

    @staticmethod
    def _validate_phase(phase):
        """
        Validate 'phase' input to be a PowerCycleTimeline instance.
        """
        PowerCyclePhase.validate_class(phase)
        return phase

    @staticmethod
    def _validate_load_set(load_set):
        """
        Validate 'load_set' input to be a list of 'PowerLoad' instances.
        """
        load_set = super(PhaseLoad, PhaseLoad).validate_list(load_set)
        for element in load_set:
            PowerLoad.validate_class(element)
        return load_set

    @staticmethod
    def _validate_normalize(normalize):
        """
        Validate 'normalize' input to be a list of boolean values.
        """
        normalize = super(PhaseLoad, PhaseLoad).validate_list(normalize)
        for element in normalize:
            if not isinstance(element, (bool)):
                raise PhaseLoadError("normalize")
        return normalize

    def _sanity(self):
        """
        Validate instance to have 'load_set' and 'normalize' attributes
        of same length.
        """
        if len(self.load_set) != len(self.normalize):
            self._issue_error("sanity")

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def _compute_normalized_set(self):
        phase = self.phase
        load_set = self.load_set
        normalize = self.normalize
        number_of_load_elements = len(load_set)

        normalized_set = []
        for e in range(number_of_load_elements):
            powerload = load_set[e]
            normalization_flag = normalize[e]
            if normalization_flag:
                powerload._normalize_time(phase.duration)
            normalized_set.append(powerload)

        return normalized_set

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'curve' method of the 'PowerLoad' class
        to the 'PowerLoad' instance that is created by the sum of all
        'PowerLoad' objects stored in the 'load_set' attribute.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        normalized_set = self._compute_normalized_set()
        resulting_load = sum(normalized_set)
        curve = resulting_load.curve(time)
        return curve

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def _refine_intrinsic_time(self, n_points):
        normalized_set = self._compute_normalized_set()
        number_of_load_elements = len(normalized_set)

        preallocated_time = []
        for e in range(number_of_load_elements):
            powerload = normalized_set[e]
            refined_time = powerload._refine_intrinsic_time(n_points)
            preallocated_time = preallocated_time + refined_time

        complete_time = self._unique_and_sorted_vector(preallocated_time)
        return complete_time

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PhaseLoad' curve, built using the 'load_set' and
        'normalize' attributes that define the instance. The number of
        points interpolated in each curve segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in the 'load_set' attribute.

        Parameters
        ----------
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PowerLoad' instances (computed with their respective
            'model' entries), that summed result in the normal plotted
            curve. Plotted as secondary plots, as defined in
            'PowerCycleABC' class. By default this input is set to
            'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'PowerLoad' class.
        """
        ax = validate_axes(ax)
        n_points = self._validate_n_points(n_points)

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        name = self.name
        computed_time = self._refine_intrinsic_time(n_points)
        computed_curve = self.curve(computed_time)

        list_of_plot_objects = []

        # Plot curve as line
        label = name + " (curve)"
        plot_object = ax.plot(
            computed_time,
            computed_curve,
            label=label,
            **final_kwargs,
        )
        list_of_plot_objects.append(plot_object)

        # Add descriptive text next to curve
        plot_object = self._add_text_to_point_in_plot(
            ax,
            name,
            computed_time,
            computed_curve,
            **kwargs,
        )
        list_of_plot_objects.append(plot_object)

        if detailed:
            normalized_set = self._compute_normalized_set()
            number_of_load_elements = len(normalized_set)

            for e in range(number_of_load_elements):
                powerload = normalized_set[e]
                powerload._make_secondary_in_plot()
                current_plot_list = powerload.plot(ax=ax)
                list_of_plot_objects.append(current_plot_list)

        return list_of_plot_objects


class PulseLoad(NetPowerABC):
    """
    Representation of the total power load during a pulse.

    Defines the pulse load with a set of 'PhaseLoad' instances.

    Parameters
    ----------
    name: str
        Description of the 'PulseLoad' instance.
    load_set: PhaseLoad | list[PhaseLoad]
        Ordered collection of instances of the 'PhaseLoad' class that
        define the 'PulseLoad' object.

    Attributes
    ----------
    pulse: PowerCyclePulse
        Pulse specification, determined by the 'phase' attributes of the
        'PhaseLoad' instances used to define the 'PulseLoad'.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }

    # Defaults for detailed plots
    _detailed_defaults = {
        "c": "k",  # Line color
        "lw": 1,  # Line width
        "ls": "--",  # Line style
    }

    def __init__(self, name, load_set):

        super().__init__(name)

        self.load_set = self._validate_load_set(load_set)
        self.pulse = self._build_pulse()

    @staticmethod
    def _validate_load_set(load_set):
        """
        Validate 'load_set' input to be a list of 'PhaseLoad' instances.
        """
        load_set = super(PulseLoad, PulseLoad).validate_list(load_set)
        for element in load_set:
            PhaseLoad.validate_class(element)
        return load_set

    def _build_pulse(self):
        """
        Build pulse from 'PowerCyclePhase' instances stored in the
        'phase' attributes of each 'PhaseLoad' instance in the
        'load_set' ordered list.
        """
        name = self.name
        load_set = self.load_set
        phase_set = []
        for load in load_set:
            phase = load.phase
            phase_set.append(phase)
        name = "Pulse for " + name
        pulse = PowerCyclePulse(name, phase_set)
        return pulse

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'curve' method of the 'PowerLoad' class
        to the 'PowerLoad' instance that is created by the sum of all
        'PowerLoad' objects stored in the 'load_set' attribute.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """

        """
        pulse = self.pulse
        load_set = self.load_set

        phase_set = pulse.phase_set
        number_of_phases = len(phase_set)

        for p in range(number_of_phases):
            phase = phase_set[p]
            phaseload = load_set[p]

            intrinsic_time =

        n_loads = len(load_set)
        for i in range(n_loads):
            normalization_flag = normalize[i]
            powerload = load_set[i]
            if normalization_flag:
                powerload._normalize_time(phase.duration)

        resulting_load = sum(load_set)
        curve = resulting_load.curve(time)
        return curve
        """
        pass

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def _refine_intrinsic_time(self, n_points):
        pass

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PulseLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in the 'load_set' attribute.

        Parameters
        ----------
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PowerLoad' instances (computed with their respective
            'model' entries), that summed result in the normal plotted
            curve. Plotted as secondary plots, as defined in
            'PowerCycleABC' class. By default this input is set to
            'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'PowerLoad' class.
        """
