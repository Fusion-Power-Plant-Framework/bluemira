# COPYRIGHT PLACEHOLDER

"""
Classes for the definition of power loads in the power cycle model.
"""
import copy
from enum import Enum
from typing import List, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import EPS
from bluemira.power_cycle.base import PowerCycleLoadABC
from bluemira.power_cycle.errors import (
    LoadDataError,
    PhaseLoadError,
    PowerLoadError,
    PulseLoadError,
)
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import (
    unnest_list,
    validate_axes,
    validate_list,
    validate_numerical,
    validate_vector,
)


class LoadData(PowerCycleLoadABC):
    """
    Class to store a set of time and data vectors, to be used in
    creating 'PowerLoad' objects.

    Takes a pair of (time,data) vectors and creates a 'LoadData' object
    used to build power load objects to represent the time evolution
    of a given power in the plant.
    Instances of this class do not specify any dependence between the
    data points it stores, so no method is defined for altering (e.g.
    applying a multiplicative efficiency) or calculating related values
    (e.g. interpolation). Instead, these actions are performed with
    objects of the 'PowerLoad' class, that are built with instances of
    'LoadData'.

    Parameters
    ----------
    name: str
        Description of the 'LoadData' instance.
    time: int | float | list[int | float]
        List of time values that define the LoadData. [s]
    data: int | float | list[int | float]
        List of power values that define the LoadData. [W]

    Properties
    ----------
    intrinsic_time: list[int | float]
        Deep copy of the list stored in the 'time' attribute.
    """

    def __init__(
        self,
        name,
        time: Union[int, float, List[Union[int, float]]],
        data: Union[int, float, List[Union[int, float]]],
    ):
        super().__init__(name)

        self.data = validate_list(data)
        self.time = validate_list(time)
        self._is_increasing(self.time)

        if len(self.data) != len(self.time):
            raise LoadDataError("sanity")

        self._norm = []  # Memory for time normalization
        self._shift = []  # Memory for time shifting

    @staticmethod
    def _is_increasing(parameter):
        """
        Validate a parameter for creation of a class instance to be an
        (not necessarily strictly) increasing list.
        """
        if not all(i <= j for i, j in zip(parameter, parameter[1:])):
            raise LoadDataError("increasing")
        return parameter

    @classmethod
    def null(cls):
        """
        Instantiates a null version of the class.
        """
        return cls("Null LoadData", time=[0, 1], data=[0, 0])

    def _normalise_time(self, new_end_time):
        """
        Normalize values stored in the 'time' attribute, so that the
        last time value coincides with 'new_end_time'.
        Stores the normalization factor in the attribute '_norm', which
        is always initialized as an empty list.
        """
        old_time = self.time
        old_end_time = old_time[-1]
        norm = new_end_time / old_end_time
        new_time = [norm * t for t in old_time]
        self._is_increasing(new_time)
        self.time = new_time
        self._norm.append(norm)

    def _shift_time(self, time_shift):
        """
        Shift all time values in the 'time' attribute by the numerical
        value 'time_shift'.
        Stores the shifting factor in the attribute '_shift', which
        is always initialized as an empty list.
        """
        time_shift = validate_numerical(time_shift)
        self.time = [t + time_shift for t in self.time]
        self._shift.append(time_shift)

    def make_consumption_explicit(self):
        """
        Modifies the instance by turning every non-positive value stored
        in 'data' into its opposite.
        """
        for i, value in enumerate(self.data):
            if value > 0:
                self.data[i] = -value

    @property
    def intrinsic_time(self):
        """
        Deep copy of the time vector contained in the 'time' attribute.
        """
        return copy.deepcopy(self.time)

    def plot(self, ax=None, **kwargs):
        """
        Plot the points that define the 'LoadData' instance.

        This method applies the 'matplotlib.pyplot.scatter' imported
        method to the vectors that define the 'LoadData' instance. The
        default options for this plot are defined as class attributes,
        but can be overridden.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        **kwargs: dict
            Options for the 'scatter' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.scatter', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
        """
        ax = validate_axes(ax)

        # Set each default options in kwargs, if not specified
        return ax, self.plot_obj(
            ax,
            self.time,
            self.data,
            self.name,
            kwargs,
            {**self._scatter_kwargs, **kwargs},
            False,
        )


class LoadModel(Enum):
    """
    Members define possible models used by the methods defined in the
    'PowerLoad' class to compute values between load definition points.

    The 'name' of a member is a 'str' that roughly describes the
    interpolation behavior, while its associated 'value' is a 'str' that
    specifies which kind of interpolation is applied when calling the
    imported 'scipy.interpolate.interp1d' method.
    """

    RAMP = "linear"  # 'interp1d' linear interpolation
    STEP = "previous"  # 'interp1d' previous-value interpolation


class PowerLoad(PowerCycleLoadABC):
    """
    Generic representation of a power load.

    Defines a power load with a set of 'LoadData' instances. Each
    instance must be accompanied by a 'LoadModel' specification, used to
    compute additional values between data points. This enables the
    instance to compute time-dependent curves.

    Instances of the 'PowerLoad' class can be added to each other, and
    a list of them can be summed. Instances can also be multiplied and
    divided by scalar numerical values.

    Parameters
    ----------
    name: str
        Description of the 'PowerLoad' instance.
    loaddata_set: LoadData | list[LoadData]
        Collection of instances of the 'LoadData' class that define
        the 'PowerLoad' object.
    loadmodel_set: LoadModel | list[LoadModel]
        Mathematical loadmodel used to compute values between
        'loaddata_set' definition points.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'LoadData' objects contained in the
        'loaddata_set' attribute, ordered and with no repetitions.
    """

    _n_points = 100

    def __init__(
        self,
        name,
        loaddata_set: Union[LoadData, List[LoadData]],
        loadmodel_set: Union[LoadModel, List[LoadModel]],
    ):
        super().__init__(name)

        self.loaddata_set = self._validate_loaddata_set(loaddata_set)
        self.loadmodel_set = self._validate_loadmodel_set(loadmodel_set)

        if len(self.loaddata_set) != len(self.loadmodel_set):
            raise PowerLoadError("sanity")

    @staticmethod
    def _validate_loaddata_set(loaddata_set):
        """
        Validate 'loaddata_set' input to be a list of 'LoadData'
        objects.
        """
        loaddata_set = validate_list(loaddata_set)
        for element in loaddata_set:
            LoadData.validate_class(element)
        return loaddata_set

    @staticmethod
    def _validate_loadmodel_set(loadmodel_set):
        """
        Validate 'loadmodel_set' input to be a list of 'LoadModel'
        objects.
        """
        loadmodel_set = validate_list(loadmodel_set)
        for element in loadmodel_set:
            if not isinstance(element, LoadModel):
                element_class = type(element)
                raise PowerLoadError(
                    "loadmodel",
                    "One of the arguments provided is an instance of "
                    f"the {element_class!r} class instead.",
                )
        return loadmodel_set

    @classmethod
    def null(cls):
        """
        Instantiates an null version of the class.
        """
        return cls(
            "Null PowerLoad",
            LoadData.null(),
            LoadModel["RAMP"],
        )

    @staticmethod
    def _single_curve(loaddata, loadmodel, time):
        """
        Method that applies the 'scipy.interpolate.interp1d' imported
        method to a single instance of the 'LoadData' class. The kind
        of interpolation is determined by the 'loadmodel' input. Values
        are returned at the times specified in the 'time' input, with
        any out-of-bound values set to zero.
        """
        try:
            interpolation_kind = loadmodel.value
        except AttributeError:
            raise PowerLoadError("loadmodel")

        interpolation_operator = interp1d(
            loaddata.time,
            loaddata.data,
            kind=interpolation_kind,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )

        return interpolation_operator(time)

    @staticmethod
    def _validate_curve_input(time):
        """
        Validate the 'time' input for the 'curve' method to be a list of
        numeric values. In this case, the elements of 'time' can be
        negative.
        """
        return validate_vector(time)

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'scipy.interpolate.interp1d' imported
        method to each 'LoadData' object stored in the 'loaddata_set'
        attribute and sums the results. The kind of interpolation is
        determined by each respective value in the 'loadmodel_set'
        attribute. Any out-of-bound values are set to zero.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        time = self._validate_curve_input(time)
        curve = np.zeros(len(time))

        for loaddata, loadmodel in zip(self.loaddata_set, self.loadmodel_set):
            curve += self._single_curve(loaddata, loadmodel, time)
        return curve

    def _normalise_time(self, new_end_time):
        """
        Normalize the time of all 'LoadData' objects stored in the
        'loaddata_set' attribute, so that their last time values
        coincide with 'new_end_time'.
        """
        for ld in self.loaddata_set:
            ld._normalise_time(new_end_time)

    def _shift_time(self, time_shift):
        """
        Shift the 'time' attribute of all 'LoadData' objects in the
        'loaddata_set' attribute by the numerical value 'time_shift'.
        """
        for ld in self.loaddata_set:
            ld._shift_time(time_shift)

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'loaddata_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.loaddata_set)

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'LoadData' objects contained in the 'loaddata_set'
        attribute, ordered and with no repetitions.
        """
        return self._build_time_from_load_set(self.loaddata_set)

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PowerLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'matplotlib.pyplot.plot' imported
        method to a list of values built using the 'curve' method.
        The default options for this plot are defined as class
        attributes, but can be overridden.

        This method can also plot the individual 'LoadData' objects
        stored in the 'loaddata_set' attribute that define the
        'PowerLoad' instance.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'LoadData' objects (computed with their respective
            'loadmodel_set' entries), that summed result in the normal
            plotted curve. These objects are plotted as secondary plots,
            as defined in 'PowerCycleABC' class. By default this input
            is set to 'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        list_of_plot_objects: list
            List of plot objects created by the 'matplotlib' package.
            The first element of the list is the plot object created
            using the 'pyplot.plot', while the second element of the
            list is the plot object created using the 'pyplot.text'
            method.
            If the 'detailed' argument is set to 'True', the list
            continues to include the lists of plot objects created by
            the 'LoadData' class, with the addition of plotted curves
            for the visualization of the model selected for each load.
        """
        ax = validate_axes(ax)

        # Set each default options in kwargs, if not specified
        final_kwargs = {**self._plot_kwargs, **kwargs}

        computed_time = self._refine_vector(
            self.intrinsic_time, self._validate_n_points(n_points)
        )
        computed_curve = self.curve(computed_time)

        list_of_plot_objects = self.plot_obj(
            computed_time, computed_curve, self.name, kwargs, final_kwargs, True
        )

        if detailed:
            for ld, lm in zip(self.loaddata_set, self.loadmodel_set):
                current_curve = self._single_curve(
                    ld,
                    lm,
                    computed_time,
                )

                # Plot current LoadData with seconday kwargs
                ld._make_secondary_in_plot()
                ax, current_plot_list = ld.plot(ax=ax)

                # Plot current curve as line with secondary kwargs
                kwargs.update(ld._plot_kwargs)
                plot_object = ax.plot(
                    computed_time,
                    current_curve,
                    **kwargs,
                )
                current_plot_list.append(plot_object)

                list_of_plot_objects.append(current_plot_list)

        return ax, list_of_plot_objects

    def __add__(self, other):
        """
        The addition of 'PowerLoad' instances creates a new 'PowerLoad'
        instance with joined 'loaddata_set' and 'loadmodel_set'
        attributes.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)
        return PowerLoad(
            "Resulting PowerLoad",
            this.loaddata_set + other.loaddata_set,
            this.loadmodel_set + other.loadmodel_set,
        )

    def __mul__(self, number):
        """
        An instance of the 'PowerLoad' class can only be multiplied by
        scalar numerical values.
        The multiplication of a 'PowerLoad' instance by a number
        multiplies all values in the 'data' attributes of 'LoadData'
        objects stored in the 'loaddata_set' by that number.
        """
        number = validate_numerical(number)
        other = copy.deepcopy(self)
        for loaddata in other.loaddata_set:
            multiplied_data = [d * number for d in loaddata.data]
            loaddata.data = multiplied_data
        return other

    def __truediv__(self, number):
        """
        An instance of the 'PowerLoad' class can only be divided by
        scalar numerical values.
        The division of a 'PowerLoad' instance by a number
        divides all values in the 'data' attributes of 'LoadData'
        objects stored in the 'loaddata_set' by that number.
        """
        number = validate_numerical(number)
        other = copy.deepcopy(self)
        for loaddata in other.loaddata_set:
            divided_data = [d / number for d in loaddata.data]
            loaddata.data = divided_data
        return other


class PhaseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse phase.

    Defines the phase load with a set of 'PowerLoad' instances. Each
    instance must be accompanied by a 'normalise' specification, used to
    indicate whether that power load must have its curve normalised in
    time in respect to the 'duration' of the 'phase' Parameter. This
    enables the instance to adjust the evolution of power loads
    accordingly, if changes occur to the plant pulse.

    Parameters
    ----------
    name: str
        Description of the 'PhaseLoad' instance.
    phase: PowerCyclePhase
        Pulse phase specification, that determines in which phase the
        load happens.
    powerload_set: PowerLoad | list[PowerLoad]
        Collection of instances of the 'PowerLoad' class that define
        the 'PhaseLoad' object.
    normalise: bool | list[bool]
        List of boolean values that defines which elements of
        'powerload_set' have their time-dependence normalised in respect
        to the phase duration. A value of 'True' forces a normalization,
        while a value of 'False' does not and time values beyond the
        phase duration are ignored.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        'powerload_set' attribute, ordered and with no repetitions.
    normalised_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        '_normalised_set' attribute, ordered and with no repetitions.
    """

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

    def __init__(self, name, phase, powerload_set, normalise):
        super().__init__(name)

        self.phase = self._validate_phase(phase)
        self.powerload_set = self._validate_powerload_set(powerload_set)
        self.normalise = self._validate_normalise(normalise)

        if len(self.powerload_set) != len(self.normalise):
            raise PhaseLoadError("sanity")

    @staticmethod
    def _validate_phase(phase):
        """
        Validate 'phase' input to be a PowerCycleTimeline instance.
        """
        PowerCyclePhase.validate_class(phase)
        return phase

    @staticmethod
    def _validate_powerload_set(powerload_set):
        """
        Validate 'powerload_set' input to be a list of 'PowerLoad'
        instances.
        """
        powerload_set = validate_list(powerload_set)
        for element in powerload_set:
            PowerLoad.validate_class(element)
        return powerload_set

    @staticmethod
    def _validate_normalise(normalise):
        """
        Validate 'normalise' input to be a list of boolean values.
        """
        normalise = validate_list(normalise)
        for element in normalise:
            if not isinstance(element, (bool)):
                raise PhaseLoadError(
                    "normalise",
                    f"Element {element!r} of 'normalise' list is an "
                    f"instance of the {type(element)!r} class instead.",
                )
        return normalise

    @classmethod
    def null(cls, phase):
        """
        Instantiates an null version of the class.
        """
        return cls(
            f"Null PhaseLoad for phase {phase.name}",
            phase,
            PowerLoad.null(),
            True,
        )

    @property
    def _normalised_set(self):
        """
        Modified 'powerload_set' attribute, in which all times are
        normalised in respect to the 'duration' of the 'phase'
        attribute.
        """
        normalised_set = copy.deepcopy(self.powerload_set)
        for index in np.where(self.normalise)[0]:
            normalised_set[index]._normalise_time(self.phase.duration)
        return normalised_set

    def _curve(self, time, primary=False):
        """
        If primary, build curve in respect to 'normalised_set'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'powerload_set', since set will already have been normalised.
        """
        return sum(self._normalised_set if primary else self.powerload_set).curve(time)

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'curve' method of the 'PowerLoad' class
        to the 'PowerLoad' instance that is created by the sum of all
        'PowerLoad' objects stored in the 'powerload_set' attribute.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        return self._curve(time, primary=True)

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'powerload_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.powerload_set)

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the 'powerload_set'
        attribute (i.e. all times are their original values).
        """
        return self._build_time_from_load_set(self.powerload_set)

    @property
    def normalised_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_normalised_set'
        attribute (i.e. all times are normalised in respect to the phase
        duration).
        """
        return self._build_time_from_load_set(self._normalised_set)

    def _plot(self, primary=False, ax=None, n_points=None, **kwargs):
        """
        If primary, plot in respect to 'normalised_time'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'intrinsic_time', since set will already have been normalised.
        """
        ax = validate_axes(ax)

        computed_time = self._refine_vector(
            self.normalised_time if primary else self.intrinsic_time,
            self._validate_n_points(n_points),
        )

        return ax, self.plot_obj(
            ax,
            computed_time,
            self._curve(computed_time, primary=primary),
            self.name,
            kwargs,
            {**self._plot_kwargs, **kwargs},
            True,
        )

    def _plot_as_secondary(self, ax=None, n_points=None, **kwargs):
        return self._plot(primary=False, ax=ax, n_points=n_points, **kwargs)

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PhaseLoad' curve, built using the 'powerload_set' and
        'normalise' attributes that define the instance. The number of
        points interpolated in each curve segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in the 'powerload_set' attribute.

        Parameters
        ----------
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PowerLoad' objects, that summed result in the normal
            plotted curve. These objects are plotted as secondary plots,
            as defined in 'PowerCycleABC' class. By default this input
            is set to 'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
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
        ax, list_of_plot_objects = self._plot(
            primary=True,
            ax=ax,
            n_points=n_points,
            **kwargs,
        )

        if detailed:
            for normal_load in self._normalised_set:
                normal_load._make_secondary_in_plot()
                ax, plot_list = normal_load.plot(ax=ax)
                list_of_plot_objects.append(plot_list)

        return ax, list_of_plot_objects

    def __add__(self, other):
        """
        The addition of 'PhaseLoad' instances creates a new 'PhaseLoad'
        instance with joined 'powerload_set' and 'normalise' attributes,
        but only if its phases are the same.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)

        if this.phase != other.phase:
            raise PhaseLoadError(
                "addition",
                "The phases of this PhaseLoad addition represent "
                f"{this.phase.name!r} and {other.phase.name!r} "
                "respectively.",
            )
        return PhaseLoad(
            f"Resulting PhaseLoad for phase {this.phase.name!r}",
            this.phase,
            this.powerload_set + other.powerload_set,
            this.normalise + other.normalise,
        )


class PulseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse.

    Defines the pulse load with a set of 'PhaseLoad' instances. The list
    of 'PhaseLoad' objects given as a parameter to be stored in the
    'phaseload_set' attribute must be provided in the order they are
    expected to occur during a pulse. This ensures that the correct
    'PowerCyclePulse' object is created and stored in the 'pulse'
    attribute. This enables the instance to shift power loads in time
    accordingly.

    Time shifts of each phase load in the 'phaseload_set' occurs AFTER
    the normalization performed by each 'PhaseLoad' object. In short,
    'PulseLoad' curves are built by joining each individual 'PhaseLoad'
    curve after performing the following the manipulations in the order
    presented below:
        1) normalization, given the 'normalise' attribute of each
            'PhaseLoad', with normalization equal to the 'duration'
            of its 'phase' attribute;
        2) shift, given the sum of the 'duration' attributes of
            the 'phase' of each 'PhaseLoad' that comes before it.

    Parameters
    ----------
    name: str
        Description of the 'PulseLoad' instance.
    pulse: PowerCyclePulse
        Pulse specification, that determines the necessary phases to
        be characterized by 'PhaseLoad' objects.
    phaseload_set: PhaseLoad | list[PhaseLoad]
        Collection of 'PhaseLoad' objects that define the 'PulseLoad'
        instance. Upon initialization, phase loads are permuted to have
        the same order as the phases in the 'phase_set' attribute of
        the 'pulse' parameter. Missing cases are treated with the
        creation of an null phase load.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        'phaseload_set' attribute, ordered and with no repetitions.
    shifted_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        '_shifted_set' attribute, ordered and with no repetitions.
    """

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

    # Defaults for delimiter plots
    _delimiter_defaults = {
        "c": "darkorange",  # Line color
    }

    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * EPS

    def __init__(self, name, pulse, phaseload_set):
        super().__init__(name)

        pulse = self._validate_pulse(pulse)
        phaseload_set = self._validate_phaseload_set(phaseload_set, pulse)

        self.pulse = pulse
        self.phaseload_set = phaseload_set

    @staticmethod
    def _validate_pulse(pulse):
        PowerCyclePulse.validate_class(pulse)
        return pulse

    @staticmethod
    def _validate_phaseload_set(phaseload_set, pulse):
        """
        Validate 'phaseload_set' input to be a list of 'PhaseLoad'
        instances. Multiple phase loads for the same phase are added.
        Mssing phase loads are filled with a null instance.
        """
        phaseload_set = validate_list(phaseload_set)

        validated_phaseload_set = []
        phase_library = pulse.build_phase_library()
        for phase_in_pulse in phase_library.values():
            phaseloads_for_phase = []
            for phaseload in phaseload_set:
                PhaseLoad.validate_class(phaseload)

                phase_of_phaseload = phaseload.phase
                if phase_of_phaseload == phase_in_pulse:
                    phaseloads_for_phase.append(phaseload)

            no_phaseloads_were_added = len(phaseloads_for_phase) == 0
            if no_phaseloads_were_added:
                null_phaseload = PhaseLoad.null(phase_in_pulse)
                phaseloads_for_phase = null_phaseload

            phaseloads_for_phase = validate_list(phaseloads_for_phase)

            single_phaseload = sum(phaseloads_for_phase)
            validated_phaseload_set.append(single_phaseload)

        return validated_phaseload_set

    @classmethod
    def null(cls, pulse):
        """
        Instantiates an null version of the class.
        """
        phase_library = pulse.build_phase_library()
        return cls(
            f"Null PhaseLoad for pulse {pulse.name}",
            pulse,
            [PhaseLoad.null(phase) for phase in phase_library.values()],
        )

    def _build_pulse_from_phaseload_set(self):
        """
        Build pulse from 'PowerCyclePhase' objects stored in the
        'phase' attributes of each 'PhaseLoad' instance in the
        'phaseload_set' list.
        """
        return PowerCyclePulse(
            f"Pulse for {self.name}",
            [phaseload.phase for phaseload in self.phaseload_set],
        )

    @property
    def _shifted_set(self):
        """
        Modified 'phaseload_set' attribute, in which all times of each
        'PhaseLoad' object are shifted by the sum of 'duration' values
        of the 'phase' attribute of each 'PhaseLoad' that comes before
        it in the 'phaseload_set' attribute.
        Shifts are applied to the '_normalised_set' property of each
        'PhaseLoad' object.
        """
        phaseload_set = copy.deepcopy(self.phaseload_set)

        time_shift = 0
        shifted_set = []
        for phaseload in phaseload_set:
            normalised_set = phaseload._normalised_set
            for normal_load in normalised_set:
                normal_load._shift_time(time_shift)
            phaseload.powerload_set = normalised_set

            shifted_set.append(phaseload)
            time_shift += phaseload.phase.duration

        return shifted_set

    def curve(self, time):
        """
        Create a curve by calculating phase load values at the specified
        times.

        This method applies the 'curve' method of the 'PhaseLoad' class
        to each object stored in the '_shifted_set' attribute, and
        returns the sum of all individual curves created.

        The last point of each 'PhaseLoad' curve is shifted by a minimal
        time 'epsilon', defined as a class attribute, to avoid an
        overlap with the first point of the curve in the following phase
        and a super-position of loads at that point.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        shifted_set = self._shifted_set

        curve = []
        modified_time = []
        for shifted_load in shifted_set:
            intrinsic_time = shifted_load.intrinsic_time

            max_t = max(intrinsic_time)
            min_t = min(intrinsic_time)
            load_time = [t for t in time if (min_t <= t) and (t <= max_t)]

            load_time[-1] = load_time[-1] - self.epsilon
            load_curve = shifted_load._curve(load_time, primary=False)

            modified_time.append(load_time)
            curve.append(load_curve)

        modified_time = unnest_list(modified_time)
        curve = unnest_list(curve)

        return modified_time, curve

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'phaseload_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.phaseload_set)

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PhaseLoad' objects contained in the 'phaseload_set'
        attribute (i.e. all times are their original values).
        """
        return self._build_time_from_load_set(self.phaseload_set)

    @property
    def shifted_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_shifted_set'
        attribute (i.e. all times are shifted in respect to the
        duration of previous phases).
        """
        return self._build_time_from_load_set(self._shifted_set)

    def _plot_phase_delimiters(self, ax=None):
        """
        Add vertical lines to plot to specify where the phases of a
        pulse end.
        """
        ax = validate_axes(ax)
        axis_limits = ax.get_ylim()

        shifted_set = self._shifted_set

        default_delimiter_kwargs = self._delimiter_defaults
        delimiter_kwargs = default_delimiter_kwargs

        default_line_kwargs = self._detailed_defaults
        line_kwargs = {**default_line_kwargs, **delimiter_kwargs}

        default_text_kwargs = self._text_kwargs
        text_kwargs = {**default_text_kwargs, **delimiter_kwargs}

        list_of_plot_objects = []
        for shifted_load in shifted_set:
            intrinsic_time = shifted_load.intrinsic_time
            last_time = intrinsic_time[-1]

            label = "Phase delimiter for " + shifted_load.phase.name

            plot_object = ax.plot(
                [last_time, last_time],
                axis_limits,
                label=label,
                **line_kwargs,
            )
            list_of_plot_objects.append(plot_object)

            plot_object = ax.text(
                last_time,
                axis_limits[-1],
                "End of " + shifted_load.phase.name,
                label=label,
                **text_kwargs,
            )
            list_of_plot_objects.append(plot_object)

        return ax, list_of_plot_objects

    def plot(self, ax=None, n_points=None, detailed=False, **kwargs):
        """
        Plot a 'PulseLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.

        This method applies the 'plot' method of the 'PowerLoad' class
        to the resulting load created by the 'curve' method.

        This method can also plot the individual 'PowerLoad' objects
        stored in each 'PhaseLoad' instance of the 'phaseload_set'
        attribute.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'PhaseLoad' instances, that summed result in the normal
            plotted curve. Plotted as secondary plots, as defined in
            'PowerCycleABC' class. By default this input is set to
            'False'.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
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

        # Set each default options in kwargs, if not specified
        default_plot_kwargs = self._plot_kwargs
        final_kwargs = {**default_plot_kwargs, **kwargs}

        time_to_plot = self.shifted_time
        modified_time, computed_curve = self.curve(
            self._refine_vector(time_to_plot, self._validate_n_points(n_points))
        )

        list_of_plot_objects = self.plot_obj(
            ax, modified_time, computed_curve, self.name, kwargs, final_kwargs
        )

        if detailed:
            for shifted_load in self._shifted_set:
                shifted_load._make_secondary_in_plot()
                ax, plot_list = shifted_load._plot_as_secondary(ax=ax)
                list_of_plot_objects.append(plot_list)

            # Add phase delimiters
            ax, delimiter_list = self._plot_phase_delimiters(ax=ax)
            list_of_plot_objects += delimiter_list

        return ax, list_of_plot_objects

    def __add__(self, other):
        """
        The addition of 'PulseLoad' instances can only be performed if
        their pulses are equal. It returns a new 'PulseLoad' instance
        with a 'phaseload_set' that contains the addition of the
        respective 'PhaseLoad' objects in each original instance.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)
        if this.pulse != other.pulse:
            raise PulseLoadError(
                "addition",
                "The pulses of this PulseLoad addition represent "
                f"{this.pulse.name!r} and {other.pulse.name!r} "
                "respectively.",
            )

        return PulseLoad(
            f"Resulting PulseLoad for pulse {this.pulse.name!r}",
            this.pulse,
            this.phaseload_set + other.phaseload_set,
        )


class ScenarioLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a scenario.

    Defines the phase load with a set of 'PulseLoad' instances. Each
    instance must be accompanied by a 'repetition' specification, used
    to indicate how many times that pulse load is repeated in the
    scenario before a new set of pulse loads starts. This enables the
    instance to adjust the evolution of pulse loads accordingly, if
    changes occur to the plant scenario.

    Parameters
    ----------
    name: str
        Description of the 'ScenarioLoad' instance.
    scenario: 'PowerCycleScenario'
        Scenario specification, that determines the necessary pulses to
        be characterized by 'PulseLoad' objects.
    pulseload_set: PulseLoad | list[PulseLoad]
        Collection of instances of the 'PulseLoad' class that define
        the 'ScenarioLoad' object.

    Attributes
    ----------
    scenario: PowerCycleScenario
        Scenario specification, determined by the 'pulse' attributes of
        the 'PulseLoad' instances used to define the 'ScenarioLoad'.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PulseLoad' objects contained in the
        'pulseload_set' attribute, ordered and with no repetitions.
    timeline_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PulseLoad' objects contained in the
        '_timeline_set' attribute, ordered and with no repetitions.
    """

    @property
    def intrinsic_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PulseLoad' objects contained in the 'pulseload_set'
        attribute (i.e. all times are their original values).
        """
        # return self._build_time_from_load_set(self.pulseload_set)

    @intrinsic_time.setter
    def intrinsic_time(self, value) -> None:
        pass

    @property
    def timeline_time(self):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the '_timeline_set'
        attribute (i.e. all times are shifted in respect to the
        duration of previous pulses).
        """
        # return self._build_time_from_load_set(self._timeline_set)
