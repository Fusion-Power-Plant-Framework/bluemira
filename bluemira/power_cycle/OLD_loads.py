"""
Classes to create loads in the power cycle model.
"""
import numpy as np

# Import Power Cycle packages
from bluemira.power_cycle.OLD_base import PowerCycleABC, PowerCycleError, classproperty
from bluemira.power_cycle.OLD_timeline import PowerCyclePhase
from bluemira.power_cycle.tools import add_dict_entries, validate_axes


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
        curve = np.zeros(n_time)

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
        kwargs = add_dict_entries(kwargs, default)

        # Validate axes
        ax = validate_axes(ax)

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
