from typing import Literal


from abc import ABC
from typing import Iterable
import numpy as np

from bluemira.base.constants import EPS
from typing import ClassVar, Dict, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np

from bluemira.power_cycle.refactor.load_manager import (
    LoadType,
)


#### PowerCycleLoadConfig
class PlotMixin:
    def _plot_load(self, ax, x, y, name, plot_or_scatter=True, **kwargs):
        if plot_or_scatter:
            plot_type = ax.plot
            lab = "(curve)"
        else:
            plot_type = ax.scatter
            lab = "(data)"
        plot_type(x, y, label=f"{type(self).__name__} {name} {lab}", **kwargs)
        ax.legend()
        return ax


class Tmp:
    def plot(
        self,
        ax=None,
        n_points=100,
        color: Literal["r", "b"] = "r",
        detailed=False,
        **kwargs,
    ):
        """
        Plot a 'PowerLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.


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

        """
        computed_time = interpolate(self.build_timeseries(), n_points)

        ax = self._plot_load(
            validate_axes(ax),
            computed_time,
            self.load_total(computed_time),
            self.name,
            kwargs,
            True,
        )

        if detailed:
            for ld in self.loads.values():
                # Plot current LoadData with seconday kwargs
                ld.plot(ax=ax, marker="x")

                # Plot current curve as line with secondary kwargs
                kwargs.update({"color": "k", "linewidth": 1, "linestyle": "--"})
                ax.plot(
                    computed_time,
                    ld.interpolate(computed_time),
                    **kwargs,
                )

        return ax

    ### PowerCycleSystemLoadConfig

    def plot(self, ax=None, n_points=100, detailed=False, **kwargs):
        for load in self.active.values():
            ax = load.plot(
                ax=ax, n_points=n_points, detailed=detailed, color="r", **kwargs
            )
        for load in self.reactive.values():
            load.plot(ax=ax, n_points=n_points, detailed=detailed, color="b", **kwargs)
        return ax


###


def _add_text_to_point_in_plot(ax, function_name, name, x, y, index=0, **kwargs):
    ax.text(
        x[index],
        y[index],
        f"{name} ({function_name})",
        label=f"{name} (name)",
        **{**{"color": "k", "size": 10, "rotation": 45}, **kwargs},
    )


def plot_phase_breakdowns(phase_breakdowns):
    curve = []
    modified_time = []
    time = interpolate(self.shifted_time, n_points)
    for shifted_load in self._shifted_set:
        intrinsic_time = shifted_load.intrinsic_time
        max_t = max(intrinsic_time)
        min_t = min(intrinsic_time)

        load_time = time[(time >= min_t) & (time <= max_t)]
        load_time[-1] = load_time[-1] - self.epsilon

        modified_time.append(load_time)
        curve.append(shifted_load._curve(load_time, primary=False))

    modified_time = np.concatenate(modified_time)
    computed_curve = np.concatenate(curve, axis=-1)

    ax = _plot_load(validate_axes(ax), modified_time, computed_curve, self.name, kwargs)

    if detailed:
        # make secondary plot
        self._text_index = 0
        self._plot_kwargs = {"color": "k", "linewidth": 1, "linestyle": "--"}
        self._scatter_kwargs = {"color": "k", "s": 100, "marker": "x"}
        self._text_kwargs = {"color": "k", "size": 10, "rotation": 45}
        for shifted_load in self._shifted_set:
            shifted_load._plot(primary=False, ax=ax, n_points=100, **kwargs)

        # Add phase delimiters

        axis_limits = ax.get_ylim()
        _delimiter_defaults = {"color": "darkorange"}
        line_kwargs = {
            **{"color": "k", "linewidth": 1, "linestyle": "--"},
            **_delimiter_defaults,
        }
        text_kwargs = {
            **{"color": "k", "size": 10, "rotation": 45},
            **_delimiter_defaults,
        }

        for shifted_load in self._shifted_set:
            intrinsic_time = shifted_load.intrinsic_time
            last_time = intrinsic_time[-1]

            label = f"Phase delimiter for {shifted_load.phase.name}"

            ax.plot(
                [last_time, last_time],
                axis_limits,
                label=label,
                **line_kwargs,
            )

            ax.text(
                last_time,
                axis_limits[-1],
                f"End of {shifted_load.phase.name}",
                label=label,
                **text_kwargs,
            )

    return ax


# COPYRIGHT PLACEHOLDER

"""
Classes for the definition of power loads in the power cycle model.
"""


class PowerCycleLoadABC(ABC):
    """
    Abstract base class for classes in the Power Cycle module that are
    used to represent and sum power loads.
    """

    # Default number of points (for any plotting method)
    _n_points = 50

    # Index of (time,data) points used as location for 'text'
    _text_index = -1

    # Pyplot defaults (kwargs arguments for 'matplotlib.pyplot' methods)
    _plot_kwargs = {"color": "k", "linewidth": 2, "linestyle": "-"}
    _scatter_kwargs = {"color": "k", "s": 100, "marker": "x"}
    _text_kwargs = {"color": "k", "size": 10, "rotation": 45}

    def __eq__(self, other):
        """
        Power Cycle objects should be considered equal even if their
        'name' and 'label' attributes are different.
        """
        if type(self) != type(other):
            return False

        attributes = list(self.__dict__.keys())
        for attr in ["name", "label"]:
            attributes.pop(attributes.index(attr))

        for attr in attributes:
            check = getattr(self, attr) == getattr(other, attr)
            if isinstance(check, Iterable):
                if not all(check):
                    return False
            elif not check:
                return False
        return True


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
    _plot_defaults = {"color": "k", "linewidth": 2, "linestyle": "-"}

    # Defaults for detailed plots
    _detailed_defaults = {"color": "k", "linewidth": 1, "linestyle": "--"}

    def __init__(self, name, phase, powerload_set, normalise):
        self.name = name
        self.phase = phase
        self.powerload_set = powerload_set
        self.normalise = normalise

    def plot(self, ax=None, n_points=100, detailed=False, primary=True, **kwargs):
        """
        If primary, plot in respect to 'normalised_time'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'intrinsic_time', since set will already have been normalised.
        """
        computed_time = interpolate(
            self._normalised_time_of_set if primary else self.intrinsic_time,
            n_points,
        )

        ax = self._plot_load(
            validate_axes(ax),
            computed_time,
            sum(self._normalised_time_of_set if primary else self.powerload_set).curve(
                computed_time
            ),
            self.name,
            kwargs,
            True,
        )
        if detailed:
            for normal_load in self._normalised_time_of_set:
                normal_load._make_secondary_in_plot()
                normal_load.plot(ax=ax)
        return ax


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
        '_ss' attribute, ordered and with no repetitions.
    """

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {"color": "k", "linewidth": 2, "linestyle": "-"}

    # Defaults for detailed plots
    _detailed_defaults = {"color": "k", "linewidth": 1, "linestyle": "--"}

    # Defaults for delimiter plots
    _delimiter_defaults = {"color": "darkorange"}

    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * EPS
    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * EPS


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict]


class PowerCycleScenario:
    def plot(self, unit="MW"):
        # TODO include repeat pulses, also currently all pulses are plotted on a separate axis
        x = int(np.ceil(np.sqrt(len(self.pulses))))
        _, axes = plt.subplots(x, x)
        if not isinstance(axes, Iterable):
            axes = np.array([axes])
        lab_col = None

        for phases, ax in zip(self.pulses.values(), axes.flat):
            start_time = 0
            for phase in phases:
                _, lab_col = phase.plot_data(
                    ax=ax, phase_start=start_time, unit=unit, lab_col=lab_col
                )
                start_time += phase.duration
        plt.show()

        return axes

    @property
    def pulse_durations(self) -> Dict[str, float]:
        """Duration of all pulses"""
        durations_arrays = {
            k: np.array([self.phase_time[phase.config.name] for phase in phases])
            for k, phases in self.pulses.items()
        }
        return {k: np.sum(v) for k, v in durations_arrays.items()}


class PowerCyclePhase:
    """A single temporal phase of a pulse."""

    def __init__(
        self,
        phase_duration: float,
        time_resolution: int = 100,
    ):
        self._duration = phase_duration
        self.time_resolution = time_resolution

    @property
    def duration(self) -> float:
        """Duration of phase"""
        return self._duration

    @duration.setter
    def duration(self, value: float):
        self._duration = value
        self._update_phase_duration()

    @property
    def time_resolution(self) -> int:
        """Number of time steps to model the phase"""
        return self._time_resolution

    @time_resolution.setter
    def time_resolution(self, value: int):
        self._time_resolution = value
        self._update_phase_duration()
        self._time = np.linspace(0, 1, num=self._time_resolution)

    def _update_phase_duration(self):
        self._phase_duration = np.linspace(0, self._duration, num=self._time_resolution)

    def plot_data(self, ax=None, phase_start=0, unit="MW", lab_col=None):
        """Plot the phase as a stacked plot"""
        if ax is None:
            _, ax = plt.subplots()

        active = self._get_data_arrays(self.loads[LoadType.ACTIVE], unit=unit)
        reactive = self._get_data_arrays(self.loads[LoadType.REACTIVE], unit=unit)

        ax.plot(
            self._phase_duration + phase_start,
            np.sum(list(active.values()), axis=0),
            label="active" if lab_col is None else "_active",
            color="k",
        )

        labels, colours, lab_col = self._define_labels_colours(
            ax,
            [a.rsplit(":", 1)[0] for tive in (active, reactive) for a in tive],
            lab_col,
        )

        stack = np.cumsum(
            np.row_stack(list(active.values()) + list(reactive.values())),
            axis=0,
        )

        x = self._phase_duration + phase_start
        coll = ax.fill_between(
            x,
            0,
            stack[0, :],
            facecolor=next(colours),
            label=next(labels, None),
            alpha=0.7,
        )
        coll.sticky_edges.y[:] = [0]
        for i in range(len(stack) - 1):
            _fillbetween_plot(ax, x, stack, i, next(colours), next(labels, None))

        ax.legend()
        return ax, lab_col

    def _define_labels_colours(self, ax, labels, lab_col=None):
        u_labels, u_ind, u_rlabels = np.unique(
            labels, return_index=True, return_inverse=True
        )
        if lab_col is None:
            col = [ax._get_lines.get_next_color() for _ in u_labels]

            labs = iter(
                [lab if ind in u_ind else f"_{lab}" for ind, lab in enumerate(labels)]
            )
            lab_col = dict(zip(u_labels, col))
        else:
            if u_labs := set(u_labels) - lab_col.keys():
                for new_l in u_labs:
                    lab_col[new_l] = ax._get_lines.get_next_color()

            col = [lab_col[name] for name in u_labels]
            labs = iter(
                [
                    lab if lab in u_labs and ind in u_ind else f"_{lab}"
                    for ind, lab in enumerate(labels)
                ]
            )

        cols = iter([col[i] for i in u_rlabels])

        return labs, cols, lab_col


def _fillbetween_plot(ax, x, y, i, fc, lab):
    return ax.fill_between(x, y[i, :], y[i + 1, :], facecolor=fc, label=lab, alpha=0.7)
    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * EPS
