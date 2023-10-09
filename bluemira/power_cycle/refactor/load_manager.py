from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.power_cycle.errors import PowerLoadError
from bluemira.power_cycle.refactor.base import Config, LibraryConfigDescriptor
from bluemira.power_cycle.tools import read_json, validate_axes

if TYPE_CHECKING:
    from pathlib import Path


def interpolate(vector: npt.NDArray, n_points: int):
    """
    Add points between each point in a vector.
    """
    if n_points == 0:
        return vector

    return np.concatenate(
        [
            *(
                np.linspace(vector[s], vector[s + 1], n_points + 1, endpoint=False)
                for s in range(len(vector) - 1)
            ),
            np.atleast_1d(vector[-1]),
        ]
    )


class PlotMixin:
    def _plot_load(self, ax, x, y, name, kwargs, plot_or_scatter=True):
        if plot_or_scatter:
            plot_type = ax.plot
            lab = "(curve)"
        else:
            plot_type = ax.scatter
            lab = "(data)"
        plot_type(x, y, label=f"{type(self).__name__} {name} {lab}", **kwargs)
        ax.legend()
        return ax


def _add_text_to_point_in_plot(
    function_name,
    axes,
    name,
    x_list,
    y_list,
    text_kwargs=None,
    **kwargs,
):
    index = 0
    _text_kwargs = text_kwargs or {"color": "k", "size": 10, "rotation": 45}
    text_to_be_added = f"{name} ({function_name})"
    label_of_text_object = f"{name} (name)"

    # Fall back on default kwargs if wrong keys are passed
    try:
        axes.text(
            x_list[index],
            y_list[index],
            text_to_be_added,
            label=label_of_text_object,
            **{**_text_kwargs, **kwargs},
        )
    except AttributeError:
        bluemira_warn("Unknown key word argument falling back to plot defaults")
        axes.text(
            x_list[index],
            y_list[index],
            text_to_be_added,
            label=label_of_text_object,
            **_text_kwargs,
        )


class LoadType(Enum):
    ACTIVE = auto()
    REACTIVE = auto()


class LoadModel(Enum):
    """
    Members define possible models used.

    Maps model names to 'interp1d' interpolation behaviour.
    """

    RAMP = "linear"
    STEP = "previous"


@dataclass
class PowerCycleSubLoadConfig(Config, PlotMixin):
    """Power cycle sub load config"""

    time: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    model: Optional[Union[LoadModel, str]] = None
    unit: str = "W"
    description: str = ""

    def __post_init__(self):
        if any(isinstance(i, type(None)) for i in (self.time, self.data, self.model)):
            self = self.null()
        for var_name in ("time", "data"):
            var = getattr(self, var_name)
            if not isinstance(var, np.ndarray):
                setattr(self, var_name, np.array(var))
        if isinstance(self.model, str):
            self.model = LoadModel[self.model.upper()]
        if self.data.size != self.time.size:
            raise ValueError(f"time and data must be the same length for {self.name}")
        if any(np.diff(self.time) < 0):
            raise ValueError("time must increase")

        self.data = raw_uc(self.data, self.unit, "W")
        self.unit = "W"

    @classmethod
    def null(cls):
        return cls(
            "Null SubLoad",
            time=np.arange(2),
            data=np.zeros(2),
            model=LoadModel.RAMP,
        )

    def interpolate(self, time: np.ndarray):
        """
        Interpolate subload for a given time vector

        Notes
        -----
        The interpolation type is set by subload.model.
        Any out-of-bound values are set to zero.
        """
        return interp1d(
            self.time,
            self.data,
            kind=self.model.value,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )(time)

    def plot(self, ax=None, color="k", linewidth=2, linestyle="-", **kwargs):
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
        kwargs["color"] = color
        kwargs["linewidth"] = linewidth
        kwargs["linestyle"] = linestyle
        # Set each default options in kwargs, if not specified
        return self._plot_load(
            validate_axes(ax),
            self.time,
            self.data,
            self.name,
            kwargs,
            False,
        )


@dataclass
class PowerCycleLoadConfig(Config, PlotMixin):
    phases: list[str]
    normalise: list[bool]
    consumption: bool
    efficiencies: dict  # todo  another dataclass
    loads: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleSubLoadConfig
    )
    description: str = ""

    def __post_init__(self):
        if len(self.phases) != len(self.normalise):
            raise ValueError(
                f"phases and normlise must be the same length for {self.name}"
            )

    def get_load_data_with_efficiencies(self):
        data = self.get_explicit_data_consumption()
        for name in data:
            for efficiency in self.efficiencies.values():
                if self.consumption:
                    eff = 1 / efficiency
                data[name] *= eff
        return data

    def get_explicit_data_consumption(self):
        return {
            load.name: -load.data if self.consumption else load.data
            for load in self.loads.values()
        }

    def build_timeseries(self):
        try:
            return np.unique(np.concatenate([ld.time for ld in self.loads.values()]))
        except ValueError:
            if not self.loads:
                raise PowerLoadError(f"{self.name} has no loads") from None
            raise

    def load_total(self, timeseries: np.ndarray):
        return np.sum(
            [subload.interpolate(timeseries) for subload in self.loads.values()],
            axis=0,
        )

    def __radd__(self, other):
        """
        The reverse addition operator, to enable the 'sum' method for
        children classes that define the '__add__' method.
        """
        return self.__add__(other)

    def __add__(self, other: Union[PowerCycleLoadConfig, float]):
        this = copy.deepcopy(self)
        if isinstance(other, float):
            for load in this.loads.values():
                load.data += other
            return this

        other = copy.deepcopy(other)

        """
        The addition of 'PowerLoad' instances creates a new 'PowerLoad'
        instance with joined 'loaddata_set' and 'loadmodel_set'
        attributes.
        """
        return PowerLoad(
            "Resulting PowerLoad",
            this.loaddata_set + other.loaddata_set,
            this.loadmodel_set + other.loadmodel_set,
        )
        ...
        return None

    def __mul__(self, other: float):
        this = copy.deepcopy(self)
        for load in this.loads.values():
            load.data *= other
        return this

    def __truediv__(self, other: float):
        this = copy.deepcopy(self)
        for load in this.loads.values():
            load.data /= other
        return this

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


@dataclass
class PowerCycleSystemLoadConfig:
    name: str
    reactive: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleLoadConfig
    )
    active: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleLoadConfig
    )
    description: str = ""

    def plot(self, ax=None, n_points=100, detailed=False, **kwargs):
        for load in self.active.values():
            ax = load.plot(
                ax=ax, n_points=n_points, detailed=detailed, color="r", **kwargs
            )
        for load in self.reactive.values():
            load.plot(ax=ax, n_points=n_points, detailed=detailed, color="b", **kwargs)
        return ax


@dataclass
class PowerCycleManagerConfig(Config):
    config_path: str
    systems: dict[str, PowerCycleSystemLoadConfig]
    description: str = ""


def create_manager_configs(manager_config_path: Union[Path, str]):
    manager_configs = {}
    for key, val in read_json(manager_config_path).items():
        json_contents = read_json(val["config_path"])
        val["systems"] = {
            system: PowerCycleSystemLoadConfig(name=system, **json_contents[system])
            for system in val["systems"]
        }
        manager_configs[key] = PowerCycleManagerConfig(name=key, **val)
    return manager_configs


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


def _shift_time(subload, time):
    new_subload = copy.deepcopy(subload)
    new_subload.time += time
    new_subload._shift = time
    return new_subload


def _normalise_time(subload, new_end_time):
    new_subload = copy.deepcopy(subload)
    new_subload._norm = new_end_time / subload.time[-1]
    new_subload.time *= new_subload._norm
    return new_subload


def _normalise_time(load, new_end_time):
    for subload in load.loads.values():
        subload, _normalise_time(new_end_time)


def _shift_time(load, new_end_time):
    for subload in load.loads.values():
        subload, _shift_time(new_end_time)
