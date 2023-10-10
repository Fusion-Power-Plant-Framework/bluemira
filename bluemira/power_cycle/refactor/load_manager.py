from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.errors import PowerLoadError
from bluemira.power_cycle.refactor.base import Config, LibraryConfigDescriptor
from bluemira.power_cycle.tools import create_axes, read_json

if TYPE_CHECKING:
    from pathlib import Path


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
        # Set each default options in kwargs, if not specified
        return self._plot_load(
            create_axes(ax),
            self.time,
            self.data,
            self.name,
            False,
            **{**{"color": "k", "linewidth": 2, "linestyle": "-"}, **kwargs},
        )


@dataclass
class PowerCycleLoadConfig(Config):
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
