from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.errors import PowerLoadError
from bluemira.power_cycle.refactor.base import (
    Config,
    Descriptor,
    LibraryConfigDescriptor,
)
from bluemira.power_cycle.refactor.time import (
    PhaseConfig,
    PowerCycleBreakdownConfig,
    PulseConfig,
    ScenarioConfig,
    ScenarioConfigDescriptor,
)
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
        if any(i is None for i in (self.time, self.data, self.model)):
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
    subloads: List[str]
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

    def load_total(self, timeseries: np.ndarray, unit=None):
        return np.sum(
            [
                subload.interpolate(timeseries)
                if unit is None
                else raw_uc(subload.interpolate(timeseries), subload.unit, unit)
                for subload in self.loads.values()
            ],
            axis=0,
        )


class LoadConfigDescriptor(Descriptor):
    """Config descriptor for use with dataclasses"""

    def __init__(
        self,
        *,
        library: Config,
    ):
        self.library = library

    def __get__(self, obj: Any, _) -> Dict[str, Config]:
        """Get the config"""
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Dict[str, Union[Config, Dict]],
    ):
        """Setup the config"""
        for k, v in value.items():
            if not isinstance(v, self.library):
                value[k] = self.library(name=k, **v)

        setattr(obj, self._name, value)


@dataclass
class SubLoadLibrary:
    load_type: LoadType
    loads: LoadConfigDescriptor = LoadConfigDescriptor(library=PowerCycleSubLoadConfig)


@dataclass
class LoadLibrary:
    load_type: LoadType
    loads: LoadConfigDescriptor = LoadConfigDescriptor(library=PowerCycleLoadConfig)


@dataclass
class PowerCycleSubSystem:
    name: str
    reactive_loads: List[str]
    active_loads: List[str]
    description: str = ""


@dataclass
class PowerCycleSystem(Config):
    subsystems: List[str]
    description: str = ""


@dataclass
class PowerCycleLibraryConfig:
    load: Dict[LoadType, LoadLibrary]
    subload: Dict[LoadType, SubLoadLibrary]
    scenario: ScenarioConfigDescriptor = ScenarioConfigDescriptor()
    pulse: LibraryConfigDescriptor = LibraryConfigDescriptor(library_config=PulseConfig)
    phase: LibraryConfigDescriptor = LibraryConfigDescriptor(library_config=PhaseConfig)
    breakdown: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleBreakdownConfig
    )
    system: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleSystem
    )
    subsystem: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleSubSystem
    )

    def __post_init__(self):
        bl_keys = self.breakdown.keys()
        ss_keys = self.subsystem.keys()
        for ph_c in self.phase.values():
            if unknown := ph_c.breakdown - bl_keys:
                raise ValueError(f"Unknown breakdown configurations {unknown}")
        for sys_c in self.system.values():
            if unknown := sys_c.subsystems - ss_keys:
                raise ValueError(f"Unknown subsystem configurations {unknown}")
        for s_sys_c in self.subsystem.values():
            for entry in LoadType:
                if (
                    unknown := getattr(s_sys_c, f"{entry.name.lower()}_loads")
                    - self.load[entry].loads.keys()
                ):
                    raise ValueError(
                        f"Unknown load configurations in subsystem {unknown}"
                    )
        for load, subload in zip(self.load.values(), self.subload.values()):
            for sl in load.loads.values():
                if unknown := sl.subloads - subload.loads.keys():
                    raise ValueError(f"Unknown subload configurations in load {unknown}")

    def import_breakdown_data(self, breakdown_duration_params):
        for br in self.breakdown.values():
            if isinstance(br.duration, str):
                br.duration = getattr(
                    breakdown_duration_params, br.duration.replace("-", "_")
                )

    @classmethod
    def from_json(cls, manager_config_path: Union[Path, str]):
        json_content = read_json(manager_config_path)

        libraries = {
            "load": {},
            "subload": {},
        }
        for load_type in LoadType:
            libraries["subload"][load_type] = SubLoadLibrary(
                load_type, json_content[f"{load_type.name.lower()}_subload_library"]
            )

            libraries["load"][load_type] = LoadLibrary(
                load_type, json_content[f"{load_type.name.lower()}_load_library"]
            )

        return cls(
            scenario=ScenarioConfig(**json_content["scenario"]),
            pulse={
                k: PulseConfig(name=k, **v)
                for k, v in json_content["pulse_library"].items()
            },
            phase={
                k: PhaseConfig(name=k, **v)
                for k, v in json_content["phase_library"].items()
            },
            breakdown={
                k: PowerCycleBreakdownConfig(name=k, **v)
                for k, v in json_content["breakdown_library"].items()
            },
            system={
                k: PowerCycleSystem(name=k, **v)
                for k, v in json_content["system_library"].items()
            },
            subsystem={
                k: PowerCycleSubSystem(name=k, **v)
                for k, v in json_content["sub_system_library"].items()
            },
            **libraries,
        )

    def make_phase(self, phase: str):
        phase_config = self.phase[phase]
        return phase

    def make_pulse(self, pulse: str):
        return {phase: self.make_phase(phase) for phase in self.pulse[pulse].phases}

    def make_scenario(self):
        pulses = {pulse: self.make_pulse(pulse) for pulse in self.scenario.pulses.keys()}
        # TODO deal with repeat pulses
        return pulses
