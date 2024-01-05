# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle net loads"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.tools import read_json

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


@dataclass
class Config:
    """Configuration base dataclass"""

    name: str


class LoadType(Enum):
    """Possibly load types"""

    ACTIVE = auto()
    REACTIVE = auto()

    @classmethod
    def from_str(cls, load_type: Union[str, LoadType]) -> LoadType:
        """Create loadtype from str"""
        if isinstance(load_type, str):
            return cls[load_type.upper()]
        return load_type

    @property
    def as_str(self) -> str:
        """Load type as a string"""
        return f"{self.name.lower()}_loads"


class LoadModel(Enum):
    """
    Members define possible models used.

    Maps model names to 'interp1d' interpolation behaviour.
    """

    RAMP = "linear"
    STEP = "previous"


@dataclass
class ScenarioConfig(Config):
    """Power cycle scenario config"""

    pulses: dict[str, int]
    description: str = ""


@dataclass
class PulseConfig(Config):
    """Power cycle pulse config"""

    phases: list[str]
    description: str = ""


@dataclass
class PhaseConfig(Config):
    """Power cycle phase config"""

    operation: str
    subphases: list[str]
    description: str = ""


@dataclass
class PowerCycleLoadConfig(Config):
    """Power cycle load config"""

    consumption: bool
    efficiencies: dict  # todo  another dataclass?
    subloads: List[str]
    description: str = ""


@dataclass
class PowerCycleSubSystem(Config):
    """Power cycle sub system config"""

    reactive_loads: List[str]
    active_loads: List[str]
    description: str = ""


@dataclass
class PowerCycleSystem(Config):
    """Power cycle system config"""

    subsystems: List[str]
    description: str = ""


class Descriptor:
    """Data class property descriptor"""

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = f"_{name}"


class ScenarioConfigDescriptor(Descriptor):
    """Scenario config descriptor for use with dataclasses"""

    def __get__(self, obj: Any, _) -> ScenarioConfig:
        """Get the scenario config"""
        return getattr(obj, self._name)

    def __set__(self, obj: Any, value: Union[dict, ScenarioConfig]):
        """Set the scenario config"""
        if not isinstance(value, ScenarioConfig):
            value = ScenarioConfig(**value)

        setattr(obj, self._name, value)


class LibraryConfigDescriptor(Descriptor):
    """Config descriptor for use with dataclasses"""

    def __init__(self, *, config: Type[Config]):
        self.config = config

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
            if not isinstance(v, self.config):
                value[k] = self.config(name=k, **v)

        setattr(obj, self._name, value)


def interpolate_extra(vector: npt.NDArray, n_points: int):
    """
    Add points between each point in a vector.
    """
    if n_points == 0:
        return vector

    return np.concatenate([
        *(
            np.linspace(vector[s], vector[s + 1], n_points + 1, endpoint=False)
            for s in range(len(vector) - 1)
        ),
        np.atleast_1d(vector[-1]),
    ])


@dataclass
class PowerCycleSubPhase(Config):
    """SubPhase Config"""

    duration: Union[float, str]
    reactive_loads: list[str] = field(default_factory=list)
    active_loads: list[str] = field(default_factory=list)
    unit: str = "s"
    description: str = ""
    reference: str = ""

    def __post_init__(self):
        """Enforce unit conversion"""
        if isinstance(self.duration, (float, int)):
            self.duration = raw_uc(self.duration, self.unit, "second")
            self.unit = "s"


@dataclass
class PowerCycleSubLoad(Config):
    """Power cycle sub load config"""

    time: npt.ArrayLike
    data: npt.ArrayLike
    model: Union[LoadModel, str]
    unit: str = "W"
    description: str = ""
    normalised: bool = True

    def __post_init__(self):
        """Validate subload"""
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
        """Create empty subload"""
        return cls(
            "Null SubLoad",
            time=np.arange(2),
            data=np.zeros(2),
            model=LoadModel.RAMP,
        )

    def interpolate(
        self, time: npt.ArrayLike, end_time: Optional[float] = None
    ) -> np.ndarray:
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
        )(time if end_time is None else np.array(time) * end_time)


@dataclass
class SubLoadLibrary:
    """Subload collector"""

    load_type: LoadType
    loads: LibraryConfigDescriptor = LibraryConfigDescriptor(config=PowerCycleSubLoad)


@dataclass
class LoadLibrary:
    """Load collector"""

    load_type: LoadType
    loads: LibraryConfigDescriptor = LibraryConfigDescriptor(config=PowerCycleLoadConfig)


class Loads:
    """Loads of a phase"""

    def __init__(
        self,
        load_config: Dict[LoadType, Dict[str, PowerCycleLoadConfig]],
        subloads: Dict[LoadType, Dict[str, PowerCycleSubLoad]],
    ):
        self.load_config = load_config
        self.subloads = subloads

    @staticmethod
    def _normalise_timeseries(
        time: np.ndarray, end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[float]]:
        if min(time) < 0:
            raise NotImplementedError("Negative time not supported")

        if max(time) > 1:
            mx_time = max(time)
            return time / mx_time, mx_time if end_time is None else end_time
        return time, end_time

    def get_load_data_with_efficiencies(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W]
        end_time:
            for unnormalised subloads this assures the subload is
            applied at the right point in time
        """
        load_type = LoadType.from_str(load_type)
        data = self.get_explicit_data_consumption(timeseries, load_type, unit, end_time)
        for load_conf in self.load_config[load_type].values():
            for eff in load_conf.efficiencies.values():
                c_eff = 1 / eff if load_conf.consumption else eff
                for sl in load_conf.subloads:
                    data[sl] *= c_eff

        return data

    def get_explicit_data_consumption(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get data with consumption resulting in a negative load

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W]
        end_time:
            for unnormalised subloads this assures the subload is
            applied at the right point in time
        """
        load_type = LoadType.from_str(load_type)
        subload = self.get_interpolated_loads(timeseries, load_type, unit, end_time)
        return {
            sl: -subload[sl] if load_conf.consumption else subload[sl]
            for load_conf in self.load_config[load_type].values()
            for sl in load_conf.subloads
        }

    def build_timeseries(self, end_time: Optional[float] = None) -> np.ndarray:
        """Build a combined time series based on subloads"""
        times = []
        for lt in self.subloads.values():
            for ld in lt.values():
                if ld.normalised:
                    times.append(ld.time)
                else:
                    times.append(
                        ld.time / (max(ld.time) if end_time is None else end_time)
                    )
        return np.unique(np.concatenate(times))

    def get_interpolated_loads(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get loads for a given time series

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W]
        end_time:
            for unnormalised subloads this assures the subload is
            applied at the right point in time
        """
        timeseries, end_time = self._normalise_timeseries(timeseries, end_time)
        load_type = LoadType.from_str(load_type)
        subload = self.subloads[load_type]
        return {
            sl: subload[sl].interpolate(timeseries, end_time)
            if unit is None
            else raw_uc(
                subload[sl].interpolate(timeseries, end_time),
                subload[sl].unit,
                unit,
            )
            for load in self.load_config[load_type].values()
            for sl in load.subloads
        }

    def load_total(
        self,
        timeseries: np.ndarray,
        load_type: Union[str, LoadType],
        unit: Optional[str] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Total load for each timeseries point for a given load_type

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W]
        end_time:
            for unnormalised subloads this assures the subload is
            applied at the right point in time
        """
        return np.sum(
            list(
                self.get_load_data_with_efficiencies(
                    timeseries, load_type, unit, end_time
                ).values()
            ),
            axis=0,
        )


@dataclass
class Phase:
    """Phase container"""

    config: PhaseConfig
    subphases: Dict[str, PowerCycleSubPhase]
    loads: Loads

    def __post_init__(self):
        """Validate duration"""
        if self.duration < 0:
            raise ValueError(
                f"{self.config.name} phase duration must be positive: {self.duration}s"
            )

    @property
    def duration(self):
        """Duration of phase"""
        return getattr(np, self.config.operation)([
            s_ph.duration for s_ph in self.subphases.values()
        ])


@dataclass
class PowerCycleLibraryConfig:
    """Power Cycle Configuration"""

    load: Dict[LoadType, LoadLibrary]
    subload: Dict[LoadType, SubLoadLibrary]
    scenario: ScenarioConfigDescriptor = ScenarioConfigDescriptor()
    pulse: LibraryConfigDescriptor = LibraryConfigDescriptor(config=PulseConfig)
    phase: LibraryConfigDescriptor = LibraryConfigDescriptor(config=PhaseConfig)
    subphase: LibraryConfigDescriptor = LibraryConfigDescriptor(
        config=PowerCycleSubPhase
    )
    system: LibraryConfigDescriptor = LibraryConfigDescriptor(config=PowerCycleSystem)
    subsystem: LibraryConfigDescriptor = LibraryConfigDescriptor(
        config=PowerCycleSubSystem
    )

    def check_config(self):
        """Check powercycle configuration"""
        ph_keys = self.phase.keys()
        bl_keys = self.subphase.keys()
        ss_keys = self.subsystem.keys()
        r_loads = self.load[LoadType.REACTIVE].loads.keys()
        a_loads = self.load[LoadType.ACTIVE].loads.keys()
        # scenario has known pulses
        if unknown_pulse := self.scenario.pulses.keys() - self.pulse.keys():
            raise ValueError(f"Unknown pulses {unknown_pulse}")
        # pulses have known phases
        for pulse in self.pulse.values():
            if unknown_phase := pulse.phases - ph_keys:
                raise ValueError(f"Unknown phases {unknown_phase}")
        # phases have known subphases
        for ph_c in self.phase.values():
            if unknown_s_ph := ph_c.subphases - bl_keys:
                raise ValueError(f"Unknown subphase configurations {unknown_s_ph}")
        # subphases have known loads
        for subphase in self.subphase.values():
            if unknown_r_load := subphase.reactive_loads - r_loads:
                raise ValueError(f"Unknown reactive loads {unknown_r_load}")
            if unknown_a_load := subphase.active_loads - a_loads:
                raise ValueError(f"Unknown reactive loads {unknown_a_load}")

        # systems have known subsystems
        for sys_c in self.system.values():
            if unknown := sys_c.subsystems - ss_keys:
                raise ValueError(f"Unknown subsystem configurations {unknown}")
        # subsystems have known loads
        for s_sys_c in self.subsystem.values():
            for entry in LoadType:
                if (
                    unknown := getattr(s_sys_c, entry.as_str)
                    - self.load[entry].loads.keys()
                ):
                    raise ValueError(
                        f"Unknown load configurations in subsystem {unknown}"
                    )
        # loads have known subloads
        for load, subload in zip(self.load.values(), self.subload.values()):
            for sl in load.loads.values():
                if unknown := sl.subloads - subload.loads.keys():
                    raise ValueError(f"Unknown subload configurations in load {unknown}")

    def import_subphase_data(self, subphase_duration_params):
        """Import subphase data"""
        for s_ph in self.subphase.values():
            if isinstance(s_ph.duration, str):
                s_ph.duration = getattr(
                    subphase_duration_params, s_ph.duration.replace("-", "_")
                )

    def add_load_config(
        self,
        load_type: Union[str, LoadType],
        subphases: Union[str, Iterable[str]],
        load_config: PowerCycleLoadConfig,
    ):
        """Add load config"""
        load_type = LoadType.from_str(load_type)
        self.load[load_type].loads[load_config.name] = load_config
        if isinstance(subphases, str):
            subphases = [subphases]
        for subphase in subphases:
            getattr(self.subphase[subphase], load_type.as_str).append(load_config.name)

    def add_subload(
        self,
        load_type: Union[str, LoadType],
        subload: PowerCycleSubLoad,
    ):
        """Add subload"""
        self.subload[LoadType.from_str(load_type)].loads[subload.name] = subload

    @classmethod
    def from_json(cls, manager_config_path: Union[Path, str]):
        """Create configuration from json"""
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
            subphase={
                k: PowerCycleSubPhase(name=k, **v)
                for k, v in json_content["subphase_library"].items()
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

    def make_phase(self, phase: str, *, check=True) -> Phase:
        """Create a single phase object"""
        if check:
            self.check_config()

        phase_config = self.phase[phase]
        subphases = {k: self.subphase[k] for k in phase_config.subphases}
        phase_loads = {}
        phase_subloads = {}
        for loadtype in LoadType:
            phase_loads[loadtype] = {}
            phase_subloads[loadtype] = {}
            subloads = self.subload[loadtype].loads
            for subphase in subphases.values():
                for ld in getattr(subphase, loadtype.as_str):
                    load = self.load[loadtype].loads[ld]
                    phase_loads[loadtype][ld] = load
                    for sl in load.subloads:
                        phase_subloads[loadtype][sl] = subloads[sl]

        return Phase(phase_config, subphases, Loads(phase_loads, phase_subloads))

    def make_pulse(self, pulse: str, *, check=True) -> Dict[str, Phase]:
        """Create a pulse dictionary"""
        if check:
            self.check_config()
        return {
            phase: self.make_phase(phase, check=False)
            for phase in self.pulse[pulse].phases
        }

    def make_scenario(self) -> Dict[str, Dict[str, Union[float, Dict[str, Phase]]]]:
        """Create a scenario dictionary"""
        self.check_config()
        return {
            pulse: {"repeat": reps, "data": self.make_pulse(pulse, check=False)}
            for pulse, reps in self.scenario.pulses.items()
        }
