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
class Efficiency:
    """Efficiency data container"""

    value: float
    desc: str = ""
    reactive: Optional[bool] = None


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


class PhaseEfficiencyDescriptor(Descriptor):
    """Efficiency descriptor for use with dataclasses"""

    def __get__(self, obj: Any, _) -> List[Efficiency]:
        """Get the config"""
        if obj is None:
            return dict
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: List[Union[Dict[str, Union[str, float, bool]], Efficiency]],
    ):
        """Setup the config"""
        if callable(value):
            value = value()
        for k, val in value.items():
            for no, v in enumerate(val):
                if not isinstance(v, Efficiency):
                    value[k][no] = Efficiency(**v)

        setattr(obj, self._name, value)


class LoadEfficiencyDescriptor(Descriptor):
    """Efficiency descriptor for use with dataclasses"""

    def __get__(self, obj: Any, _) -> List[Efficiency]:
        """Get the config"""
        if obj is None:
            return list
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: List[Union[Dict[str, Union[str, float, bool]], Efficiency]],
    ):
        """Setup the config"""
        if callable(value):
            value = value()
        for no, v in enumerate(value):
            if not isinstance(v, Efficiency):
                value[no] = Efficiency(**v)

        setattr(obj, self._name, value)


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
class PowerCycleSubPhase(Config):
    """SubPhase Config"""

    duration: Union[float, str]
    loads: list[str] = field(default_factory=list)
    efficiencies: PhaseEfficiencyDescriptor = PhaseEfficiencyDescriptor()
    unit: str = "s"
    description: str = ""
    reference: str = ""

    def __post_init__(self):
        """Enforce unit conversion"""
        if isinstance(self.duration, (float, int)):
            self.duration = raw_uc(self.duration, self.unit, "second")
            self.unit = "s"


@dataclass
class PowerCycleSystem(Config):
    """Power cycle system config"""

    subsystems: List[str]
    description: str = ""


@dataclass
class PowerCycleSubSystem(Config):
    """Power cycle sub system config"""

    loads: List[str]
    description: str = ""


@dataclass
class PowerCycleLoad(Config):
    """Power cycle load config"""

    time: npt.ArrayLike = field(default_factory=lambda: np.arange(2))
    reactive_data: npt.ArrayLike = field(default_factory=lambda: np.zeros(2))
    active_data: npt.ArrayLike = field(default_factory=lambda: np.zeros(2))
    efficiencies: LoadEfficiencyDescriptor = LoadEfficiencyDescriptor()
    model: Union[LoadModel, str] = LoadModel.RAMP
    unit: str = "W"
    description: str = ""
    normalised: bool = True
    consumption: bool = True

    def __post_init__(self):
        """Validate load"""
        for var_name in ("time", "reactive_data", "active_data"):
            var = getattr(self, var_name)
            if not isinstance(var, np.ndarray):
                setattr(self, var_name, np.array(var))
        if isinstance(self.model, str):
            self.model = LoadModel[self.model.upper()]
        for data in (self.reactive_data, self.active_data):
            if data.size != self.time.size:
                raise ValueError(
                    f"time and data must be the same length for {self.name}:" f"{data}"
                )
        if any(np.diff(self.time) < 0):
            raise ValueError("time must increase")

        self.reactive_data = raw_uc(self.reactive_data, self.unit, "W")
        self.active_data = raw_uc(self.active_data, self.unit, "W")
        self.unit = "W"

    def interpolate(
        self,
        time: npt.ArrayLike,
        end_time: Optional[float] = None,
        load_type: Union[str, LoadType] = LoadType.ACTIVE,
    ) -> np.ndarray:
        """
        Interpolate load for a given time vector

        Notes
        -----
        The interpolation type is set by subload.model.
        Any out-of-bound values are set to zero.
        """
        if isinstance(load_type, str):
            load_type = LoadType[load_type.upper()]
        return interp1d(
            self.time,
            getattr(self, f"{load_type.name}_data"),
            kind=self.model.value,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )(time if end_time is None else np.array(time) * end_time)


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

    loads: dict[PowerCycleLoad]
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
        sph_keys = self.subphase.keys()
        ss_keys = self.subsystem.keys()
        loads = self.loads.keys()
        # scenario has known pulses
        if unknown_pulse := self.scenario.pulses.keys() - self.pulse.keys():
            raise ValueError(f"Unknown pulses {unknown_pulse}")
        # pulses have known phases
        for pulse in self.pulse.values():
            if unknown_phase := pulse.phases - ph_keys:
                raise ValueError(f"Unknown phases {unknown_phase}")
        # phases have known subphases
        for ph_c in self.phase.values():
            if unknown_s_ph := ph_c.subphases - sph_keys:
                raise ValueError(f"Unknown subphase configurations {unknown_s_ph}")
        # subphases have known loads
        for subphase in self.subphase.values():
            if unknown_load := subphase.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")

        # systems have known subsystems
        for sys_c in self.system.values():
            if unknown := sys_c.subsystems - ss_keys:
                raise ValueError(f"Unknown subsystem configurations {unknown}")
        # subsystems have known loads
        for s_sys_c in self.subsystem.values():
            if unknown_load := s_sys_c.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")

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

    @classmethod
    def from_json(cls, manager_config_path: Union[Path, str]):
        """Create configuration from json"""
        json_content = read_json(manager_config_path)
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
            loads={
                k: PowerCycleLoad(name=k, **v)
                for k, v in json_content["load_library"].items()
            },
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
