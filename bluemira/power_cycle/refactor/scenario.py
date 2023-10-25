from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Protocol, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.power_cycle.refactor.load_manager import (
    LoadType,
    PowerCycleLoadConfig,
    create_manager_configs,
)
from bluemira.power_cycle.refactor.loads import PulseSystemLoad
from bluemira.power_cycle.refactor.time import ScenarioBuilderConfig
from bluemira.utilities.tools import flatten_iterable


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict]


class PowerCycleScenario:
    def __init__(
        self,
        scenario_config: Union[str, Path],
        manager_config: Union[str, Path],
        breakdown_data: Optional[
            IsDataclass  # Probably will end up being a parameterframe
        ],
    ):
        self.manager_configs = create_manager_configs(manager_config)

        self.scenario_config = ScenarioBuilderConfig.from_file(scenario_config)
        self.scenario_config.import_breakdown_data(breakdown_data)
        self.phase_time = self.scenario_config.build_phase_breakdowns()
        self.pulse_configs = self.scenario_config.get_scenario_pulses()

        self.pulse_system_loads = {
            k: PulseSystemLoad(self.pulse_configs[k], self.manager_configs)
            for k in self.pulse_configs
        }
        self._create_scenario()

    def _create_scenario(self):
        """Create all the phases for all the pulses"""
        self.pulses = {}
        for pulse_name, phase_list in self.pulse_configs.items():
            phases = self.pulses[pulse_name] = []
            for pulse_system_load in self.pulse_system_loads.values():
                for phase_config in phase_list:
                    phases.append(
                        PowerCyclePhase(
                            phase_config,
                            pulse_system_load,
                            phase_duration=self.phase_time[phase_config.name],
                        )
                    )

    def plot(self, unit="MW"):
        # TODO include repeat pulses, also currently all pulses are plotted on a separate axis
        x = int(np.ceil(np.sqrt(len(self.pulses))))
        _, axes = plt.subplots(x, x)
        if not isinstance(axes, list):
            axes = [axes]
        for phases, ax in zip(self.pulses.values(), flatten_iterable(axes)):
            start_time = 0
            for phase in phases:
                phase.plot_data(ax=ax, phase_start=start_time, unit=unit)
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
        phase_config,
        pulse_system_load,
        phase_duration: float,
        time_resolution: int = 100,
    ):
        self.config = phase_config
        self._duration = phase_duration
        self.time_resolution = time_resolution

        self.loads = self._setup_phase(pulse_system_load)

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

    def _setup_phase(self, pulse_system_load):
        return {
            loadtype: [
                load
                for sysloads in pulse_system_load.psl[loadtype].values()
                for loads in sysloads.values()
                for load in loads
                if self.config.name in load.phases
            ]
            for loadtype in LoadType
        }

    def plot_data(self, ax=None, phase_start=0, unit="MW"):
        """Plot the phase as a stackplot"""
        # TODO improve colouring and add legend
        if ax is None:
            _, ax = plt.subplots()

        active = self._get_data_arrays(self.loads[LoadType.ACTIVE], unit=unit)
        active_offset = np.sum(active, axis=0)
        ax.stackplot(
            self._phase_duration + phase_start,
            active,
        )
        ax.stackplot(
            self._phase_duration + phase_start,
            [
                reactive_load + active_offset
                for reactive_load in self._get_data_arrays(
                    self.loads[LoadType.REACTIVE], unit=unit
                )
            ],
        )
        return ax

    def _get_data_arrays(self, load_configs: List[PowerCycleLoadConfig], unit="MW"):
        return [
            self._collect_data(load_config, unit=unit) for load_config in load_configs
        ]

    def _collect_data(self, load_config: PowerCycleLoadConfig, unit="MW"):
        normalised = load_config.normalise[load_config.phases.index(self.config.name)]
        if normalised:
            return load_config.load_total(self._time, unit=unit)
        raise NotImplementedError

    def __repr__(self) -> str:
        """Representation of the class"""
        return f"<{type(self).__name__} {self.config.name}: {self.config.description}>"
