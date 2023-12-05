from pathlib import Path
from typing import ClassVar, Dict, Iterable, Optional, Protocol, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.power_cycle.refactor.load_manager import (
    LoadType,
    PowerCycleLoadConfig,
)

# create_manager_configs,
from bluemira.power_cycle.refactor.loads import PulseSystemLoad
from bluemira.power_cycle.refactor.time import ScenarioBuilderConfig


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
            loadtype: {
                f"{sysname}:{loadsname}:{load.name}": load
                for sysname, sysloads in pulse_system_load.psl[loadtype].items()
                for loadsname, loads in sysloads.items()
                for load in loads
                if self.config.name in load.phases
            }
            for loadtype in LoadType
        }

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

    def _get_data_arrays(self, load_configs: Dict[str, PowerCycleLoadConfig], unit="MW"):
        return {
            name: self._collect_data(load_config, unit=unit)
            for name, load_config in load_configs.items()
        }

    def _collect_data(self, load_config: PowerCycleLoadConfig, unit="MW"):
        normalised = load_config.normalise[load_config.phases.index(self.config.name)]
        if normalised:
            return load_config.load_total(self._time, unit=unit)
        raise NotImplementedError

    def __repr__(self) -> str:
        """Representation of the class"""
        return f"<{type(self).__name__} {self.config.name}: {self.config.description}>"

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
