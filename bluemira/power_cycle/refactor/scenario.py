from pathlib import Path
from typing import ClassVar, Dict, Optional, Protocol, Union

import numpy as np

from bluemira.power_cycle.refactor.load_manager import create_manager_configs
from bluemira.power_cycle.refactor.loads import PulseSystemLoad
from bluemira.power_cycle.refactor.time import (
    ScenarioBuilderConfig,
    build_phase_breakdowns,
    pulse_phase_durations,
)


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
        self.pulses = self.scenario_config.get_scenario_pulses()

        self.pulse_system_loads = {
            k: PulseSystemLoad(self.pulses[k], self.manager_configs) for k in self.pulses
        }
        self._create_scenario()

    def _create_scenario(self):
        # combine scenario_config and pulse_system_loads so that
        # we have continuous pulses with repeats etc
        ...

    def plot(self):
        ...

    def tmp(self):
        phase_breakdowns = build_phase_breakdowns(self.scenario_config)

        durations_arrays = {
            k: pulse_phase_durations(phases, phase_breakdowns)
            for k, phases in self.pulses.items()
        }
        pulse_durations = {k: np.sum(v) for k, v in durations_arrays.items()}
