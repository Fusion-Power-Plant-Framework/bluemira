from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.refactor.load_manager import (
    PowerCycleLoadConfig,
    PowerCycleSubLoadConfig,
    create_manager_configs,
)
from bluemira.power_cycle.refactor.time import (
    ScenarioBuilderConfig,
    build_phase_breakdowns,
    get_scenario_pulses,
    pulse_phase_durations,
)


@dataclass
class PowerCycleDurationParameters:
    CS_recharge_time: float = raw_uc(5, "minute", "second")
    pumpdown_time: float = raw_uc(10, "minute", "second")
    ramp_up_time: float = 157
    ramp_down_time: float = 157


@dataclass
class PowerCyclePhaseLoads:
    CS: PowerCycleLoadConfig = PowerCycleLoadConfig("", [], [], True, {}, {})
    TF: PowerCycleLoadConfig = PowerCycleLoadConfig("", [], [], True, {}, {})
    PF: PowerCycleLoadConfig = PowerCycleLoadConfig("", [], [], True, {}, {})


manager_configs = create_manager_configs("manager_config_complete.json")

scenario_config = ScenarioBuilderConfig.from_file("scenario_config.json")
scenario_config.import_breakdown_data(PowerCycleDurationParameters())

phase_breakdowns = build_phase_breakdowns(scenario_config)

pulses = get_scenario_pulses(scenario_config)
durations_arrays = {
    k: pulse_phase_durations(phases, phase_breakdowns) for k, phases in pulses.items()
}
pulse_durations = {k: np.sum(v) for k, v in durations_arrays.items()}
