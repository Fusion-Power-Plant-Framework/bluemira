from __future__ import annotations

from dataclasses import dataclass

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.refactor.load_manager import PowerCycleLoadConfig
from bluemira.power_cycle.refactor.scenario import PowerCycleScenario


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


scenario = PowerCycleScenario(
    "scenario_config.json",
    "manager_config_complete.json",
    PowerCycleDurationParameters(),
)
