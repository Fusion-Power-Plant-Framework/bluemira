from __future__ import annotations

from dataclasses import dataclass

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.net import (
    PowerCycleLibraryConfig,
    PowerCycleLoadConfig,
    PowerCycleSubLoad,
    interpolate_extra,
)


@dataclass
class PowerCycleDurationParameters:
    CS_recharge_time: float = raw_uc(5, "minute", "second")
    pumpdown_time: float = raw_uc(10, "minute", "second")
    ramp_up_time: float = 157
    ramp_down_time: float = 157


config = PowerCycleLibraryConfig.from_json("scenario_config.json")
config.import_breakdown_data(PowerCycleDurationParameters())

config.add_load_config(
    "active",
    PowerCycleLoadConfig("CS", ["d2f"], True, {}, ["cs_power"], "something made up"),
)

config.add_subload(
    "active", PowerCycleSubLoad("cs_power", [0, 1], [10, 20], "RAMP", "MW", "dunno")
)

phase = config.make_phase("dwl")

normalised_time = interpolate_extra(phase.loads.build_timeseries(), 5 - 2)
active_loads = phase.loads.get_load_data_with_efficiencies(
    normalised_time, "active", "MW"
)
active_load_total = phase.loads.load_total(normalised_time, "active", "MW")

reactive_loads = phase.loads.get_load_data_with_efficiencies(
    normalised_time, "reactive", "MW"
)
reactive_load_total = phase.loads.load_total(normalised_time, "reactive", "MW")

timeseries = normalised_time * phase.duration
