# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle example"""

# %%
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

# from _tiago_.tools import pp
from bluemira.base.file import get_bluemira_root
from bluemira.base.reactor_config import ReactorConfig
from bluemira.power_cycle.net import (
    PowerCycle,
    interpolate_extra,
)

# %% [markdown]
# # Power Cycle EU-DEMO 2017 Baseline example
#
# First, read and build the config as in the simple example. Then build the
# power cycle.
#
# Second, extract the standard pulse from the power cycle and get loads of
# each individual phase. Then sum load arrays to write the total load during
# the pulse.
#
# Third, get the total load.
# %%
config_path = Path(get_bluemira_root(), "examples", "power_cycle")
powercycle_config = ReactorConfig(config_path / "EUDEMO17_config.json", None)
power_cycle = PowerCycle(**powercycle_config.config_for("Power Cycle"), durations={})


def get_pulse_load(
    power_cycle: PowerCycle,
    pulse_label: str,
    load_type: str,
    load_unit: str,
    extra_points: int = 0,
):
    """Get the total load for a specific type and unit."""
    pulse = power_cycle.get_pulse(pulse_label)
    phase_order = power_cycle.pulse_library.root[pulse_label].phases

    starting_time = 0
    phase_timeseries = OrderedDict()
    phase_loads_for_type = OrderedDict()
    phase_total_for_type = OrderedDict()

    pulse_timeseries = []
    pulse_loads_for_type = {}
    for p_name in phase_order:
        phase = pulse[p_name]
        p_times = phase.timeseries()
        new_times = interpolate_extra(p_times, n_points=extra_points)
        p_loads = phase.load(load_type, load_unit, timeseries=new_times)
        p_total = phase.total_load(load_type, load_unit, timeseries=new_times)

        timeseries = new_times + starting_time
        phase_timeseries[p_name] = timeseries
        pulse_timeseries.append(timeseries)
        starting_time = timeseries[-1]

        phase_loads_for_type[p_name] = p_loads
        phase_total_for_type[p_name] = p_total

        for l_name, load in p_loads.items():
            if l_name not in pulse_loads_for_type:
                pulse_loads_for_type[l_name] = load
            else:
                last_load = pulse_loads_for_type[l_name]
                pulse_loads_for_type[l_name] = np.concatenate([last_load, load])
    # pp(phase_loads_for_type)
    # pp(phase_total_for_type)
    # pp(pulse_loads_for_type)

    pulse_total_for_type = list(pulse_loads_for_type.values())
    pulse_total_for_type = np.sum(pulse_total_for_type, axis=0)
    # pp(pulse_total_for_type)

    return (
        phase_timeseries,
        phase_loads_for_type,
        phase_total_for_type,
        pulse_timeseries,
        pulse_loads_for_type,
        pulse_total_for_type,
    )


(
    phase_timeseries,
    phase_loads_active,
    phase_total_active,
    pulse_timeseries,
    pulse_loads_active,
    pulse_total_active,
) = get_pulse_load(
    power_cycle,
    pulse_label="std",
    load_type="active",
    load_unit="MW",
    extra_points=0,
)
# pp(phase_timeseries)
# pp(phase_loads_active)
# pp(phase_total_active)
# pp(pulse_timeseries)
# pp(pulse_loads_active)
# pp(pulse_total_active)

# TEMP: add turbine load to pulse, because it is zero due to bug
turbine_power = 790 * 0.95 * 0.85
consumption_Minucci_SSEN_HCPB_ftt = sum([
    -12.2,  # TFV
    -3,  # TER
    -3 * 6,  # ECH + NBI + ICH upkeeps
    -6.1,  # DGC
    -9.7,  # VV PHTS
    -165.6,  # HCPB PHTS
    -2.3,  # VVPS
    -19.5,  # DIV+LIM PHTS
    -(5 + 4.6 + 3),  # RM, Assembly, Waste
    -12,  # BOP
    -3.1,  # Site
    -101.8,  # Cryogenics
    -21,  # EPS upkeep
    # TEMP: add "eps_peak" to `plb` subphase to also model PPEN
    -54.8,  # Buildings
    -3.6,  # Plant Control
    -90.9,  # Auxiliaries
    -32,  # Switchyard estimates
])  # MW
net_ftt = turbine_power + consumption_Minucci_SSEN_HCPB_ftt

# Add turbine power to the total active load
# phase_total_active = OrderedDict(
#     (k, v + turbine_power) for k, v in phase_total_active.items()
#     )
# pulse_total_active += turbine_power
# pp(turbine_power)
# pp(consumption_Minucci_SSEN_HCPB_ftt)
# pp(net_ftt)

# Last value should be 21 in 'eps_upkeep': array([-21., -21.,  -0.])
print(phase_loads_active["ftt"]["eps_upkeep"])  # MW
