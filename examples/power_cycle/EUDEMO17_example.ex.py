# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle example"""

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from bluemira.base.file import get_bluemira_root
from bluemira.base.reactor_config import ReactorConfig
from bluemira.power_cycle.net import (
    PowerCycle,
    interpolate_extra,
)

# from _tiago_.tools import pp

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
reactor_config = ReactorConfig(config_path / "EUDEMO17_config.json", None)
power_cycle = PowerCycle(**reactor_config.config_for("Power Cycle"), durations={})


def get_pulse_load(
    power_cycle: PowerCycle,
    pulse_label: str,
    load_type: str,
    load_unit: str,
    extra_points: int = 0,
):
    """Get the total load for a specific type and unit."""
    pulse = power_cycle.get_pulse(pulse_label)
    # pp(pulse)

    phase_loads_for_type = {}
    phase_total_for_type = {}
    pulse_loads_for_type = {}
    for p_name, phase in pulse.items():
        p_times = phase.timeseries()
        new_times = interpolate_extra(p_times, n_points=extra_points)
        p_loads = phase.load(load_type, load_unit, timeseries=new_times)
        p_total = phase.total_load(load_type, load_unit, timeseries=new_times)

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
        phase_loads_for_type,
        phase_total_for_type,
        pulse_loads_for_type,
        pulse_total_for_type,
    )


(phase_loads_active, phase_total_active, pulse_loads_active, pulse_total_active) = (
    get_pulse_load(
        power_cycle,
        pulse_label="std",
        load_type="active",
        load_unit="MW",
        extra_points=10,
    )
)

# pp(phase_loads_active)
# pp(phase_total_active)
# pp(pulse_loads_active)
# pp(pulse_total_active)
