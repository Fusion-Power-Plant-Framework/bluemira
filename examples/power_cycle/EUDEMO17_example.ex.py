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
# interest.
# %%
config_path = Path(get_bluemira_root(), "examples", "power_cycle")
reactor_config = ReactorConfig(config_path / "EUDEMO17_config.json", None)
power_cycle = PowerCycle(**reactor_config.config_for("Power Cycle"), durations={})

pulse = power_cycle.get_pulse("std")
# pp(pulse)

timeseries = {
    name: interpolate_extra(phase.timeseries(), 1) for name, phase in pulse.items()
}
# pp(timeseries)

pulse_loads_active = {}
for name, phase in pulse.items():
    ts = timeseries[name]
    pulse_loads_active[name] = phase.load("active", "MW", timeseries=ts)
# pp(pulse_loads_active)
