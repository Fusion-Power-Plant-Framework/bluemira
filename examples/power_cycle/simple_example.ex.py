# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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

from bluemira.base.reactor_config import ReactorConfig
from bluemira.power_cycle.net import (
    Efficiency,
    Load,
    PowerCycle,
    interpolate_extra,
)

# %% [markdown]
# # Power Cycle example
#
# Firstly we read in the build config and extract the config for the PowerCycle.
# We import any subphase durations needed for the config.
# In principle these could come from other parts of the reactor design.
# %%

reactor_config = ReactorConfig(Path(__file__).parent / "scenario_config.json", None)
config = PowerCycle(
    **{
        **reactor_config.config_for("Power Cycle"),
        "durations": {
            "cs_recharge_time": 300,
            "pumpdown_time": 600,
            "ramp_up_time": 157,
            "ramp_down_time": 157,
        },
    },
)

# %% [markdown]
# We can then dynamically add a new load to a specific subphase of the config.
# %%

config.add_load(
    "cs_power",
    load=Load(
        data={"active": [1, 2], "reactive": [10, 20]},
        efficiencies=[Efficiency(value=0.1)],
        description="something made up",
    ),
    subphases=["cru", "bri"],
    subphase_efficiency=[Efficiency(value={"reactive": 0.2})],
)

# %% [markdown]
# Once the config is created we can now pull out the data for a specific phase.
#
# Below we have interpolated the timeseries and pulled out the active and reactive loads
# %%

phase = config.get_phase("dwl")

timeseries = interpolate_extra(phase.timeseries(), 1)

active_loads = phase.load("active", "MW", timeseries=timeseries)
active_load_total = phase.total_load("active", "MW", timeseries=timeseries)

# %% [markdown]
# Note for reactive loads the unit is 'var' (volt-ampere reactive). Although numerically
# identical to a watt it is the wrong unit for reactive loads.
# %%

reactive_loads = phase.load("reactive", "Mvar", timeseries=timeseries)
reactive_load_total = phase.total_load("reactive", "Mvar", timeseries=timeseries)
