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

from dataclasses import dataclass

from bluemira.power_cycle.net import (
    PowerCycleLibraryConfig,
    PowerCycleLoadConfig,
    PowerCycleSubLoad,
    interpolate_extra,
)

# %% [markdown]
# # Power Cycle example

# %%


@dataclass
class PowerCycleDurationParameters:
    """Dummy power cycle duration parameters [s]"""

    CS_recharge_time: float = 300
    pumpdown_time: float = 600
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
