# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle example"""

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def get_pulse_data(
    power_cycle: PowerCycle,
    pulse_label: str,
    load_type: str,
    load_unit: str,
    extra_points: int = 0,
):
    """Get the total load for a specific type and unit."""
    pulse = power_cycle.get_pulse(pulse_label)
    phase_order = pulse._config.phases

    phase_timeseries = pulse.phase_timeseries()
    pulse_timeseries = pulse.timeseries()  # BUG WHEN ARGUMENT FOR PULSE.LOAD
    pulse_timeseries = interpolate_extra(pulse_timeseries, n_points=extra_points)
    # pp(phase_timeseries)
    # pp(pulse_timeseries)

    phase_loads_for_type = {}
    phase_total_for_type = {}

    pulse_times = [interpolate_extra(t, n_points=extra_points) for t in phase_timeseries]
    pulse_loads_for_type = pulse.load(
        load_type,
        load_unit,
        timeseries=pulse_times,
    )
    pulse_total_for_type = pulse.total_load(
        load_type,
        load_unit,
        timeseries=pulse_times,
    )

    for p_index, p_name in enumerate(phase_order):
        phase = pulse.phases[p_name]
        p_times = phase.timeseries()
        if not all(p_times == phase_timeseries[p_index]):
            print("error")

        new_times = interpolate_extra(p_times, n_points=extra_points)
        p_loads = phase.load(load_type, load_unit, timeseries=new_times)
        phase_loads_for_type[p_name] = p_loads

        p_total = phase.total_load(load_type, load_unit, timeseries=new_times)
        phase_total_for_type[p_name] = p_total

    # pp(phase_loads_for_type)
    # pp(pulse_loads_for_type)
    # pp(phase_total_for_type)
    # pp(pulse_total_for_type)

    return {
        "pulse_label": pulse_label,
        "load_type": load_type,
        "load_unit": load_unit,
        "phase_order": phase_order,
        "phase_timeseries": phase_timeseries,
        "pulse_timeseries": pulse_timeseries,
        "phase_loads": phase_loads_for_type,
        "pulse_loads": pulse_loads_for_type,
        "phase_total": phase_total_for_type,
        "pulse_total": pulse_total_for_type,
    }


def plot_pulse_data(pulse_data, load_alpha=0.7, total_alpha=1.0):
    """Create plots for each phase and the whole pulse showing load vectors."""
    pulse_label = pulse_data["pulse_label"]
    load_type = pulse_data["load_type"]

    load_names = list(pulse_data["pulse_loads"].keys())
    colors = plt.cm.hsv(np.linspace(0, 1, len(load_names)))
    color_map = {load_name: colors[i] for i, load_name in enumerate(load_names)}

    n_plots = len(pulse_data["phase_order"])
    n_row = n_col = int(np.ceil(np.sqrt(n_plots)))
    fig_phase, axes = plt.subplots(n_row, n_col, figsize=(14, 8))
    axes = axes.flatten()

    handles, labels = [], []

    for i, phase_name in enumerate(pulse_data["phase_order"]):
        ax = axes[i]
        for load_name, load_values in pulse_data["phase_loads"][phase_name].items():
            line = ax.plot(
                pulse_data["phase_timeseries"][i],
                load_values,
                label=load_name,
                color=color_map[load_name],
                linestyle="--",
                alpha=load_alpha,
            )[0]
            if load_name not in labels:
                handles.append(line)
                labels.append(load_name)

        total_line = ax.plot(
            pulse_data["phase_timeseries"][i],
            pulse_data["phase_total"][phase_name],
            label="Net",
            linewidth=3,
            color="black",
            alpha=total_alpha,
        )[0]
        if "Net" not in labels:
            handles.append(total_line)
            labels.append("Net")

        ax.set_title(f"Phase: {phase_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{load_type.capitalize()} Power [{pulse_data['load_unit']}]")
        ax.grid(visible=True, alpha=0.3)

        # if i == 0:
        #     ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig_phase.savefig(
        f"{pulse_label}_{load_type}_phases.pdf",
        bbox_inches="tight",
    )

    fig_pulse, ax = plt.subplots(1, 1, figsize=(14, 6))
    for load_name, load_values in pulse_data["pulse_loads"].items():
        ax.plot(
            pulse_data["pulse_timeseries"],
            load_values,
            label=load_name,
            color=color_map[load_name],
            linestyle="--",
            alpha=load_alpha,
        )
    ax.plot(
        pulse_data["pulse_timeseries"],
        pulse_data["pulse_total"],
        label="Net",
        linewidth=3,
        color="black",
        alpha=total_alpha,
    )
    ax.set_title(f"Pulse: {pulse_label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{load_type.capitalize()} Power [{pulse_data['load_unit']}]")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    fig_pulse.savefig(
        f"{pulse_label}_{load_type}_pulse.pdf",
        bbox_inches="tight",
    )

    plt.show()
    return fig_phase, fig_pulse


def export_pulse_data(pulse_data, filename="pulse_data.json"):
    """Recursively convert arrays to lists and export pulse to a JSON file."""

    def convert_arrays_to_lists(obj):
        if isinstance(obj, np.ndarray):
            return obj.flatten().tolist()  # Convert to 1D list
        if isinstance(obj, dict):
            return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_arrays_to_lists(item) for item in obj]
        return obj

    pulse_json = convert_arrays_to_lists(pulse_data)
    pulse_json = json.dumps(pulse_json, indent=4)
    Path(filename).write_text(pulse_json)


pulse_active_data = get_pulse_data(
    power_cycle,
    pulse_label="std",
    load_type="active",
    load_unit="MW",
    extra_points=0,
)
export_filename = config_path / "EUDEMO17_pulse_active_data.json"
export_pulse_data(pulse_active_data, filename=export_filename)
plot_pulse_data(pulse_active_data)


# def get_pulse_data(
#     power_cycle: PowerCycle,
#     pulse_label: str,
#     load_type: str,
#     load_unit: str,
#     extra_points: int = 0,
# ):
#     """Get the total load for a specific type and unit."""
#     pulse = power_cycle.get_pulse(pulse_label)
#     phase_order = power_cycle.pulse_library.root[pulse_label].phases

#     starting_time = 0
#     phase_timeseries = OrderedDict()
#     phase_loads_for_type = OrderedDict()
#     phase_total_for_type = OrderedDict()

#     pulse_timeseries = []
#     pulse_loads_for_type = {}
#     for p_name in phase_order:
#         phase = pulse[p_name]
#         p_times = phase.timeseries()
#         new_times = interpolate_extra(p_times, n_points=extra_points)
#         p_loads = phase.load(load_type, load_unit, timeseries=new_times)
#         p_total = phase.total_load(load_type, load_unit, timeseries=new_times)

#         timeseries = new_times + starting_time
#         phase_timeseries[p_name] = timeseries
#         pulse_timeseries.append(timeseries)
#         starting_time = timeseries[-1]

#         phase_loads_for_type[p_name] = p_loads
#         phase_total_for_type[p_name] = p_total

#         for l_name, load in p_loads.items():
#             if l_name not in pulse_loads_for_type:
#                 pulse_loads_for_type[l_name] = load
#             else:
#                 last_load = pulse_loads_for_type[l_name]
#                 pulse_loads_for_type[l_name] = np.concatenate([last_load, load])
#     # pp(phase_loads_for_type)
#     # pp(phase_total_for_type)
#     # pp(pulse_loads_for_type)

#     pulse_timeseries = np.concatenate(pulse_timeseries)
#     pulse_total_for_type = list(pulse_loads_for_type.values())
#     pulse_total_for_type = np.sum(pulse_total_for_type, axis=0)
#     # pp(pulse_total_for_type)

#     return {
#         "pulse_label": pulse_label,
#         "load_type": load_type,
#         "load_unit": load_unit,
#         "phase_order": phase_order,
#         "phase_timeseries": phase_timeseries,
#         "phase_loads": phase_loads_for_type,
#         "phase_total": phase_total_for_type,
#         "pulse_timeseries": pulse_timeseries,
#         "pulse_loads": pulse_loads_for_type,
#         "pulse_total": pulse_total_for_type,
#     }


# pulse_active_data = get_pulse_data(
#     power_cycle,
#     pulse_label="std",
#     load_type="active",
#     load_unit="MW",
#     extra_points=0,
# )
# phase_timeseries = pulse_active_data["phase_timeseries"]
# phase_loads_active = pulse_active_data["phase_loads"]
# phase_total_active = pulse_active_data["phase_total"]
# pulse_timeseries = pulse_active_data["pulse_timeseries"]
# pulse_loads_active = pulse_active_data["pulse_loads"]
# pulse_total_active = pulse_active_data["pulse_total"]
# # pp(phase_timeseries)
# # pp(phase_loads_active)
# # pp(phase_total_active)
# # pp(pulse_timeseries)
# # pp(pulse_loads_active)
# # pp(pulse_total_active)


# # TEMP: add turbine load to pulse, because it is zero due to bug
# production_minucci_turbine_hcpb = 790 * 0.95 * 0.85
# consumption_minucci_ssen_hcpb_ftt = sum([
#     -12.2,  # TFV
#     -3,  # TER
#     -3 * 6,  # ECH + NBI + ICH upkeeps
#     -6.1,  # DGC
#     -9.7,  # VV PHTS
#     -165.6,  # HCPB PHTS
#     -2.3,  # VVPS
#     -19.5,  # DIV+LIM PHTS
#     -(5 + 4.6 + 3),  # RM, Assembly, Waste
#     -12,  # BOP
#     -3.1,  # Site
#     -101.8,  # Cryogenics
#     -21,  # EPS upkeep
#     # TEMP: add "eps_peak" to `plb` subphase to also model PPEN
#     -54.8,  # Buildings
#     -3.6,  # Plant Control
#     -90.9,  # Auxiliaries
#     -32,  # Switchyard estimates
# ])  # MW
# net_ftt = production_minucci_turbine_hcpb + consumption_minucci_ssen_hcpb_ftt
# # pp(net_ftt)

# # Remove turbine power to plot only consumption
# pulse_consumption = pulse_total_active - production_minucci_turbine_hcpb
# # pp(pulse_consumption)
