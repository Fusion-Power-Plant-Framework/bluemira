# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Script used to analyse T retention data
"""

import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.fuel_cycle.blocks import FuelCycleComponent
from bluemira.fuel_cycle.tools import (
    convert_flux_to_flow,
    fit_sink_data,
    piecewise_sqrt_threshold,
)

plot_defaults()

PATH = get_bluemira_path("fuel_cycle/blanket_fw_T_retention", subfolder="data")

# Compiles the data from the files
data = {}
for file in os.listdir(PATH):
    if Path(file).suffix == ".json":
        short_name = Path(file).stem
        with open(Path(PATH, file)) as fh:
            data[short_name] = json.load(fh)

# Convert the data to arrays and inventories to kg
for v in data.values():
    v["time"] = np.array(v["time"])
    v["inventory"] = np.array(v["inventory"]) / 1000

# Plot the original data
f, (ax, ax2) = plt.subplots(1, 2)
for k, v in data.items():
    ax.plot(v["time"], v["inventory"], "s", marker="o", label=k)

# Fit the data with a sqrt threshold model
for v in data.values():
    p_opt = fit_sink_data(v["time"], v["inventory"], method="sqrt", plot=False)

    x_fit = np.linspace(0, max(v["time"]), 50)
    y_fit = piecewise_sqrt_threshold(x_fit, *p_opt)
    v["p_opt"] = p_opt
    v["x_fit"] = x_fit
    v["y_fit"] = y_fit

# Plot the fits
for k, v in data.items():
    label = ""
    if "HCPB" in k:
        label += "HCPB"
    elif "WCLL" in k:
        label += "WCLL"
    if "Lower" in k:
        label += " lower"
    elif "Upper" in k:
        label += " upper"
    a = v["p_opt"][0]
    label += f" fit: {a:.3f}" + "$\\times\\sqrt{t}$"
    ax.plot(v["x_fit"], v["y_fit"], linestyle="--", label=label)

ax.set_xlabel("Time [fpy]")
ax.set_ylabel("Inventory [kg]")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# Now build an example TCycleComponent for the HCPB upper
# with a constant mass flux equivalent to that modelled

for k, v in data.items():
    label = ""
    if "HCPB" in k:
        label += "HCPB"
    elif "WCLL" in k:
        label += "WCLL"
    if "Lower" in k:
        label += " lower"
    elif "Upper" in k:
        label += " upper"

    t = np.linspace(0, max(v["time"]), 1000)
    if "Upper" in k:
        flux = 1e20
    elif "Lower" in k:
        flux = 1e19
    m_flow = convert_flux_to_flow(flux, 1400)
    m = m_flow * np.ones(1000)

    component = FuelCycleComponent(
        label, t, v["p_opt"][0], v["p_opt"][2], retention_model="sqrt_bathtub"
    )
    component.add_in_flow(m)
    component.run()
    ax2.plot(t, component.inventory, label=label + " sqrt model")

ax2.set_xlabel("Time [fpy]")
ax2.set_ylabel("Inventory [kg]")
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])
ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
