# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

from bluemira.base.file import get_bluemira_path
from bluemira.fuel_cycle.blocks import FuelCycleComponent
from bluemira.fuel_cycle.tools import (
    convert_flux_to_flow,
    fit_sink_data,
    piecewise_sqrt_threshold,
)


@pytest.mark.classplot
class TestFuelCycleComponent:
    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots()

    def test_bathtub(self):
        t = np.linspace(0, 30, 1900)
        m = 5e-7 * np.ones(1900)

        component = FuelCycleComponent("test", t, 0.995, 1.11, retention_model="bathtub")
        component.add_in_flow(m)
        component.run()

        self.ax.plot(t, component.inventory, label="linear")

    def test_sqrtbathtub(self):
        t = np.linspace(0, 60, 1900)

        m_flow = convert_flux_to_flow(1e20, 1400)
        m = m_flow * np.ones(1900)

        component = FuelCycleComponent(
            "test", t, 0.33, 1.3, retention_model="sqrt_bathtub"
        )
        component.add_in_flow(m)
        component.run()

        self.ax.plot(t, component.inventory, label="sqrt")
        self.ax.legend()


class TestSqrtFittedSinks:
    def test_fits(self):
        path = Path(
            get_bluemira_path("fuel_cycle/blanket_fw_T_retention", subfolder="data")
        )

        data = {}
        for file in path.iterdir():
            if file.suffix == ".json":
                with open(file) as fh:
                    data[file.stem] = json.load(fh)

        # Convert the data to arrays and inventories to kg
        for v in data.values():
            v["time"] = np.array(v["time"])
            v["inventory"] = np.array(v["inventory"]) / 1000

        # Fit the data with a sqrt threshold model
        for v in data.values():
            p_opt = fit_sink_data(v["time"], v["inventory"], method="sqrt", plot=False)

            x_fit = np.linspace(0, max(v["time"]), 50)
            y_fit = piecewise_sqrt_threshold(x_fit, *p_opt)
            v["p_opt"] = p_opt
            v["x_fit"] = x_fit
            v["y_fit"] = y_fit

        # Now build an example TCycleComponent for the HCPB upper
        # with a constant mass flux equivalent to that modelled

        _, ax = plt.subplots()

        r_2_values = []

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

            # We have to switch off decay in the model in order to check it
            # matches with the data (which don't include decay effects).
            component = FuelCycleComponent(
                label,
                t,
                v["p_opt"][0],
                v["p_opt"][2],
                retention_model="sqrt_bathtub",
                _testing=True,
            )
            component.add_in_flow(m)
            component.run()

            # Crude goodness of fit test
            interpolation = interp1d(v["time"], v["inventory"])
            y_interp = interpolation(t)[:-1]
            y_model = component.inventory[:-1]
            y_mean = np.mean(y_interp)
            ss_tot = np.sum(y_interp - y_mean**2)
            ss_res = np.sum((y_interp - y_model) ** 2)
            r_2 = 1 - ss_res / ss_tot

            ax.plot(t[:-1], y_interp, label=label)
            ax.plot(t[:-1], y_model, label=label + " fit", linestyle="--")

            r_2_values.append(r_2)

        ax.set_xlabel("time [fpy]")
        ax.set_ylabel("inventory [kg]")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        assert np.all(np.array(r_2_values) > 0.9995)
