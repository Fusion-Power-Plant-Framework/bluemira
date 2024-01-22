# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.reactor_config import ReactorConfig
from bluemira.power_cycle.net import (
    LoadType,
    Loads,
    Phase,
    PhaseConfig,
    PowerCycleLibraryConfig,
    PowerCycleLoadConfig,
    SubPhaseConfig,
    interpolate_extra,
)


def test_LoadType_from_str():
    assert LoadType.from_str("active") == LoadType.ACTIVE
    assert LoadType.from_str("reactive") == LoadType.REACTIVE


def test_interpolate_extra_returns_the_correct_length():
    arr = interpolate_extra(np.arange(5), 5)
    assert arr.size == 25
    assert max(arr) == 4
    assert min(arr) == 0


def test_SubPhaseConfig_duration():
    pcb = SubPhaseConfig("name", 5, unit="hours")
    assert pcb.duration == 18000
    assert pcb.unit == "s"


class TestPowerCycleSubLoad:
    def test_interpolate(self):
        pcsl = PowerCycleLoadConfig(
            "name", np.array([0, 0.5, 1]), np.arange(3), model="ramp"
        )
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1], load_type="reactive"),
            np.array([0, 0.2, 0.4, 0.6, 2]),
        )

        pcsl = PowerCycleLoadConfig(
            "name", np.array([0, 0.5, 2]), active_data=np.arange(3), model="ramp"
        )
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1], 10),
            np.array([0, 1 + 1 / 3, 2, 0, 0]),
        )

    def test_validation_raises_ValueError(self):
        with pytest.raises(ValueError, match="time and data"):
            PowerCycleLoadConfig(
                "name", np.array([0, 0.1, 1]), np.zeros(2), model="ramp"
            )

        with pytest.raises(ValueError, match="time and data"):
            PowerCycleLoadConfig("name", [0, 0.1, 1], np.zeros(2), model="ramp")

        with pytest.raises(ValueError, match="time must increase"):
            PowerCycleLoadConfig("name", [0, 1, 0.1], np.zeros(3), model="ramp")

        pcsl = PowerCycleLoadConfig(
            "name", [0, 0.1, 1], np.zeros(3), model="ramp", unit="MW"
        )
        assert np.allclose(pcsl.time, np.array([0, 0.1, 1]))
        assert pcsl.unit == "W"


class TestLoads:
    @classmethod
    def setup_class(cls):
        reactor_config = ReactorConfig(
            Path(__file__).parent / "test_data" / "scenario_config.json", None
        )
        cls._config = PowerCycleLibraryConfig.from_dict(
            reactor_config.config_for("Power Cycle"),
            {
                "cs_recharge_time": 300,
                "pumpdown_time": 600,
                "ramp_up_time": 157,
                "ramp_down_time": 157,
            },
        )
        cls._loads = cls._config.get_phase("dwl").loads

    def setup_method(self):
        self.loads = deepcopy(self._loads)

    def test_build_timeseries(self):
        assert np.allclose(self.loads.build_timeseries(), [0, 1])
        assert np.allclose(self.loads.build_timeseries(200), [0, 0.6, 1])

    @pytest.mark.parametrize(
        ("time", "et", "res1", "res2"),
        [
            (
                np.array([0, 0.005, 0.6, 1]),
                200,
                np.array([-4.7, -4.7, -0.0, -0.0]),
                np.array([-10.2, -10.2, -10.2, -0.0]),
            ),
            (
                np.array([0, 0.005, 0.6, 1]),
                None,
                np.full(4, -4.7),
                np.full(4, -10.2),
            ),
            (
                np.array([0, 1, 120, 200]),
                None,
                np.array([-4.7, -4.7, -0.0, -0.0]),
                np.array([-10.2, -10.2, -10.2, -0.0]),
            ),
        ],
    )
    def test_get_load_data_with_efficiencies(self, time, et, res1, res2):
        load_data = self.loads.get_load_data_with_efficiencies(
            time, "reactive", "MW", end_time=et
        )
        assert np.allclose(load_data["vv"], res1)
        # not normalised
        assert np.allclose(load_data["eps_upk"], res2)

    @pytest.mark.parametrize(
        ("time", "et", "res"),
        [
            (np.array([0, 0.6, 1]), 200, np.array([165.4, -10.2, 0.0])),
            (np.array([0, 120, 200]), None, np.array([165.4, -10.2, 0.0])),
            (np.array([0, 0.6, 1]), None, np.full(3, 165.4)),
        ],
    )
    def test_get_load_total(self, time, et, res):
        assert np.allclose(
            self.loads.load_total(time, "reactive", "MW", end_time=et), res
        )


class TestPhase:
    def test_duration_validation_and_extraction(self):
        phase = Phase(
            PhaseConfig("dwl", "max", ["a", "b"]),
            {"a": SubPhaseConfig("a", 5), "b": SubPhaseConfig("b", 10)},
            Loads({
                "name": PowerCycleLoadConfig(
                    "name", np.array([0, 0.5, 1]), np.arange(3), model="ramp"
                )
            }),
        )

        assert phase.duration == 10


class TestPowerCycleLibraryConfig:
    @classmethod
    def setup_class(cls):
        reactor_config = ReactorConfig(
            Path(__file__).parent / "test_data" / "scenario_config.json", None
        )
        cls._config = PowerCycleLibraryConfig.from_dict(
            reactor_config.config_for("Power Cycle"),
            {
                "cs_recharge_time": 300,
                "pumpdown_time": 600,
                "ramp_up_time": 157,
                "ramp_down_time": 157,
            },
        )

    def setup_method(self):
        self.config = deepcopy(self._config)

    def test_get_scenario(self):
        scenario = self.config.get_scenario()

        assert scenario["std"]["repeat"] == 1
        assert len(scenario["std"]["data"].keys()) == 4
        assert all(isinstance(val, Phase) for val in scenario["std"]["data"].values())
        assert scenario["std"]["data"]["dwl"].subphases.keys() == {"csr", "pmp"}

        sph = scenario["std"]["data"]["dwl"].subphases
        assert scenario["std"]["data"]["dwl"].loads.loads.keys() == set(
            sph["csr"].loads + sph["pmp"].loads
        )

    def test_import_subphase_data(self):
        assert self.config.subphase["csr"].duration == 300

    def test_add_load_config(self):
        self.config.add_load_config(
            PowerCycleLoadConfig(
                "cs_power",
                [0, 1],
                [10, 20],
                model="RAMP",
                unit="MW",
                description="dunno",
            ),
            ["cru", "bri"],
        )
        assert np.allclose(self.config.loads["cs_power"].reactive_data, [10e6, 20e6])
        assert self.config.loads["cs_power"].unit == "W"
