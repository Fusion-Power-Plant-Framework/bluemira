# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bluemira.power_cycle.net import (
    LoadLibrary,
    LoadModel,
    LoadType,
    Loads,
    Phase,
    PhaseConfig,
    PowerCycleBreakdown,
    PowerCycleLibraryConfig,
    PowerCycleLoadConfig,
    PowerCycleSubLoad,
    ScenarioConfig,
    ScenarioConfigDescriptor,
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


def test_PowerCycleBreakdown_duration():
    pcb = PowerCycleBreakdown("name", 5, unit="hours")
    assert pcb.duration == 18000
    assert pcb.unit == "s"


class TestPowerCycleSubLoad:
    def test_null(self):
        pcsl = PowerCycleSubLoad.null()
        assert np.allclose(pcsl.time, np.arange(2))
        assert np.allclose(pcsl.data, np.zeros(2))
        assert pcsl.model == LoadModel.RAMP

    def test_interpolate(self):
        pcsl = PowerCycleSubLoad("name", np.array([0, 0.5, 1]), np.arange(3), "ramp")
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1]), np.array([0, 0.2, 0.4, 0.6, 2])
        )

        pcsl = PowerCycleSubLoad("name", np.array([0, 0.5, 2]), np.arange(3), "ramp")
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1], 10),
            np.array([0, 1 + 1 / 3, 2, 0, 0]),
        )

    def test_validation_raises_ValueError(self):
        with pytest.raises(ValueError, match="time and data"):
            PowerCycleSubLoad("name", np.array([0, 0.1, 1]), np.zeros(2), "ramp")

        with pytest.raises(ValueError, match="time and data"):
            PowerCycleSubLoad("name", [0, 0.1, 1], np.zeros(2), "ramp")

        with pytest.raises(ValueError, match="time must increase"):
            PowerCycleSubLoad("name", [0, 1, 0.1], np.zeros(3), "ramp")

        pcsl = PowerCycleSubLoad("name", [0, 0.1, 1], np.zeros(3), "ramp", "MW")
        assert np.allclose(pcsl.time, np.array([0, 0.1, 1]))
        assert pcsl.unit == "W"


class TestDescriptors:
    def test_LibraryConfigDescriptor_returns_correct_type(self):
        pclc_dict = {
            "test": {
                "consumption": True,
                "efficiencies": {"test": 0.5},
                "subloads": ["test_load"],
            }
        }
        l_lib = LoadLibrary(
            LoadType.ACTIVE,
            pclc_dict,
        )

        assert all(isinstance(val, PowerCycleLoadConfig) for val in l_lib.loads.values())
        assert l_lib.loads.keys() == {"test"}
        assert l_lib.loads["test"].consumption
        assert l_lib.loads["test"].subloads == ["test_load"]
        assert l_lib.loads["test"].efficiencies["test"] == pytest.approx(0.5)

    def test_ScenarioConfigDescriptor_returns_correct_type(self):
        @dataclass
        class Test:
            sc: ScenarioConfigDescriptor = ScenarioConfigDescriptor()

        t_sc = Test({"name": "sc_1", "pulses": {"std_pulse": 1}, "description": "test"})
        assert isinstance(t_sc.sc, ScenarioConfig)
        assert t_sc.sc.pulses["std_pulse"] == 1
        assert t_sc.sc.description == "test"


@dataclass
class PowerCycleDurationParameters:
    CS_recharge_time: float = 300
    pumpdown_time: float = 600
    ramp_up_time: float = 157
    ramp_down_time: float = 157


class TestLoads:
    @classmethod
    def setup_class(cls):
        cls._config = PowerCycleLibraryConfig.from_json(
            Path(__file__).parent / "test_data" / "scenario_config.json"
        )
        cls._config.import_breakdown_data(PowerCycleDurationParameters())
        cls._loads = cls._config.make_phase("dwl").loads

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
            (np.array([0, 0.6, 1]), 200, np.array([368.295, -10.2, 0.0])),
            (np.array([0, 120, 200]), None, np.array([368.295, -10.2, 0.0])),
            (np.array([0, 0.6, 1]), None, np.full(3, 368.295)),
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
            {"a": PowerCycleBreakdown("a", 5), "b": PowerCycleBreakdown("b", 10)},
            Loads(  # Dummy load
                {LoadType.ACTIVE: {}, LoadType.REACTIVE: {}},
                {LoadType.ACTIVE: {}, LoadType.REACTIVE: {}},
            ),
        )

        assert phase.duration == 10


class TestPowerCycleLibraryConfig:
    @classmethod
    def setup_class(cls):
        cls._config = PowerCycleLibraryConfig.from_json(
            Path(__file__).parent / "test_data" / "scenario_config.json"
        )
        cls._config.import_breakdown_data(PowerCycleDurationParameters())

    def setup_method(self):
        self.config = deepcopy(self._config)

    def test_make_scenario(self):
        scenario = self.config.make_scenario()

        assert scenario["std"]["repeat"] == 1
        assert len(scenario["std"]["data"].keys()) == 4
        assert all(isinstance(val, Phase) for val in scenario["std"]["data"].values())

    def test_import_breakdown_data(self):
        assert self.config.breakdown["csr"].duration == 300

    def test_add_subload(self):
        self.config.add_subload(
            "active",
            PowerCycleSubLoad("cs_power", [0, 1], [10, 20], "RAMP", "MW", "dunno"),
        )
        assert np.allclose(
            self.config.subload[LoadType.ACTIVE].loads["cs_power"].data, [10e6, 20e6]
        )
        assert self.config.subload[LoadType.ACTIVE].loads["cs_power"].unit == "W"

    def test_add_load(self):
        name = "CS"
        breakdowns = ["cru", "bri"]
        self.config.add_load_config(
            "active",
            breakdowns,
            PowerCycleLoadConfig(name, True, {}, ["cs_power"], "something made up"),
        )
        assert all(name in self.config.breakdown[k].active_loads for k in breakdowns)
        assert self.config.load[LoadType.ACTIVE].loads["CS"].subloads == ["cs_power"]
