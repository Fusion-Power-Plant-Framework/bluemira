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
    Efficiency,
    Load,
    LoadLibrary,
    LoadTypeOptions,
    Phase,
    PhaseConfig,
    PowerCycle,
    SubPhase,
    SubphaseLibrary,
    interpolate_extra,
    make_power_cycle,
)


def test_LoadTypeOptions_from_str():
    assert LoadTypeOptions("active") == LoadTypeOptions.ACTIVE
    assert LoadTypeOptions("reactive") == LoadTypeOptions.REACTIVE


def test_interpolate_extra_returns_the_correct_length():
    arr = interpolate_extra(np.arange(5), 5)
    assert arr.size == 25
    assert max(arr) == 4
    assert min(arr) == 0


def test_SubPhaseConfig_duration():
    pcb = SubPhase(duration=5, unit="hours")
    assert pcb.duration == 18000
    assert pcb.unit == "s"


def test_make_powercycle():
    pc = make_power_cycle("s_name")
    assert pc.scenario.name == "s_name"
    pc = make_power_cycle(
        "s_name",
        {"std": 1, "other": 2},
        {"std": {"phases": ["one"]}, "other": {"phases": ["one"]}},
        {"one": {"operation": "max", "subphases": ["spone"]}},
        {"spone": {"duration": 5, "loads": ["lone"]}},
        loads={"lone": {"time": [0, 1, 2], "data": [2, 0, 4]}},
    )
    sc = pc.get_scenario()
    assert isinstance(sc["std"]["data"]["one"], Phase)
    assert isinstance(sc["other"]["data"]["one"], Phase)
    ph = pc.get_phase("one")
    assert np.allclose(ph.timeseries(), [0, 5, 10])
    assert np.allclose(ph.load("active")["lone"], [-2, -1, 0])
    assert np.allclose(ph.total_load("active"), [-2, -1, 0])
    assert ph.duration == 5

    pc = make_power_cycle(
        "s_name",
        systems={"sone": {"subsystems": ["ssone"]}},
        subsystems={"ssone": {"loads": ["lone"]}},
        loads={"lone": {"time": [0, 1, 2], "data": [2, 0, 4]}},
    )
    # Systems are not yet used...
    assert pc.system_library.root.keys() == {"sone"}
    assert pc.sub_system_library.root.keys() == {"ssone"}


class TestLoad:
    def test_interpolate(self):
        pcsl = Load(
            time=np.array([0, 0.5, 1]), data={"reactive": np.arange(3)}, model="ramp"
        )
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1], load_type="reactive"),
            np.array([0, 0.2, 0.4, 0.6, 2]),
        )
        pcsl2 = Load(time=np.array([0, 0.5, 1]), data=np.arange(3), model="ramp")
        assert np.allclose(
            pcsl2.interpolate([0, 0.1, 0.2, 0.3, 1], load_type="reactive"),
            np.array([0, 0.2, 0.4, 0.6, 2]),
        )
        assert np.allclose(
            pcsl2.interpolate([0, 0.1, 0.2, 0.3, 1], load_type="active"),
            np.array([0, 0.2, 0.4, 0.6, 2]),
        )

        pcsl = Load(
            time=np.array([0, 0.5, 2]),
            data={"active": np.arange(3)},
            model="ramp",
            normalised=False,
        )
        assert np.allclose(
            pcsl.interpolate([0, 0.1, 0.2, 0.3, 1], 10),
            np.array([0, 1 + 1 / 3, 2, 0, 0]),
        )

    def test_validation_raises_ValueError(self):
        with pytest.raises(ValueError, match="time and data"):
            Load(time=np.array([0, 0.1, 1]), data=np.zeros(2), model="ramp")

        with pytest.raises(ValueError, match="time and data"):
            Load(time=[0, 0.1, 1], data=np.zeros(2), model="ramp")

        with pytest.raises(ValueError, match="time must increase"):
            Load(time=[0, 1, 0.1], data=np.zeros(3), model="ramp")

        pcsl = Load(time=[0, 0.1, 1], data=np.zeros(3), model="ramp", unit="MW")
        assert np.allclose(pcsl.time[LoadTypeOptions.ACTIVE], np.array([0, 0.1, 1]))
        assert np.allclose(pcsl.time[LoadTypeOptions.REACTIVE], np.array([0, 0.1, 1]))
        assert pcsl.unit == "W"


class TestSubPhase:
    @classmethod
    def setup_class(cls):
        reactor_config = ReactorConfig(
            Path(__file__).parent / "test_data" / "scenario_config.json", None
        )
        cls._config = PowerCycle(
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
        cls._subphases = cls._config.get_phase("dwl").subphases

    def setup_method(self):
        self.subphases = deepcopy(self._subphases)

    @pytest.mark.parametrize("load_type", ["active", "reactive", None])
    @pytest.mark.parametrize("end_time", [200, None])
    @pytest.mark.parametrize("consumption", [True, False, None])
    def test_build_timeseries(self, load_type, end_time, consumption):
        for sp in self.subphases.root.values():
            assert np.allclose(
                sp.build_timeseries(
                    load_library=self._config.load_library,
                    load_type=load_type,
                    end_time=end_time,
                    consumption=consumption,
                ),
                [0, 0.6, 1]
                if consumption in {True, None} and end_time == 200
                else [0, 1],
            )

    @pytest.mark.parametrize(
        ("time", "et", "res1", "res2"),
        [
            (
                np.array([0, 0.005, 0.6, 1]),
                200,
                np.full(4, -4.7),
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
                np.full(4, -4.7),
                np.array([-10.2, -10.2, -10.2, -0.0]),
            ),
        ],
    )
    def test_get_load_data_with_efficiencies(self, time, et, res1, res2):
        for sp in self.subphases.root.values():
            load_data = sp.get_load_data_with_efficiencies(
                self._config.load_library, time, "reactive", unit="MW", end_time=et
            )
            assert np.allclose(load_data["vv"], res1)
            # not normalised
            assert np.allclose(load_data["eps_upk"], res2)

    def test_get_load_data_with_efficiencies_unset_efficiency_is_one(self):
        for sp in self.subphases.root.values():
            load_data = sp.get_load_data_with_efficiencies(
                self._config.load_library,
                np.array([0, 0.005, 0.6, 1]),
                "active",
                unit="MW",
            )
            assert np.allclose(load_data["turb"], 750 * 0.85)


class TestPhaseFromScenario:
    @classmethod
    def setup_class(cls):
        reactor_config = ReactorConfig(
            Path(__file__).parent / "test_data" / "scenario_config.json", None
        )
        cls._config = PowerCycle(
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
        cls._phase = cls._config.get_phase("dwl")

    def setup_method(self):
        self.phase = deepcopy(self._phase)

    @pytest.mark.parametrize("load_type", ["active", "reactive", None])
    @pytest.mark.parametrize("consumption", [True, False, None])
    def test_build_timeseries(self, load_type, consumption):
        assert np.allclose(
            self.phase.timeseries(load_type, consumption=consumption),
            [0, 120, 600] if consumption in {True, None} else [0, 600],
        )

    @pytest.mark.parametrize(
        ("time", "res1", "res2"),
        [
            (
                np.array([0, 0.005, 0.6, 1]),  # seen as already normalised time
                np.full(4, -4.7),
                np.array([-10.2, -10.2, 0.0, 0.0]),
            ),
            (
                np.array([0, 0.005, 0.6, 100]),
                np.full(4, -4.7),
                np.array([-10.2, -10.2, -10.2, 0.0]),
            ),
            (
                np.array([0, 1, 120, 200]),
                np.full(4, -4.7),
                np.array([-10.2, -10.2, 0.0, 0.0]),
            ),
        ],
    )
    def test_get_load(self, time, res1, res2):
        load_data = self.phase.load("reactive", "MW", timeseries=time)
        assert np.allclose(load_data["vv"], res1)
        # not normalised
        assert np.allclose(load_data["eps_upk"], res2)

    @pytest.mark.parametrize(
        ("time", "res"),
        [
            (np.array([0, 0.6, 1]), np.array([107.175, 117.375, 117.375])),
            (np.array([0, 120, 150]), np.array([107.175, 117.375, 117.375])),
            (np.array([0, 0.6, 600]), np.array([107.175, 107.175, 117.375])),
        ],
    )
    def test_get_load_total(self, time, res):
        assert np.allclose(self.phase.total_load("reactive", "MW", timeseries=time), res)


class TestPhase:
    def setup_method(self):
        self.phase = Phase(
            PhaseConfig(operation="max", subphases=["a", "b"]),
            SubphaseLibrary(
                a=SubPhase(duration=5, loads=["name", "name"]),
                b=SubPhase(
                    duration=10,
                    loads=["name2"],
                    efficiencies={"name2": [Efficiency(value=0.1)]},
                ),
            ),
            LoadLibrary(
                name=Load(time=np.array([0, 0.5, 1]), data=np.arange(3), model="ramp"),
                name2=Load(
                    time=np.array([0, 0.2, 1]),
                    data=np.arange(3),
                    model="ramp",
                    consumption=False,
                ),
            ),
        )

    def test_duration_validation_and_extraction(self):
        assert self.phase.duration == 10

    @pytest.mark.parametrize("load_type", ["active", "reactive", None])
    @pytest.mark.parametrize("consumption", [True, False, None])
    def test_build_timeseries(self, load_type, consumption):
        assert np.allclose(
            self.phase.timeseries(load_type=load_type, consumption=consumption),
            [0, 2, 5, 10]
            if consumption is None
            else [0, 5, 10]
            if consumption
            else [0, 2, 10],
        )

    @pytest.mark.parametrize("consumption", [True, False, None])
    @pytest.mark.parametrize("load_type", ["active", "reactive"])
    def test_total_load(self, load_type, consumption):
        assert np.allclose(
            self.phase.total_load(
                load_type, timeseries=[0, 2, 5, 10], consumption=consumption
            ),
            [0.0, -0.7, -1.8625, -3.8]
            if consumption is None
            else [0.0, -0.8, -2.0, -4.0]
            if consumption
            else [0.0, 0.1, 0.1375, 0.2],
        )

    @pytest.mark.parametrize("consumption", [True, False, None])
    @pytest.mark.parametrize("load_type", ["active", "reactive"])
    def test_load(self, load_type, consumption):
        for _ in range(2):  # run the duplicates twice doesnt keep doubling load
            res = self.phase.load(
                load_type, timeseries=[0, 2, 5, 10], consumption=consumption
            )
            if (name := res.get("name", None)) is not None:
                assert np.allclose(name, np.array([-0.0, -0.8, -2.0, -4.0]))
            if (name2 := res.get("name2", None)) is not None:
                assert np.allclose(name2, np.array([0.0, 0.1, 0.1375, 0.2]))


class TestPowerCycle:
    @classmethod
    def setup_class(cls):
        reactor_config = ReactorConfig(
            Path(__file__).parent / "test_data" / "scenario_config.json", None
        )
        cls._config = PowerCycle(
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

    def setup_method(self):
        self.config = deepcopy(self._config)

    def test_get_scenario(self):
        scenario = self.config.get_scenario()

        assert scenario["std"]["repeat"] == 1
        assert len(scenario["std"]["data"].keys()) == 4
        assert all(isinstance(val, Phase) for val in scenario["std"]["data"].values())
        assert scenario["std"]["data"]["dwl"].subphases.root.keys() == {"csr", "pmp"}

        sph = scenario["std"]["data"]["dwl"].subphases.root
        assert scenario["std"]["data"]["dwl"].loads.root.keys() == set(
            sph["csr"].loads + sph["pmp"].loads
        )

    def test_import_subphase_data(self):
        assert self.config.subphase_library.root["csr"].duration == 300

    def test_add_load_config(self):
        self.config.add_load(
            "cs_power",
            Load(
                time=[0, 1], data=[10, 20], model="RAMP", unit="MW", description="dunno"
            ),
            ["cru", "bri"],
        )
        assert np.allclose(
            self.config.load_library.root["cs_power"].data[LoadTypeOptions.REACTIVE],
            [10e6, 20e6],
        )
        assert self.config.load_library.root["cs_power"].unit == "W"
