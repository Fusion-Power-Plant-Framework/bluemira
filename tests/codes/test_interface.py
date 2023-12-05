# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from dataclasses import dataclass
from typing import ClassVar

from bluemira.base.parameter_frame import Parameter
from bluemira.codes import interface
from bluemira.codes.interface import BaseRunMode, NoOpTask
from bluemira.codes.params import MappedParameterFrame, ParameterMapping


class NoOpRunMode(BaseRunMode):
    RUN = 0


class NoOpSolver(interface.CodesSolver):
    name = "MyTestSolver"
    setup_cls = NoOpTask
    run_cls = NoOpTask
    teardown_cls = NoOpTask
    run_mode_cls = NoOpRunMode


@dataclass
class Params(MappedParameterFrame):
    param1: Parameter[float]
    param2: Parameter[int]

    _mappings: ClassVar = {
        "param1": ParameterMapping("ext1", send=True, recv=True, unit="MW"),
        "param2": ParameterMapping("ext2", "ext3", send=False, recv=False),
    }

    @property
    def mappings(self):
        return self._mappings

    @classmethod
    def from_defaults(cls):
        return super().from_defaults({})


class TestCodesSolver:
    def test_modify_mappings_updates_send_recv_values_of_params(self):
        params = Params.from_dict(
            {
                "param1": {"value": 0.1, "unit": "m"},
                "param2": {"value": 5, "unit": "dimensionless"},
            }
        )
        solver = NoOpSolver(params)

        solver.modify_mappings(
            {
                "param1": {"recv": True, "send": False},
                "param2": {"recv": False, "send": True},
            }
        )

        assert solver.params.mappings["param1"].send is False
        assert solver.params.mappings["param1"].recv is True
        assert solver.params.mappings["param2"].send is True
        assert solver.params.mappings["param2"].recv is False

    def test_no_defaults_are_set_to_None(self):
        params = Params.from_defaults()
        assert params.param1.value is None
        assert params.param2.value is None
        assert params.param1.unit == "W"
        assert params.param2.unit == ""
        assert params.mappings["param1"].name == "ext1"
        assert params.mappings["param1"].out_name == "ext1"
        assert params.mappings["param2"].name == "ext2"
        assert params.mappings["param2"].out_name == "ext3"
