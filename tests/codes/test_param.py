# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass

import pytest

from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.params import ParameterMapping, make_mapped_default_parameter_frame


class TestParameterMapping:
    def setup(self):
        self.pm = ParameterMapping("Name", send=True, recv=False)

    @pytest.mark.parametrize(
        "attr, value",
        zip(
            ["name", "mynewattr", "_frozen", "unit"],
            ["NewName", "Hello", ["custom", "list"], "MW"],
        ),
    )
    def test_no_keyvalue_change(self, attr, value):

        with pytest.raises(KeyError):
            setattr(self.pm, attr, value)

    def test_value_change(self):

        for var in ["send", "recv"]:
            with pytest.raises(ValueError):
                setattr(self.pm, var, "A string")

        assert self.pm.send
        assert not self.pm.recv
        self.pm.send = False
        self.pm.recv = True
        assert not self.pm.send
        assert self.pm.recv

    def test_tofrom_dict(self):
        assert self.pm == ParameterMapping.from_dict(self.pm.to_dict())


@dataclass
class MyDF:
    a = 1
    b = "hello"
    c = True


@dataclass
class MyPF(ParameterFrame):
    A: Parameter[float]
    B: Parameter[str]
    C: Parameter[bool]
    D: Parameter[float]
    E: Parameter[str]
    F: Parameter[bool]


mappings = {
    "A": ParameterMapping("a", send=True, recv=False, unit="m"),
    "B": ParameterMapping("b", send=True, recv=False, unit=""),
    "C": ParameterMapping("c", send=True, recv=False, unit=""),
    "D": ParameterMapping("d", send=False, recv=False, unit="m"),
    "E": ParameterMapping("e", send=False, recv=False, unit=""),
    "F": ParameterMapping("f", send=False, recv=False, unit=""),
}


class TestDefaultPM:
    def test_mapped_default_pm_sets_unknown_values(self):
        params = make_mapped_default_parameter_frame(MyDF, mappings, MyPF)
        assert params.D.value == 0
        assert params.D.unit == "m"
        assert params.E.value == "0"
        assert params.E.unit == ""
        assert params.F.value is False
        assert params.F.unit == ""

    def test_mapped_default_pm_sets_known_values(self):
        params = make_mapped_default_parameter_frame(MyDF, mappings, MyPF)
        assert params.A.value == 1
        assert params.A.unit == "m"
        assert params.B.value == "hello"
        assert params.B.unit == ""
        assert params.C.value is True
        assert params.C.unit == ""
