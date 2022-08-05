# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

from bluemira.base.builder import Builder


class ParamClass:
    def __init__(self, param_1, param_2) -> None:
        self.param_1 = param_1
        self.param_2 = param_2

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class StubBuilder(Builder):
    param_cls = ParamClass

    def build(self):
        return super().build()


class TestBuilder:
    _params = {
        "param_1": {"name": "param_1", "unit": "m", "value": 1},
        "param_2": {"name": "param_2", "unit": "T", "value": 2},
    }

    def test_default_name_is_class_name_sans_builder(self):
        builder = StubBuilder(self._params, {})

        assert builder.name == "Stub"
