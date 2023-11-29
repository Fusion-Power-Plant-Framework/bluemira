# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from typing import ClassVar

from bluemira.base.builder import Builder
from bluemira.base.components import Component


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
    _params: ClassVar = {
        "param_1": {"unit": "m", "value": 1},
        "param_2": {"unit": "T", "value": 2},
    }

    def test_default_name_is_class_name_sans_builder(self):
        builder = StubBuilder(self._params, {})

        assert builder.name == "Stub"

    def test_default_name_is_class_name_sans_build_config(self):
        builder = StubBuilder(self._params, None)

        assert builder.name == "Stub"

    def test_component_tree(self):
        builder = StubBuilder(self._params, {})
        component = builder.component_tree(
            xz=[Component("p1")], xy=[Component("p2")], xyz=[Component("p3")]
        )
        assert len(component.descendants) == 6
        assert len(component.children) == 3
        assert [ch.name for ch in component.children] == ["xz", "xy", "xyz"]
        # xyz child's view is not changed from the default
        assert [
            desc.plot_options.view_placement.label for desc in component.descendants
        ] == [
            "xzy",
            "xzy",
            "xyz",
            "xyz",
            "xzy",
            "xzy",
        ]
