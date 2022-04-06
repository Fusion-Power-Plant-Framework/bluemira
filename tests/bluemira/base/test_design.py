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

"""
Tests for the design module.
"""

import copy
from unittest.mock import MagicMock

import pytest

import tests
from bluemira.base.design import Design
from bluemira.base.error import DesignError


class TestDesign:
    build_config = {
        "Plasma": {
            "class": "MakeParameterisedShape",
            "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
            "variables_map": {
                "r_0": "R_0",
                "a": "A",
            },
            "label": "Shape",
        },
        "TF Coils": {
            "class": "MakeParameterisedShape",
            "param_class": "PrincetonD",
            "variables_map": {
                "x1": "r_tf_in_centre",
                "x2": {
                    "value": "r_tf_out_centre",
                    "lower_bound": 8.0,
                },
                "dz": 0.0,
            },
            "label": "Shape",
        },
    }
    params = {
        "Name": "Test Design",
        "R_0": (9.0, "Input"),
        "A": (3.5, "Input"),
        "r_tf_in_centre": (5.0, "Input"),
        "r_tf_out_centre": (15.0, "Input"),
    }

    def test_builders(self):
        design = Design(self.params, self.build_config)
        component = design.run()

        assert component is not None

        assert [child.name for child in component.children] == ["Plasma", "TF Coils"]

        plasma_component = component.get_component("Plasma")
        assert [child.name for child in plasma_component.children] == ["Shape"]

        tf_coils_component = component.get_component("TF Coils")
        assert [child.name for child in tf_coils_component.children] == ["Shape"]

        if tests.PLOTTING:
            component.plot_2d()

    def test_stage_usage(self):
        design = Design(self.params, self.build_config)
        with pytest.raises(DesignError):
            design.stage

        design._stage.append("test")
        design._stage.append("test2")

        assert design.stage == "test2"

    def test_design_stage_decorator(self):
        fake_self = MagicMock()
        fake_self._stage = []

        @Design.design_stage("MOCK STAGE1")
        def func(self):
            return self._stage.copy(), self._stage, func2(self)

        @Design.design_stage("MOCK STAGE2")
        def func2(self):
            return self._stage.copy()

        stage_copy1, stage1, stage_copy2 = func(fake_self)
        assert stage_copy1 == ["MOCK STAGE1"]
        assert stage_copy2 == ["MOCK STAGE1", "MOCK STAGE2"]
        assert stage1 == []

    def test_params_validation(self):
        bad_params = copy.deepcopy(self.params)
        bad_params.pop("Name")
        with pytest.raises(DesignError):
            Design(bad_params, self.build_config)
