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
Test for the closed first wall, without a divertor.
"""

import copy

from bluemira.builders.EUDEMO.first_wall import WallBuilder
from bluemira.builders.EUDEMO.first_wall.first_wall import _WALL_MODULE_REF


class TestWall:

    _default_variables_map = {
        "x1": {  # ib radius
            "value": "r_fw_ib_in",
        },
        "x2": {  # ob radius
            "value": "r_fw_ob_in",
        },
    }

    _default_config = {
        "param_class": f"{_WALL_MODULE_REF}::WallPolySpline",
        "variables_map": _default_variables_map,
        "runmode": "mock",
        "name": "First Wall",
        "problem_class": f"{_WALL_MODULE_REF}::MinimiseLength",
    }

    _params = {
        "Name": "First Wall Example",
        "plasma_type": "SN",
        "R_0": (9.0, "Input"),
        "kappa_95": (1.6, "Input"),
        "r_fw_ib_in": (5.8, "Input"),
        "r_fw_ob_in": (12.1, "Input"),
        "A": (3.1, "Input"),
    }

    def test_builder_has_name_from_config(self):
        config = copy.deepcopy(self._default_config)
        config["name"] = "New name"

        builder = WallBuilder(self._params, build_config=config)

        assert builder.name == "New name"

    def test_built_component_contains_physical_component_in_xz(self):
        builder = WallBuilder(self._params, build_config=self._default_config)

        component = builder()

        xy_component = component.get_component("xz", first=False)
        assert len(xy_component) == 1
        wall_components = xy_component[0].get_component(
            WallBuilder.COMPONENT_WALL_BOUNDARY, first=False
        )
        assert len(wall_components) == 1

    def test_physical_component_shape_is_closed(self):
        builder = WallBuilder(self._params, build_config=self._default_config)

        component = builder()

        assert component.get_component(
            WallBuilder.COMPONENT_WALL_BOUNDARY
        ).shape.is_closed()

    def test_component_height_derived_from_params_in_mock_mode(self):
        params = copy.deepcopy(self._params)
        params.update(
            {"R_0": (10.0, "Input"), "kappa_95": (2.0, "Input"), "A": (2.0, "Input")}
        )

        builder = WallBuilder(params, build_config=self._default_config)
        component = builder()

        bounding_box = component.get_component(
            WallBuilder.COMPONENT_WALL_BOUNDARY
        ).shape.bounding_box
        # expected_height = 2*(R_0/A)*kappa_95 = 20
        assert bounding_box.z_max - bounding_box.z_min == 20.0
