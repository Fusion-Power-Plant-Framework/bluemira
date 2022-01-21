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
import copy

from bluemira.builders.EUDEMO.first_wall import FirstWallBuilder


class TestFirstWallBuilder:

    _default_variables_map = {
        "x1": {  # ib radius
            "value": "r_fw_ib_in",
        },
        "x2": {  # ob radius
            "value": "r_fw_ob_in",
        },
    }

    _default_config = {
        "param_class": "bluemira.builders.EUDEMO.first_wall::FirstWallPolySpline",
        "variables_map": _default_variables_map,
        "runmode": "mock",
        "name": "First Wall",
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

        builder = FirstWallBuilder(self._params, build_config=config)

        assert builder.name == "New name"

    def test_built_component_contains_physical_component_in_xz(self):
        builder = FirstWallBuilder(self._params, build_config=self._default_config)

        component = builder(self._params)

        xy_component = component.get_component("xz", first=False)
        assert len(xy_component) == 1
        assert len(xy_component[0].get_component("first_wall", first=False)) == 1

    def test_component_height_derived_from_params(self):
        params = copy.deepcopy(self._params)
        params.update(
            {"R_0": (10.0, "Input"), "kappa_95": (2.0, "Input"), "A": (2.0, "Input")}
        )

        builder = FirstWallBuilder(self._params, build_config=self._default_config)
        component = builder(params)

        bounding_box = component.get_component("first_wall").shape.bounding_box
        # expected_height = 2*(R_0/A)*kappa_95 = 20
        assert bounding_box.z_max - bounding_box.z_min == 20.0
