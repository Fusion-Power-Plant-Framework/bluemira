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

import pytest

from bluemira.base.builder import Builder
from bluemira.base.components import Component
from bluemira.base.error import BuilderError


class NoParamBuilder(Builder):
    def build(self, **kwargs) -> Component:
        return super().build(**kwargs)

    def reinitialise(self, params, **kwargs) -> None:
        return super().reinitialise(params, **kwargs)


class ABuilder(Builder):
    _required_params = [
        "P_el_net",
        "R_0",
        "n_TF",
    ]

    def build(self, **kwargs) -> Component:
        return super().build(**kwargs)

    def reinitialise(self, params, **kwargs) -> None:
        return super().reinitialise(params, **kwargs)


class TestBuilder:
    def test_name(self):
        params = {}
        build_config = {
            "name": "TestBuilder",
        }
        builder = NoParamBuilder(params, build_config)
        assert builder.name == build_config["name"]

    def test_required_params(self):
        params = {
            "P_el_net": {
                "value": 500,
                "description": "Make some fusion",
            },
            "R_0": 9.0,
            "n_TF": {
                "value": 16,
                "description": "Number of TF coils needed for this study",
            },
        }
        build_config = {"name": "TestBuilder"}
        builder = ABuilder(params, build_config)
        builder_dict = builder._params.to_dict(verbose=True)
        for key, var in params.items():
            if isinstance(var, dict):
                for param_key in var.keys():
                    assert params[key][param_key] == builder_dict[key][param_key]

    def test_required_params_missing(self):
        params = {}
        build_config = {"name": "TestBuilder"}
        with pytest.raises(
            BuilderError,
            match="Required parameters P_el_net, R_0, n_TF not provided to Builder",
        ):
            ABuilder(params, build_config)

        params = {
            "P_el_net": {
                "value": 500,
                "description": "Make some fusion",
            },
            "n_TF": {
                "value": 16,
                "description": "Number of TF coils needed for this study",
            },
        }
        with pytest.raises(
            BuilderError, match="Required parameters R_0 not provided to Builder"
        ):
            ABuilder(params, build_config)
