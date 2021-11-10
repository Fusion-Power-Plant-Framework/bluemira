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
Tests for shape builders
"""

import pytest

import matplotlib.pyplot as plt
import numpy as np

from bluemira.builders.shapes import MakeParameterisedShape

import tests


class TestMakeParameterisedShape:
    def test_builder(self):
        params = {
            "r_tf_in_centre": (5.0, "Input"),
            "r_tf_out_centre": (9.0, "Input"),
        }
        build_config = {
            "name": "TF Shape",
            "param_class": "PrincetonD",
            "variables_map": {
                "x1": "r_tf_in_centre",
                "x2": {
                    "value": "r_tf_out_centre",
                    "lower_bound": 8.0,
                },
                "dz": 0.0,
            },
            "target": "TF Coils/xz/Shape",
        }
        builder = MakeParameterisedShape(params, build_config)
        component = builder.build(params)
        assert component is not None

        target_split = build_config["target"].split("/")
        target_path = "/".join(target_split)
        component_name = target_split[-1]

        if tests.PLOTTING:
            shape = component[0][1].shape.discretize()
            plt.plot(*shape.T[0::2])
            plt.gca().set_aspect("equal")
            plt.show()

        assert component[0][0] == target_path
        assert component[0][1].name == component_name

        discr = component[0][1].shape.discretize()
        assert min(discr.T[0]) == pytest.approx(params["r_tf_in_centre"][0], abs=1e-3)
        assert max(discr.T[0]) == pytest.approx(params["r_tf_out_centre"][0], abs=1e-3)
        assert np.average(discr.T[1]) == pytest.approx(
            build_config["variables_map"]["dz"], abs=1e-3
        )
