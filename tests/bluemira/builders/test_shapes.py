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

import numpy as np
from bluemira.base.components import PhysicalComponent

from bluemira.geometry.optimisation import GeometryOptimisationProblem

from bluemira.builders.shapes import MakeParameterisedShape, MakeOptimisedShape

import tests


class TestMakeParameterisedShape:
    def test_builder(self):
        params = {
            "r_tf_in_centre": (5.0, "Input"),
            "r_tf_out_centre": (9.0, "Input"),
        }
        build_config = {
            "name": "TF Coils",
            "param_class": "PrincetonD",
            "variables_map": {
                "x1": "r_tf_in_centre",
                "x2": {
                    "value": "r_tf_out_centre",
                    "lower_bound": 8.0,
                },
                "dz": {"value": 0.0, "fixed": True},
            },
            "label": "Shape",
        }
        builder = MakeParameterisedShape(params, build_config)
        component = builder(params)

        assert component is not None

        name = build_config["name"]
        label = build_config["label"]

        assert component.name == name

        child: PhysicalComponent = component.get_component("Shape")
        assert child is not None
        assert child.shape.label == label

        discr = child.shape.discretize()
        assert min(discr.x) == pytest.approx(params["r_tf_in_centre"][0], abs=1e-3)
        assert max(discr.x) == pytest.approx(params["r_tf_out_centre"][0], abs=1e-3)
        assert np.average(discr.z) == pytest.approx(
            build_config["variables_map"]["dz"]["value"], abs=1e-2
        )

        if tests.PLOTTING:
            component.plot_2d()


class MinimiseLength(GeometryOptimisationProblem):
    """
    A simple geometry optimisation problem that minimises length without constraints.
    """

    def calculate_length(self, x):
        """
        Calculate the length of the GeometryParameterisation
        """
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        """
        Objective function is the length of the parameterised shape.
        """
        length = self.calculate_length(x)

        if grad.size > 0:
            # Only called if a gradient-based optimiser is used
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=length
            )

        return length


class TestMakeOptimisedShape:
    @pytest.mark.parametrize(
        "problem_class",
        ["tests.bluemira.builders.test_shapes::MinimiseLength", MinimiseLength],
    )
    def test_builder(self, problem_class):
        params = {
            "r_tf_in_centre": (5.0, "Input"),
            "r_tf_out_centre": (9.0, "Input"),
        }
        build_config = {
            "name": "TF Coils",
            "param_class": "PrincetonD",
            "variables_map": {
                "x1": {
                    "value": "r_tf_in_centre",
                    "upper_bound": 6.0,
                },
                "x2": {
                    "value": "r_tf_out_centre",
                    "lower_bound": 8.0,
                },
                "dz": {"value": 0.0, "fixed": True},
            },
            "problem_class": problem_class,
            "label": "Shape",
        }
        builder = MakeOptimisedShape(params, build_config)
        component = builder(params)

        assert component is not None

        name = build_config["name"]
        label = build_config["label"]

        assert component.name == name

        child: PhysicalComponent = component.get_component("Shape")
        assert child is not None
        assert child.shape.label == label

        discr = child.shape.discretize()
        assert min(discr.x) == pytest.approx(
            build_config["variables_map"]["x1"]["upper_bound"], abs=1e-3
        )
        assert max(discr.x) == pytest.approx(
            build_config["variables_map"]["x2"]["lower_bound"], abs=1e-3
        )
        assert np.average(discr.z) == pytest.approx(
            build_config["variables_map"]["dz"]["value"], abs=1e-2
        )

        if tests.PLOTTING:
            component.plot_2d()
