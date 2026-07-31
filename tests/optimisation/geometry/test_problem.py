# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from dataclasses import dataclass

import pytest

from bluemira.geometry.optimisation import GeomOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_circle
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_variables import OptVariable, OptVariablesFrame, ov


@dataclass
class CircleOptVariables(OptVariablesFrame):
    """Optimisation variables for a circle in the xz-plane."""

    radius: OptVariable = ov("radius", 10, 1e-8, 15)
    centre_x: OptVariable = ov("centre_x", 0, -10, 10)
    centre_z: OptVariable = ov("centre_z", 0, 0, 10)


class Circle(GeometryParameterisation):
    param_cls: type[CircleOptVariables] = CircleOptVariables

    def create_shape(self, label: str = "") -> BluemiraWire:
        return make_circle(
            self.variables["radius"].value,
            center=(
                self.variables["centre_x"].value,
                0,
                self.variables["centre_z"].value,
            ),
            axis=(0, 1, 0),
            label=label,
        )


class MaxCircleLenOptProblem(GeomOptimisationProblem):
    def objective(self, geom: Circle) -> float:
        return -geom.create_shape().length


class TestGeomOptimisationProblem:
    @pytest.mark.parametrize("alg", ["SLSQP", "SLSQP_SCIPY"])
    def test_maximise_circle_len_with_kozs(self, alg):
        op = MaxCircleLenOptProblem()
        circle = Circle({
            "radius": {"value": 1},
            "centre_x": {"value": 1},
            "centre_z": {"value": 1},
        })

        result = op.optimise(circle, algorithm=alg, opt_conditions={"xtol_rel": 1e-12})

        assert result.geom.variables["radius"].value == pytest.approx(15)
        assert result.geom.variables["centre_x"].value == pytest.approx(1)
        assert result.geom.variables["centre_z"].value == pytest.approx(1)
