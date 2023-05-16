# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

from typing import Tuple

import pytest

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_circle
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation import GeomOptimisationProblem
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables


class Circle(GeometryParameterisation):
    def __init__(self, radius: float, centre: Tuple[float, float]):
        opt_vars = OptVariables(
            [
                BoundedVariable("radius", radius, 1e-8, 15),
                BoundedVariable("centre_x", centre[0], -10, 10),
                BoundedVariable("centre_z", centre[1], -10, 10),
            ]
        )
        super().__init__(opt_vars)

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
    def test_maximise_circle_len_with_kozs(self):
        op = MaxCircleLenOptProblem()
        circle = Circle(1, (1, 1))

        result = op.optimise(
            circle, algorithm="SLSQP", opt_conditions={"xtol_rel": 1e-12}
        )

        assert result.geom.variables["radius"].value == pytest.approx(15)
        assert result.geom.variables["centre_x"].value == pytest.approx(1)
        assert result.geom.variables["centre_z"].value == pytest.approx(1)
