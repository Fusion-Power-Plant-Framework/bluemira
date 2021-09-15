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
import numpy as np

from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.opt_variables import OptVariables, BoundedVariable
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry.parameterisations import (
    princeton_d,
    PrincetonD,
    GeometryParameterisation,
)
from bluemira.geometry.tools import make_polygon
from bluemira.geometry._deprecated_tools import get_perimeter
from bluemira.geometry.wire import BluemiraWire


class TestGeometryParameterisation:
    def test_subclass(self):
        class TestPara(GeometryParameterisation):
            def __init__(self):
                variables = OptVariables(
                    [
                        BoundedVariable("a", 0, -1, 1),
                        BoundedVariable("b", 2, 0, 4),
                        BoundedVariable("c", 4, 2, 6, fixed=True),
                    ],
                    frozen=True,
                )
                super().__init__("test", variables)

            def create_shape(self, **kwargs):
                return BluemiraWire(
                    make_polygon(
                        [
                            [self.variables["a"], 0, 0],
                            [self.variables["b"], 0, 0],
                            [self.variables["c"], 0, 1],
                            [self.variables["a"], 0, 1],
                        ]
                    )
                )

        t = TestPara()


class TestPrincetonD:
    @pytest.mark.parametrize("x1", [1, 4, 5])
    @pytest.mark.parametrize("x2", [6, 10, 200])
    @pytest.mark.parametrize("dz", [0, 100])
    def test_princeton_d(self, x1, x2, dz):
        x, z = princeton_d(x1, x2, dz, 500)
        assert len(x) == 500
        assert len(z) == 500
        assert np.isclose(np.min(x), x1)
        assert np.isclose(np.max(x), x2, rtol=1e-3)
        # check symmetry
        assert np.isclose(np.mean(z), dz)
        assert np.allclose(x[:250], x[250:][::-1])

    def test_error(self):
        with pytest.raises(GeometryParameterisationError):
            princeton_d(10, 3, 0)

    def test_parameterisation(self):
        p = PrincetonD()
        p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
        p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
        p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)

        wire = p.create_shape()
        array = p.create_array(n_points=200)

        assert np.isclose(wire.length, get_perimeter(*array), rtol=1e-3)

    def test_bad_behaviour(self):
        p = PrincetonD()
        with pytest.raises(OptVariablesError):
            p.variables.add_variable(BoundedVariable("new", 0, 0, 0))

        with pytest.raises(OptVariablesError):
            p.variables.remove_variable("x1")
