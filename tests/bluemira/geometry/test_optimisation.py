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

import numpy as np

from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.utilities.optimiser import Optimiser
from bluemira.geometry.parameterisations import TripleArc, PrincetonD


class TestOptimisationProblem(GeometryOptimisationProblem):
    def __init__(self, parameterisation, optimiser, constrain_shape=True):
        super().__init__(parameterisation, optimiser)
        if constrain_shape:
            self.optimiser.add_ineq_constraints(
                self.parameterisation.shape_ineq_constraints,
                1e-6 * np.ones(self.parameterisation.n_ineq_constraints),
            )

    def calculate_length(self, x):
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length

    def f_objective(self, x, grad):
        value = self.calculate_length(x)

        if grad.size > 0:
            grad[:] = self.optimiser.approx_derivative(
                self.calculate_length, x, f0=value
            )
        return value


class TestGeometryOptimisationProblem:
    @classmethod
    def setup_class(cls):
        parameterisation = TripleArc(
            {
                "x1": {"value": 3.2, "lower_bound": 3.0, "fixed": False},
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions={
                "max_eval": 1000,
            },
        )
        problem = TestOptimisationProblem(
            parameterisation, optimiser, constrain_shape=False
        )

        problem.solve()
        cls.ref_length = parameterisation.create_shape().length

    def test_dummy_constraint(self):
        parameterisation = TripleArc(
            {
                "x1": {"value": 3.2, "lower_bound": 3.0, "fixed": False},
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions={
                "max_eval": 1000,
            },
        )
        problem = TestOptimisationProblem(parameterisation, optimiser)

        assert problem.parameterisation.variables.n_free_variables == 7
        assert problem.parameterisation.variables._fixed_variable_indices == []

        problem.solve()
        length = parameterisation.create_shape().length
        assert np.isclose(length, self.ref_length)

    def test_fixed_var(self):
        parameterisation = TripleArc(
            {
                "x1": {"value": 3.2, "fixed": True},
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions={
                "max_eval": 1000,
            },
        )
        problem = TestOptimisationProblem(parameterisation, optimiser)

        assert problem.parameterisation.variables.n_free_variables == 6
        assert problem.parameterisation.variables._fixed_variable_indices == [0]

        problem.solve()
