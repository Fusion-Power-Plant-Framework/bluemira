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
import pytest

from bluemira.geometry.optimisation import GeometryOptimisationProblem, MinimiseLength
from bluemira.geometry.parameterisations import PictureFrame, TripleArc
from bluemira.geometry.tools import make_circle
from bluemira.utilities.optimiser import Optimiser


class DummyOptimisationProblem(GeometryOptimisationProblem):
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
                "x1": {
                    "value": 3.2,
                    "lower_bound": 3.0,
                    "upper_bound": 3.2,
                    "fixed": False,
                },
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        cls.opt_conditions = {
            "max_eval": 1000,
            "ftol_rel": 1e-6,
            "xtol_rel": 1e-12,
            "xtol_abs": 1e-12,
        }
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions=cls.opt_conditions,
        )
        problem = DummyOptimisationProblem(parameterisation, optimiser)
        problem.apply_shape_constraints()
        problem.solve()
        cls.ref_length = parameterisation.create_shape().length

    def test_repetition(self):
        parameterisation = TripleArc(
            {
                "x1": {
                    "value": 3.2,
                    "lower_bound": 3.0,
                    "upper_bound": 3.2,
                    "fixed": False,
                },
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions=self.opt_conditions,
        )
        problem = DummyOptimisationProblem(parameterisation, optimiser)
        problem.apply_shape_constraints()
        assert problem.parameterisation.variables.n_free_variables == 7
        assert problem.parameterisation.variables._fixed_variable_indices == []

        problem.solve()
        length = parameterisation.create_shape().length
        assert np.isclose(length, self.ref_length)

    def test_fixed_var(self):
        parameterisation = TripleArc(
            {
                "x1": {
                    "value": 3.2,
                    "lower_bound": 3.0,
                    "upper_bound": 3.2,
                    "fixed": True,
                },
                "dz": {"value": -0.5, "upper_bound": -0.3},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions=self.opt_conditions,
        )
        problem = DummyOptimisationProblem(parameterisation, optimiser)
        problem.apply_shape_constraints()
        assert problem.parameterisation.variables.n_free_variables == 6
        assert problem.parameterisation.variables._fixed_variable_indices == [0]

        problem.solve()
        length = parameterisation.create_shape().length
        assert np.isclose(length, self.ref_length)

    def test_fixed_var_con(self):
        parameterisation = TripleArc(
            {
                "x1": {
                    "value": 3.2,
                    "lower_bound": 3.0,
                    "upper_bound": 3.2,
                    "fixed": True,
                },
                "dz": {"value": -0.5, "upper_bound": -0.3},
                "a1": {"value": 100, "fixed": True},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions=self.opt_conditions,
        )
        problem = DummyOptimisationProblem(parameterisation, optimiser)

        assert problem.parameterisation.variables.n_free_variables == 5
        assert problem.parameterisation.variables._fixed_variable_indices == [0, 5]

        problem.solve()

        assert problem.parameterisation.variables["a1"].value == 100


class TestMinimiseLength:
    def test_minimise_with_keep_out_zone(self):
        # Create a PictureFrame with un-rounded edges (a rectangle) and
        # a circular keep-out zone within it.
        # We expect the rectangle to contract such that the distance
        # between the parallel edges is equal to the diameter of the
        # keep-out zone.
        koz_radius = 4.5
        koz_center = [10, 0, 0]
        keep_out_zone = make_circle(radius=koz_radius, center=koz_center, axis=[0, 1, 0])
        parameterisation = PictureFrame(
            {
                # Make sure bounds are set within the keep-out zone so
                # we know it's doing some work
                "x1": {"value": 4.5, "upper_bound": 6, "lower_bound": 3},
                "x2": {"value": 16, "upper_bound": 17.5, "lower_bound": 14.5},
                "z1": {"value": 8, "upper_bound": 15, "lower_bound": 2.5},
                "z2": {"value": -6, "upper_bound": -2.5, "lower_bound": -15},
                "ri": {"value": 0, "fixed": True},
                "ro": {"value": 0, "fixed": True},
            }
        )
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions={
                "max_eval": 100,
                "ftol_rel": 1e-4,
                "xtol_rel": 1e-8,
                "xtol_abs": 1e-8,
            },
        )
        problem = MinimiseLength(
            parameterisation, optimiser, keep_out_zone=keep_out_zone
        )

        problem.solve()

        optimised_shape = problem.parameterisation.create_shape()
        np.testing.assert_array_almost_equal(
            list(optimised_shape.center_of_mass), koz_center, decimal=2
        )
        bounds = optimised_shape.bounding_box
        assert bounds.x_max - bounds.x_min == pytest.approx(2 * koz_radius, rel=0.01)
        assert bounds.z_max - bounds.z_min == pytest.approx(2 * koz_radius, rel=0.01)
