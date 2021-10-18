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
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.optimiser import Optimiser

import nlopt

nlopt.srand(13436547564)


class MyProblem(GeometryOptimisationProblem):
    """
    Minimise the length of a geometry parameterisation
    """

    def f_objective(self, x, grad=None):
        """
        This is the signature for an objective function. If grad=None and a gradient-
        based optimiser is used, the gradient of the objective function is calculated
        under the hood.
        """
        self.update_parameterisation(x)
        return self.parameterisation.create_shape().length


# Here we solve the problem with a gradient-based optimisation algorithm (SLSQP)
# The gradients are automatically calculated under the hood
parameterisation_1 = PrincetonD()
slsqp_optimiser = Optimiser("SLSQP", 3, {}, {"ftol_rel": 1e18})
problem = MyProblem(parameterisation_1, slsqp_optimiser)
problem.solve()

# Now let's do the same with an optimisation algorithm that doesn't require gradients
parameterisation_2 = PrincetonD()
cobyla_optimiser = Optimiser("COBYLA", 3, {}, {"ftol_rel": 1e-3, "max_eval": 1000})
problem = MyProblem(parameterisation_2, cobyla_optimiser)
problem.solve()


class MyConstrainedProblem(GeometryOptimisationProblem):
    def __init__(self, parameterisation, optimiser, ineq_con_tolerances):
        super().__init__(parameterisation, optimiser)
        self.optimiser.add_ineq_constraints(self.f_ineq_constraints, ineq_con_tolerances)

    def f_objective(self, x, grad=None):
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        return length

    def f_ineq_constraints(self, constraint, x, grad=None):
        """
        This is the signature for an inequality constraint. If grad=None and a gradient-
        based optimiser is used, the jacobian of the constraint function is calculated
        under the hood.
        """
        self.update_parameterisation(x)
        length = self.parameterisation.create_shape().length
        constraint[:] = np.array([40 - length, 40 - length])
        return constraint


# square = make_polygon([[5, 0, -2], [8, 0, -2], [8, 0, 2], [5, 0, 2]], closed=True)
parameterisation_3 = PrincetonD()
slsqp_optimiser = Optimiser("SLSQP", 3, {}, {"ftol_rel": 1e-3, "max_eval": 1000})
problem = MyConstrainedProblem(parameterisation_3, slsqp_optimiser, 1e-3 * np.ones(2))
problem.solve()
