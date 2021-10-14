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
Geometry optimisation classes and tools
"""

import abc
import numpy as np

from bluemira.geometry.tools import distance_to
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.opt_tools import Optimiser


def signed_distance_function(shape1, shape2):
    in_or_out = 1.0
    min_distance = distance_to(shape1, shape2)
    return in_or_out * min_distance


class GeometryOptimisationProblem(abc.ABC):
    """
    Geometry optimisation class.

    Parameters
    ----------
    parameterisation: GeometryParameterisation
        Geometry parameterisation instance to use in the optimisation problem
    optimiser: Optimiser
        Optimiser instance to use in the optimisation problem
    """

    def __init__(self, parameterisation: GeometryParameterisation, optimiser: Optimiser):
        self.parameterisation = parameterisation
        self.optimiser = optimiser
        self.optimiser.set_initial_value(
            self.parameterisation.variables.get_normalised_values()
        )
        self.optimiser.set_lower_bounds(np.zeros(optimiser.n_variables))
        self.optimiser.set_upper_bounds(np.ones(optimiser.n_variables))
        self.optimiser.set_objective_function(self.f_objective)

    def update_parameterisation(self, x):
        self.parameterisation.variables.set_values_from_norm(x)

    f_objective = None
    f_eq_constraints = None
    f_ineq_constraints = None

    def solve(self):
        x_star = self.optimiser.optimise()
        self.update_parameterisation(x_star)

    # @abc.abstractmethod
    # def f_objective(self, x):
    #     pass

    # @abc.abstractmethod
    # def f_eq_constraints(self, x):
    #     pass

    # @abc.abstractmethod
    # def f_ineq_constraints(self, x):
    #     pass
