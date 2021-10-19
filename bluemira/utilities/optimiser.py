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
Static API to optimisation library
"""
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from bluemira.utilities._nlopt_api import NLOPTOptimiser
from bluemira.utilities.error import InternalOptError


__all__ = ["approx_derivative", "Optimiser"]


class Optimiser(NLOPTOptimiser):
    def optimise(self, x0=None):
        """
        Run the optimisation problem.

        Parameters
        ----------
        x0: Optional[np.ndarray]
            Starting solution vector

        Returns
        -------
        x_star: np.ndarray
            Optimal solution vector
        """
        if x0 is None:
            x0 = 0.5 * np.ones(self.n_variables)

        x_star = super().optimise(x0)
        self.check_constraints(x_star)
        return x_star

    def approx_derivative(self, function, x, f0):
        return approx_derivative(
            function, x, bounds=[self.lower_bounds, self.upper_bounds], f0=f0
        )

    def set_objective_function(self, f_objective):
        if f_objective is None:
            return

        super().set_objective_function(f_objective)

    def add_eq_constraint(self, f_constraint, tolerance):
        if f_constraint is None:
            return
        super().add_eq_constraint(f_constraint, tolerance)

    def add_ineq_constraint(self, f_constraint, tolerance):
        if f_constraint is None:
            return
        super().add_ineq_constraint(f_constraint, tolerance)

    def add_ineq_constraints(self, f_constraint, tolerance):
        if f_constraint is None:
            return
        super().add_ineq_constraints(f_constraint, tolerance)

    def check_constraints(self, x):
        """
        Check that the constraints have been met.
        """

        c_values = []
        tolerances = []
        for constraint, tolerance in zip(self.constraints, self.constraint_tols):
            c_values.extend(constraint(np.zeros(len(tolerance)), x, np.empty(0)))
            tolerances.extend(tolerance)

        c_values = np.array(c_values)
        tolerances = np.array(tolerances)

        if not np.all(c_values < tolerances):
            raise InternalOptError(
                "Some constraints have not been adequately satisfied."
            )
