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

from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import UpdateableConstraint
from bluemira.equilibria.optimisation.objectives import CoilCurrentsObjective
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise


class MinimalCurrentCOP(CoilsetOptimisationProblem):
    """
    Bounded, constrained, minimal current optimisation problem.

    Parameters
    ----------
    coilset:
        Coilset to optimise
    eq:
        Equilibrium object to optimise the currents for
    max_currents:
        Current bounds vector [A]
    opt_conditions:
        Optimiser conditions
    opt_algorithm:
        optimiser algorithm
    opt_parameters:
        Optimiser specific parameters,
        see https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters
        Otherwise, the parameters can be founded by digging through the source code.
    constraints:
        List of optimisation constraints to apply to the optimisation problem
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        max_currents: Optional[npt.ArrayLike] = None,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: Optional[Dict[str, Union[float, int]]] = None,
        opt_parameters: Optional[Dict[str, float]] = None,
        constraints: Optional[List[UpdateableConstraint]] = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_conditions = opt_conditions
        self.opt_algorithm = opt_algorithm
        self.opt_parameters = opt_parameters
        self._constraints = [] if constraints is None else constraints

    def optimise(self, x0: Optional[npt.NDArray] = None, fixed_coils: bool = True):
        """
        Run the optimisation problem

        Parameters
        ----------
        fixed_coils:
            Whether or not to update to coilset response matrices

        Returns
        -------
        coilset: CoilSet
            Optimised CoilSet
        """
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            initial_state, n_states = self.read_coilset_state(
                self.eq.coilset, self.scale
            )
            _, _, initial_currents = np.array_split(initial_state, n_states)
            x0 = np.clip(initial_currents, *self.bounds)

        objective = CoilCurrentsObjective()
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            algorithm=self.opt_algorithm,
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
