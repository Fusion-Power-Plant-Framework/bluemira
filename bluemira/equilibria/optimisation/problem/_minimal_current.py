# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from typing import Dict, List, Optional

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
    constraints:
        List of optimisation constraints to apply to the optimisation problem
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        max_currents: Optional[npt.ArrayLike] = None,
        opt_conditions: Optional[Dict[str, float]] = None,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        constraints: Optional[List[UpdateableConstraint]] = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_conditions = opt_conditions
        self.opt_algorithm = opt_algorithm
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
            f_objective=objective.f_objective,
            df_objective=objective.df_objective,
            x0=x0,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
