# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import UpdateableConstraint
from bluemira.equilibria.optimisation.objectives import MaximiseConnectionLength
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.optimisation import optimise


class MaximiseConnectionLengthCOP(CoilsetOptimisationProblem):
    """
    Bounded, constrained, minimal current optimisation problem.

    Parameters
    ----------
    eq:
        Equilibrium object to optimise the currents for
    optimiser:
        Optimiser object to use
    max_currents:
        Current bounds vector [A]
    constraints:
        List of optimisation constraints to apply to the optimisation problem
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        double_null: bool = False,
        lower: bool = True,
        outer: bool = True,
        psi_n_tol: float = 1e-6,
        delta_start: float = 0.01,
        plasma_facing_boundary: Optional[Union[Grid, Coordinates]] = None,
        max_currents: Optional[npt.ArrayLike] = None,
        opt_algorithm: str = "SLSQP",
        opt_conditions: Optional[Dict[str, float]] = None,
        opt_parameters: Optional[Dict[str, float]] = None,
        constraints: Optional[List[UpdateableConstraint]] = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.double_null = double_null
        self.lower = lower
        self.outer = outer
        self.psi_n_tol = psi_n_tol
        self.delta_start = delta_start
        self.plasma_facing_boundary = plasma_facing_boundary
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_conditions = opt_conditions
        self.opt_algorithm = opt_algorithm
        self.opt_parameters = (opt_parameters,)
        self._args = {
            "eq": self.eq,
            "scale": self.scale,
            "double_null": self.double_null,
            "lower": self.lower,
            "outer": self.outer,
            "psi_n_tol": self.psi_n_tol,
            "delta_start": delta_start,
            "plasma_facing_boundary": self.plasma_facing_boundary,
        }
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

        objective = MaximiseConnectionLength(**self._args)
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            # opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
