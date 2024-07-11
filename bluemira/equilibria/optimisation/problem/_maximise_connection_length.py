# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import UpdateableConstraint
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
        lower: bool = True,
        plasma_facing_boundary: Grid | Coordinates | None = None,
        calculation_method: str = "flux_surface_geometry",
        psi_n_tol: float = 1e-6,
        delta_start: float = 0.01,
        opt_algorithm: str = "SLSQP",
        max_currents: npt.ArrayLike | None = None,
        opt_conditions: dict[str, float] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.lower = lower
        self.plasma_facing_boundary = plasma_facing_boundary
        self.calculation_method = calculation_method
        self.psi_n_tol = psi_n_tol
        self.delta_start = delta_start
        self.opt_algorithm = opt_algorithm
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_conditions = opt_conditions or self._opt_condition_defaults({
            "max_eval": 100
        })
        self.opt_parameters = (
            {"initial_step": 0.03} if opt_parameters is None else opt_parameters
        )
        self._constraints = [] if constraints is None else constraints
        self._args = {
            "eq": self.eq,
            "scale": self.scale,
            "lower": self.lower,
            "psi_n_tol": self.psi_n_tol,
            "delta_start": delta_start,
            "plasma_facing_boundary": self.plasma_facing_boundary,
            "calculation_method": self.calculation_method,
        }

    def optimise(
        self, x0: npt.NDArray | None = None, *, fixed_coils: bool = True
    ) -> CoilsetOptimiserResult:
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
            cs_opt_state = self.coilset.get_optimisation_state(current_scale=self.scale)
            x0 = np.clip(cs_opt_state.currents, *self.bounds)
        else:
            x0 = np.clip(x0 / self.scale, *self.bounds)

        objective = MaximiseConnectionLength(**self._args)
        eq_constraints, ineq_constraints = self._make_numerical_constraints(self.coilset)
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        opt_currents = opt_result.x
        self.coilset.set_optimisation_state(
            opt_currents=opt_currents,
            current_scale=self.scale,
        )
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
