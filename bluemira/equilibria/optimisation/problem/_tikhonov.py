# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraintSet,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.objectives import RegularisedLsqObjective, tikhonov
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
    EqCoilsetOptimisationProblem,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise
from bluemira.utilities.tools import floatify


class TikhonovCurrentCOP(EqCoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents subject to maximum current bounds.

    Coilset currents optimised using objectives.regularised_lsq_objective as
    objective function.

    Parameters
    ----------
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    gamma:
        Tikhonov regularisation parameter in units of [A⁻¹].
    opt_algorithm:
        Optimiser algorithm
    opt_conditions:
        optimiser conditions
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    opt_parameters:
        optimisation parameters
    max_currents:
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    constraints:
        Optional list of UpdatableConstraint objects storing
        information about constraints that must be satisfied
        during the coilset optimisation, to be provided to the
        optimiser.
    """

    def __init__(
        self,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        gamma: float,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: dict[str, float | int] | None = None,
        opt_parameters: dict[str, float] | None = None,
        max_currents: npt.ArrayLike | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        super().__init__(
            eq,
            opt_algorithm,
            max_currents=max_currents,
            opt_conditions=opt_conditions,
            constraints=constraints,
            opt_parameters={"initial_step": 0.03, **(opt_parameters or {})},
            targets=targets,
        )
        self.gamma = gamma

    def optimise(self, x0=None, *, fixed_coils=True) -> CoilsetOptimiserResult:
        """
        Solve the optimisation problem

        Parameters
        ----------
        fixed_coils:
            Whether or not to update to coilset response matrices

        Returns
        -------
        coilset:
            Optimised CoilSet
        """
        # Scale the control matrix and magnetic field targets vector by weights.
        self.targets(self.eq, I_not_dI=True, fixed_coils=fixed_coils)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            cs_opt_state = self.coilset.get_optimisation_state(current_scale=self.scale)
            x0 = np.clip(cs_opt_state.currents, *self.bounds)
        else:
            x0 = np.clip(x0 / self.scale, *self.bounds)

        objective = RegularisedLsqObjective(
            scale=self.scale,
            a_mat=a_mat,
            b_vec=b_vec,
            gamma=self.gamma,
            currents_expand_mat=self.eq.coilset._opt_currents_expand_mat,
        )
        eq_constraints, ineq_constraints = self._make_numerical_constraints(self.coilset)
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            algorithm=self.opt_algorithm,
            opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

        opt_currents = opt_result.x
        self.coilset.set_optimisation_state(
            opt_currents=opt_currents, current_scale=self.scale
        )

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)


class UnconstrainedTikhonovCurrentGradientCOP(CoilsetOptimisationProblem):
    """
    Unbounded, unconstrained, analytically optimised current gradient vector for minimal
    error to the L2-norm of a set of magnetic constraints (used here as targets).

    This is useful for getting a preliminary Equilibrium

    Parameters
    ----------
    coilset:
        CoilSet object to optimise with
    eq:
        Equilibrium object to optimise for
    targets:
        Set of magnetic constraints to minimise the error for
    gamma:
        Tikhonov regularisation parameter [1/A]
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        gamma: float,
    ):
        self.coilset = coilset
        self.eq = eq
        self.targets = targets
        self.gamma = gamma

    def optimise(self, **_) -> CoilsetOptimiserResult:
        """
        Optimise the prescribed problem.

        Notes
        -----
        The weight vector is used to scale the response matrix and
        constraint vector. The weights are assumed to be uncorrelated, such that the
        weight matrix W_ij used to define (for example) the least-squares objective
        function (Ax - b)ᵀ W (Ax - b), is diagonal, such that
        weights[i] = w[i] = sqrt(W[i,i]).
        """  # noqa: DOC201
        # Scale the control matrix and magnetic field targets vector by weights.
        self.targets(self.eq, I_not_dI=False)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()

        c_cs = self.eq.coilset.get_control_coils()

        # Optimise currents using analytic expression for optimum.
        current_adjustment = tikhonov(
            a_mat, b_vec, self.gamma, currents_expand_mat=c_cs._opt_currents_expand_mat
        )

        # Update parameterisation (coilset).
        opt_currents = c_cs._opt_currents + current_adjustment
        c_cs.set_optimisation_state(opt_currents=opt_currents, current_scale=1.0)

        f_x = floatify(
            np.linalg.norm(a_mat @ c_cs.current - b_vec)
            + np.linalg.norm(self.gamma * c_cs.current)
        )
        return CoilsetOptimiserResult(
            coilset=c_cs, f_x=f_x, n_evals=0, history=[], constraints_satisfied=True
        )
