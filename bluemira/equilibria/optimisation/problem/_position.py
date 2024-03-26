# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraintSet,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.objectives import (
    RegularisedLsqObjective,
)
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise
from bluemira.utilities.positioning import PositionMapper


class CoilsetPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents and positions
    subject to maximum current bounds and positions bounded within
    a provided region.

    Coil currents and positions are optimised simultaneously.

    Parameters
    ----------
    coilset:
        Coilset to optimise.
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    position_mapper:
        Position mappings of coil regions
    max_currents:
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    gamma:
        Tikhonov regularisation parameter in units of [A⁻¹].
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    opt_parameters:
        Optimiser specific parameters,
        see https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters
        Otherwise, the parameters can be founded by digging through the source code.
    constraints:
        List of optimisation constraints to apply to the optimisation problem

    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        position_mapper: PositionMapper,
        max_currents: npt.ArrayLike | None = None,
        gamma=1e-8,
        opt_algorithm: AlgorithmType = Algorithm.SBPLX,
        opt_conditions: dict[str, float] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        self.eq = eq
        self.coilset = eq.coilset
        self.targets = targets
        self.position_mapper = position_mapper
        self.bounds = self.get_mapped_state_bounds(max_currents)
        self.gamma = gamma
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults({
            "max_eval": 100
        })
        self.opt_parameters = opt_parameters
        self._constraints = [] if constraints is None else constraints

    def optimise(self, x0: Optional[npt.NDArray] = None, **_) -> CoilsetOptimiserResult:
        """
        Run the optimisation.

        Returns
        -------
        The result of the optimisation.
        """
        if x0 is None:
            # Get initial state and apply region mapping to coil positions.
            cs_opt_state = self.coilset.get_optimisation_state(
                self.position_mapper.interpolator_names, current_scale=self.scale
            )
            initial_mapped_positions = self.position_mapper.to_L(
                cs_opt_state.xs, cs_opt_state.zs
            )

            len_mapped_pos = len(initial_mapped_positions)
            x0 = np.concatenate((initial_mapped_positions, cs_opt_state.currents))

        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=lambda vector: self.objective(vector, len_mapped_pos),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            algorithm=self.opt_algorithm,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

        # Updates the coilset with the final optimised state vector
        self.objective(opt_result.x, len_mapped_pos)

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)

    def objective(self, vector: npt.NDArray[np.float64], len_mapped_pos: int) -> float:
        """
        Least-squares objective with Tikhonov regularisation term.

        Parameters
        ----------
        vector:
            The new coilset state vector.

        Returns
        -------
        The figure of merit being minimised.
        """
        # Update the coilset with the new state vector
        opt_mapped_positions, opt_currents = np.array_split(vector, len_mapped_pos)
        coil_position_map = self.position_mapper.to_xz_dict(opt_mapped_positions)

        self.coilset.set_optimisation_state(
            opt_currents, coil_position_map, current_scale=self.scale
        )

        # Update target
        self.eq._remap_greens()

        # Scale the control matrix and constraint vector by weights.
        self.targets(self.eq, I_not_dI=True, fixed_coils=False)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()

        objective = RegularisedLsqObjective(
            scale=self.scale,
            a_mat=a_mat,
            b_vec=b_vec,
            gamma=self.gamma,
            current_sym_mat=self.coilset._optimisation_currents_sym_mat,
        )
        return objective.f_objective(opt_currents)

    def get_mapped_state_bounds(
        self, max_currents: npt.ArrayLike | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mapped bounds on the coilset state vector from the coil regions and
        maximum coil currents.

        Parameters
        ----------
        max_currents:
            Maximum allowed current for each independent coil current in coilset [A].
            If specified as a float, the float will set the maximum allowed current
            for all coils.

        Returns
        -------
        bounds: np.array
            Array containing state vectors representing lower and upper bounds
            for coilset state degrees of freedom.
        """
        # Get mapped position bounds from PositionMapper
        opt_dimension = self.position_mapper.dimension
        lower_pos_bounds, upper_pos_bounds = (
            np.zeros(opt_dimension),
            np.ones(opt_dimension),
        )
        current_bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)

        return (
            np.concatenate((lower_pos_bounds, current_bounds[0])),
            np.concatenate((upper_pos_bounds, current_bounds[1])),
        )
