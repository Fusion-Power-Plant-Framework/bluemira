# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Equilibria Optimisation base module
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSetSymmetryStatus
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.constraints import UpdateableConstraint
from bluemira.optimisation._algorithm import Algorithm, AlgorithmDefaultConditions

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet
    from bluemira.optimisation._optimiser import OptimiserResult
    from bluemira.optimisation.typing import ConstraintT


@dataclass
class CoilsetOptimiserResult:
    """Coilset optimisation result object"""

    coilset: CoilSet
    """The optimised coilset."""
    f_x: float
    """The evaluation of the optimised parameterisation."""
    n_evals: int
    """The number of evaluations of the objective function in the optimisation."""
    history: list[tuple[np.ndarray, float]] = field(repr=False)
    """
    The history of the parametrisation at each iteration.

    The first element of each tuple is the parameterisation (x), the
    second is the evaluation of the objective function at x (f(x)).
    """
    constraints_satisfied: bool | None = None
    """
    Whether all constraints have been satisfied to within the required tolerance.

    Is ``None`` if constraints have not been checked.
    """

    @classmethod
    def from_opt_result(
        cls, coilset: CoilSet, opt_result: OptimiserResult
    ) -> CoilsetOptimiserResult:
        """Make a coilset optimisation result from a normal optimisation result."""
        return cls(
            coilset=coilset,
            f_x=opt_result.f_x,
            n_evals=opt_result.n_evals,
            history=opt_result.history,
            constraints_satisfied=opt_result.constraints_satisfied,
        )


class CoilsetOptimisationProblem(abc.ABC):
    """
    Abstract base class for coilset optimisation problems.

    Subclasses should provide an optimise() method that
    returns an optimised coilset object, optimised according
    to a specific objective function for that subclass.
    """

    def _opt_condition_defaults(
        self, default_cond: dict[str, float | int]
    ) -> dict[str, float | int]:
        algorithm = (
            Algorithm[self.opt_algorithm]
            if not isinstance(self.opt_algorithm, Algorithm)
            else self.opt_algorithm
        )

        return {
            **getattr(AlgorithmDefaultConditions(), algorithm.name).to_dict(),
            **default_cond,
        }

    @abc.abstractmethod
    def optimise(self, **kwargs) -> CoilsetOptimiserResult:
        """Run the coilset optimisation."""

    @property
    def coilset(self) -> CoilSet:
        """The optimisation problem coilset"""
        return self._coilset

    @coilset.setter
    def coilset(self, value: CoilSet):
        self._coilset = value

    @staticmethod
    def get_current_bounds(
        coilset: CoilSet, max_currents: npt.ArrayLike | None, current_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gets the scaled current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        coilset:
            Coilset to fetch current bounds for.
        max_currents:
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.
            If the coils have current density limits that are more restrictive than these
            coil currents, the smaller current limit of the two will be used for each
            coil.
        current_scale:
            Factor to scale coilset currents down when returning scaled current limits.

        Returns
        -------
        current_bounds: (np.narray, np.narray)
            Tuple of arrays containing lower and upper bounds for currents
            permitted in each control coil.
        """
        cc = coilset.get_control_coils()

        n_cc_opt_currents = cc.n_current_optimisable_coils
        scaled_input_current_limits = np.inf * np.ones(n_cc_opt_currents)

        if max_currents is not None:
            input_current_limits = np.asarray(max_currents)
            input_size = np.size(input_current_limits)
            if input_size in {1, n_cc_opt_currents}:
                scaled_input_current_limits = input_current_limits / current_scale
            else:
                raise EquilibriaError(
                    f"Length of max_currents {input_size} array provided to "
                    "the optimiser is not equal to the number of "
                    f"optimisable control currents present {n_cc_opt_currents}."
                )

        # Get the current limits from coil current densities

        # if a coil has no jmax, then the current is limited by the max current provided
        # or default to inf
        # if a coil has jmax and is fixed (sized), then the current is limited by
        # jmax * area
        # if a coil is not fixed (sized) and it has jmax, then the current is limited
        # by the max current provided or defaults to inf

        opt_coils_max_currents = cc.get_max_current()[cc._opt_currents_inds]

        # Limit the control current magnitude by the smaller of the two limits
        control_current_limits = np.minimum(
            scaled_input_current_limits, opt_coils_max_currents
        )
        return (-control_current_limits, control_current_limits)

    def set_current_bounds(self, max_currents: npt.NDArray[np.float64]):
        """
        Set the current bounds on this instance.

        Parameters
        ----------
        max_currents:
            Vector of maximum currents [A]
        """
        n_control_currents = len(self.coilset.current[self.coilset._control_ind])
        if len(max_currents) != n_control_currents:
            raise ValueError(
                "Length of maximum current vector must be equal to the number of"
                " controls."
            )

        # TODO: sort out this interface
        upper_bounds = np.abs(max_currents) / self.scale
        lower_bounds = -upper_bounds
        self.bounds = (lower_bounds, upper_bounds)

    def update_magnetic_constraints(
        self, *, I_not_dI: bool = True, fixed_coils: bool = True
    ):
        """
        Update the magnetic optimisation constraints with the state of the Equilibrium
        """
        if not hasattr(self, "_constraints"):
            return
        for constraint in self._constraints:
            if isinstance(constraint, UpdateableConstraint):
                constraint.prepare(self.eq, I_not_dI=I_not_dI, fixed_coils=fixed_coils)
            if "scale" in constraint._args:
                constraint._args["scale"] = self.scale

    def _make_numerical_constraints(
        self, coilset: CoilSet
    ) -> tuple[list[ConstraintT], list[ConstraintT]]:
        """Build the numerical equality and inequality constraint dictionaries."""
        if (constraints := getattr(self, "_constraints", None)) is None:
            return [], []
        equality = []
        inequality = []
        for constraint in constraints:
            f = constraint.f_constraint()

            f_c = f.f_constraint
            df_c = getattr(f, "df_constraint", None)

            if coilset._opt_currents_symmetry_status in {
                CoilSetSymmetryStatus.FULL,
                CoilSetSymmetryStatus.PARTIAL,
            }:
                # wrap the constraint function
                @functools.wraps(f.f_constraint)
                def wrapped_f_c(x, f=f):
                    return f.f_constraint(coilset._opt_currents_expand_mat @ x)

                f_c = wrapped_f_c

                # wrap the derivative function
                if (
                    df_c is None
                    or coilset._opt_currents_symmetry_status
                    is CoilSetSymmetryStatus.PARTIAL
                ):
                    # if partially symmetric, the derivative function
                    # is set to None as there is no analytical solution
                    df_c = None
                else:
                    # if fully symmetric, the result of the derivative
                    # function is multiplied by the repetition matrix
                    # (to reduce the shape to the number of current optimisable coils)
                    # by multiplying the result with the repetition matrix
                    # and dividing by 2 (as the values are added together)
                    def wrapped_df_c(x, f=f):
                        df_res = f.df_constraint(coilset._opt_currents_expand_mat @ x)
                        return df_res @ coilset._opt_currents_sym_reduce_mat

                    df_c = wrapped_df_c

            d: ConstraintT = {
                "f_constraint": f_c,
                "df_constraint": df_c,
                "tolerance": constraint.tolerance,
            }
            # TODO: tidy this up, so the interface guarantees this works!
            if getattr(constraint, "constraint_type", "inequality") == "equality":
                equality.append(d)
            else:
                inequality.append(d)

        return equality, inequality

    @property
    def scale(self) -> float:
        """Problem scaling value"""
        return 1e6
