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
        self, default_cond=dict[str, float | int]
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
    def read_coilset_state(
        coilset: CoilSet, current_scale: float
    ) -> tuple[npt.NDArray[np.float64], int]:
        """
        Reads the input coilset and generates the state vector as an array to represent
        it.

        Parameters
        ----------
        coilset: Coilset
            Coilset to be read into the state vector.
        current_scale: float
            Factor to scale coilset currents down by for population of coilset_state.
            Used to minimise round-off errors in optimisation.

        Returns
        -------
        State vector containing substate (position and current)
        information for each coil.

        Number of substates (blocks) in the state vector.
        """
        substates = 3
        cc = coilset.get_control_coils()
        x, z = cc._optimisation_positions
        currents = cc._optimisation_currents / current_scale

        coilset_state = np.concatenate((x, z, currents))
        return coilset_state, substates

    @staticmethod
    def set_coilset_state(
        coilset: CoilSet, coilset_state: npt.NDArray[np.float64], current_scale: float
    ):
        """
        Set the optimiser coilset from a provided state vector.

        Parameters
        ----------
        coilset:
            Coilset to set from state vector.
        coilset_state:
            State vector representing degrees of freedom of the coilset,
            to be used to update the coilset.
        current_scale:
            Factor to scale state vector currents up by when setting
            coilset currents.
            Used to minimise round-off errors in optimisation.
        """
        x, z, currents = np.array_split(coilset_state, 3)

        cc = coilset.get_control_coils()
        cc.x = x
        cc.z = z
        cc.current = currents * current_scale

    @staticmethod
    def get_state_bounds(
        x_bounds: tuple[npt.NDArray, npt.NDArray],
        z_bounds: tuple[npt.NDArray, npt.NDArray],
        current_bounds: tuple[npt.NDArray, npt.NDArray],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray]:
        """
        Get bounds on the state vector from provided bounds on the substates.

        Parameters
        ----------
        x_bounds:
            Tuple containing lower and upper bounds on the radial coil positions.
        z_bounds:
            Tuple containing lower and upper bounds on the vertical coil positions.
        current_bounds:
            Tuple containing bounds on the coil currents.

        Returns
        -------
        Array containing state vectors representing lower and upper bounds
        for coilset state degrees of freedom.
        """
        lower_bounds = np.concatenate((x_bounds[0], z_bounds[0], current_bounds[0]))
        upper_bounds = np.concatenate((x_bounds[1], z_bounds[1], current_bounds[1]))
        return np.array([lower_bounds, upper_bounds])

    @staticmethod
    def get_current_bounds(
        coilset: CoilSet, max_currents: npt.ArrayLike, current_scale: float
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

        opt_coils_max_currents = cc.get_max_current()[cc._optimisation_currents_inds]

        # Limit the control current magnitude by the smaller of the two limits
        control_current_limits = np.minimum(
            scaled_input_current_limits, opt_coils_max_currents
        )
        # todo: shouldn't the first argument be 0'S?
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
        self,
    ) -> tuple[list[ConstraintT], list[ConstraintT]]:
        """Build the numerical equality and inequality constraint dictionaries."""
        if (constraints := getattr(self, "_constraints", None)) is None:
            return [], []
        equality = []
        inequality = []
        for constraint in constraints:
            f_constraint = constraint.f_constraint()
            d: ConstraintT = {
                "f_constraint": f_constraint.f_constraint,
                "df_constraint": getattr(f_constraint, "df_constraint", None),
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
