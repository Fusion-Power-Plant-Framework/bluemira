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

"""
OptimisationProblems for coilset design.

New optimisation schemes for the coilset can be provided by subclassing
from CoilsetOP, which is an abstract base class for OptimisationProblems
that use a coilset as their parameterisation object.

Subclasses must provide an optimise() method that returns an optimised
coilset according to a given optimisation objective function.
As the exact form of the state vector that is optimised is often
specific to each objective function, each subclass of CoilsetOP is
generally also specific to a given objective function, since
the method used to map the coilset object to the state vector
(and additional required arguments) will generally differ in each case.

"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import bluemira.equilibria.opt_objectives as objectives
from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.opt_constraints import (
    FieldConstraints,
    MagneticConstraintSet,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.constraints import ConstraintFunction
from bluemira.equilibria.optimisation.objectives import (
    ObjectiveFunction,
    RegularisedLsqObjective,
)
from bluemira.equilibria.positioner import RegionMapper
from bluemira.optimisation import optimise
from bluemira.utilities.opt_tools import regularised_lsq_fom, tikhonov
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from bluemira.utilities.positioning import PositionMapper

__all__ = [
    "UnconstrainedTikhonovCurrentGradientCOP",
    "TikhonovCurrentCOP",
    "CoilsetPositionCOP",
    "NestedCoilsetPositionCOP",
]


class CoilsetOptimisationProblem:
    """
    Abstract base class for OptimisationProblems for the coilset.
    Provides helper methods and utilities for OptimisationProblems
    using a coilset as their parameterisation object.

    Subclasses should provide an optimise() method that
    returns an optimised coilset object, optimised according
    to a specific objective function for that subclass.
    """

    @property
    def coilset(self):
        return self._parameterisation

    @coilset.setter
    def coilset(self, value: CoilSet):
        self._parameterisation = value

    @staticmethod
    def read_coilset_state(coilset, current_scale):
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
        coilset_state: np.array
            State vector containing substate (position and current)
            information for each coil.
        substates: int
            Number of substates (blocks) in the state vector.
        """
        substates = 3
        x, z = coilset.position
        currents = coilset.current / current_scale

        coilset_state = np.concatenate((x, z, currents))
        return coilset_state, substates

    @staticmethod
    def set_coilset_state(coilset, coilset_state, current_scale):
        """
        Set the optimiser coilset from a provided state vector.

        Parameters
        ----------
        coilset: Coilset
            Coilset to set from state vector.
        coilset_state: np.array
            State vector representing degrees of freedom of the coilset,
            to be used to update the coilset.
        current_scale: float
            Factor to scale state vector currents up by when setting
            coilset currents.
            Used to minimise round-off errors in optimisation.
        """
        x, z, currents = np.array_split(coilset_state, 3)

        coilset.x = x
        coilset.z = z
        coilset.current = currents * current_scale

    @staticmethod
    def get_state_bounds(x_bounds, z_bounds, current_bounds):
        """
        Get bounds on the state vector from provided bounds on the substates.

        Parameters
        ----------
        x_bounds: tuple
            Tuple containing lower and upper bounds on the radial coil positions.
        z_bounds: tuple
            Tuple containing lower and upper bounds on the vertical coil positions.
        current_bounds: tuple
            Tuple containing bounds on the coil currents.

        Returns
        -------
        bounds: np.array
            Array containing state vectors representing lower and upper bounds
            for coilset state degrees of freedom.
        """
        lower_bounds = np.concatenate((x_bounds[0], z_bounds[0], current_bounds[0]))
        upper_bounds = np.concatenate((x_bounds[1], z_bounds[1], current_bounds[1]))
        bounds = np.array([lower_bounds, upper_bounds])
        return bounds

    @staticmethod
    def get_current_bounds(coilset, max_currents, current_scale):
        """
        Gets the scaled current vector bounds. Must be called prior to optimise.

        Parameters
        ----------
        coilset: Coilset
            Coilset to fetch current bounds for.
        max_currents: float or np.ndarray
            Maximum magnitude of currents in each coil [A] permitted during optimisation.
            If max_current is supplied as a float, the float will be set as the
            maximum allowed current magnitude for all coils.
            If the coils have current density limits that are more restrictive than these
            coil currents, the smaller current limit of the two will be used for each
            coil.
        current_scale: float
            Factor to scale coilset currents down when returning scaled current limits.

        Returns
        -------
        current_bounds: (np.narray, np.narray)
            Tuple of arrays containing lower and upper bounds for currents
            permitted in each control coil.
        """
        n_control_currents = len(coilset.current[coilset._control_ind])
        scaled_input_current_limits = np.inf * np.ones(n_control_currents)

        if max_currents is not None:
            input_current_limits = np.asarray(max_currents)
            input_size = np.size(np.asarray(input_current_limits))
            if input_size == 1 or input_size == n_control_currents:
                scaled_input_current_limits = input_current_limits / current_scale
            else:
                raise EquilibriaError(
                    "Length of max_currents array provided to optimiser is not"
                    "equal to the number of control currents present."
                )

        # Get the current limits from coil current densities
        coilset_current_limits = np.infty * np.ones(n_control_currents)
        coilset_current_limits[coilset._flag_sizefix] = coilset.get_max_current()[
            coilset._flag_sizefix
        ]

        # Limit the control current magnitude by the smaller of the two limits
        control_current_limits = np.minimum(
            scaled_input_current_limits, coilset_current_limits
        )
        current_bounds = (-control_current_limits, control_current_limits)

        return current_bounds

    def set_current_bounds(self, max_currents: np.ndarray):
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
                "Length of maximum current vector must be equal to the number of controls."
            )

        # TODO: sort out this interface
        upper_bounds = np.abs(max_currents) / self.scale
        lower_bounds = -upper_bounds
        self.bounds = (lower_bounds, upper_bounds)

    def update_magnetic_constraints(
        self, I_not_dI: bool = True, fixed_coils: bool = True
    ):
        """
        Update the magnetic optimisation constraints with the state of the Equilibrium
        """
        if self._constraints is not None:
            for constraint in self._constraints:
                if isinstance(constraint, UpdateableConstraint):
                    constraint.prepare(
                        self.eq, I_not_dI=I_not_dI, fixed_coils=fixed_coils
                    )
                if "scale" in constraint._args:
                    constraint._args["scale"] = self.scale


class TikhonovCurrentCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents subject to maximum current bounds.

    Coilset currents optimised using objectives.regularised_lsq_objective as
    objective function.

    Parameters
    ----------
    coilset: CoilSet
        Coilset to optimise.
    eq: Equilibrium
        Equilibrium object used to update magnetic field targets.
    targets: MagneticConstraintSet
        Set of magnetic field targets to use in objective function.
    gamma: float (default = 1e-8)
        Tikhonov regularisation parameter in units of [A⁻¹].
    max_currents Union[float, np.ndarray] (default = None)
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    optimiser: bluemira.utilities.optimiser.Optimiser
        Optimiser object to use for constrained optimisation.
    constraints: List[OptimisationConstraint] (default: None)
        Optional list of OptimisationConstraint objects storing
        information about constraints that must be satisfied
        during the coilset optimisation, to be provided to the
        optimiser.
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        gamma: float,
        opt_algorithm: str = "SLSQP",
        opt_conditions: Dict[str, Union[float, int]] = {
            "xtol_rel": 1e-4,
            "xtol_abs": 1e-4,
            "ftol_rel": 1e-4,
            "ftol_abs": 1e-4,
            "max_eval": 100,
        },
        opt_parameters: Dict[str, Any] = {"initial_step": 0.03},
        max_currents: Optional[npt.ArrayLike] = None,
        constraints: Optional[List[ConstraintFunction]] = None,
    ):
        self.scale = 1e6  # current_scale
        self.coilset = coilset
        self.eq = eq
        self.targets = targets
        self.gamma = gamma
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions
        self.opt_parameters = opt_parameters
        self._constraints = constraints

    def optimise(self, x0=None, fixed_coils=True):
        """
        Solve the optimisation problem

        Parameters
        ----------
        fixed_coils: True
            Whether or not to update to coilset response matrices

        Returns
        -------
        coilset: CoilSet
            Optimised CoilSet
        """
        # Scale the control matrix and magnetic field targets vector by weights.
        self.targets(self.eq, I_not_dI=True, fixed_coils=fixed_coils)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            initial_state, n_states = self.read_coilset_state(self.coilset, self.scale)
            _, _, initial_currents = np.array_split(initial_state, n_states)
            x0 = np.clip(initial_currents, *self.bounds)

        objective = RegularisedLsqObjective(
            scale=self.scale,
            a_mat=a_mat,
            b_vec=b_vec,
            gamma=self.gamma,
        )
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            algorithm=self.opt_algorithm,
            opt_parameters=self.opt_parameters,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return self.coilset
