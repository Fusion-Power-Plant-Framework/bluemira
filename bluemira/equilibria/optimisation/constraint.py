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
import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.optimisation import Algorithm, optimise
from bluemira.optimisation.typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


# TODO(hsaunders1904): should probably move this to optimisation module
class Constraint(abc.ABC):
    @abc.abstractmethod
    def f_constraint(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def df_constraint(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def tolerance(self) -> npt.NDArray:
        raise NotImplementedError


class L2NormConstraint(Constraint):
    def __init__(
        self,
        tolerance: npt.NDArray,
        a_mat: npt.NDArray,
        b_vec: npt.NDArray,
        scale: float,
        target_value: float,
    ) -> None:
        self._tolerance = tolerance
        self.a = a_mat
        self.b = b_vec
        self.scale = scale
        self.target_value = target_value

    def f_constraint(self, x: npt.NDArray) -> npt.NDArray:
        vector = self.scale * x
        residual = self.a @ vector - self.b
        constraint = residual.T @ residual - self.target_value
        print(f"{constraint=}")
        return constraint

    def df_constraint(self, x: npt.NDArray) -> npt.NDArray:
        df_c = 2 * self.scale * (self.a.T @ self.a @ x - self.a.T @ self.b)
        print(f"{df_c=}")
        return df_c

    def tolerance(self) -> npt.NDArray:
        return self._tolerance


class CoilSetConstraint(abc.ABC):
    @abc.abstractmethod
    def control_response(self, coilset: CoilSet):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def constraint_target(self) -> float:
        pass

    @abc.abstractmethod
    def constraint(self, coilset: CoilSet) -> Constraint:
        """The constraint object to pass to the optimiser."""

    @abc.abstractproperty
    def length(self) -> int:
        """The length of the constraint vector."""

    @abc.abstractproperty
    def weights(self) -> npt.NDArray:
        """The weights for each element in the constraint array."""


class CoilSetConstraintSet:
    r"""Wrapper around a list of :class:`.CoilSetConstraint`\s"""

    def __init__(self, constraints: List[CoilSetConstraint]) -> None:
        self._constraints = constraints

    def get_weighted_arrays(self, coilset: CoilSet, eq: Equilibrium):
        weights = self.weight_matrix()
        weighted_a = weights[:, np.newaxis] * self.control_matrix(coilset)
        weighted_b = weights * self.b(eq)
        return weights, weighted_a, weighted_b

    def b(self, eq: Equilibrium) -> npt.NDArray:
        return self.target(eq) - self.background(eq)

    @property
    def constraint_length(self) -> int:
        """The cumulative size of the constraint set."""
        return sum(c.length for c in self._constraints)

    def weight_matrix(self) -> npt.NDArray:
        """
        Build the weight matrix used in an optimisation.

        This is assumed to be diagonal.
        """
        # TODO(hsaunders1904): how can this be diagonal if it's 1D?
        return np.concatenate([c.weights for c in self._constraints])

    def control_matrix(self, coilset: CoilSet) -> npt.NDArray:
        """Build the control response matrix used in optimisation."""
        return np.vstack([c.control_response(coilset) for c in self._constraints])

    def target(self, eq: Equilibrium) -> npt.NDArray:
        """The constraint target value vector."""
        return np.concatenate(
            [np.full(c.length, c.constraint_target()) for c in self._constraints]
        )

    def background(self, eq: Equilibrium) -> npt.NDArray:
        """The background value vector."""
        return np.concatenate([c.evaluate() for c in self._constraints])

    # def update_psi_boundary(self, psi_boundary: npt.ArrayLike) -> None:
    #     """
    #     Update the target value for all `PsiBoundaryConstraint`s.

    #     Parameters
    #     ----------
    #     psi_bndry:
    #         The target psi boundary value [V.s/rad]
    #     """
    #     boundary_array = np.array(psi_boundary)
    #     for constraint in self._constraints:
    #         if isinstance(constraint, PsiBoundaryConstraint):
    #             constraint.set_constraint_target(boundary_array)


# class PsiBoundaryConstraint(CoilSetConstraint):
#     pass


class IsofluxConstraint(CoilSetConstraint):
    def __init__(
        self,
        x: npt.ArrayLike,
        z: npt.ArrayLike,
        ref_x: float,
        ref_z: float,
        eq: Equilibrium,
        constraint_value: float = 0.0,
        weights: npt.ArrayLike = 1.0,
        tolerance: float = 1e-6,
    ):
        self.x = np.atleast_1d(x)
        self.z = np.atleast_1d(z)
        self.ref_x = ref_x
        self.ref_z = ref_z
        self.eq = eq
        self.constraint_value = constraint_value
        self._weights = (
            np.full_like(self.x, weights)
            if np.isscalar(weights)
            else np.atleast_1d(weights)
        )
        self.tolerance = np.atleast_1d(tolerance)
        # self.constraint = L2NormConstraint

        # TODO: validate x, z and weights have equal length

    def control_response(self, coilset: CoilSet):
        return coilset.psi_response(self.x, self.z, control=True) - coilset.psi_response(
            self.ref_x, self.ref_z, control=True
        )

    def evaluate(self, I_not_dI: bool = True) -> npt.NDArray:
        if I_not_dI:
            return np.atleast_1d(self.eq.plasma.psi(self.x, self.z))
        return np.atleast_1d(self.eq.psi(self.x, self.z))

    def constraint_target(self, I_not_dI: bool = True) -> float:
        if I_not_dI:
            return float(self.eq.plasma.psi(self.ref_x, self.ref_z))
        return float(self.eq.psi(self.ref_x, self.ref_z))

    @property
    def length(self) -> int:
        """The size of the constraint vector."""
        return len(self.x) if hasattr(self.x, "__len__") else 1

    @property
    def weights(self) -> npt.NDArray:
        return self._weights

    def constraint(self, coilset: CoilSet) -> Constraint:
        a_mat = self.control_response(coilset)
        b_vec = self.constraint_target() - self.evaluate()
        return L2NormConstraint(
            a_mat=a_mat,
            b_vec=b_vec,
            target_value=0,
            tolerance=self.tolerance,
            scale=1,
        )


@dataclass
class CoilSetOptimiserResult:
    # eq: Equilibrium
    coilset: CoilSet
    n_evals: int
    history: List[Tuple[float, np.ndarray]]
    constraints_satisfied: Optional[bool]
    f_x: float


class CoilSetOptimisationProblem(abc.ABC):
    @abc.abstractmethod
    def objective(self, coilset) -> float:
        pass

    def df_objective(self, coilset) -> npt.NDArray:
        pass

    def constraints(self) -> CoilSetConstraintSet:
        raise NotImplementedError

    def lower_bounds(self, coilset: CoilSet = None) -> npt.ArrayLike:
        return -np.inf

    def upper_bounds(self, coilset: CoilSet = None) -> npt.ArrayLike:
        return np.inf

    def pre_optimise(self):
        return None

    def optimise(
        self,
        coilset: CoilSet,
        *,
        algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
        check_constraints: bool = True,
        check_constraints_warn: bool = True,
    ) -> CoilSetOptimiserResult:
        self.pre_optimise()
        print(f"{self.a_mat=}")
        print(f"{self.b_vec=}")
        bounds = (self.lower_bounds(coilset), self.upper_bounds(coilset))
        bounds = tuple(b / 1e6 for b in bounds)
        print(f"{bounds=}")
        initial_currents = np.clip(coilset.current, *bounds) / 1e6
        print(f"{initial_currents=}")
        opt_conditions = {
            "xtol_rel": 1e-4,
            "xtol_abs": 1e-4,
            "ftol_rel": 1e-4,
            "ftol_abs": 1e-4,
            "max_eval": 100,
        }
        opt_parameters = {"initial_step": 0.03}
        result = optimise(
            _to_objective(self.objective, coilset),
            x0=initial_currents,
            df_objective=_to_df_objective(self.df_objective, coilset),
            bounds=bounds,
            # ineq_constraints=[
            #     _to_constraint(c, coilset) for c in self.constraints()._constraints
            # ],
            algorithm=algorithm,
            opt_conditions=opt_conditions,
            opt_parameters=opt_parameters,
            keep_history=keep_history,
            check_constraints=check_constraints,
            check_constraints_warn=check_constraints_warn,
        )

        coilset.get_control_coils().current = result.x * 1e6
        return CoilSetOptimiserResult(
            coilset=coilset,
            n_evals=result.n_evals,
            f_x=result.f_x,
            history=result.history,
            constraints_satisfied=result.constraints_satisfied,
        )

    def update_magnetic_constraint(
        self, I_not_dI: bool = True, fixed_coils: bool = True
    ):
        pass

    @staticmethod
    def read_state(coilset: CoilSet) -> npt.NDArray:
        x, z = coilset.position
        currents = coilset.current
        return np.concatenate((x, z, currents))

    def bounds_of_currents(
        self, coilset: CoilSet, max_currents: npt.ArrayLike
    ) -> npt.NDArray:
        n_control_currents = len(coilset.current[coilset._control_ind])
        scaled_input_current_limits = np.inf * np.ones(n_control_currents)

        if max_currents is not None:
            input_current_limits = np.asarray(max_currents)
            input_size = np.size(np.asarray(input_current_limits))
            if input_size == 1 or input_size == n_control_currents:
                scaled_input_current_limits = input_current_limits
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
        return control_current_limits


def _set_coil_state(coilset: CoilSet, state: npt.NDArray) -> CoilSet:
    x, z, currents = np.array_split(state, 3)
    coilset.x = x
    coilset.z = z
    coilset.current = currents
    return coilset


def _to_objective(f: Callable[[CoilSet], float], coilset: CoilSet) -> ObjectiveCallable:
    """Convert a coilset objective function to a normal one."""

    def objective(x: npt.NDArray) -> float:
        state = TikhonovCurrentCOP.read_state(coilset)
        state[-11:] = x * 1e6
        _set_coil_state(coilset, state)
        return f(coilset)

    return objective


def _to_df_objective(
    df: Callable[[CoilSet], npt.NDArray], coilset: CoilSet
) -> OptimiserCallable:
    """Convert a coilset objective gradient to a normal one."""

    def df_objective(x: npt.NDArray) -> npt.NDArray:
        state = TikhonovCurrentCOP.read_state(coilset)
        state[-11:] = x * 1e6
        _set_coil_state(coilset, state)
        return df(coilset)

    return df_objective


def _to_constraint(coil_constraint: CoilSetConstraint, coilset: CoilSet) -> ConstraintT:
    constraint = coil_constraint.constraint(coilset)
    return {
        "f_constraint": constraint.f_constraint,
        "df_constraint": constraint.df_constraint,
        "tolerance": constraint.tolerance(),
    }


class TikhonovCurrentCOP(CoilSetOptimisationProblem):
    hack_ctr = 0

    def __init__(
        self,
        eq: Equilibrium,
        coilset: CoilSet,
        targets: List[CoilSetConstraint],
        gamma: float,
    ):
        self.gamma = gamma
        self.eq = eq
        self.coilset = coilset
        self._targets = targets
        self.a_mat, self.b_vec = self.get_a_mat_b_vec()
        print(f"{self.a_mat=}, {self.b_vec=}")
        self.hack_ctr += 1

    def pre_optimise(self):
        self.a_mat, self.b_vec = self.get_a_mat_b_vec()

    def objective(self, coilset) -> float:
        from bluemira.equilibria.opt_objectives import regularised_lsq_fom

        x = self.read_state(coilset)[-11:]  # TODO(hsaunders1904): normalize/scaling
        print(f"{x=}")
        a_mat, b_vec = self.a_mat, self.b_vec
        fom = regularised_lsq_fom(x, a_mat, b_vec, self.gamma)[0]
        print(f"{fom=}")
        return fom

    def df_objective(self, coilset) -> npt.NDArray:
        x = self.read_state(coilset)[-11:]
        a_mat, b_vec = self.a_mat, self.b_vec
        jac = 2 * a_mat.T @ a_mat @ x / len(b_vec)
        jac -= 2 * a_mat.T @ b_vec / len(b_vec)
        jac += 2 * self.gamma * self.gamma * x
        jac *= 1e6
        print(f"{jac=}")
        return jac

    def lower_bounds(self, coilset: CoilSet) -> npt.NDArray:
        return -self.upper_bounds(coilset)

    def upper_bounds(self, coilset: CoilSet) -> npt.NDArray:
        return self.bounds_of_currents(coilset, coilset.get_max_current())

    def get_a_mat_b_vec(self):
        constraint_set = self.constraints()
        return constraint_set.get_weighted_arrays(self.coilset, self.eq)[1:]

    def constraints(self) -> CoilSetConstraintSet:
        return CoilSetConstraintSet(self._targets)
