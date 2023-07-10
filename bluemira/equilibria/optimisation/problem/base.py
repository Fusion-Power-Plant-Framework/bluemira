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
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.constraint import (
    CoilSetConstraint,
    CoilSetConstraintSet,
)
from bluemira.optimisation import Algorithm, optimise
from bluemira.optimisation.typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


@dataclass
class CoilSetOptimiserResult:
    coilset: CoilSet
    n_evals: int
    history: List[Tuple[float, np.ndarray]]
    constraints_satisfied: Optional[bool]
    f_x: float


class CoilSetOptimisationProblem(abc.ABC):
    @abc.abstractmethod
    def objective(self, coilset: CoilSet) -> float:
        pass

    def df_objective(self, coilset: CoilSet) -> npt.NDArray:
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
        state = CoilSetOptimisationProblem.read_state(coilset)
        state[-11:] = x * 1e6
        _set_coil_state(coilset, state)
        return f(coilset)

    return objective


def _to_df_objective(
    df: Callable[[CoilSet], npt.NDArray], coilset: CoilSet
) -> OptimiserCallable:
    """Convert a coilset objective gradient to a normal one."""

    def df_objective(x: npt.NDArray) -> npt.NDArray:
        state = CoilSetOptimisationProblem.read_state(coilset)
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
