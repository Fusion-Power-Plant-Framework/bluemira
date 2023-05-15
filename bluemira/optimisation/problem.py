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
"""Interface for defining an optimisation problem."""
import abc
from typing import Any, Callable, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np

from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._optimise import OptimiserResult, optimise
from bluemira.optimisation.typing import ConstraintT


class OptimisationProblem(abc.ABC):
    """Interface for a an optimisation problem."""

    @abc.abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """The objective function to minimise."""

    def df_objective(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the objective function at ``x``."""

    def eq_constraints(self) -> List[ConstraintT]:
        """The equality constraints on the optimisation."""
        return []

    def ineq_constraints(self) -> List[ConstraintT]:
        """The inequality constraints on the optimisation."""
        return []

    def bounds(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """The lower and upper bounds of the optimisation parameters."""
        return None

    def optimise(
        self,
        x0: np.ndarray,
        *,
        algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
    ) -> OptimiserResult:
        """Perform the optimisation."""
        return optimise(
            self.objective,
            df_objective=self.__overridden_or_default(self.df_objective, None),
            x0=x0,
            algorithm=algorithm,
            opt_conditions=opt_conditions,
            opt_parameters=opt_parameters,
            bounds=self.bounds(),
            eq_constraints=self.eq_constraints(),
            ineq_constraints=self.ineq_constraints(),
            keep_history=keep_history,
        )

    __T1 = TypeVar("__T1", bound=Callable[[Any], Any])
    __T2 = TypeVar("__T2")

    def __overridden_or_default(self, f: __T1, default: __T2) -> Union[__T1, __T2]:
        """
        If the given object is not a member of this class return a default.

        This can be used to decide whether a function has been overridden or not.
        Which is useful in this class for the ``df_objective`` case, where overriding
        the method is possible, but not necessary. We want it to appear in the class
        interface, but we want to be able to tell if it's been overridden so we can
        use an approximate gradient if it has not been.
        """
        if self.__is_overridden(f):
            return f
        return default

    def __is_overridden(self, f: __T1) -> bool:
        """Determine if the given object is a member of this class or not."""
        try:
            this_f = getattr(OptimisationProblem, f.__name__)
        except AttributeError:
            return False
        return f is this_f
