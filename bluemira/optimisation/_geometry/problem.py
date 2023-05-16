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
"""Interface for defining a geometry-based optimisation problem."""

import abc
from typing import Any, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation import Algorithm, optimise_geometry
from bluemira.optimisation._geometry.optimise import GeomOptimiserResult
from bluemira.optimisation._geometry.typing import GeomConstraintT
from bluemira.optimisation.problem import OptimisationProblemBase

_GeomT = TypeVar("_GeomT", bound=GeometryParameterisation)


class GeomOptimisationProblem(abc.ABC, OptimisationProblemBase):
    """
    Interface for a geometry optimisation problem.

    This is an alternative to running a geometry optimisation using the
    :func:`.optimise_geometry` function.
    """

    @abc.abstractmethod
    def objective(self, geom: _GeomT) -> float:
        """The objective function to minimise."""

    def df_objective(self, geom: _GeomT) -> np.ndarray:
        """The derivative of the objective function."""
        raise NotImplementedError

    def eq_constraints(self) -> List[GeomConstraintT]:
        """List of equality constraints for the optimisation."""
        return []

    def ineq_constraints(self) -> List[GeomConstraintT]:
        """List of inequality constraints for the optimisation."""
        return []

    def keep_out_zones(self) -> List[Union[BluemiraWire, Tuple[BluemiraWire, int]]]:
        """List of geometric keep-out zones."""
        return []

    def keep_in_zones(self) -> List[Union[BluemiraWire, Tuple[BluemiraWire, int]]]:
        """List of geometric keep-in zones."""
        return []

    def optimise(
        self,
        geom: _GeomT,
        *,
        algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
    ) -> GeomOptimiserResult[_GeomT]:
        """
        Run the geometry optimisation.

        Parameters
        ----------
        geom:
            The geometry to optimise the parameters of. The existing
            parameterisation is used as the initial guess in the
            optimisation.
        algorithm:
            The optimisation algorithm to use, by default ``Algorithm.SLSQP``.
        opt_conditions:
            The stopping conditions for the optimiser. Supported conditions
            are:

                * ftol_abs: float
                * ftol_rel: float
                * xtol_abs: float
                * xtol_rel: float
                * max_eval: int
                * max_time: float
                * stop_val: float

            (default: {"max_eval": 2000})
        opt_parameters:
            The algorithm-specific optimisation parameters.
        keep_history:
            Whether or not to record the history of the optimisation
            parameters at each iteration. Note that this can significantly
            impact the performance of the optimisation.
            (default: False)

        Returns
        -------
        The result of the optimisation.
        """
        koz, koz_discr = self.__split_zones_and_discr(self.keep_out_zones())
        kiz, kiz_discr = self.__split_zones_and_discr(self.keep_in_zones())
        df_objective = self._overridden_or_default(
            self.df_objective, GeomOptimisationProblem, None
        )
        return optimise_geometry(
            geom,
            f_objective=self.objective,
            df_objective=df_objective,
            keep_out_zones=koz,
            keep_in_zones=kiz,
            algorithm=algorithm,
            opt_conditions=opt_conditions,
            opt_parameters=opt_parameters,
            eq_constraints=self.eq_constraints(),
            ineq_constraints=self.ineq_constraints(),
            keep_history=keep_history,
            koz_discretisation=koz_discr,
            kiz_discretisation=kiz_discr,
        )

    def __split_zones_and_discr(
        self, zones_and_discr: Iterable[Union[BluemiraWire, Tuple[BluemiraWire, int]]]
    ) -> Tuple[List[BluemiraWire], List[int]]:
        """
        Split the zones and their discretizations.

        Set the discretization to 100 if one is not given for a zone.
        """
        zones = []
        zone_discr = []
        for zone in zones_and_discr:
            if isinstance(zone, BluemiraWire):
                zones.append(zone)
                zone_discr.append(100)
            else:
                zones.append(zone[0])
                zone_discr.append(zone[1])
        return zones, zone_discr
