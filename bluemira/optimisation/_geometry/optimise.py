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
from dataclasses import asdict, dataclass
from itertools import repeat
from typing import Any, Generic, Iterable, Mapping, Optional, Tuple, TypeVar, Union

import numpy as np

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._geometry import _tools
from bluemira.optimisation._geometry.typing import (
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.optimisation._optimise import optimise
from bluemira.optimisation._optimiser import OptimiserResult

_GeomT = TypeVar("_GeomT", bound=GeometryParameterisation)


@dataclass
class GeomOptimiserResult(OptimiserResult, Generic[_GeomT]):
    """Container for the result of a geometry optimisation."""

    geom: _GeomT


def optimise_geometry(
    geom: _GeomT,
    f_objective: GeomOptimiserObjective,
    df_objective: Optional[GeomOptimiserCallable] = None,
    *,
    keep_out_zones: Iterable[BluemiraWire] = (),
    keep_in_zones: Iterable[BluemiraWire] = (),
    algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    eq_constraints: Iterable[GeomConstraintT] = (),
    ineq_constraints: Iterable[GeomConstraintT] = (),
    keep_history: bool = False,
    koz_discretisation: Union[int, Iterable[int]] = 100,
    kiz_discretisation: Union[int, Iterable[int]] = 100,
) -> GeomOptimiserResult[_GeomT]:
    r"""
    Minimise the given objective function for a geometry parameterisation.

    Parameters
    ----------
    geom:
        The geometry to optimise the parameters of. The existing
        parameterisation is used as the initial guess in the
        optimisation.
    f_objective:
        The objective function to minimise. Must take as an argument a
        `GeometryParameterisation` and return a float.
    df_objective:
        The derivative of the objective function, by default None. If
        not given, an approximation of the derivative is made using
        the 'central differences' method.
        This argument is ignored if a non-gradient based algorithm is
        used.
    keep_out_zones:
        An iterable of closed wires, defining areas the geometry must
        not intersect.
    keep_in_zones:
        An iterable list of closed wires, defining areas the geometry
        must wholly lie within.
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
    bounds:
        The upper and lower bounds for the optimisation parameters.
        The first array being the lower bounds, the second the upper.
    eq_constraints:
        The equality constraints for the optimiser.
        A dict with keys:

            * f_constraint: the constraint function.
            * tolerance: the tolerances in each constraint dimension.
            * df_constraint (optional): the derivative of the constraint
              function. If not given, a numerical approximation of the
              gradient is made (if a gradient is required).

        A constraint is a vector-valued, non-linear, inequality
        constraint of the form $f_{c}(x) \le 0$.

        The constraint function should have the form ``f(g) -> y``,
        where:

            * `g` is a geometry parameterisation.
            * `y` is a numpy array containing the values of the
              constraint at the current parameterisation of ``g``.
              It must have size $m$, where $m$ is the dimensionality of
              the constraint.

        The tolerance array must have the same dimensionality as the
        constraint.

        The gradient function should have the same form as the
        constraint function, however its output should have size
        $n \cross m$ where $m$ is the dimensionality of the constraint.

        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    ineq_constraints:
        The geometric inequality constraints for the optimiser.
        This argument has the same form as the `eq_constraint` argument,
        but each constraint is of the form $f_{c}(x) \le 0$.

        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    keep_history:
        Whether or not to record the history of the optimisation
        parameters at each iteration. Note that this can significantly
        impact the performance of the optimisation.
        (default: False)
    koz_discretisation:
        The number of points to discretise the keep-out zone(s) over.
        If this is an int, all keep-out zones will be discretised with
        the same number of points. If this is an iterable, each i-th
        keep-out zone is discretised using value in the i-th item.
        The iterable should have the same number of items as
        ``keep_out_zones``.
    kiz_discretisation:
        The number of points to discretise the keep-in zone(s) over.
        If this is an int, all keep-in zones will be discretised with
        the same number of points. If this is an iterable, each i-th
        keep-in zone is discretised using value in the i-th item.
        The iterable should have the same number of items as
        ``keep_in_zones``.

    Returns
    -------
    The result of the optimisation.
    """
    dimensions = geom.variables.n_free_variables
    f_obj = _tools.to_objective(f_objective, geom)
    if df_objective is not None:
        df_obj = _tools.to_optimiser_callable(df_objective, geom)
    else:
        df_obj = None
    ineq_constraints_list = _tools.get_shape_ineq_constraint(geom)
    for constraint in ineq_constraints:
        ineq_constraints_list.append(constraint)
    for koz, discr in zip_with_scalar(keep_out_zones, koz_discretisation):
        ineq_constraints_list.append(
            _tools.make_keep_out_zone_constraint(koz, n_discr=discr)
        )
    for kiz, discr in zip_with_scalar(keep_in_zones, kiz_discretisation):
        ineq_constraints_list.append(
            _tools.make_keep_in_zone_constraint(kiz, n_discr=discr)
        )

    result = optimise(
        f_obj,
        dimensions=dimensions,
        x0=geom.variables.get_normalised_values(),
        df_objective=df_obj,
        algorithm=algorithm,
        opt_conditions=opt_conditions,
        opt_parameters=opt_parameters,
        bounds=(np.zeros(dimensions), np.ones(dimensions)),
        eq_constraints=[_tools.to_constraint(c, geom) for c in eq_constraints],
        ineq_constraints=[_tools.to_constraint(c, geom) for c in ineq_constraints_list],
        keep_history=keep_history,
    )
    # Make sure we update the geometry with the result, the last
    # geometry update may not have been with the optimum result
    geom.variables.set_values_from_norm(result.x)
    return GeomOptimiserResult(**asdict(result), geom=geom)


_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def zip_with_scalar(
    it1: Iterable[_T1], it2: Union[Iterable[_T2], _T2]
) -> Iterable[Tuple[_T1, _T2]]:
    """Return an iterator that zips an iterable with another, or with a scalar."""
    if not isinstance(it2, Iterable):
        it2 = repeat(it2)
    return zip(it1, it2)
