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
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._geometry._typing import (
    GeomConstraintT,
    GeomOptimiserCallable,
)
from bluemira.optimisation._geometry.parameterisations import INEQ_CONSTRAINT_REGISTRY
from bluemira.optimisation._optimise import optimise
from bluemira.optimisation._optimiser import OptimiserResult
from bluemira.optimisation._typing import ConstraintT, OptimiserCallable
from bluemira.optimisation.error import GeometryOptimisationError


@dataclass
class GeomOptimiserResult(OptimiserResult):
    """Container for the result of a geometry optimisation."""

    geom: GeometryParameterisation


def optimise_geometry(
    geom: GeometryParameterisation,
    f_objective: GeomOptimiserCallable,  # TODO(hsaunders1904): typing is wrong here
    df_objective: Optional[GeomOptimiserCallable] = None,
    keep_out_zones: Iterable[BluemiraWire] = (),
    keep_in_zones: Iterable[BluemiraWire] = (),
    algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
    opt_conditions: Optional[Dict] = None,
    opt_parameters: Optional[Dict] = None,
    eq_constraints: Iterable[GeomConstraintT] = (),
    ineq_constraints: Iterable[GeomConstraintT] = (),
    keep_history: bool = False,
) -> GeomOptimiserResult:
    r"""
    Minimise the given objective function for a geometry parameterisation.

    Parameters
    ----------
    geom: GeometryParameterisation
        The geometry to optimise the parameters of.
    f_objective: GeomOptimiserCallable
        The objective function to minimise. Must take as an argument a
        `GeometryParameterisation`and return a numpy array.
            TODO(hsaunders1904): should this not return a scalar?
    df_objective: Optional[GeomOptimiserCallable], optional
        The derivative of the objective function, by default None. If
        not given, an approximation of the derivative is made using
        the 'central differences' method.
        This argument is ignored if a non-gradient based algorithm is
        used.
    keep_out_zones: Iterable[BluemiraWire], optional
        An iterable of closed wires, defining areas the geometry must
        not intersect.
    keep_in_zones: Iterable[BluemiraWire], optional
        An iterable list of closed wires, defining areas the geometry
        must wholly lie within.
    algorithm : Union[Algorithm, str], optional
        The optimisation algorithm to use, by default `Algorithm.SLSQP`.
    opt_conditions: Optional[Dict]
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
    opt_parameters: Optional[Dict]
        The algorithm-specific optimisation parameters.
    bounds: Tuple[np.ndarray, np.ndarray]
        The upper and lower bounds for the optimisation parameters.
        The first array being the lower bounds, the second the upper.
    eq_constraints: Iterable[GeomConstraintT]
        The equality constraints for the optimiser.
        A dict with keys:

            * f_constraint: the constraint function.
            * tolerance: the tolerances in each constraint dimension.
            * df_constraint (optional): the derivative of the constraint
              function. If not given, a numerical approximation of the
              gradient is made (if a gradient is required).

        A constraint is a vector-valued, non-linear, inequality
        constraint of the form $f_{c}(x) \le 0$.

        The constraint function should have the form `f(g) -> y`,
        where:

            * `g` is a geometry parameterisation.
            * `y` is a numpy array containing the values of the
              constraint at the current parameterisation of `g`.
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

    ineq_constraints: Iterable[GeomConstraintT]
        The geometric inequality constraints for the optimiser.
        This argument has the same form as the `eq_constraint` argument,
        but each constraint is in the form $f_{c}(x) \le 0$.

        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    keep_history: bool
        Whether or not to record the history of the optimisation
        parameters at each iteration. Note that this can significantly
        impact the performance of the optimisation.
        (default: False)

    Returns
    -------
    GeomOptimiserResult
        The result of the optimisation.
    """
    dimensions = geom.variables.n_free_variables
    f_obj = _to_optimiser_callable(f_objective, geom)
    if df_objective is not None:
        df_obj = _to_optimiser_callable(df_objective, geom)
    else:
        df_obj = None
    ineq_constraints_list = _get_shape_ineq_constraint(geom)
    for constraint in ineq_constraints:
        ineq_constraints_list.append(constraint)
    for koz in keep_out_zones:
        ineq_constraints_list.append(_make_keep_out_zone_constraint(koz))
    for kiz in keep_in_zones:
        ineq_constraints_list.append(_make_keep_in_zone_constraint(kiz))

    result = optimise(
        f_obj,
        dimensions=dimensions,
        x0=geom.variables.get_normalised_values(),
        df_objective=df_obj,
        algorithm=algorithm,
        opt_conditions=opt_conditions,
        opt_parameters=opt_parameters,
        bounds=(np.zeros(dimensions), np.ones(dimensions)),
        eq_constraints=[_to_constraint(c, geom) for c in eq_constraints],
        ineq_constraints=[_to_constraint(c, geom) for c in ineq_constraints_list],
        keep_history=keep_history,
    )
    return GeomOptimiserResult(**asdict(result), geom=geom)


def _to_optimiser_callable(
    geom_callable: GeomOptimiserCallable, geom: GeometryParameterisation
) -> OptimiserCallable:
    """Convert a geometry objective function to a normal objective function."""

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable(geom)

    return f


def calculate_signed_distance(parameterisation, n_shape_discr, koz_points):
    """
    Signed distance from the parameterised shape to the keep-out zone.
    """
    shape = parameterisation.create_shape()
    s = shape.discretize(ndiscr=n_shape_discr).xz
    return signed_distance_2D_polygon(s.T, koz_points.T).T


def _make_keep_out_zone_constraint(koz: BluemiraWire) -> GeomConstraintT:
    """Make a keep-out zone inequality constraint from a wire."""
    if not koz.is_closed():
        raise GeometryOptimisationError(
            f"Keep-out zone with label '{koz.label}' is not closed."
        )
    koz_points = koz.discretize(100, byedges=True).xz

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return calculate_signed_distance(geom, n_shape_discr=100, koz_points=koz_points)

    return {"f_constraint": _f_constraint, "tolerance": np.full(100, 1e-3)}


def _make_keep_in_zone_constraint(koz: BluemiraWire) -> GeomConstraintT:
    """Make a keep-in zone inequality constraint from a wire."""
    if not koz.is_closed():
        raise GeometryOptimisationError(
            f"Keep-in zone with label '{koz.label}' is not closed."
        )
    koz_points = koz.discretize(100, byedges=True).xz

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return -calculate_signed_distance(geom, n_shape_discr=100, koz_points=koz_points)

    return {"f_constraint": _f_constraint, "tolerance": np.full(100, 1e-3)}


def _get_shape_ineq_constraint(geom: GeometryParameterisation) -> List[GeomConstraintT]:
    return INEQ_CONSTRAINT_REGISTRY.get(type(geom), [])


def _to_constraint(
    geom_constraint: GeomConstraintT, geom: GeometryParameterisation
) -> ConstraintT:
    """Convert a geometry constraint to a normal one."""

    def _f_constraint(x: np.ndarray) -> np.ndarray:
        geom.variables.set_values_from_norm(x)
        return geom_constraint["f_constraint"](geom)

    constraint = {
        "f_constraint": _f_constraint,
        "tolerance": geom_constraint["tolerance"],
    }

    if "df_constraint" in geom_constraint:

        def _df_constraint(x: np.ndarray) -> np.ndarray:
            geom.variables.set_values_from_norm(x)
            return geom_constraint["df_constraint"](geom)

        constraint["df_constraint"] = _df_constraint

    return constraint
