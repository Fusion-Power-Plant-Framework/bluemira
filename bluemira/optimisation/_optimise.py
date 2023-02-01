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
"""Definition of the generic `optimise` function."""

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._nlopt import NloptOptimiser
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation._typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


def optimise(
    f_objective: ObjectiveCallable,
    dimensions: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    df_objective: Optional[OptimiserCallable] = None,
    algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eq_constraints: Iterable[ConstraintT] = (),
    ineq_constraints: Iterable[ConstraintT] = (),
    keep_history: bool = False,
) -> OptimiserResult:
    r"""
    Find the parameters that minimise the given objective function.

    Parameters
    ----------
    f_objective: Callable[[Arg(np.ndarray, 'x')], np.ndarray]
        The objective function to minimise.
    dimensions: Optional[int]
        The dimensionality of the problem. This or `x0` must be given.
    x0: Optional[np.ndarray]
        The initial guess for the optimisation parameters. This or
        `dimensions` must be given, if both are given, `x0.size` must be
        equal to `dimensions`.
    df_objective: Optional[Callable[[Arg(np.ndarray, 'x')], np.ndarray]]
        The derivative of the objective function.
    algorithm: Optional[Union[Algorithm, str]]
        The optimisation algorithm to use. See enum
        :obj:`optimisation.Algorithm` for supported algorithms.
        (default: "SLSQP")
    opt_conditions: Optional[Mapping[str, Union[int, float]]]
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
    opt_parameters: Optional[Mapping[str, Any]]
        The algorithm-specific optimisation parameters.
    bounds: Tuple[np.ndarray, np.ndarray]
        The upper and lower bounds for the optimisation parameters.
        The first array being the lower bounds, the second the upper.
    eq_constraints: Iterable[ConstraintT]
        The equality constraints for the optimiser.
        A dict with keys:

            * f_constraint: the constraint function.
            * tolerance: the tolerances in each constraint dimension.
            * df_constraint (optional): the derivative of the constraint
              function. If not given, a numerical approximation of the
              gradient is made (if a gradient is required).

        A constraint is a vector-valued, non-linear, inequality
        constraint of the form $f_{c}(x) \le 0$.

        The constraint function should have the form `f(x) -> y`,
        where:

            * `x` is a numpy array of the optimisation parameters.
            * `y` is a numpy array containing the values of the
              constraint at `x`, with size $m$, where $m$ is the
              dimensionality of the constraint.

        The tolerance array must have the same dimensionality as the
        constraint.

        The gradient function should have the same form as the
        constraint function, however its output should have size
        $n \cross m$ where $m$ is the dimensionality of the constraint.

        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    ineq_constraints: Iterable[ConstraintT]
        The inequality constraints for the optimiser.
        This argument has the same form as the `eq_constraint` argument,
        but each constraint is in the form $f_{c}(x) \le 0$.

        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    """
    if dimensions is None:
        if x0 is not None:
            dimensions = x0.size
        else:
            raise ValueError("Must give argument 'dimension' or 'x0'.")
    else:
        if x0 is not None and x0.size != dimensions:
            raise ValueError("Size of 'x0' and 'dimensions' must agree.")

    if opt_conditions is None:
        opt_conditions = {"max_eval": 2000}

    optimiser = _make_optimiser(
        f_objective,
        dimensions,
        df_objective,
        algorithm,
        opt_conditions,
        opt_parameters,
        bounds,
        eq_constraints,
        ineq_constraints,
        keep_history,
    )
    return optimiser.optimise(x0)


def _make_optimiser(
    f_objective: ObjectiveCallable,
    dimensions: int,
    df_objective: Optional[OptimiserCallable] = None,
    algorithm: Union[Algorithm, str] = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eq_constraints: Iterable[ConstraintT] = (),
    ineq_constraints: Iterable[ConstraintT] = (),
    keep_history: bool = False,
) -> Optimiser:
    """Make a new optimiser object."""
    opt = NloptOptimiser(
        algorithm,
        dimensions,
        f_objective=f_objective,
        df_objective=df_objective,
        opt_conditions=opt_conditions,
        opt_parameters=opt_parameters,
        keep_history=keep_history,
    )
    for constraint in eq_constraints:
        opt.add_eq_constraint(**constraint)
    for constraint in ineq_constraints:
        opt.add_ineq_constraint(**constraint)
    if bounds:
        opt.set_lower_bounds(bounds[0])
        opt.set_upper_bounds(bounds[1])
    return opt
