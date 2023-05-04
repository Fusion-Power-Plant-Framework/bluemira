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
import copy
from typing import List

import numpy as np

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation._geometry._typing import (
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.optimisation._geometry.parameterisations import INEQ_CONSTRAINT_REGISTRY
from bluemira.optimisation._typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)
from bluemira.optimisation.error import GeometryOptimisationError

_DEFAULT_ZONE_DISCR = 100  # points
_DEFAULT_ZONE_TOL = 1e-3


def to_objective(
    geom_objective: GeomOptimiserObjective, geom: GeometryParameterisation
) -> ObjectiveCallable:
    """Convert a geometry objective function to a normal objective function."""

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_objective(geom)

    return f


def to_optimiser_callable(
    geom_callable: GeomOptimiserCallable,
    geom: GeometryParameterisation,
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.
    """

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable(geom)

    return f


def to_constraint(
    geom_constraint: GeomConstraintT, geom: GeometryParameterisation
) -> ConstraintT:
    """Convert a geometry constraint to a normal one."""
    constraint: ConstraintT = {
        "f_constraint": to_optimiser_callable(geom_constraint["f_constraint"], geom),
        "tolerance": geom_constraint["tolerance"],
    }
    if "df_constraint" in geom_constraint:
        constraint["df_constraint"] = to_optimiser_callable(
            geom_constraint["df_constraint"], geom
        )

    return constraint


def calculate_signed_distance(parameterisation, n_shape_discr, zone_points):
    """
    Signed distance from the parameterised shape to the keep-out/in zone.
    """
    shape = parameterisation.create_shape()
    s = shape.discretize(ndiscr=n_shape_discr).xz
    return signed_distance_2D_polygon(s.T, zone_points.T).T


def make_keep_out_zone_constraint(
    koz: BluemiraWire,
    n_discr: int = _DEFAULT_ZONE_DISCR,
    tol: float = _DEFAULT_ZONE_TOL,
) -> GeomConstraintT:
    """Make a keep-out zone inequality constraint from a wire."""
    if not koz.is_closed():
        raise GeometryOptimisationError(
            f"Keep-out zone with label '{koz.label}' is not closed."
        )
    koz_points = koz.discretize(n_discr, byedges=True).xz

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return calculate_signed_distance(
            geom, n_shape_discr=n_discr, zone_points=koz_points
        )

    return {"f_constraint": _f_constraint, "tolerance": np.full(n_discr, tol)}


def make_keep_in_zone_constraint(
    kiz: BluemiraWire,
    n_discr: int = _DEFAULT_ZONE_DISCR,
    tol: float = _DEFAULT_ZONE_TOL,
) -> GeomConstraintT:
    """Make a keep-in zone inequality constraint from a wire."""
    if not kiz.is_closed():
        raise GeometryOptimisationError(
            f"Keep-in zone with label '{kiz.label}' is not closed."
        )
    kiz_points = kiz.discretize(n_discr, byedges=True).xz

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return -calculate_signed_distance(
            geom, n_shape_discr=n_discr, zone_points=kiz_points
        )

    return {"f_constraint": _f_constraint, "tolerance": np.full(n_discr, tol)}


def get_shape_ineq_constraint(geom: GeometryParameterisation) -> List[GeomConstraintT]:
    """
    Retrieve the inequality constraints registered for the given parameterisation.

    If no constraints are registered, return an empty list.
    """
    constraints = INEQ_CONSTRAINT_REGISTRY.get(type(geom), [])
    if constraints:
        return copy.deepcopy(constraints)
    return constraints
