# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from dataclasses import dataclass

import numpy as np

from bluemira.geometry.optimisation.typing import (
    GeomClsOptimiserCallable,
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation.error import GeometryOptimisationError
from bluemira.optimisation.typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


@dataclass
class KeepOutZone:
    """Definition of a keep-out zone for a geometry optimisation."""

    wire: BluemiraWire
    """Closed wire defining the keep-out zone."""
    byedges: bool = True
    """Whether to discretise the keep-out zone by edges or not."""
    dl: float | None = None
    """
    The discretisation length for the keep-out zone.

    This overrides ``n_discr`` if given.
    """
    n_discr: int = 100
    """The number of points to discretise the keep-out zone into."""
    shape_n_discr: int = 100
    """The number of points to discretise the geometry being optimised into."""
    tol: float = 1e-8
    """The tolerance for the keep-out zone constraint."""


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


def to_optimiser_callable_from_cls(
    geom_callable: GeomClsOptimiserCallable,
    geom: GeometryParameterisation,
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.
    """

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable()

    return f


def to_constraint(
    geom_constraint: GeomConstraintT, geom: GeometryParameterisation
) -> ConstraintT:
    """Convert a geometry constraint to a normal one."""
    constraint: ConstraintT = {
        "f_constraint": to_optimiser_callable(geom_constraint["f_constraint"], geom),
        "df_constraint": None,
        "tolerance": geom_constraint["tolerance"],
    }
    if df_constraint := geom_constraint.get("df_constraint", None):
        constraint["df_constraint"] = to_optimiser_callable(df_constraint, geom)
    return constraint


def calculate_signed_distance(
    parameterisation: GeometryParameterisation,
    n_shape_discr: int,
    zone_points: np.ndarray,
) -> np.ndarray:
    """
    Signed distance from the parameterised shape to the keep-out/in zone.
    """
    shape = parameterisation.create_shape()
    # Note that we do not discretise by edges here, as the number of
    # points must remain constant so the size of constraint vectors
    # remain constant.
    s = shape.discretise(n_shape_discr, byedges=False).xz
    return signed_distance_2D_polygon(s.T, zone_points.T).T


def make_keep_out_zone_constraint(koz: KeepOutZone) -> GeomConstraintT:
    """Make a keep-out zone inequality constraint from a wire.

    Raises
    ------
    GeometryOptimisationError
        Koz wire is not closed
    """
    if not koz.wire.is_closed():
        raise GeometryOptimisationError(
            f"Keep-out zone with label '{koz.wire.label}' is not closed."
        )
    koz_points = koz.wire.discretise(koz.n_discr, byedges=koz.byedges, dl=koz.dl).xz
    # Note that we do not allow discretisation using 'dl' or 'byedges'
    # for the shape being optimised. The size of the constraint cannot
    # change within an optimisation loop (NLOpt will error) and these
    # options do not guarantee a constant number of discretised points.
    shape_n_discr = koz.shape_n_discr

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return calculate_signed_distance(
            geom,
            n_shape_discr=shape_n_discr,
            zone_points=koz_points,
        )

    return {"f_constraint": _f_constraint, "tolerance": np.full(shape_n_discr, koz.tol)}


def get_shape_ineq_constraint(geom: GeometryParameterisation) -> list[ConstraintT]:
    """
    Retrieve the inequality constraints registered for the given parameterisation.

    If no constraints are registered, return an empty list.
    """
    if geom.n_ineq_constraints < 1:
        return []
    if df_constraint := getattr(geom, "df_ineq_constraint", None):
        df_constraint = to_optimiser_callable_from_cls(df_constraint, geom)
    return [
        {
            "f_constraint": to_optimiser_callable_from_cls(geom.f_ineq_constraint, geom),
            "df_constraint": df_constraint,
            "tolerance": geom.tolerance,
        }
    ]
