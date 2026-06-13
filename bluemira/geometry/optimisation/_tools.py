# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from dataclasses import dataclass

import numpy as np

from bluemira.geometry.error import OutOfBoundsError
from bluemira.geometry.optimisation.typed import (
    GeomClsOptimiserCallable,
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation import ConstraintT, ObjectiveCallable, OptimiserCallable
from bluemira.optimisation.error import GeometryOptimisationError


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
    """Convert a geometry objective function to a normal objective function.

    Returns
    -------
    :
        The objective function converted from a geometry objective function.
    """
    worst = [0.0]  # largest finite objective value seen so far

    def f(x):
        geom.variables.set_values_from_norm(x)
        try:
            value = geom_objective(geom)
        except OutOfBoundsError as exc:
            # No constructable geometry here. Penalise above the worst feasible
            # value seen, growing with how far out of bounds we are, so a
            # gradient-based optimiser is pushed back to the feasible region
            # instead of crashing on the geometry build.
            return (abs(worst[0]) + 1.0) * (1.0 + exc.excess)
        worst[0] = max(worst[0], value)
        return value

    return f


def to_optimiser_callable(
    geom_callable: GeomOptimiserCallable, geom: GeometryParameterisation
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.

    Returns
    -------
    :
        The optimiser function converted from a geometry optimiser function.
    """
    state = {"shape": None, "worst": 0.0}  # last output shape, largest magnitude

    def f(x):
        geom.variables.set_values_from_norm(x)
        try:
            value = geom_callable(geom)
        except OutOfBoundsError as exc:
            # See ``to_objective``: penalise (a constraint reads as strongly
            # violated) rather than crash on the geometry build.
            penalty = (state["worst"] + 1.0) * (1.0 + exc.excess)
            return penalty if not state["shape"] else np.full(state["shape"], penalty)
        arr = np.asarray(value, dtype=float)
        state["shape"] = arr.shape
        if arr.size:
            state["worst"] = max(state["worst"], float(np.max(np.abs(arr))))
        return value

    return f


def to_optimiser_callable_from_cls(
    geom_callable: GeomClsOptimiserCallable, geom: GeometryParameterisation
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.

    Returns
    -------
    :
        The optimiser function converted from a geometry optimiser function.
    """

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable()

    return f


def to_constraint(
    geom_constraint: GeomConstraintT, geom: GeometryParameterisation
) -> ConstraintT:
    """Convert a geometry constraint to a normal one.

    Returns
    -------
    :
        The consatraint constructed from the geometry constraint.
    """
    constraint: ConstraintT = {
        "f_constraint": to_optimiser_callable(geom_constraint["f_constraint"], geom),
        "df_constraint": None,
        "tolerance": geom_constraint["tolerance"],
    }
    if name := geom_constraint.get("name", None):
        constraint["name"] = name

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

    Returns
    -------
    :
        Signed distance from the parameterised shape to the keep-out/in zone.
    """
    shape = parameterisation.create_shape()
    # Note that we do not discretise by edges here, as the number of
    # points must remain constant so the size of constraint vectors
    # remain constant. Sample densely, then resample to the requested count
    # from a backend-independent anchor (see _canonical_loop_points).
    dense = shape.discretise(max(8 * n_shape_discr, 200), byedges=False).xz
    s = _canonical_loop_points(dense, n_shape_discr)
    return signed_distance_2D_polygon(s.T, zone_points.T).T


def _canonical_loop_points(loop: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed loop to ``n`` points in a backend-independent way.

    ``discretise`` starts at the wire's first edge and follows the wire's
    stored direction, both of which depend on the backend-specific edge
    ordering. For an otherwise identical loop this only phase-shifts (and can
    reverse) the samples, yet it scrambles the per-element keep-out-zone
    constraint vector between backends and sends the optimiser into a different
    basin. Pinning the winding, anchoring at the outer-midplane point and
    resampling at uniform arc length makes the constraint deterministic and
    smooth in the design variables.

    Parameters
    ----------
    loop:
        ``(2, M)`` densely sampled closed loop in the xz plane.
    n:
        Number of points to return.

    Returns
    -------
    :
        ``(2, n)`` points at uniform arc length, measured from the anchor.
    """
    pts = loop.T
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    # Consistent winding (counter-clockwise) via the shoelace signed area.
    x, z = pts[:, 0], pts[:, 1]
    if np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z) < 0:
        pts = pts[::-1]
    # Cumulative arc length around the closed loop.
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1)),
    ])
    length = cum[-1]
    # Anchor where the loop crosses the outboard horizontal ray from its
    # centroid (the outer-midplane point). Unlike an axis-aligned or radial
    # extremum, this crossing is unique, sits on the smooth outer wall and
    # moves continuously as the shape changes, so it never jumps between
    # optimiser iterations or between the backends' phase-shifted samplings.
    # That matters because a resample repositions every point relative to the
    # anchor, so an anchor that jumps scrambles the whole constraint vector.
    c = pts.mean(axis=0)
    dz = pts[:, 1] - c[1]
    dz_next = np.roll(dz, -1)
    outboard = (pts[:, 0] > c[0]) & (np.roll(pts[:, 0], -1) > c[0])
    crossings = np.flatnonzero((dz <= 0) & (dz_next > 0) & outboard)
    j = (
        int(crossings[np.argmax(pts[crossings, 0])])
        if len(crossings)
        else int(np.argmax(pts[:, 0]))
    )
    t = dz[j] / (dz[j] - dz_next[j]) if dz[j] != dz_next[j] else 0.0
    s0 = (cum[j] + t * (cum[j + 1] - cum[j])) % length
    # Resample at uniform arc length from the anchor.
    targets = (s0 + np.linspace(0.0, length, n, endpoint=False)) % length
    return np.vstack([
        np.interp(targets, cum, closed[:, 0]),
        np.interp(targets, cum, closed[:, 1]),
    ])


def make_keep_out_zone_constraint(koz: KeepOutZone) -> GeomConstraintT:
    """Make a keep-out zone inequality constraint from a wire.

    Returns
    -------
    :
        The inequality constraint for the keep-out zone.

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
            geom, n_shape_discr=shape_n_discr, zone_points=koz_points
        )

    return {
        "name": "KOZ",
        "f_constraint": _f_constraint,
        "tolerance": np.full(shape_n_discr, koz.tol),
    }


def get_shape_ineq_constraint(geom: GeometryParameterisation) -> list[ConstraintT]:
    """
    Retrieve the inequality constraints registered for the given parameterisation.

    If no constraints are registered, return an empty list.

    Returns
    -------
    :
        The inequality constraints registered for the given parameterisation.
    """
    if geom.n_ineq_constraints < 1:
        return []
    if df_constraint := getattr(geom, "df_ineq_constraint", None):
        df_constraint = to_optimiser_callable_from_cls(df_constraint, geom)
    return [
        {
            "name": geom.name,
            "f_constraint": to_optimiser_callable_from_cls(geom.f_ineq_constraint, geom),
            "df_constraint": df_constraint,
            "tolerance": geom.tolerance,
        }
    ]
