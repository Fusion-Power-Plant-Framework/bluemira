# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CadQuery backend for bluemira.

Implements the same public interface as _freecadapi.py using CadQuery's
free-function / direct Shape API (no Workplane state). ``show_cad``
delegates to polyscope; placements go through the :class:`_CQPlacement`
adapter (a drop-in for FreeCAD's ``Base.Placement``).
"""

from __future__ import annotations

import contextlib
import math
from itertools import pairwise
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import (
    BRepAdaptor_CompCurve,
    BRepAdaptor_Curve,
    BRepAdaptor_Surface,
)
from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_BuilderAlgo,
    BRepAlgoAPI_Section,
    BRepAlgoAPI_Splitter,
)
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
)
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet2d
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from OCP.GProp import GProp_GProps
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Line,
)
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Surface
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Wire
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import (
    TopAbs_FACE,
    TopAbs_IN,
    TopAbs_REVERSED,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_HSequenceOfShape, TopTools_ListOfShape
from OCP.TopoDS import TopoDS
from OCP.gp import (
    gp_Ax1,
    gp_Ax2,
    gp_Dir,
    gp_Pln,
    gp_Pnt,
    gp_Pnt2d,
    gp_Trsf,
    gp_Vec,
)

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes._cadqueryapi._aliases import (
    _ANGLE_PARALLEL_TOL,
    _GEOM_NEAR_ZERO_TOL,
    _OCC_DEFAULT_TOL,
    _POINT_COINCIDENCE_TOL,
    apiCompound,
    apiEdge,
    apiShape,
    apiShell,
    apiSolid,
    apiWire,
)
from bluemira.codes.error import FreeCADError, InvalidCADInputsError
from bluemira.geometry.error import GeometryError

if TYPE_CHECKING:
    from collections.abc import Iterable


class _apiFaceMeta(type):
    """Metaclass so ``isinstance(x, apiFace)`` works for plain cq.Face objects."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, cq.Face)


def _face_from_wires_tolerant(outer: cq.Wire, inner: list) -> cq.Face:
    """Build a face from *outer* (+ optional *inner* hole wires).

    Tries three constructors in sequence, from strictest to most forgiving:

    1. ``BRepBuilderAPI_MakeFace(outer, OnlyPlane=False)`` — OCCT's native
       ``BRepLib_FindSurface`` over all surface types (plane, cylinder,
       cone, sphere, torus, bspline). Succeeds when the wire's edges carry
       pcurves on a common non-planar surface, e.g. side faces of a swept
       solid. Equivalent to FreeCAD's ``Part.Face(wire)``.
    2. ``cq.Face.makeFromWires`` — planar-only with ``OnlyPlane=True`` and
       strict confusion tolerance; fast path for clean planar wires.
    3. SVD-fitted plane + ``BRepBuilderAPI_MakeFace(pln, wire)`` — tolerant
       planar fallback for wires that are planar by construction but
       carry out-of-plane floating-point noise above OCCT's default
       confusion threshold (~1e-7).

    The non-planar path (1) has to come first, because ``cq.Face.makeFromWires``
    succeeds on curved-but-near-planar wires by projecting them onto a flat
    plane — silently destroying the surface curvature and giving wrong
    ``.Area`` / ``.Volume`` for any reconstructed solid. The TF coil
    insulation-shell rebuild hit this: 2 of 8 faces (the large swept sides)
    would fall into the SVD-fit fallback and produce a 68 m² deficit in the
    reconstructed shell area.
    """
    if not inner:
        try:
            builder = BRepBuilderAPI_MakeFace(outer.wrapped, False)
            builder.Build()
            if builder.IsDone():
                return cq.Face(builder.Face())
        except Exception:  # noqa: BLE001, S110
            pass  # fall through to planar paths below
    else:  # we have inner wire(s), check if they are valid
        o_bounds = np.array(bounding_box(outer))

        for hole_wire in inner:
            h_bounds = np.array(bounding_box(hole_wire))

            is_outside_mins = np.any(h_bounds[:3] < o_bounds[:3] - _OCC_DEFAULT_TOL)
            is_outside_maxs = np.any(h_bounds[3:] > o_bounds[3:] + _OCC_DEFAULT_TOL)

            if is_outside_mins or is_outside_maxs:
                raise GeometryError(
                    "Topological error: inner wire is located partially or "
                    "completely outside the bounds of the outer wire."
                )

    try:
        return cq.Face.makeFromWires(outer, inner)
    except ValueError as exc:
        if "not planar" not in str(exc):
            raise

    pts: list[list[float]] = []
    for w in [outer, *inner]:
        for v in w.Vertices():
            c = v.Center()
            pts.append([c.x, c.y, c.z])
    arr = np.asarray(pts, dtype=float)
    centroid = arr.mean(axis=0)
    _, _, vh = np.linalg.svd(arr - centroid, full_matrices=False)
    normal = vh[2]
    pln = gp_Pln(gp_Pnt(*centroid), gp_Dir(*normal))

    builder = BRepBuilderAPI_MakeFace(pln, outer.wrapped, True)
    for w in inner:
        builder.Add(w.wrapped)
    builder.Build()
    if not builder.IsDone():
        raise GeometryError(
            f"Tolerant planar face build failed: {builder.Error()}"
        ) from None
    face = cq.Face(builder.Face())
    fix_shape(face)
    return face


class apiFace(metaclass=_apiFaceMeta):
    """Drop-in for ``cq.Face``.

    Calling ``apiFace(wire)`` with a ``cq.Wire`` uses ``makeFromWires``
    instead of the raw OCC constructor that FreeCAD's ``Part.Face(wire)`` used.
    On numerically non-planar wires it falls back to an SVD-fitted plane +
    ``BRepBuilderAPI_MakeFace`` path so slight construction noise doesn't
    reject faces that are planar by design.
    """

    def __new__(cls, obj=None):
        if isinstance(obj, cq.Wire):
            return _face_from_wires_tolerant(obj, [])
        if isinstance(obj, (list, tuple)):
            wires = list(obj)
            return _face_from_wires_tolerant(wires[0], wires[1:])
        if obj is None:
            return cq.Face.__new__(cq.Face)
        return cq.Face(obj)


EPS = 1e-8

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_cq_vectors(points: list | np.ndarray) -> list[cq.Vector]:
    return [cq.Vector(float(p[0]), float(p[1]), float(p[2])) for p in points]


def _vector_to_numpy(v: cq.Vector) -> np.ndarray:
    arr = np.array([v.x, v.y, v.z])
    arr[np.abs(arr) < _GEOM_NEAR_ZERO_TOL] = 0.0
    return arr


# ---------------------------------------------------------------------------
# Geometry creation
# ---------------------------------------------------------------------------


def make_polygon(points: list | np.ndarray) -> apiWire:
    """Make a polygon wire from points, dropping consecutive duplicates.

    OCC's ``BRepBuilderAPI_MakePolygon`` rejects vertices closer than
    ``Precision::Confusion`` (~1e-7); FreeCAD's ``Part.makePolygon``
    silently collapses them. Match FreeCAD's contract by removing the
    duplicates (a 4-point polygon with one repeated vertex collapses to a
    triangle, not a self-intersecting shape). If the whole input collapses
    to a single point, emit a degenerate 2-vertex wire instead of erroring.
    """
    vecs = _to_cq_vectors(points)
    deduped: list[cq.Vector] = []
    for v in vecs:
        if deduped and deduped[-1].sub(v).Length < _OCC_DEFAULT_TOL:
            continue
        deduped.append(v)
    if len(deduped) < 2 and len(vecs) >= 2:  # noqa: PLR2004
        deduped = [
            vecs[0],
            cq.Vector(vecs[0].x + _OCC_DEFAULT_TOL, vecs[0].y, vecs[0].z),
        ]
    return cq.Wire.makePolygon(deduped)


def interpolate_bspline(
    points: list | np.ndarray,
    *,
    closed: bool = False,
    start_tangent: Iterable | None = None,
    end_tangent: Iterable | None = None,
) -> apiWire:
    """
    Make an interpolating B-spline wire through the given points.

    Parameters
    ----------
    points:
        Sequence of 3-D points.
    closed:
        Whether to close the spline.
    start_tangent:
        Tangent direction at the first point (requires end_tangent too).
    end_tangent:
        Tangent direction at the last point (requires start_tangent too).
    """
    pnts = _to_cq_vectors(points)

    if len(pnts) < 2:  # noqa: PLR2004
        raise InvalidCADInputsError("interpolate_bspline: not enough points")

    if np.allclose(
        [pnts[0].x, pnts[0].y, pnts[0].z],
        [pnts[-1].x, pnts[-1].y, pnts[-1].z],
        rtol=EPS,
        atol=0,
    ):
        if len(pnts) > 2:  # noqa: PLR2004
            if not closed:
                bluemira_warn("interpolate_bspline: equal endpoints forced Closed")
            closed = True
            pnts = pnts[:-1]
        else:
            raise InvalidCADInputsError(
                "interpolate_bspline: Invalid pointslist (len == 2 and first == last)"
            )

    tangents = None
    if start_tangent is not None and end_tangent is not None:
        # Pass exactly two tangents → makeSpline uses the start/end-only path
        tangents = [cq.Vector(*start_tangent), cq.Vector(*end_tangent)]
    elif start_tangent is not None or end_tangent is not None:
        bluemira_warn(
            "You must set both start and end tangencies or neither. Tangencies ignored."
        )

    try:
        edge = cq.Edge.makeSpline(pnts, tangents=tangents, periodic=closed)
        return cq.Wire.assembleEdges([edge])
    except Exception as exc:
        raise FreeCADError(f"CadQuery was unable to make a spline: {exc}") from exc


def make_face(wire: apiWire) -> apiFace:
    """Make a planar face bounded by *wire*."""
    try:
        face = cq.Face.makeFromWires(wire)
    except Exception as exc:
        raise FreeCADError(f"An invalid face has been generated: {exc}") from exc
    if not face.isValid():
        raise FreeCADError("An invalid face has been generated")
    return face


# ---------------------------------------------------------------------------
# Shape transformations
# ---------------------------------------------------------------------------


def revolve_shape(
    shape: apiShape,
    base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    degree: float = 180.0,
) -> apiShape:
    """Revolve *shape* around an axis defined by *base* and *direction*.

    * **Face** input → Solid (with planar end-caps when degree < 360°), matching
      FreeCAD's ``Part.Face.revolve`` behaviour.
    * **Wire / Edge** input → Shell (lateral surface only, no end-caps), matching
      FreeCAD's ``Part.Wire.revolve`` behaviour.  This is implemented via OCC's
      ``BRepPrimAPI_MakeRevol`` which does not add caps for open angles.
    """
    try:
        if isinstance(shape, cq.Face):
            axis_start = cq.Vector(*base)
            axis_end = cq.Vector(
                base[0] + direction[0],
                base[1] + direction[1],
                base[2] + direction[2],
            )
            outer_wire = shape.outerWire()
            inner_wires = shape.innerWires()
            return cq.Solid.revolve(
                outer_wire, inner_wires, degree, axis_start, axis_end
            )
        # Wire / Edge: use OCC directly to get a Shell without end-caps.
        n = np.asarray(direction, dtype=float)
        n /= np.linalg.norm(n)
        axis = gp_Ax1(
            gp_Pnt(*[float(x) for x in base]),
            gp_Dir(*n.tolist()),
        )
        maker = BRepPrimAPI_MakeRevol(shape.wrapped, axis, math.radians(degree))
        maker.Build()
        return cq.Shape.cast(maker.Shape())
    except FreeCADError:
        raise
    except Exception as exc:
        raise FreeCADError(f"CadQuery revolve failed: {exc}") from exc


def _check_path_tangent_continuity(path: apiWire, tol: float = 1e-6) -> None:
    """Raise ``FreeCADError`` if the path has a non-tangent-continuous join.

    Mirrors FreeCAD's ``Part.Wire.makePipeShell`` precondition: at every
    interior vertex of the sweep path, the end-tangent of the incoming
    edge and the start-tangent of the outgoing edge must agree (within a
    dot-product tolerance). OCCT's ``BRepOffsetAPI_MakePipeShell`` does
    not enforce this itself and will happily sweep along a kinked polyline,
    producing a self-intersecting / kinked solid. FreeCAD raises on this
    case, so we do too.
    """
    edges = ordered_edges(path)
    pairs = list(pairwise(edges))
    if is_closed(path):
        # Closed path: also check the seam (last edge -> first edge), otherwise
        # a kink at the closure goes undetected and OCCT silently sweeps a
        # self-intersecting solid. Mirrors _freecadapi._wire_edges_tangent.
        pairs.append((edges[-1], edges[0]))
    for e_prev, e_next in pairs:
        a_prev = BRepAdaptor_Curve(e_prev.wrapped)
        a_next = BRepAdaptor_Curve(e_next.wrapped)
        p_end = gp_Pnt()
        t_end = gp_Vec()
        a_prev.D1(a_prev.LastParameter(), p_end, t_end)
        p_start = gp_Pnt()
        t_start = gp_Vec()
        a_next.D1(a_next.FirstParameter(), p_start, t_start)
        mag = t_end.Magnitude() * t_start.Magnitude()
        if mag == 0:
            continue
        cos_angle = t_end.Dot(t_start) / mag
        if cos_angle < 1.0 - tol:
            raise FreeCADError(
                "sweep_shape: path is not tangent-continuous at an interior "
                f"vertex (cos(angle)={cos_angle:.6f})."
            )


def sweep_shape(
    profiles: Iterable[apiWire],
    path: apiWire,
    *,
    solid: bool = True,
    frenet: bool = True,
    transition: int = 0,
) -> apiShell | apiSolid:
    """Sweep one or more *profiles* along *path*."""
    if isinstance(profiles, apiWire):
        # Single wire → wrap as one-element list. Avoids the trap that
        # ``list(cq.Wire)`` iterates the wire's edges (not what we want).
        profiles = [profiles]
    elif not isinstance(profiles, list):
        profiles = list(profiles)

    closures = [p.IsClosed() for p in profiles]
    all_closed = all(closures)
    none_closed = not any(closures)

    if not all_closed and not none_closed:
        raise FreeCADError("You cannot mix open and closed profiles when sweeping.")

    if none_closed and solid:
        bluemira_warn(
            "You cannot sweep open profiles and expect a Solid result. Disabling this."
        )
        solid = False

    _check_path_tangent_continuity(path)

    # bluemira's sweep semantics match FreeCAD's ``Part.Wire.makePipeShell``:
    # all profiles are section profiles along the path (multi-section sweep),
    # not an outer+inner wires pair. For a single profile this degenerates to
    # a plain pipe sweep.
    try:
        if len(profiles) == 1:
            transition_mode = ["transformed", "right", "round"][transition]
            result = cq.Solid.sweep(
                profiles[0],
                [],
                path,
                makeSolid=solid,
                isFrenet=frenet,
                transitionMode=transition_mode,
            )
        else:
            result = cq.Solid.sweep_multi(
                profiles, path, makeSolid=solid, isFrenet=frenet
            )
    except Exception as exc:
        raise FreeCADError(f"CadQuery sweep failed: {exc}") from exc

    # Multi-section sweeps on composite profiles produce shells whose adjacent
    # faces do not share topological edges — ``isValid()`` still returns True,
    # but downstream boolean/section operations silently drop individual faces
    # because there is no shared boundary for them to propagate across. Sewing
    # stitches the shell back together via vertex/edge merging within
    # tolerance, restoring a single coherent boundary.
    if solid and len(profiles) > 1:
        result = _sewn_solid(result)
    if solid and not result.isValid():
        fix_shape(result)

    if solid:
        return result
    return result.Shells()[0]


def _sewn_solid(solid: apiSolid, tolerance: float = 1e-3) -> apiSolid:
    """Rebuild *solid* by sewing its faces into a shell and constructing a solid.

    ``BRepBuilderAPI_Sewing`` reconnects faces that share geometric boundaries
    within *tolerance* but not TopoDS vertices/edges — a common side-effect of
    multi-section sweeps on composite profiles. Returns the original solid if
    sewing fails to produce a single valid shell.
    """
    sewer = BRepBuilderAPI_Sewing(tolerance)
    for f in solid.Faces():
        sewer.Add(f.wrapped)
    sewer.Perform()
    sewn = sewer.SewedShape()
    # Sewing may yield a Shell, a Compound of Shells, or the original shapes;
    # we only rebuild when we end up with exactly one Shell.
    shells = cq.Shape.cast(sewn).Shells() if not sewn.IsNull() else []
    if len(shells) != 1:
        return solid
    maker = BRepBuilderAPI_MakeSolid(shells[0].wrapped)
    if not maker.IsDone():
        return solid
    return cq.Solid(maker.Solid())


def offset_wire(
    wire: apiWire,
    thickness: float,
    join: str = "intersect",
    *,
    open_wire: bool = True,
) -> apiWire:
    """
    Offset *wire* by *thickness* in its plane.

    Parameters
    ----------
    wire:
        Wire to offset (must be planar and non-straight).
    thickness:
        Offset distance. Positive = outward, negative = inward.
    join:
        Corner style: "arc", "intersect", or "tangent".
    open_wire:
        For open wires, keep offset open (True) or close it (False).
    """
    if not thickness:
        return wire

    if _wire_is_straight(wire):
        raise InvalidCADInputsError("Cannot offset a straight line.")

    if not _wire_is_planar(wire):
        raise InvalidCADInputsError("Cannot offset a non-planar wire.")

    # CadQuery offset2D kind: 'arc' | 'intersection' | 'tangent'
    _join_map = {"arc": "arc", "intersect": "intersection", "tangent": "tangent"}
    kind = _join_map[join.lower()]

    if join.lower() == "tangent":
        bluemira_warn(
            f"Join type: {join} may be unstable. Consider 'arc' or 'intersect'."
        )

    if wire.IsClosed() and open_wire:
        open_wire = False

    try:
        result_wires = wire.offset2D(thickness, kind=kind)
    except Exception as exc:
        raise FreeCADError(
            f"CadQuery was unable to make an offset of wire: {exc}"
        ) from exc

    if not result_wires:
        raise FreeCADError("offset_wire: no result produced")

    result = result_wires[0]

    if not open_wire and not result.IsClosed():
        raise FreeCADError("offset failed to close wire")

    return result


# ---------------------------------------------------------------------------
# Shape validation helpers
# ---------------------------------------------------------------------------


def _wire_is_straight(wire: apiWire) -> bool:
    """True if the wire is a single straight line segment."""
    edges = wire.Edges()
    if len(edges) != 1:
        return False
    edge = edges[0]
    verts = edge.Vertices()
    if len(verts) != 2:  # noqa: PLR2004
        return False
    p1 = _vector_to_numpy(verts[0].Center())
    p2 = _vector_to_numpy(verts[1].Center())
    chord = np.linalg.norm(p2 - p1)
    return bool(np.isclose(chord, wire.Length(), rtol=1e-5, atol=1e-8))


def _wire_is_planar(wire: apiWire) -> bool:
    """True if all vertices of the wire lie in a single plane."""
    try:
        face = cq.Face.makeFromWires(wire)
        return face.geomType() == "PLANE"
    except Exception:  # noqa: BLE001
        return False


def _edges_tangent(edge1: apiEdge, edge2: apiEdge, tol: float = 0.3) -> bool:
    """True if two consecutive edges are not sharply discontinuous at their junction.

    The default tolerance ``tol=0.3`` (cos_angle > 0.7, i.e. angle < ~45°) is
    intentionally generous: it rejects obvious kinks (like 90° polygon corners,
    cos=0) while accepting smooth parametric paths whose junction tangents may
    differ slightly due to numerical evaluation or moderate curvature changes.
    """
    try:
        t1 = _vector_to_numpy(edge1.tangentAt(edge1.paramAt(1.0)))
        t2 = _vector_to_numpy(edge2.tangentAt(edge2.paramAt(0.0)))
        cos_angle = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + EPS)
        return bool(cos_angle > 1.0 - tol)
    except Exception:  # noqa: BLE001
        return True  # conservative: don't block sweep on uncertainty


def _wire_edges_tangent(wire: apiWire, atol: float = 1e-4) -> bool:
    """True if all consecutive edges in the wire are tangent.

    Uses wire-level tangents at small offsets from each edge junction to avoid
    issues with OCC edge orientation vs wire orientation.
    """
    edges = wire.Edges()
    if len(edges) <= 1:
        return True

    total = wire.Length()
    if total < _GEOM_NEAR_ZERO_TOL:
        return True

    # Build cumulative arc lengths at each edge boundary using ordered edges.
    oe = ordered_edges(wire)
    if not oe:
        oe = edges

    cum = [0.0]
    for e in oe:
        cum.append(cum[-1] + e.Length())

    eps = min(1e-6 * total, 1e-9)

    for i in range(1, len(oe)):
        d = cum[i]  # junction at this arc length
        # Tangent just before and just after the junction
        d_before = max(0.0, d - eps)
        d_after = min(total, d + eps)
        t_before = wire.tangentAt(d_before, mode="length")
        t_after = wire.tangentAt(d_after, mode="length")
        v1 = np.array([t_before.x, t_before.y, t_before.z])
        v2 = np.array([t_after.x, t_after.y, t_after.z])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < EPS or n2 < EPS:
            continue
        cos_a = np.dot(v1, v2) / (n1 * n2)
        angle = math.acos(max(-1.0, min(1.0, cos_a)))
        if not np.isclose(angle, 0.0, rtol=0.0, atol=atol):
            return False

    return True


# ---------------------------------------------------------------------------
# Shape properties
# ---------------------------------------------------------------------------


def length(obj: apiShape) -> float:
    """Total length of the shape (sum of all edge lengths).

    Sum per-edge ``Length()`` rather than the composite ``cq.Wire.Length()``:
    on CadQuery 2.7 / OCP 7.8 the composite-curve adaptor used by
    ``cq.Wire.Length()`` segfaults on certain wires assembled from
    ``BRepAlgoAPI_Section`` output (short, near-degenerate edges). Per-edge
    ``BRepAdaptor_Curve``-based length is equivalent and stable.
    """
    if isinstance(obj, cq.Edge):
        return obj.Length()
    return sum(e.Length() for e in obj.Edges())


def _occ_face_area(topods_face) -> float:
    """Compute the surface area of a TopoDS_Face via OCC mass properties."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(topods_face, props)
    return props.Mass()


def _cq_area_prop(self) -> float:
    """Area property for cq.Face.

    FreeCAD exposes this as a property, CadQuery as a method — this shim
    bridges the two.
    """
    inner = self.innerWires()
    if inner:
        outer_area = _occ_face_area(cq.Face.makeFromWires(self.outerWire()).wrapped)
        hole_area = sum(_occ_face_area(cq.Face.makeFromWires(w).wrapped) for w in inner)
        return outer_area - hole_area
    return _occ_face_area(self.wrapped)


def area(obj: apiShape) -> float:
    """Surface area of the shape.

    For faces with inner wires (holes) the area is computed as
    ``outer_area - sum(hole_areas)`` to avoid OCC quadrature inconsistency
    between the annular face and independently computed sub-face areas.
    """
    if isinstance(obj, cq.Face):
        inner = obj.innerWires()
        if inner:
            outer_area = _occ_face_area(cq.Face.makeFromWires(obj.outerWire()).wrapped)
            hole_area = sum(
                _occ_face_area(cq.Face.makeFromWires(w).wrapped) for w in inner
            )
            return outer_area - hole_area
        return obj.Area  # property (monkey-patched) → float
    return obj.Area()  # method on Wire/Solid/Shell/Edge → float


def volume(obj: apiShape) -> float:
    """Volume of the shape.

    Wires and edges are 1-D objects with zero volume; OCC's ``Volume()`` on them
    returns the arc length, so we short-circuit here.
    """
    if isinstance(obj, (cq.Wire, cq.Edge)):
        return 0.0
    return obj.Volume()


def center_of_mass(obj: apiShape) -> np.ndarray:
    """Centre of mass as a numpy array."""
    c = obj.Center()
    return _vector_to_numpy(c)


def is_closed(obj: apiShape) -> bool:
    """True if the shape is closed.

    For compounds (e.g. the result of boolean wire fuse), a compound is
    considered closed if it contains exactly one wire and that wire is closed.
    """
    if isinstance(obj, cq.Compound):
        wires = obj.Wires()
        return len(wires) == 1 and wires[0].IsClosed()
    return obj.IsClosed()


def is_valid(obj) -> bool:
    """True if the shape is valid."""
    return obj.isValid()


def fix_shape(shape: apiShape, precision: float = 1e-6, min_length: float = 1e-8):
    """
    Attempt to fix *shape* in-place via OCC's ``ShapeFix_Shape``.

    Composite wires/faces assembled from multiple sub-shapes (e.g. via
    ``cq.Wire.combine``) sometimes fail ``BRepCheck`` even when each component is
    individually valid, because adjacent edges do not share vertices. ShapeFix
    reconstructs the sharing and tightens tolerances.

    The CadQuery wrapper's ``.wrapped`` attribute is reassigned so callers that
    already hold a reference to *shape* see the fixed geometry.
    """
    fixer = ShapeFix_Shape(shape.wrapped)
    fixer.SetPrecision(precision)
    fixer.SetMinTolerance(min_length)
    fixer.Perform()
    shape.wrapped = fixer.Shape()


# ---------------------------------------------------------------------------
# Shape topology accessors  (mirror freecadapi's module-level functions so
# that wrapper classes can call cadapi.wires(x) instead of x.Wires)
# ---------------------------------------------------------------------------


def orientation(obj: apiShape) -> str:
    """Return 'Forward' or 'Reversed' for the shape's OCC orientation."""
    return "Reversed" if obj.wrapped.Orientation() == TopAbs_REVERSED else "Forward"


def reverse_shape(obj: apiShape) -> apiShape:
    """Return a new shape with reversed orientation (CadQuery shapes are immutable)."""
    return type(obj)(obj.wrapped.Reversed())


def wires(obj: apiShape) -> list[apiWire]:
    """Wires contained in the shape."""
    return obj.Wires()


def ordered_edges(obj: apiShape) -> list[apiEdge]:
    """Edges of the shape in wire-connectivity order (mirrors FreeCAD OrderedEdges)."""
    try:
        explorer = BRepTools_WireExplorer(obj.wrapped)
        result = []
        while explorer.More():
            result.append(cq.Edge(explorer.Current()))
            explorer.Next()
        if result:
            return result
    except Exception:  # noqa: BLE001, S110
        pass  # fall back to storage-order edges below
    return obj.Edges()


def edges(obj: apiShape) -> list[apiEdge]:
    """Edges of the shape."""
    return obj.Edges()


def eccentricity(edge: apiEdge) -> float:
    """Return the eccentricity of an ellipse/circle edge's underlying curve.

    Returns 0.0 for a circle, e = sqrt(1 - (b/a)²) for an ellipse. Raises
    for any other curve type — the concept doesn't apply to line, bspline,
    bezier, etc.
    """
    adaptor = BRepAdaptor_Curve(edge.wrapped)
    curve_type = adaptor.GetType()
    if curve_type == GeomAbs_Ellipse:
        return adaptor.Ellipse().Eccentricity()
    if curve_type == GeomAbs_Circle:
        return 0.0
    raise GeometryError(
        f"eccentricity: edge curve is not an ellipse or circle ({curve_type})."
    )


def faces(obj: apiShape) -> list[apiFace]:
    """Faces of the shape."""
    return obj.Faces()


def shells(obj: apiShape) -> list[apiShell]:
    """Shells of the shape."""
    return obj.Shells()


def solids(obj: apiShape) -> list[apiSolid]:
    """Solids of the shape."""
    return obj.Solids()


def wire_from_edges(edge_list: list[apiEdge]) -> apiWire:
    """Create a wire from a list of edges."""
    return cq.Wire.assembleEdges(edge_list)


def wire_from_wires(wire_list: list[apiWire]) -> apiWire:
    """Create a single wire from a list of connected wires."""
    if len(wire_list) == 1:
        return wire_list[0]
    result = cq.Wire.combine(wire_list)
    return result[0] if isinstance(result, list) else result


def arrange_edges(old_wire: apiWire, new_wire: apiWire) -> apiWire:
    """Flip edges of *new_wire* whose orientation disagrees with *old_wire*.

    Mirrors FreeCAD's helper of the same name: sort both wires' edges into
    connectivity order, then walk paired edges and reverse any new-wire edge
    whose TopoDS orientation flag differs from the old-wire edge at the same
    position. Used to stabilise offset results whose edges come back with
    inconsistent orientation.
    """
    old_edges = ordered_edges(old_wire)
    new_edges = ordered_edges(new_wire)
    adjusted: list[apiEdge] = []
    for i, new_edge in enumerate(new_edges):
        if i < len(old_edges) and (
            new_edge.wrapped.Orientation() != old_edges[i].wrapped.Orientation()
        ):
            adjusted.append(reverse_shape(new_edge))
        else:
            adjusted.append(new_edge)
    try:
        return cq.Wire.assembleEdges(adjusted)
    except Exception:  # noqa: BLE001
        return new_wire


# ---------------------------------------------------------------------------
# Distance / spatial queries
# ---------------------------------------------------------------------------


def dist_to_shape(
    shape1: apiShape, shape2: apiShape
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Minimum distance between two shapes.

    Returns
    -------
    dist:
        Minimum distance.
    vectors:
        List of (point_on_shape1, point_on_shape2) numpy arrays.
    """
    dss = BRepExtrema_DistShapeShape(shape1.wrapped, shape2.wrapped)
    dss.Perform()
    dist = dss.Value()
    vectors = []
    for i in range(1, dss.NbSolution() + 1):
        p1 = dss.PointOnShape1(i)
        p2 = dss.PointOnShape2(i)
        vectors.append((
            np.array([p1.X(), p1.Y(), p1.Z()]),
            np.array([p2.X(), p2.Y(), p2.Z()]),
        ))
    return dist, vectors


# Error handling
# ---------------------------------------------------------------------------


def catch_caderr(new_error_type):
    """Translate FreeCADError raised inside the decorated function into
    *new_error_type*. Mirrors the FreeCAD backend so callers get a uniform
    error-translation contract regardless of backend.
    """

    def argswrap(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FreeCADError as fe:
                raise new_error_type(fe.args[0]) from fe

        return wrapper

    return argswrap


# ---------------------------------------------------------------------------
# Shape predicates
# ---------------------------------------------------------------------------


def is_null(obj: apiShape) -> bool:
    """True if the shape is null."""
    return obj.wrapped.IsNull()


def is_same(obj1: apiShape, obj2: apiShape) -> bool:
    """True if the two shapes share the same underlying TShape."""
    return obj1.wrapped.IsSame(obj2.wrapped)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def bounding_box(obj: apiShape) -> tuple[float, float, float, float, float, float]:
    """Return (xmin, ymin, zmin, xmax, ymax, zmax)."""
    bb = obj.BoundingBox()
    return bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax


def optimal_bounding_box(
    obj: apiShape,
) -> tuple[float, float, float, float, float, float]:
    """Alias for bounding_box (CadQuery does not expose a tighter variant)."""
    return bounding_box(obj)


# ---------------------------------------------------------------------------
# Vertex / point accessors
# ---------------------------------------------------------------------------


def edge_tangent_at(edge: apiEdge, param: float) -> np.ndarray:
    """Return the unit tangent of *edge* at normalised parameter *param* in [0, 1]."""
    return _vector_to_numpy(edge.tangentAt(param))


def start_point(obj: apiShape) -> np.ndarray:
    """Start point of the first ordered edge (not orientation-aware)."""
    return _vector_to_numpy(ordered_edges(obj)[0].startPoint())


def end_point(obj: apiShape) -> np.ndarray:
    """End point of the last ordered edge (not orientation-aware)."""
    return _vector_to_numpy(ordered_edges(obj)[-1].endPoint())


def ordered_vertexes(obj: apiShape) -> np.ndarray:
    """Vertices in connectivity order along a wire."""
    edge_list = ordered_edges(obj)
    pts = [_vector_to_numpy(e.startPoint()) for e in edge_list]
    if not obj.IsClosed():
        pts.append(_vector_to_numpy(edge_list[-1].endPoint()))
    return np.array(pts)


def vertexes(obj: apiShape) -> np.ndarray:
    """All vertices of the shape as (N, 3) array."""
    return np.array([_vector_to_numpy(v.Center()) for v in obj.Vertices()])


# ---------------------------------------------------------------------------
# Face normal
# ---------------------------------------------------------------------------


def normal_at(face: apiFace, alpha_1: float = 0.0, alpha_2: float = 0.0) -> np.ndarray:
    """Normal vector of a face at parametric coordinates (u, v)."""
    surf = BRepAdaptor_Surface(face.wrapped)
    pnt = gp_Pnt()
    du = gp_Vec()
    dv = gp_Vec()
    surf.D1(alpha_1, alpha_2, pnt, du, dv)
    normal = du.Crossed(dv)
    normal.Normalize()
    return np.array([normal.X(), normal.Y(), normal.Z()])


# Compound / shell / solid builders
# ---------------------------------------------------------------------------


def make_shell(faces: list[apiFace]) -> apiShell:
    """Create a shell from a list of faces."""
    sewer = BRepBuilderAPI_Sewing()
    for f in faces:
        sewer.Add(f.wrapped)
    sewer.Perform()
    return cq.Shell(sewer.SewedShape())


def make_solid(shell: apiShell) -> apiSolid:
    """Create a solid from a shell."""
    builder = BRepBuilderAPI_MakeSolid(shell.wrapped)
    return cq.Solid(builder.Solid())


# ---------------------------------------------------------------------------
# Wire utilities
# ---------------------------------------------------------------------------


def wire_closure(wire: apiWire) -> apiWire | None:
    """Return a line-segment wire closing an open wire, or None if already closed."""
    if wire.IsClosed():
        return None
    edge_list = ordered_edges(wire)
    p_end = edge_list[-1].endPoint()
    p_start = edge_list[0].startPoint()
    return cq.Wire.assembleEdges([cq.Edge.makeLine(p_end, p_start)])


def close_wire(wire: apiWire) -> apiWire:
    """Return the wire closed with a straight line if not already closed.

    Uses ``BRepBuilderAPI_MakeWire`` to preserve insertion order of the
    edges. ``cq.Wire.assembleEdges`` internally runs
    ``ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s``, which rearranges
    edges by connectivity regardless of input order — e.g. closing a
    ``bottom → right → top`` open wire with a ``left`` edge would yield
    the cycle ``bottom, left, right, top`` instead of
    ``bottom, right, top, left``. Downstream code (``extrude_shape`` side
    faces, orientation contract tests) depends on the insertion order.
    """
    if wire.IsClosed():
        return wire
    edge_list = ordered_edges(wire)
    p_end = edge_list[-1].endPoint()
    p_start = edge_list[0].startPoint()
    closing = cq.Edge.makeLine(p_end, p_start)
    maker = BRepBuilderAPI_MakeWire()
    for e in [*edge_list, closing]:
        maker.Add(e.wrapped)
    maker.Build()
    if not maker.IsDone():
        return cq.Wire.assembleEdges([*edge_list, closing])
    return cq.Wire(maker.Wire())


def discretise(w: apiWire, ndiscr: int = 10, dl: float | None = None) -> np.ndarray:
    """Sample a wire into an array of (N, 3) points.

    Uses ``BRepAdaptor_CompCurve`` + ``GCPnts_UniformAbscissa`` once on the
    whole wire. ``cq.Wire.positionAt`` in contrast constructs a fresh
    adaptor and recomputes the total arc length on every call — O(N²) per
    discretisation, which made multi-thousand-point samples (used by
    ``force_wire_to_spline`` on long revolved wires) dominate reactor-build
    runtime (~70 x slower than FreeCAD for a simple VVTS build).
    """
    if dl is None:
        if ndiscr < 2:  # noqa: PLR2004
            raise ValueError("ndiscr must be >= 2.")
    else:
        if dl <= 0.0:
            raise ValueError("dl must be > 0.")
        total = w.Length()
        ndiscr = max(math.ceil(total / dl + 1), 2)

    adaptor = BRepAdaptor_CompCurve(w.wrapped)
    sampler = GCPnts_UniformAbscissa(adaptor, ndiscr)
    if not sampler.IsDone() or sampler.NbPoints() < 2:  # noqa: PLR2004
        # Fall back to the slow parameter-space loop if GCPnts fails
        # (e.g. degenerate wire with zero length).
        params = np.linspace(0.0, 1.0, ndiscr)
        pts = np.array([_vector_to_numpy(w.positionAt(t)) for t in params])
    else:
        n = sampler.NbPoints()
        pts = np.empty((n, 3))
        for i in range(n):
            p = adaptor.Value(sampler.Parameter(i + 1))
            pts[i] = (p.X(), p.Y(), p.Z())
    if w.IsClosed():
        pts[-1] = pts[0]
    return pts


def discretise_by_edges(
    w: apiWire, ndiscr: int = 10, dl: float | None = None
) -> np.ndarray:
    """Sample each edge individually and concatenate.

    Uses a single ``BRepAdaptor_CompCurve(w.wrapped)`` and converts each
    target arc-length to a parameter via ``GCPnts_AbscissaPoint``. The
    previous ``cq.Wire.positionAt`` loop rebuilt the adaptor and recomputed
    the wire's total length on every call (O(N²) per discretisation);
    optimiser inner loops calling this at 200+ points per iteration
    dominated the RippleConstrainedLengthGOP runtime.
    """
    total = w.Length()
    if dl is None:
        dl = total / float(ndiscr)
    elif dl <= 0.0:
        raise ValueError("dl must be > 0.")

    # Sample directly along the parent wire's arc-length parameterisation,
    # per edge, so we inherit the wire's traversal order and orientation.
    # cq.Wire.assembleEdges([e]) strips the edge's orientation flag — hence
    # no per-edge sub-wires.
    adaptor = BRepAdaptor_CompCurve(w.wrapped)
    first_param = adaptor.FirstParameter()

    # Convert per-edge arc-length boundaries to adaptor parameters once
    # (one Newton solve per boundary), then let GCPnts_UniformAbscissa
    # sample each edge's parameter range in a single call — avoids the
    # one-Newton-per-sampled-point cost of GCPnts_AbscissaPoint.
    edge_lengths = [e.Length() for e in ordered_edges(w)]
    cum_abscissa = np.cumsum([0.0, *edge_lengths])
    edge_params = [first_param]
    for abscissa in cum_abscissa[1:]:
        finder = GCPnts_AbscissaPoint(adaptor, abscissa, first_param)
        edge_params.append(
            finder.Parameter() if finder.IsDone() else adaptor.LastParameter()
        )

    output = []
    last_pts = None
    for i, e_len in enumerate(edge_lengths):
        if e_len < _OCC_DEFAULT_TOL:
            continue
        n = max(math.ceil(e_len / dl + 1), 2)
        sampler = GCPnts_UniformAbscissa(adaptor, n, edge_params[i], edge_params[i + 1])
        if sampler.IsDone() and sampler.NbPoints() >= 2:  # noqa: PLR2004
            pts = []
            for k in range(1, sampler.NbPoints() + 1):
                p = adaptor.Value(sampler.Parameter(k))
                pts.append(np.array([p.X(), p.Y(), p.Z()]))
        else:
            # Fallback to the slow per-point positionAt loop for pathological
            # edges that GCPnts can't sample.
            a0, a1 = cum_abscissa[i] / total, cum_abscissa[i + 1] / total
            pts = [_vector_to_numpy(w.positionAt(t)) for t in np.linspace(a0, a1, n)]
        # Dedup numerically-coincident consecutive samples. GCPnts_UniformAbscissa
        # occasionally emits its final samples at the same parameter near the
        # endpoint (small-Δs numerical latch), producing duplicates that break
        # downstream vector_lengthnorm / scipy spline setup.
        deduped = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - deduped[-1]) > _POINT_COINCIDENCE_TOL:
                deduped.append(p)
        pts = deduped
        output += pts[:-1]
        last_pts = pts

    if w.IsClosed():
        output.append(output[0])
    elif last_pts:
        output.append(last_pts[-1])

    return np.array(output)


def wire_value_at(wire: apiWire, distance: float) -> np.ndarray:
    """Return the point a given arc-length distance along the wire."""
    total = wire.Length()
    if distance <= 0.0:
        return start_point(wire)
    if distance >= total:
        return end_point(wire)
    return _vector_to_numpy(wire.positionAt(distance / total))


def wire_parameter_at(wire: apiWire, vertex, tolerance: float = 1e-8) -> float:
    """Return the normalised arc-length parameter [0,1] for a point on the wire."""
    wire_1, _ = split_wire(wire, vertex, tolerance)
    if wire_1 is None:
        return 0.0
    return wire_1.Length() / wire.Length()


def split_wire(
    wire: apiWire, vertex, tolerance: float
) -> tuple[apiWire | None, apiWire | None]:
    """Split a wire at the point nearest to *vertex*."""
    vertex = np.asarray(vertex, dtype=float)
    vertex_shape = cq.Vertex.makeVertex(*vertex)

    dist, _ = dist_to_shape(wire, vertex_shape)
    if dist > tolerance:
        raise FreeCADError(
            f"Vertex is not close enough to the wire: distance {dist} > {tolerance}"
        )

    all_edges = ordered_edges(wire)

    # Find the *closest* edge to the vertex (mirrors FreeCAD's
    # _get_closest_edge_idx).  Picking the first edge within ``tolerance`` is
    # wrong when the caller passes a large tolerance (e.g. VERY_BIG) — with a
    # lax gate, a vertex at the wire's end would still "hit" edge 0, whose
    # curve projection then falls outside the edge's parameter range and the
    # split collapses to param = 0.
    split_idx = 0
    best_dist = float("inf")
    for i, e in enumerate(all_edges):
        e_wire = cq.Wire.assembleEdges([e])
        d, _ = dist_to_shape(e_wire, vertex_shape)
        if d < best_dist:
            best_dist = d
            split_idx = i

    edges_1: list[cq.Edge] = []
    edges_2: list[cq.Edge] = []

    for i, e in enumerate(all_edges):
        if i < split_idx:
            edges_1.append(e)
            continue
        if i > split_idx:
            edges_2.append(e)
            continue

        # Split-edge: project the vertex onto the edge's curve to find t_split.
        adaptor = BRepAdaptor_Curve(e.wrapped)
        t0 = adaptor.FirstParameter()
        t1 = adaptor.LastParameter()
        curve = adaptor.Curve().Curve()
        pnt = gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2]))
        proj = GeomAPI_ProjectPointOnCurve(pnt, curve, t0, t1)
        if proj.NbPoints() > 0:
            t_split = proj.LowerDistanceParameter()
        else:
            # Projection fell outside [t0, t1]: snap to the nearer endpoint.
            p0 = adaptor.Value(t0)
            p1 = adaptor.Value(t1)
            d0 = p0.SquareDistance(pnt)
            d1 = p1.SquareDistance(pnt)
            t_split = t0 if d0 <= d1 else t1
        t_split = max(t0, min(t1, t_split))
        if t_split - t0 > _ANGLE_PARALLEL_TOL:
            edges_1.append(cq.Edge(BRepBuilderAPI_MakeEdge(curve, t0, t_split).Edge()))
        if t1 - t_split > _ANGLE_PARALLEL_TOL:
            edges_2.append(cq.Edge(BRepBuilderAPI_MakeEdge(curve, t_split, t1).Edge()))

    wire_1 = cq.Wire.assembleEdges(edges_1) if edges_1 else None
    wire_2 = cq.Wire.assembleEdges(edges_2) if edges_2 else None
    return wire_1, wire_2


# ---------------------------------------------------------------------------
# Affine transforms
# ---------------------------------------------------------------------------


def _apply_trsf_inplace(shape: apiShape, trsf: gp_Trsf) -> apiShape:
    """Apply an OCC transform to *shape* in-place and return it.

    ``base.py``'s ``scale`` / ``translate`` / ``rotate`` methods call these
    functions but discard the return value, expecting in-place mutation (that is
    how FreeCAD's API works).  CadQuery shapes are normally immutable, but we
    can update their ``wrapped`` attribute directly.
    """
    new_wrapped = BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape()
    shape.wrapped = cq.Shape.cast(new_wrapped).wrapped
    return shape


def scale_shape(shape: apiShape, factor: float) -> apiShape:
    """Scale *shape* in-place by *factor* and return it."""
    trsf = gp_Trsf()
    trsf.SetScaleFactor(factor)
    return _apply_trsf_inplace(shape, trsf)


def translate_shape(shape: apiShape, vector) -> apiShape:
    """Translate *shape* in-place by *vector* and return it."""
    vec = [float(x) for x in vector]
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(*vec))
    return _apply_trsf_inplace(shape, trsf)


def rotate_shape(
    shape: apiShape,
    base=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    degree: float = 180.0,
) -> apiShape:
    """Rotate *shape* in-place and return it."""
    n = np.asarray(direction, dtype=float)
    n /= np.linalg.norm(n)
    ax = gp_Ax1(gp_Pnt(*[float(x) for x in base]), gp_Dir(*n.tolist()))
    trsf = gp_Trsf()
    trsf.SetRotation(ax, math.radians(degree))
    return _apply_trsf_inplace(shape, trsf)


def mirror_shape(
    shape: apiShape,
    base,
    direction,
) -> apiShape:
    """Return a mirrored copy of the shape about the plane defined by base+direction."""
    pnt = gp_Pnt(*[float(x) for x in base])
    drc = gp_Dir(*[float(x) for x in direction])
    trsf = gp_Trsf()
    trsf.SetMirror(gp_Ax2(pnt, drc))
    builder = BRepBuilderAPI_Transform(shape.wrapped, trsf, True)
    return cq.Shape.cast(builder.Shape())


def extrude_shape(shape: apiShape, vec) -> apiShape:
    """Extrude a shape along a vector."""
    v = gp_Vec(*[float(x) for x in vec])
    builder = BRepPrimAPI_MakePrism(shape.wrapped, v)
    return cq.Shape.cast(builder.Shape())


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------


def boolean_fuse(shapes: list, *, remove_splitter: bool = True) -> apiShape:
    """Boolean union of a list of shapes."""
    if not isinstance(shapes, list):
        raise TypeError(f"{shapes} is not a list.")
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes required.")
    result = shapes[0].fuse(*shapes[1:])

    if all(isinstance(s, cq.Face) for s in shapes) and isinstance(result, cq.Compound):
        # OCC's fuse on faces that merely touch at an edge produces a Compound
        # of disconnected faces — UnifySameDomain can't merge them because they
        # don't share any TopoDS edge. Sew the inputs first to establish edge
        # sharing, then unify to collapse coplanar neighbours into one face.
        sewing = BRepBuilderAPI_Sewing(1e-6)
        for s in shapes:
            sewing.Add(s.wrapped)
        sewing.Perform()
        sewn = cq.Shape.cast(sewing.SewedShape())
        unified = _unify_same_domain(sewn)
        if isinstance(unified, cq.Face):
            result = unified
        else:
            unified_faces = _collect_subshapes(unified, cq.Face)
            if len(unified_faces) == 1:
                result = unified_faces[0]

    if remove_splitter:
        with contextlib.suppress(Exception):
            result = result.clean()

    # For wire inputs: OCC fuse returns a compound; try to assemble a single wire.
    if all(isinstance(s, cq.Wire) for s in shapes):
        result_wires = result.Wires()
        if len(result_wires) == 1:
            return result_wires[0]
        if result_wires:
            with contextlib.suppress(Exception):
                combined = cq.Wire.combine(result_wires)
                return combined[0] if isinstance(combined, list) else combined

    # Boolean fuse on coplanar/touching faces routinely yields a compound that
    # fails BRepCheck (overlapping face boundaries, vertex sharing not yet
    # established). ShapeFix repairs it, mirroring what BluemiraFace does on
    # face creation.
    if not result.isValid():
        fix_shape(result)

    # OCC wraps fuse results in a compound even when the inputs are homogeneous
    # and the output is a single shape of that same kind. FreeCAD's fuse
    # unwraps in this case; mirror that here so callers that feed in Faces get
    # a Face back (not a Compound containing one Face), otherwise downstream
    # ``.boundary[0]`` access misinterprets the wrapping.
    if isinstance(result, cq.Face):
        return result

    if isinstance(result, cq.Compound):
        if all(isinstance(s, cq.Face) for s in shapes):
            faces = _collect_subshapes(result, cq.Face)
            if len(faces) == 1:
                return faces[0]
            if len(faces) > 1:
                raise GeometryError(
                    f"Boolean fuse operation on {shapes} gives more than one face."
                )
        if all(isinstance(s, cq.Solid) for s in shapes):
            solids = _collect_subshapes(result, cq.Solid)
            if len(solids) == 1:
                return solids[0]
            if len(solids) > 1:
                raise GeometryError(
                    f"Boolean fuse operation on {shapes} gives more than one solid."
                )

    return result


def _unify_same_domain(shape: apiShape) -> apiShape:
    """Merge coplanar connected faces and collinear edges via OCC UnifySameDomain."""
    try:
        unifier = ShapeUpgrade_UnifySameDomain(shape.wrapped, True, True, True)
        unifier.Build()
        return cq.Shape.cast(unifier.Shape())
    except Exception as exc:  # noqa: BLE001
        bluemira_warn(f"UnifySameDomain failed: {exc}")
        return shape


def _assemble_wires_from_edges(edges: list) -> list:
    """Reassemble a list of edges into as few connected wires as possible."""
    if not edges:
        return []
    seq = TopTools_HSequenceOfShape()
    for e in edges:
        seq.Append(e.wrapped)
    out = TopTools_HSequenceOfShape()
    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(seq, _OCC_DEFAULT_TOL, False, out)
    return [cq.Shape.cast(out.Value(i)) for i in range(1, out.Size() + 1)]


def _split_wire_by_closed_tools(wire: apiWire, tools: list[apiWire]) -> list[apiWire]:
    """Partition *wire* into pieces inside/outside each closed tool.

    OCC's ``cut(wire, wire)`` is a no-op when the two wires have no
    dimensional overlap. FreeCAD's ``boolean_cut`` compensates with a
    second ``BOPTools.SplitAPI.slice`` pass that adds topology wherever
    the tools' boundaries cross the argument. This helper reproduces
    that behaviour: run ``BRepAlgoAPI_Splitter`` to insert split vertices
    at tool-crossings, then classify each resulting edge as inside or
    outside the closed-tool region(s) and reassemble wires per class.

    Only the "outside" pieces are returned (mirroring the subtractive
    semantics of ``boolean_cut``), plus any pieces lying on the boundary.
    FreeCAD's impl returns *both* inside and outside pieces concatenated;
    we do the same so downstream callers like
    ``ITERGravitySupportBuilder._get_intersection_wire`` — which picks
    the shortest via ``min(key=length)`` — keep working unchanged.
    """
    # Run BOPAlgo_Splitter: inserts split vertices at every place a tool's
    # boundary crosses `wire`. Output edge count = input edges + 2 per tool
    # crossing; the curves are unchanged, only the topology is refined.
    splitter = BRepAlgoAPI_Splitter()
    args = TopTools_ListOfShape()
    args.Append(wire.wrapped)
    tools_list = TopTools_ListOfShape()
    for t in tools:
        tools_list.Append(t.wrapped)
    splitter.SetArguments(args)
    splitter.SetTools(tools_list)
    splitter.Build()
    result = cq.Shape.cast(splitter.Shape())

    # Build a planar Face per closed tool wire — the classifier works on a
    # Face's 2D parametric domain, not on a wire, so we need the region's
    # surface to test "inside vs outside".
    tool_faces = [cq.Face.makeFromWires(t).wrapped for t in tools]
    # ShapeAnalysis_Surface projects a 3D point onto the face's underlying
    # surface and returns the (u, v) parameters — needed to feed the
    # classifier, which takes parameter-space coordinates, not 3D.
    surf_analysers = [ShapeAnalysis_Surface(BRep_Tool.Surface_s(f)) for f in tool_faces]
    classifier = BRepClass_FaceClassifier()

    inside_edges: list = []
    outside_edges: list = []
    for edge in result.Edges():
        # Sample the edge at its mid-parameter: a single point is enough
        # because Splitter guarantees no edge straddles a tool boundary,
        # so every point on one edge has the same inside/outside state.
        adaptor = BRepAdaptor_Curve(edge.wrapped)
        umid = 0.5 * (adaptor.FirstParameter() + adaptor.LastParameter())
        mid3d = adaptor.Value(umid)
        is_inside_any = False
        for tool_face, analyser in zip(tool_faces, surf_analysers, strict=True):
            # Project 3D mid-point onto the tool face's surface → (u, v),
            # then classify that (u, v) against the face's trimming wires.
            uv = analyser.ValueOfUV(mid3d, _OCC_DEFAULT_TOL)
            classifier.Perform(tool_face, uv, _OCC_DEFAULT_TOL)
            # TopAbs_IN means the edge lies inside this tool's region; a
            # single inside-hit is enough (multi-tool union semantics).
            if classifier.State() == TopAbs_IN:
                is_inside_any = True
                break
        (inside_edges if is_inside_any else outside_edges).append(edge)

    # Reassemble edges of each class back into connected wires. Outside
    # first, then inside, matches FreeCAD's ordering — callers picking by
    # length index (e.g. `[-1]` or `min(key=length)`) stay consistent.
    return _assemble_wires_from_edges(outside_edges) + _assemble_wires_from_edges(
        inside_edges
    )


def _split_wire_at_tool_crossings(wire: apiWire, tools: list[apiWire]) -> list[apiWire]:
    """Partition *wire* into connected pieces separated by tool crossings.

    Open-tool variant of ``_split_wire_by_closed_tools``. Uses OCC's
    ``cq.Shape.cut`` (which on wire-vs-wire removes overlapping segments
    *and* inserts split vertices at point-crossings, producing a trimmed
    wire with refined topology) and then breaks the resulting edge chain
    into separate wires at vertices that weren't on the original wire —
    those are the tool-crossing points we want to split at. Pieces are
    returned sorted by length descending to match FreeCAD
    ``boolean_cut``'s ``Wires`` property ordering.
    """
    cut_result = wire.cut(*tools)
    all_edges = ordered_edges(cut_result) if cut_result.Edges() else []
    if not all_edges:
        return []

    orig_positions = [(v.X, v.Y, v.Z) for v in wire.Vertices()]

    def _is_original(point_tuple):
        px, py, pz = point_tuple
        return any(
            abs(px - qx) < _POINT_COINCIDENCE_TOL
            and abs(py - qy) < _POINT_COINCIDENCE_TOL
            and abs(pz - qz) < _POINT_COINCIDENCE_TOL
            for qx, qy, qz in orig_positions
        )

    # Walk the cut result's edges in connectivity order and start a new
    # wire whenever an edge ends at a non-original vertex (a tool crossing).
    # The final edge always ends at an original terminus, so we don't break
    # there.
    groups: list[list] = []
    current: list = []
    for i, edge in enumerate(all_edges):
        current.append(edge)
        end_pt = edge.endPoint()
        end_tuple = (end_pt.x, end_pt.y, end_pt.z)
        is_last = i == len(all_edges) - 1
        if not is_last and not _is_original(end_tuple):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    wires: list = []
    for g in groups:
        wires.extend(_assemble_wires_from_edges(g))
    wires.sort(key=lambda w: w.Length(), reverse=True)
    return wires


def boolean_cut(shape: apiShape, tools: list, *, split: bool = True) -> list[apiShape]:
    """Boolean subtraction — return list of result shapes."""
    if not isinstance(tools, list):
        tools = [tools]

    # For a 1-D wire argument vs. wire tools, OCC's raw ``cut`` has no
    # dimensional overlap to work with and returns the wire unchanged. FreeCAD
    # follows its ``cut`` with ``BOPTools.SplitAPI.slice(mode="Split")`` which
    # adds topology at tool-crossings and partitions the wire into pieces.
    # Mirror that via Splitter: classify inside/outside when every tool is a
    # closed region, otherwise just break the wire at each tool-crossing
    # vertex (open-tool case — no region to be inside/outside of).
    if (
        split
        and isinstance(shape, apiWire)
        and all(isinstance(t, apiWire) for t in tools)
    ):
        if all(t.IsClosed() for t in tools):
            return _split_wire_by_closed_tools(shape, tools)
        return _split_wire_at_tool_crossings(shape, tools)

    result = shape.cut(*tools)
    # OCC's BRepAlgoAPI_Cut can return nested compounds, and cq.Shape.Solids()
    # / .Faces() / .Wires() only inspect the immediate children. Walk the
    # whole tree instead so we recover sub-shapes buried one level deeper.
    if isinstance(shape, apiSolid):
        return _collect_subshapes(result, cq.Solid)
    if isinstance(shape, apiFace):
        return _collect_subshapes(result, cq.Face)
    if isinstance(shape, apiWire):
        return _collect_subshapes(result, cq.Wire)
    return [result]


def face_cut_holes(face: apiFace, holes: list) -> list:
    """Cut hole faces out of an outer face.

    Parity with ``_freecadapi.face_cut_holes``: no coplanar guard — callers
    guarantee the input wires are coplanar by construction.
    """
    return boolean_cut(face, holes, split=False)


_TOPABS_FOR_KIND: dict = {
    cq.Solid: TopAbs_SOLID,
    cq.Face: TopAbs_FACE,
    cq.Wire: TopAbs_WIRE,
}


def _collect_subshapes(shape: apiShape, kind: type) -> list:
    """Recursively collect all sub-shapes of *shape* that are instances of *kind*.

    CadQuery's ``.Solids()``/``.Faces()``/``.Wires()`` only walk the immediate
    children of a compound; OCC boolean operations occasionally yield compounds
    nested more than one level deep, which then appear empty. ``TopExp_Explorer``
    traverses the full topology tree and recovers all leaves of the requested
    kind.

    Special case for ``cq.Solid``: OCC boolean cuts can return a Shell inside a
    Compound (no ``TopoDS_Solid`` wrapper) when the result is a closed volume
    boundary but not yet a solid. Promote such shells via ``BRepBuilderAPI_MakeSolid``
    so callers that asked for solids get the volume they expect.
    """
    explorer = TopExp_Explorer(shape.wrapped, _TOPABS_FOR_KIND[kind])
    collected: list = []
    while explorer.More():
        collected.append(kind(explorer.Current()))
        explorer.Next()

    if collected or kind is not cq.Solid:
        return collected

    shell_exp = TopExp_Explorer(shape.wrapped, TopAbs_SHELL)
    while shell_exp.More():
        shell = cq.Shell(shell_exp.Current())
        maker = BRepBuilderAPI_MakeSolid(shell.wrapped)
        if maker.IsDone():
            promoted = cq.Solid(maker.Solid())
            # The freshly-built solid can still fail BRepCheck (orientation of
            # the shell not yet aligned as a solid boundary); ShapeFix_Shape
            # reliably repairs it.
            if not promoted.isValid():
                fix_shape(promoted)
            collected.append(promoted)
        shell_exp.Next()
    return collected


def boolean_fragments(shapes: list, tolerance: float = 0.0) -> tuple[apiCompound, list]:
    """Split shapes into their Boolean fragments (general fuse).

    Returns
    -------
    compound:
        A compound of all unique fragments
    fragment_map:
        List of lists — fragment_map[i] contains the fragments that originated
        from shapes[i]
    """
    args = TopTools_ListOfShape()
    for s in shapes:
        args.Append(s.wrapped)

    algo = BRepAlgoAPI_BuilderAlgo()
    algo.SetArguments(args)
    if tolerance > 0.0:
        algo.SetFuzzyValue(tolerance)
    algo.Build()

    if not algo.IsDone():
        raise FreeCADError("Boolean fragments operation failed")

    compound = cq.Shape.cast(algo.Shape())

    # Build fragment map: for each input, collect its Modified/Generated outputs.
    # If a shape is unmodified (no intersection), its list is empty — matching
    # FreeCAD's generalFuse behaviour where unmodified shapes yield [].
    fragment_map = []
    for s in shapes:
        frags = [cq.Shape.cast(t) for t in algo.Modified(s.wrapped)]
        if not frags:
            frags = [cq.Shape.cast(t) for t in algo.Generated(s.wrapped)]
        fragment_map.append(frags)

    return compound, fragment_map


# ---------------------------------------------------------------------------
# Loft / slice / point-in-shape
# ---------------------------------------------------------------------------


def loft(
    profiles,
    *,
    solid: bool = False,
    ruled: bool = False,
    closed: bool = False,
) -> apiShape:
    """Loft through a sequence of profiles."""
    result = cq.Solid.makeLoft(list(profiles), ruled=ruled)
    if not solid:
        return cq.Shell(result.Shells()[0].wrapped)
    return result


def _repair_closed_wire(wire: apiWire) -> apiWire:
    """Ensure *wire*'s TopoDS "Closed" flag matches its geometry.

    OCC's ``TopoDS_Wire`` carries an explicit Closed flag that is not always
    set by algorithms that produce wires (notably ``Face.outerWire()`` on the
    result of a boolean section). A geometrically closed wire whose flag is
    unset reports ``IsClosed() == False`` and breaks downstream checks
    (e.g. sweep's "cannot mix open and closed profiles"). ``ShapeFix_Wire``
    reconnects shared vertices and sets the Closed flag if appropriate.

    Skip the repair when the wire already reports closed — running ShapeFix
    on an already-good wire can produce a null-handle output on some OCC
    builds, leading to segfaults downstream.
    """
    if wire.IsClosed():
        return wire
    fixer = ShapeFix_Wire()
    fixer.Load(wire.wrapped)
    fixer.FixReorder()
    fixer.FixConnected()
    fixer.FixClosed()
    fixed = fixer.Wire()
    if fixed.IsNull():
        return wire
    return cq.Wire(fixed)


def slice_shape(shape: apiShape, plane_origin, plane_axis):
    """Slice a shape with a plane.

    For wires returns a numpy array of intersection points (N, 3).
    For solids/shells returns a list of closed intersection wires (outer wires
    of the cross-section faces).
    For faces returns the section curves as wires.
    """
    pln = gp_Pln(
        gp_Pnt(*[float(x) for x in plane_origin]),
        gp_Dir(*[float(x) for x in plane_axis]),
    )
    plane_face = BRepBuilderAPI_MakeFace(pln, -1e6, 1e6, -1e6, 1e6).Face()

    # --- Wire input: intersection points -------------------------------------
    if isinstance(shape, apiWire):
        section = BRepAlgoAPI_Section(shape.wrapped, plane_face)
        section.ComputePCurveOn1(True)
        section.Approximation(True)
        section.Build()
        if not section.IsDone() or section.Shape().IsNull():
            return np.empty((0, 3))
        verts = cq.Shape.cast(section.Shape()).Vertices()
        if not verts:
            return np.empty((0, 3))
        return np.array([_vector_to_numpy(v.Center()) for v in verts])

    # --- Solid input: cross-section faces -> closed outer wires --------------
    # Use boolean intersection with the plane face: OCC returns proper
    # cross-section faces whose outer wires are closed by construction,
    # matching FreeCAD's Part.Solid.slice behaviour.
    # (Compound inputs are intentionally not routed through this path —
    # BRepAlgoAPI_Common on a Compound+Face can leak the plane face itself
    # into the result; the Section-edge fallback below handles Compounds
    # robustly.)
    if isinstance(shape, cq.Solid):
        cq_plane = cq.Face(plane_face)
        try:
            intersection = shape.intersect(cq_plane)
        except Exception:  # noqa: BLE001
            intersection = None
        if intersection is not None and not intersection.wrapped.IsNull():
            section_wires: list = []
            for f in intersection.Faces():
                section_wires.append(_repair_closed_wire(f.outerWire()))
                # Inner wires represent holes in the cross-section face and
                # are legitimate section contours in their own right
                # (e.g. the inner rim of a donut cross-section).
                section_wires.extend(_repair_closed_wire(w) for w in f.innerWires())
            if section_wires:
                return section_wires

    # --- Face input (or Solid fallback): assemble section edges into wires ---
    section = BRepAlgoAPI_Section(shape.wrapped, plane_face)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()
    if not section.IsDone() or section.Shape().IsNull():
        return []

    result = cq.Shape.cast(section.Shape())
    wires = result.Wires()
    if wires:
        return wires

    edges = result.Edges()
    if not edges:
        return []

    edge_seq = TopTools_HSequenceOfShape()
    for e in edges:
        edge_seq.Append(e.wrapped)
    result_wires_seq = TopTools_HSequenceOfShape()
    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
        edge_seq, 1e-4, False, result_wires_seq
    )
    assembled = [
        cq.Shape.cast(result_wires_seq.Value(i))
        for i in range(1, result_wires_seq.Size() + 1)
    ]
    # For a Solid or Compound-of-Solids input, the cross-section must be a
    # closed curve by construction. If a wire comes back open, the upstream
    # solid had a torn shell; close the loop with a synthetic straight edge
    # so downstream face creation / multi-section sweeps can still proceed.
    # Shells, by contrast, legitimately produce open section wires — the
    # intersection of a plane with a shell wall is a line, not a loop — so
    # those must be passed through unchanged.
    if isinstance(shape, cq.Solid) or (
        isinstance(shape, cq.Compound) and shape.Solids()
    ):
        assembled = [_force_close_wire(w) for w in assembled]
    return assembled


def _force_close_wire(wire: apiWire) -> apiWire:
    """Close *wire* geometrically by appending a line edge from end to start."""
    if wire.IsClosed():
        return wire
    try:
        start = wire.startPoint()
        end = wire.endPoint()
    except Exception:  # noqa: BLE001
        return wire
    gap = (end - start).Length
    if gap < _POINT_COINCIDENCE_TOL:
        # already coincident — just repair the Closed flag
        return _repair_closed_wire(wire)
    bridge = cq.Edge.makeLine(end, start)
    edges = [*wire.Edges(), bridge]
    fixer = ShapeFix_Wire()
    fixer.Load(cq.Wire.assembleEdges(edges).wrapped)
    fixer.SetPrecision(max(_OCC_DEFAULT_TOL, gap * 0.5))
    fixer.SetMaxTolerance(max(1e-3, gap * 2.0))
    fixer.FixReorder()
    fixer.FixConnected()
    fixer.FixClosed()
    return cq.Wire(fixer.Wire())


def point_inside_shape(point, shape: apiShape) -> bool:
    """Return True if *point* is inside *shape*."""
    pnt = gp_Pnt(*[float(x) for x in point])

    if isinstance(shape, apiFace):
        # Project point onto the face's surface, then classify in UV space
        surf = BRep_Tool.Surface_s(shape.wrapped)
        projector = GeomAPI_ProjectPointOnSurf(pnt, surf)
        projector.Perform(pnt)
        if projector.NbPoints() == 0:
            return False
        u, v = projector.LowerDistanceParameters()
        classifier = BRepClass_FaceClassifier()
        classifier.Perform(shape.wrapped, gp_Pnt2d(u, v), _OCC_DEFAULT_TOL)
        return classifier.State() == TopAbs_IN

    classifier = BRepClass3d_SolidClassifier(shape.wrapped)
    classifier.Perform(pnt, _OCC_DEFAULT_TOL)
    return classifier.State() == TopAbs_IN


def serialise_shape(shape: apiWire) -> dict:
    """Serialise a CadQuery wire to a dict compatible with deserialise_shape."""
    edges = ordered_edges(shape)
    serialised = []
    for e in edges:
        adaptor = BRepAdaptor_Curve(e.wrapped)
        ctype = adaptor.GetType()
        if ctype == GeomAbs_Line:
            p0 = _vector_to_numpy(e.startPoint()).tolist()
            p1 = _vector_to_numpy(e.endPoint()).tolist()
            serialised.append({"LineSegment": {"StartPoint": p0, "EndPoint": p1}})
        elif ctype == GeomAbs_Circle:
            circ = adaptor.Circle()
            center = [circ.Location().X(), circ.Location().Y(), circ.Location().Z()]
            radius = circ.Radius()
            ax = circ.Axis().Direction()
            axis = [ax.X(), ax.Y(), ax.Z()]
            # XAxis is required to round-trip arc orientation (see _freecad_ax2).
            xax = circ.XAxis().Direction()
            x_axis = [xax.X(), xax.Y(), xax.Z()]
            t0 = adaptor.FirstParameter()
            t1 = adaptor.LastParameter()
            start_angle = math.degrees(t0)
            end_angle = math.degrees(t1)
            serialised.append({
                "ArcOfCircle": {
                    "Radius": radius,
                    "Center": center,
                    "StartAngle": start_angle,
                    "EndAngle": end_angle,
                    "Axis": axis,
                    "XAxis": x_axis,
                }
            })
        elif ctype == GeomAbs_BSplineCurve:
            bsp = adaptor.BSpline()
            poles = [
                [bsp.Pole(i).X(), bsp.Pole(i).Y(), bsp.Pole(i).Z()]
                for i in range(1, bsp.NbPoles() + 1)
            ]
            knots = [bsp.Knot(i) for i in range(1, bsp.NbKnots() + 1)]
            mults = [bsp.Multiplicity(i) for i in range(1, bsp.NbKnots() + 1)]
            weights = [bsp.Weight(i) for i in range(1, bsp.NbPoles() + 1)]
            serialised.append({
                "BSplineCurve": {
                    "Poles": poles,
                    "Knots": knots,
                    "Mults": mults,
                    "Degree": bsp.Degree(),
                    "Weights": weights,
                    "isPeriodic": bsp.IsPeriodic(),
                    "checkRational": bsp.IsRational(),
                }
            })
        elif ctype == GeomAbs_BezierCurve:
            bez = adaptor.Bezier()
            poles = [
                [bez.Pole(i).X(), bez.Pole(i).Y(), bez.Pole(i).Z()]
                for i in range(1, bez.NbPoles() + 1)
            ]
            serialised.append({"BezierCurve": {"Poles": poles}})
        elif ctype == GeomAbs_Ellipse:
            ell = adaptor.Ellipse()
            center = [ell.Location().X(), ell.Location().Y(), ell.Location().Z()]
            ax1 = ell.XAxis().Direction()
            major_ax = [ax1.X(), ax1.Y(), ax1.Z()]
            ax2 = ell.YAxis().Direction()
            minor_ax = [ax2.X(), ax2.Y(), ax2.Z()]
            t0 = adaptor.FirstParameter()
            t1 = adaptor.LastParameter()
            sp = e.startPoint()
            ep = e.endPoint()
            # OCC stores ellipses with the full Axis, we derive focus from semi-axes
            a, b = ell.MajorRadius(), ell.MinorRadius()
            c = math.sqrt(abs(a**2 - b**2))
            f1 = [
                center[0] + c * major_ax[0],
                center[1] + c * major_ax[1],
                center[2] + c * major_ax[2],
            ]
            serialised.append({
                "ArcOfEllipse": {
                    "Center": center,
                    "MajorRadius": a,
                    "MinorRadius": b,
                    "MajorAxis": major_ax,
                    "MinorAxis": minor_ax,
                    "StartAngle": math.degrees(t0),
                    "EndAngle": math.degrees(t1),
                    "Focus1": f1,
                    "StartPoint": [sp.x, sp.y, sp.z],
                    "EndPoint": [ep.x, ep.y, ep.z],
                }
            })
        else:
            raise NotImplementedError(f"serialise_shape: unsupported curve type {ctype}")
    return {"Wire": serialised}


def deserialise_shape(buffer: dict) -> apiWire:
    """Deserialise a dict (from serialise_shape / FreeCAD format) to a CadQuery wire."""
    # Local import: ``_curves`` is loaded after ``_core`` by ``__init__``, but
    # this function is only called at runtime, so a deferred import is enough
    # to keep ``_curves`` self-contained (no module-level dep on ``_core``).
    from bluemira.codes._cadqueryapi._curves import (  # noqa: PLC0415
        make_bezier,
        make_bspline,
        make_circle,
        make_ellipse,
    )

    for type_, v in buffer.items():
        if type_ == "Wire":
            edges = [deserialise_shape(item) for item in v]
            return wire_from_wires(edges)
        if type_ == "LineSegment":
            return make_polygon([v["StartPoint"], v["EndPoint"]])
        if type_ == "BezierCurve":
            return make_bezier(v["Poles"])
        if type_ == "BSplineCurve":
            return make_bspline(
                v["Poles"],
                v["Mults"],
                v["Knots"],
                periodic=v["isPeriodic"],
                degree=v["Degree"],
                weights=v["Weights"],
                check_rational=v["checkRational"],
            )
        if type_ == "ArcOfCircle":
            return make_circle(
                v["Radius"],
                v["Center"],
                v["StartAngle"],
                v["EndAngle"],
                v["Axis"],
                x_direction=v.get("XAxis"),
            )
        if type_ == "ArcOfEllipse":
            return make_ellipse(
                v["Center"],
                v["MajorRadius"],
                v["MinorRadius"],
                v["MajorAxis"],
                v["MinorAxis"],
                v["StartAngle"],
                v["EndAngle"],
            )
        raise NotImplementedError(f"deserialise_shape: unsupported type {type_!r}")
    return None


def fillet_wire_2D(wire: apiWire, radius: float, *, chamfer: bool = False) -> apiWire:
    """Fillet or chamfer a planar wire using OCC's BRepFilletAPI_MakeFillet2d."""
    is_closed = wire.IsClosed()
    work_wire = wire
    close_edge = None

    if not is_closed:
        # Add a dummy closing edge so BRepFilletAPI_MakeFillet2d can work on a face.
        wire_edges = ordered_edges(wire)
        start_pt = _vector_to_numpy(wire_edges[0].startPoint())
        end_pt = _vector_to_numpy(wire_edges[-1].endPoint())
        close_edge = cq.Edge.makeLine(cq.Vector(*end_pt), cq.Vector(*start_pt))
        work_wire = cq.Wire.assembleEdges([*wire_edges, close_edge])

    face = cq.Face.makeFromWires(work_wire)
    builder = BRepFilletAPI_MakeFillet2d(face.wrapped)

    exp = BRepTools_WireExplorer(work_wire.wrapped)
    all_edges = []
    while exp.More():
        all_edges.append(exp.Current())
        exp.Next()
    n = len(all_edges)

    if chamfer:
        # For open wires, only chamfer the n_orig-1 internal vertex pairs.
        # all_edges includes the dummy closing edge for open wires, so iterate only
        # the original edge pairs (first n_orig edges).
        n_orig = n if is_closed else (n - 1)  # original edge count (without dummy)
        n_pairs = n_orig if is_closed else n_orig - 1
        for i in range(n_pairs):
            e1_occ = all_edges[i]
            e2_occ = all_edges[(i + 1) % n]
            # Skip tangent (collinear) edge pairs — AddChamfer segfaults on them
            t1 = _vector_to_numpy(cq.Edge(e1_occ).tangentAt(1.0))
            t2 = _vector_to_numpy(cq.Edge(e2_occ).tangentAt(0.0))
            if np.linalg.norm(np.cross(t1, t2)) < _OCC_DEFAULT_TOL:
                continue
            builder.AddChamfer(e1_occ, e2_occ, radius, radius)
    else:
        # Skip endpoint vertices for open wires (they can't be filleted).
        skip_keys: set[tuple] = set()
        if not is_closed:
            sp = _vector_to_numpy(wire_edges[0].startPoint())
            ep = _vector_to_numpy(wire_edges[-1].endPoint())
            skip_keys = {
                tuple(round(c, 8) for c in sp),
                tuple(round(c, 8) for c in ep),
            }

        seen: set[tuple] = set()
        exp_v = TopExp_Explorer(wire.wrapped, TopAbs_VERTEX)
        while exp_v.More():
            v = TopoDS.Vertex_s(exp_v.Current())
            pt = BRep_Tool.Pnt_s(v)
            key = (round(pt.X(), 8), round(pt.Y(), 8), round(pt.Z(), 8))
            if key not in seen and key not in skip_keys:
                seen.add(key)
                builder.AddFillet(v, radius)
            exp_v.Next()

    builder.Build()
    if not builder.IsDone():
        # No fillet possible (e.g. tangent edges) — return wire unchanged
        return wire

    result = cq.Shape.cast(builder.Shape())
    result_wires = result.Wires()
    if not result_wires:
        return wire
    result_wire = result_wires[0]

    if is_closed:
        return result_wire

    # Open wire: remove the dummy closing edge from the result.
    # The dummy edge goes from end_pt to start_pt and was not at any filleted corner,
    # so it should appear unchanged in the result.
    result_edges = ordered_edges(result_wire)
    keep = []
    for e in result_edges:
        sp = _vector_to_numpy(e.startPoint())
        ep = _vector_to_numpy(e.endPoint())
        tol = radius + 1e-3
        if np.allclose(sp, end_pt, atol=tol) and np.allclose(ep, start_pt, atol=tol):
            continue  # this is the dummy closing edge
        keep.append(e)
    if not keep:
        return wire
    return cq.Wire.assembleEdges(keep)


def _piece_mass(piece: apiShape) -> float:
    """Mass of *piece* — volume for solids, area for shells/faces, length for
    wires/edges. Matches FreeCAD's ``shapeOfMaxSize`` size metric.
    """
    props = GProp_GProps()
    if isinstance(piece, (cq.Solid, cq.Compound)):
        BRepGProp.VolumeProperties_s(piece.wrapped, props)
    elif isinstance(piece, (cq.Face, cq.Shell)):
        BRepGProp.SurfaceProperties_s(piece.wrapped, props)
    else:
        BRepGProp.LinearProperties_s(piece.wrapped, props)
    return abs(props.Mass())


def _pick_dominant_dangler(pieces: list[apiShape], source_idx: int) -> apiShape:
    """Largest piece by mass; raises on a near-tie.

    Mirrors FreeCAD's ``shapeOfMaxSize`` — the connect algorithm assumes a
    single dominant main body per input. Two equal-size danglers typically
    signal symmetric through-drilling (an input pierced clean through by
    another), where silently keeping one would drop the other.
    """
    rel_tol = 1e-8
    masses = [_piece_mass(p) for p in pieces]
    max_mass = max(masses)
    ties = sum(
        1 for m in masses if (1 - rel_tol) * max_mass <= m <= (1 + rel_tol) * max_mass
    )
    if ties > 1:
        raise GeometryError(
            f"join_connect: input {source_idx} has {ties} equally-sized "
            "dangling pieces after general fuse — cannot pick a unique "
            "main body. This typically means an input is symmetrically "
            "through-drilled by another; the algorithm's assumption of a "
            "single dominant body per input is violated."
        )
    return pieces[masses.index(max_mass)]


def _piece_source_map(
    fragment_map: list[list[apiShape]],
) -> list[tuple[apiShape, set[int]]]:
    """Unique piece → set of source indices, via topological ``IsSame``."""
    unique: list[tuple[apiShape, set[int]]] = []
    for i, frags in enumerate(fragment_map):
        for frag in frags:
            match = next(
                (
                    idx
                    for idx, (p, _) in enumerate(unique)
                    if p.wrapped.IsSame(frag.wrapped)
                ),
                None,
            )
            if match is None:
                unique.append((frag, {i}))
            else:
                unique[match][1].add(i)
    return unique


def _grow_keepers_by_overlap(
    keepers: list[apiShape],
    unique_pieces: list[tuple[apiShape, set[int]]],
) -> None:
    """Add shared-overlap pieces that touch already-kept pieces, layer by layer
    (pieces with N sources grow from pieces with N-1 sources). Mutates
    *keepers* in place.
    """
    max_overlap = max((len(src) for _, src in unique_pieces), default=0)
    touch_test = keepers.copy()
    for ii in range(2, max_overlap + 1):
        ii_pieces = [p for p, src in unique_pieces if len(src) == ii]
        additions = [
            p for p in ii_pieces if any(_shapes_touch(p, k) for k in touch_test)
        ]
        if not additions:
            break
        for p in additions:
            if not any(p.wrapped.IsSame(k.wrapped) for k in keepers):
                keepers.append(p)
        touch_test = additions


def _shapes_touch(a: apiShape, b: apiShape, tol: float = 1e-6) -> bool:
    """Two shapes touch iff their minimum distance is below *tol*."""
    dist = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    dist.Perform()
    return bool(dist.IsDone()) and dist.Value() < tol


def join_connect(shapes: list, dist_tolerance: float = 1e-4) -> apiShape:
    """Connect the interiors of walled objects (pipes/shells).

    Mirrors FreeCAD's ``BOPTools.JoinAPI.connect`` (same algorithm, translated
    to OCP primitives). The motivation: a plain boolean fuse on two overlapping
    hollow tubes leaves the "plug" — the wall material of one tube that blocks
    the other tube's interior. ``connect`` removes those plugs.

    Algorithm (matches FreeCAD's Python source in ``BOPTools/JoinAPI.py``):

    1. General fuse of all inputs via :func:`boolean_fragments`. The result is a
       set of non-intersecting sub-pieces, each tagged with the inputs it came
       from.
    2. For each input, among its "dangling" pieces (pieces belonging only to
       that input), keep the largest. The plug is a smaller dangler of the
       penetrated tube and gets discarded here.
    3. Grow the keeper set by shared-overlap count: pieces belonging to exactly
       N sources are added iff they touch a piece already kept. Ensures the
       fused walls stay a connected solid at the joint.
    4. Union the keepers back into a single shape.
    """
    if not isinstance(shapes, list):
        raise TypeError(f"{shapes} is not a list.")
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    _, fragment_map = boolean_fragments(shapes, dist_tolerance)
    unique_pieces = _piece_source_map(fragment_map)

    keepers: list[apiShape] = []
    for i in range(len(shapes)):
        danglers = [p for p, src in unique_pieces if src == {i}]
        if danglers:
            keepers.append(_pick_dominant_dangler(danglers, i))

    _grow_keepers_by_overlap(keepers, unique_pieces)

    if not keepers:
        raise GeometryError("join_connect: no kept pieces after general fuse")
    if len(keepers) == 1:
        result = keepers[0]
    else:
        # Skip ``.clean()``/UnifySameDomain here: it reprojects coplanar faces
        # onto a fitted canonical surface, which shifts the volume by ~0.1 %
        # on curved pipe geometry and diverges from FreeCAD's JoinAPI output.
        result = boolean_fuse(list(keepers), remove_splitter=False)

    if result is None or not is_valid(result):
        raise GeometryError("join_connect: boolean union failed")
    return result


__all__ = [
    "apiFace",
    "area",
    "arrange_edges",
    "boolean_cut",
    "boolean_fragments",
    "boolean_fuse",
    "bounding_box",
    "catch_caderr",
    "center_of_mass",
    "close_wire",
    "deserialise_shape",
    "discretise",
    "discretise_by_edges",
    "dist_to_shape",
    "eccentricity",
    "edge_tangent_at",
    "edges",
    "end_point",
    "extrude_shape",
    "face_cut_holes",
    "faces",
    "fillet_wire_2D",
    "fix_shape",
    "interpolate_bspline",
    "is_closed",
    "is_null",
    "is_same",
    "is_valid",
    "join_connect",
    "length",
    "loft",
    "make_face",
    "make_polygon",
    "make_shell",
    "make_solid",
    "mirror_shape",
    "normal_at",
    "offset_wire",
    "optimal_bounding_box",
    "ordered_edges",
    "ordered_vertexes",
    "orientation",
    "point_inside_shape",
    "reverse_shape",
    "revolve_shape",
    "rotate_shape",
    "scale_shape",
    "serialise_shape",
    "shells",
    "slice_shape",
    "solids",
    "split_wire",
    "start_point",
    "sweep_shape",
    "translate_shape",
    "vertexes",
    "volume",
    "wire_closure",
    "wire_from_edges",
    "wire_from_wires",
    "wire_parameter_at",
    "wire_value_at",
    "wires",
]
