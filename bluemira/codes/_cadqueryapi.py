# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CadQuery backend for bluemira — experimental prototype.

Implements the same public interface as _freecadapi.py using CadQuery's
free-function / direct Shape API (no Workplane state).

Not yet covered:
  - Affine placements (stub objects exist for import compatibility)
  - FreeCADGui-based show_cad (delegated to polyscope instead)
  - join_connect with correct internal-wall removal (see its docstring TODO)
"""

from __future__ import annotations

import contextlib
import enum
import math
from collections import UserList
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_BuilderAlgo, BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
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
from OCP.GC import GC_MakeArcOfCircle
from OCP.GProp import GProp_GProps
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface, Geom_BezierCurve
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Line,
)
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Wire
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
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
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_HSequenceOfShape, TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.gp import (
    gp_Ax1,
    gp_Ax2,
    gp_Circ,
    gp_Dir,
    gp_Pln,
    gp_Pnt,
    gp_Pnt2d,
    gp_Trsf,
    gp_Vec,
)

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import FreeCADError, InvalidCADInputsError
from bluemira.geometry.error import GeometryError
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.display.palettes import ColorPalette

# ---------------------------------------------------------------------------
# Type aliases — mirror the names used throughout bluemira so that
# isinstance() checks and type annotations keep working.
# ---------------------------------------------------------------------------
apiVertex = cq.Vertex
apiVector = cq.Vector
apiEdge = cq.Edge
apiWire = cq.Wire
apiShell = cq.Shell
apiSolid = cq.Solid
apiShape = cq.Shape
apiCompound = cq.Compound

# ---------------------------------------------------------------------------
# Numerical tolerances used throughout the OCC / CadQuery interop layer.
# ---------------------------------------------------------------------------
#: Tolerance for collapsing tiny floating-point residuals to zero (matrix
#: cleanup, axis component snapping).
_GEOM_NEAR_ZERO_TOL = 1e-12
#: Generic angular / cross-product tolerance for "is this rotation effectively
#: zero" or "are two vectors parallel" checks.
_ANGLE_PARALLEL_TOL = 1e-10
#: Generic point-coincidence / parameter-equality tolerance.
_POINT_COINCIDENCE_TOL = 1e-9
#: Default tolerance for OCC algorithms that take a precision argument
#: (``BRepClass3d``, classifiers, edge length filters, sewing).
_OCC_DEFAULT_TOL = 1e-6
#: Threshold for selecting an alternative reference axis when the natural
#: choice is too close to the input direction.
_AXIS_DOMINANCE_TOL = 0.9


class _apiFaceMeta(type):
    """Metaclass so ``isinstance(x, apiFace)`` works for plain cq.Face objects."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, cq.Face)


class apiFace(metaclass=_apiFaceMeta):
    """Drop-in for ``cq.Face``.

    Calling ``apiFace(wire)`` with a ``cq.Wire`` uses ``makeFromWires``
    instead of the raw OCC constructor that FreeCAD's ``Part.Face(wire)`` used.
    """

    def __new__(cls, obj=None):
        if isinstance(obj, cq.Wire):
            return cq.Face.makeFromWires(obj)
        if isinstance(obj, (list, tuple)):
            wires = list(obj)
            outer = wires[0]
            inner = wires[1:]
            return cq.Face.makeFromWires(outer, inner)
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


def sweep_shape(
    profiles: Iterable[apiWire],
    path: apiWire,
    *,
    solid: bool = True,
    frenet: bool = True,
    transition: int = 0,
) -> apiShell | apiSolid:
    """Sweep one or more *profiles* along *path*."""
    if not isinstance(profiles, list):
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


# ---------------------------------------------------------------------------
# Tessellation / visualisation helpers
# ---------------------------------------------------------------------------


def tessellate(obj: apiShape, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Tessellate *obj* to a triangle mesh.

    Returns
    -------
    vertices:
        Float array of shape (N, 3).
    indices:
        Int array of shape (M, 3).
    """
    if tolerance <= 0.0:
        raise ValueError("Tolerance must be greater than 0.0")
    verts, tris = obj.tessellate(tolerance)
    return (
        np.array([[v.x, v.y, v.z] for v in verts], dtype=float),
        np.array(tris, dtype=int),
    )


def collect_verts_faces(
    solid: apiShape, tesselation: float = 0.1
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract tessellated vertices and face indices for polyscope display."""
    all_verts = []
    all_faces = []
    voffset = 0

    faces = solid.Faces()
    for face in faces:
        verts, tris = face.tessellate(tesselation)
        if verts:
            v_arr = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
            f_arr = np.array(tris, dtype=int) + voffset
            all_verts.append(v_arr)
            all_faces.append(f_arr)
            voffset += len(verts)

    if not all_verts:
        return None, None
    return np.vstack(all_verts), np.vstack(all_faces)


def collect_wires(
    solid: apiShape, deflection: float = 0.01, **_kwds
) -> tuple[np.ndarray, np.ndarray]:
    """Extract discretised wire vertices and edge indices for polyscope display.

    Parameters
    ----------
    deflection:
        Maximum chord-height deviation; controls point density per wire.
        (Polyscope passes this as ``Deflection=`` — absorbed by ``**_kwds``.)
    """
    all_verts = []
    all_edges = []
    voffset = 0

    for wire in solid.Wires():
        # Sample N points proportional to length; at least 10.
        n = max(10, int(wire.Length() / deflection))
        pts = [_vector_to_numpy(wire.positionAt(t)) for t in np.linspace(0.0, 1.0, n)]
        pts_arr = np.array(pts, dtype=float)
        seg_idx = np.arange(voffset, voffset + n - 1)
        all_verts.append(pts_arr)
        all_edges.append(np.column_stack([seg_idx, seg_idx + 1]))
        voffset += n

    return np.vstack(all_verts), np.vstack(all_edges)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


@dataclass
class DefaultDisplayOptions:
    """CadQuery backend display options (delegated to polyscope)."""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    material: str = "wax"
    tesselation: float = 0.05
    wires_on: bool = False
    wire_radius: float = 0.001
    smooth: bool = True

    @property
    def color(self) -> str:
        """See colour."""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour."""
        self.colour = value


def show_cad(
    parts: apiShape | list[apiShape],
    part_options: list[dict],
    labels: list[str],
    **kwargs,
):
    """
    Display CadQuery shapes via polyscope.

    Delegates to _polyscope.show_cad after swapping in our own
    collect_verts_faces / collect_wires implementations.
    """
    # Temporarily patch the collect helpers polyscope uses so that it calls
    # our CadQuery-aware versions instead of the FreeCAD ones. Imports are
    # local to avoid pulling in FreeCAD at module-load time when the user
    # has selected the cadquery backend.
    import bluemira.codes._freecadapi as _orig_cadapi  # noqa: PLC0415
    from bluemira.codes import _polyscope as ps_backend  # noqa: PLC0415

    _orig_collect_verts = _orig_cadapi.collect_verts_faces
    _orig_collect_wires = _orig_cadapi.collect_wires

    try:
        _orig_cadapi.collect_verts_faces = collect_verts_faces
        _orig_cadapi.collect_wires = collect_wires
        ps_backend.show_cad(parts, part_options, labels, **kwargs)
    finally:
        _orig_cadapi.collect_verts_faces = _orig_collect_verts
        _orig_cadapi.collect_wires = _orig_collect_wires


# ---------------------------------------------------------------------------
# Extra type aliases expected by bluemira.geometry
# ---------------------------------------------------------------------------
# CadQuery has no BSplineSurface public type; cq.Shape is the closest proxy.
apiSurface = cq.Shape


class _Vector:
    """Minimal FreeCAD Base.Vector stand-in."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        # Allow construction from a single iterable (e.g. numpy array, list, tuple)
        if hasattr(x, "__iter__"):
            x, y, z = x
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return f"_Vector({self.x}, {self.y}, {self.z})"

    def __array__(self, dtype=None):  # noqa: PLW3201 - numpy interop hook
        arr = np.array([self.x, self.y, self.z])
        return arr if dtype is None else arr.astype(dtype)

    def __eq__(self, other):
        if not isinstance(other, _Vector):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))


class _Rotation:
    def __init__(self, axis=(0, 0, 1), angle=0.0):
        self.Axis = _Vector(*axis)
        self.Angle = angle


class _HomogeneousMatrix:
    """Wraps a 4x4 numpy array; exposes `.A` as a flat list (FreeCAD Matrix.A API)."""

    def __init__(self, m: np.ndarray):
        self._m = m

    @property
    def A(self) -> list[float]:
        return list(self._m.flatten())


class _CQPlacement:
    """
    Minimal stand-in for FreeCAD's Base.Placement.

    Angle is stored internally in **radians** (matching FreeCAD's
    ``Placement.Rotation.Angle`` which returns radians).  The public
    API ``make_placement`` accepts degrees and converts on entry.
    """

    def __init__(self, base=(0, 0, 0), axis=(0, 0, 1), angle_rad=0.0):
        self.Base = _Vector(*base)
        # Normalize axis (mirrors FreeCAD which always stores a unit vector).
        _ax = np.asarray(axis, dtype=float)
        _norm = np.linalg.norm(_ax)
        if _norm > 0:
            _ax /= _norm
        self.Rotation = _Rotation(tuple(_ax), angle_rad)

    @property
    def Matrix(self) -> _HomogeneousMatrix:
        """4x4 homogeneous transformation matrix (FreeCAD Placement.Matrix API)."""
        R = self._rot_matrix()
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = [self.Base.x, self.Base.y, self.Base.z]
        return _HomogeneousMatrix(m)

    def _rot_matrix(self) -> np.ndarray:
        """3x3 rotation matrix via Rodrigues' formula (angle in radians)."""
        k = np.array(
            [self.Rotation.Axis.x, self.Rotation.Axis.y, self.Rotation.Axis.z],
            dtype=float,
        )
        norm = np.linalg.norm(k)
        if norm < _GEOM_NEAR_ZERO_TOL:
            return np.eye(3)
        k /= norm
        theta = self.Rotation.Angle  # already in radians
        c, s = math.cos(theta), math.sin(theta)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)

    def multVec(self, vec) -> _Vector:
        """Apply rotation + translation; returns a Base.Vector-compatible vector."""
        v = np.asarray(list(vec) if hasattr(vec, "__iter__") else [vec], dtype=float)
        out = self._rot_matrix() @ v + np.array([
            self.Base.x,
            self.Base.y,
            self.Base.z,
        ])
        return _Vector(*out)

    def inverse(self):
        R = self._rot_matrix()
        b = np.array([self.Base.x, self.Base.y, self.Base.z])
        inv_b = -R.T @ b
        inv = _CQPlacement(base=tuple(inv_b), angle_rad=-self.Rotation.Angle)
        inv.Rotation.Axis = self.Rotation.Axis
        return inv

    def __repr__(self):
        return f"_CQPlacement(base={self.Base}, angle_rad={self.Rotation.Angle})"


apiPlacement = _CQPlacement


class _CQPlane:
    """Minimal stand-in for FreeCAD's Part.Plane.

    Only `.Position` and `.Axis` are needed by BluemiraPlane.
    """

    def __init__(self, base=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
        self.Position = _Vector(*base)
        self.Axis = _Vector(*axis)

    def __repr__(self):
        return f"_CQPlane(Position={self.Position}, Axis={self.Axis})"


apiPlane = _CQPlane


# Stand-in for cadapi.Base used in placement.py: cadapi.Base.Vector(value)
class _BaseModule:
    Vector = _Vector


Base = _BaseModule()


def make_placement(
    base=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    angle: float = 0.0,
) -> _CQPlacement:
    """Create a placement.  *angle* is in degrees (FreeCAD convention)."""
    return _CQPlacement(base=list(base), axis=list(axis), angle_rad=math.radians(angle))


def make_placement_from_matrix(matrix) -> _CQPlacement:
    """Extract a _CQPlacement from a 4x4 homogeneous transformation matrix.

    The rotation sub-matrix is normalised before axis-angle decomposition,
    so scaled matrices are handled correctly (as the test requires).
    """
    m = np.array(matrix, dtype=float)
    R = m[:3, :3].copy()
    t = m[:3, 3]

    # Orthonormalise R via SVD so scaled R still decomposes correctly.
    U, _s, Vt = np.linalg.svd(R)
    R_norm = U @ Vt  # nearest proper rotation (det=+1 guaranteed below)
    if np.linalg.det(R_norm) < 0:
        U[:, -1] *= -1
        R_norm = U @ Vt

    # Axis-angle from rotation matrix.
    trace = np.clip(np.trace(R_norm), -1.0, 3.0)
    theta = math.acos((trace - 1.0) / 2.0)  # radians

    sin_theta = math.sin(theta)
    if abs(theta) < _ANGLE_PARALLEL_TOL:
        # θ ≈ 0 → rotation is identity; axis direction is arbitrary.
        axis = np.array([0.0, 0.0, 1.0])
    elif abs(sin_theta) < _ANGLE_PARALLEL_TOL:
        # θ ≈ π → Rodrigues inverse is singular (sin→0).
        # R + I = 2·n·nᵀ at θ=π; pick the column with largest norm and
        # normalise it to recover the axis robustly (ambiguous sign for a
        # 180° rotation is physically irrelevant).
        m_sym = R_norm + np.eye(3)
        col_norms = np.linalg.norm(m_sym, axis=0)
        k = int(np.argmax(col_norms))
        if col_norms[k] > _ANGLE_PARALLEL_TOL:
            axis = m_sym[:, k] / col_norms[k]
        else:
            axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.array([
            R_norm[2, 1] - R_norm[1, 2],
            R_norm[0, 2] - R_norm[2, 0],
            R_norm[1, 0] - R_norm[0, 1],
        ]) / (2.0 * sin_theta)
        n = np.linalg.norm(axis)
        if n > 0:
            axis /= n
        else:
            axis = np.array([0.0, 0.0, 1.0])

    return _CQPlacement(base=tuple(t), axis=tuple(axis), angle_rad=theta)


def move_placement(placement: _CQPlacement, vector) -> None:
    """Stub — no-op for import compatibility."""


def make_plane(
    base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> _CQPlane:
    """Create a plane from a base point and normal axis (axis is normalized)."""
    n = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(n)
    if norm > 0:
        n /= norm
    return _CQPlane(base=tuple(base), axis=tuple(n))


def make_plane_from_3_points(
    point1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    point2: tuple[float, float, float] = (1.0, 0.0, 0.0),
    point3: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> _CQPlane:
    """Create a plane defined by three non-collinear points."""
    p1 = np.asarray(point1, dtype=float)
    p2 = np.asarray(point2, dtype=float)
    p3 = np.asarray(point3, dtype=float)
    normal = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(normal)
    if norm < _GEOM_NEAR_ZERO_TOL:
        raise ValueError("Three points are collinear — cannot define a plane.")
    normal /= norm
    return _CQPlane(base=tuple(p1), axis=tuple(normal))


def _checked_to_numpy(obj, types, get_xyz, name):
    """Shape-(3,) or (N, 3) array of coordinates; TypeError on wrong types.

    Mirrors FreeCAD's ``@check_data_type`` contract used by
    :func:`vector_to_numpy` / :func:`vertex_to_numpy`. *get_xyz* maps one
    element to an object exposing ``.x .y .z``.
    """
    if isinstance(obj, types):
        c = get_xyz(obj)
        return np.array([c.x, c.y, c.z])
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, types) for x in obj):
        return np.array([[(c := get_xyz(x)).x, c.y, c.z] for x in obj])
    raise TypeError(f"{name} expects {types} or list thereof, got {type(obj)}")


def vector_to_numpy(v) -> np.ndarray:
    """Convert a cq.Vector / _Vector (or list thereof) to numpy array."""
    return _checked_to_numpy(v, (cq.Vector, _Vector), lambda x: x, "vector_to_numpy")


def make_vertex(x: float, y: float, z: float) -> apiVertex:
    """Construct a vertex from coordinates."""
    return cq.Vertex.makeVertex(float(x), float(y), float(z))


def vertex_to_numpy(vertexes) -> np.ndarray:
    """Convert a cq.Vertex (or list thereof) to numpy array."""
    return _checked_to_numpy(
        vertexes, (cq.Vertex,), lambda v: v.Center(), "vertex_to_numpy"
    )


def face_from_plane(plane: _CQPlane, width: float, height: float) -> cq.Face:
    """Create a rectangular face of size ``width x height`` centred at the plane base.

    Two orthogonal basis vectors in the plane are computed from the plane normal,
    then the four corners are wired into a face.
    """
    base = np.array([plane.Position.x, plane.Position.y, plane.Position.z])
    n = np.array([plane.Axis.x, plane.Axis.y, plane.Axis.z], dtype=float)
    n /= np.linalg.norm(n)

    # Build an orthonormal basis (u, v) in the plane
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(n[2]) < _AXIS_DOMINANCE_TOL
        else np.array([1.0, 0.0, 0.0])
    )
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    hw, hh = width / 2.0, height / 2.0
    corners = [
        base - hw * u - hh * v,
        base + hw * u - hh * v,
        base + hw * u + hh * v,
        base - hw * u + hh * v,
    ]
    verts = list(starmap(cq.Vector, corners))
    edges = [cq.Edge.makeLine(verts[i], verts[(i + 1) % 4]) for i in range(4)]
    wire = cq.Wire.assembleEdges(edges)
    return cq.Face.makeFromWires(wire)


def _rotation_to_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (axis, angle_rad) rotation that aligns unit vector *src* with *dst*."""
    src /= np.linalg.norm(src) + 1e-16
    dst /= np.linalg.norm(dst) + 1e-16
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < _GEOM_NEAR_ZERO_TOL:
        if dot > 0:
            return np.array([0.0, 0.0, 1.0]), 0.0
        # 180° rotation — pick arbitrary perpendicular axis
        perp = (
            np.array([1.0, 0.0, 0.0])
            if abs(src[0]) < _AXIS_DOMINANCE_TOL
            else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return axis, math.pi
    axis = cross / cross_norm
    return axis, math.acos(dot)


def placement_from_plane(plane: _CQPlane) -> _CQPlacement:
    """Convert a plane to a placement whose local z-axis aligns with the plane normal."""
    base = (plane.Position.x, plane.Position.y, plane.Position.z)
    n = np.array([plane.Axis.x, plane.Axis.y, plane.Axis.z], dtype=float)
    axis, angle_rad = _rotation_to_align(np.array([0.0, 0.0, 1.0]), n)
    return _CQPlacement(base=base, axis=tuple(axis), angle_rad=angle_rad)


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Curve constructors
# ---------------------------------------------------------------------------


def make_bezier(points: list | np.ndarray) -> apiWire:
    """Create a Bezier curve wire from a list of poles."""
    pts = np.asarray(points)
    poles = TColgp_Array1OfPnt(1, len(pts))
    for i, p in enumerate(pts):
        poles.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    curve = Geom_BezierCurve(poles)
    edge = cq.Edge(BRepBuilderAPI_MakeEdge(curve).Edge())
    return cq.Wire.assembleEdges([edge])


def make_bspline(
    poles: np.ndarray,
    mults: np.ndarray,
    knots: np.ndarray,
    *,
    periodic: bool,
    degree: int,
    weights: np.ndarray,
    check_rational: bool,
) -> apiWire:
    """Create a B-Spline wire from poles, multiplicities, and knots."""
    poles = np.asarray(poles)
    tcol_poles = TColgp_Array1OfPnt(1, len(poles))
    for i, p in enumerate(poles):
        tcol_poles.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    mults = np.asarray(mults, dtype=int)
    tcol_mults = TColStd_Array1OfInteger(1, len(mults))
    for i, m in enumerate(mults):
        tcol_mults.SetValue(i + 1, int(m))

    knots = np.asarray(knots, dtype=float)
    tcol_knots = TColStd_Array1OfReal(1, len(knots))
    for i, k in enumerate(knots):
        tcol_knots.SetValue(i + 1, float(k))

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        tcol_weights = TColStd_Array1OfReal(1, len(weights))
        for i, w in enumerate(weights):
            tcol_weights.SetValue(i + 1, float(w))
        curve = Geom_BSplineCurve(
            tcol_poles, tcol_weights, tcol_knots, tcol_mults, degree, periodic
        )
    else:
        curve = Geom_BSplineCurve(tcol_poles, tcol_knots, tcol_mults, degree, periodic)

    edge = cq.Edge(BRepBuilderAPI_MakeEdge(curve).Edge())
    return cq.Wire.assembleEdges([edge])


def make_bsplinesurface(
    poles: np.ndarray,
    mults_u: np.ndarray,
    mults_v: np.ndarray,
    knot_vector_u: np.ndarray,
    knot_vector_v: np.ndarray,
    degree_u: int,
    degree_v: int,
    weights: np.ndarray,
    *,
    periodic: bool = False,
    check_rational: bool = False,
):
    """Create a B-Spline surface from poles, multiplicities, and knots."""
    poles = np.asarray(poles)
    nrows, ncols = poles.shape[:2]
    tcol_poles = TColgp_Array2OfPnt(1, nrows, 1, ncols)
    for i in range(nrows):
        for j in range(ncols):
            p = poles[i, j]
            tcol_poles.SetValue(
                i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2]))
            )

    def _real_array(arr):
        arr = np.asarray(arr, dtype=float)
        a = TColStd_Array1OfReal(1, len(arr))
        for i, v in enumerate(arr):
            a.SetValue(i + 1, float(v))
        return a

    def _int_array(arr):
        arr = np.asarray(arr, dtype=int)
        a = TColStd_Array1OfInteger(1, len(arr))
        for i, v in enumerate(arr):
            a.SetValue(i + 1, int(v))
        return a

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        tcol_weights = TColStd_Array2OfReal(1, nrows, 1, ncols)
        for i in range(nrows):
            for j in range(ncols):
                tcol_weights.SetValue(i + 1, j + 1, float(weights[i, j]))
        surface = Geom_BSplineSurface(
            tcol_poles,
            tcol_weights,
            _real_array(knot_vector_u),
            _real_array(knot_vector_v),
            _int_array(mults_u),
            _int_array(mults_v),
            degree_u,
            degree_v,
        )
    else:
        surface = Geom_BSplineSurface(
            tcol_poles,
            _real_array(knot_vector_u),
            _real_array(knot_vector_v),
            _int_array(mults_u),
            _int_array(mults_v),
            degree_u,
            degree_v,
        )
    return surface


def _freecad_ax2(center, axis, x_direction=None):
    """Build a gp_Ax2 matching FreeCAD's angle convention.

    FreeCAD's Part.Circle computes its X-axis by projecting (1,0,0) onto
    the plane perpendicular to *axis* (falls back to (0,1,0) when axis ∥ x).
    OCC's gp_Ax2(P, N) auto-picks a different X-axis, causing angle offsets.
    Pass *x_direction* to override the derivation (needed for serialisation
    round-trip where the original X-axis is known).
    """
    n = np.asarray(axis, dtype=float)
    n /= np.linalg.norm(n)
    if x_direction is None:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 1.0 - _POINT_COINCIDENCE_TOL:
            ref = np.array([0.0, 1.0, 0.0])
        x = ref - np.dot(ref, n) * n
    else:
        x = np.asarray(x_direction, dtype=float)
    x /= np.linalg.norm(x)
    return gp_Ax2(
        gp_Pnt(*[float(v) for v in center]),
        gp_Dir(*n.tolist()),
        gp_Dir(*x.tolist()),
    )


def make_circle(
    radius: float = 1.0,
    center=(0.0, 0.0, 0.0),
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    axis=(0.0, 0.0, 1.0),
    x_direction=None,
) -> apiWire:
    """Create a circle or arc of circle with FreeCAD-compatible angle convention.

    *x_direction* pins the local X-axis for serialisation round-trip (angle
    parameters are measured against it); defaults to FreeCAD's derivation.
    """
    circ = gp_Circ(_freecad_ax2(center, axis, x_direction), radius)
    if start_angle == end_angle:
        edge = cq.Edge(BRepBuilderAPI_MakeEdge(circ).Edge())
    else:
        arc = GC_MakeArcOfCircle(
            circ, math.radians(start_angle), math.radians(end_angle), True
        )
        edge = cq.Edge(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
    return cq.Wire.assembleEdges([edge])


def make_circle_arc_3P(
    p1,
    p2,
    p3,
    axis=None,
) -> apiWire:
    """Create an arc of circle through three points."""
    try:
        edge = cq.Edge.makeThreePointArc(cq.Vector(*p1), cq.Vector(*p2), cq.Vector(*p3))
    except Exception as e:
        raise FreeCADError(str(e)) from e
    return cq.Wire.assembleEdges([edge])


def make_ellipse(
    center: list = (0.0, 0.0, 0.0),
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: list = (1.0, 0.0, 0.0),
    minor_axis: list = (0.0, 1.0, 0.0),
    start_angle: float = 0.0,
    end_angle: float = 360.0,
) -> apiWire:
    """Create an ellipse or arc of ellipse."""
    major_axis_v = cq.Vector(*major_axis).normalized()
    minor_axis_v = cq.Vector(*minor_axis).normalized()
    normal = major_axis_v.cross(minor_axis_v)
    center_v = cq.Vector(*center)

    start_angle %= 360.0
    end_angle %= 360.0
    if start_angle == end_angle:
        edge = cq.Edge.makeEllipse(
            major_radius, minor_radius, center_v, normal, major_axis_v
        )
    else:
        edge = cq.Edge.makeEllipse(
            major_radius,
            minor_radius,
            center_v,
            normal,
            major_axis_v,
            start_angle,
            end_angle,
        )
    return cq.Wire.assembleEdges([edge])


# ---------------------------------------------------------------------------
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
    """Return the wire closed with a straight line if not already closed."""
    if wire.IsClosed():
        return wire
    edge_list = ordered_edges(wire)
    p_end = edge_list[-1].endPoint()
    p_start = edge_list[0].startPoint()
    closing = cq.Edge.makeLine(p_end, p_start)
    return cq.Wire.assembleEdges([*edge_list, closing])


def discretise(w: apiWire, ndiscr: int = 10, dl: float | None = None) -> np.ndarray:
    """Sample a wire into an array of (N, 3) points."""
    total = w.Length()
    if dl is None:
        if ndiscr < 2:  # noqa: PLR2004
            raise ValueError("ndiscr must be >= 2.")
        params = np.linspace(0.0, 1.0, ndiscr)
    else:
        if dl <= 0.0:
            raise ValueError("dl must be > 0.")
        ndiscr = max(math.ceil(total / dl + 1), 2)
        params = np.linspace(0.0, 1.0, ndiscr)

    pts = np.array([_vector_to_numpy(w.positionAt(t)) for t in params])
    if w.IsClosed():
        pts[-1] = pts[0]
    return pts


def discretise_by_edges(
    w: apiWire, ndiscr: int = 10, dl: float | None = None
) -> np.ndarray:
    """Sample each edge individually and concatenate."""
    total = w.Length()
    if dl is None:
        dl = total / float(ndiscr)
    elif dl <= 0.0:
        raise ValueError("dl must be > 0.")

    # Sample directly along the parent wire's arc-length parameterisation,
    # per edge, so we inherit the wire's traversal order and orientation.
    # cq.Wire.assembleEdges([e]) strips the edge's orientation flag — hence
    # no per-edge sub-wires.
    edge_lengths = [e.Length() for e in ordered_edges(w)]
    cum = np.cumsum([0.0, *edge_lengths]) / total
    output = []
    last_pts = None
    for i, e_len in enumerate(edge_lengths):
        if e_len < _OCC_DEFAULT_TOL:
            continue
        n = max(math.ceil(e_len / dl + 1), 2)
        pts = [
            _vector_to_numpy(w.positionAt(t)) for t in np.linspace(cum[i], cum[i + 1], n)
        ]
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
        if all(isinstance(s, cq.Solid) for s in shapes):
            solids = _collect_subshapes(result, cq.Solid)
            if len(solids) == 1:
                return solids[0]

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


def boolean_cut(shape: apiShape, tools: list, *, split: bool = True) -> list[apiShape]:
    """Boolean subtraction — return list of result shapes."""
    if not isinstance(tools, list):
        tools = [tools]

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


def join_connect(shapes: list, dist_tolerance: float = 1e-4) -> apiShape:
    """Connect the interiors of walled objects (pipes/shells).

    This mirrors FreeCAD's ``Part.JoinAPI.connect``: it performs a boolean
    union of all input shapes **and** removes the internal partition faces
    that remain where one hollow solid penetrates another (the "plug" region).
    The result is a single manifold solid whose volume equals the union minus
    the overlapping cross-sectional plugs.

    .. warning::
        The current implementation is a plain boolean fuse and therefore
        produces a **wrong volume** when the input shapes overlap: the
        internal partition walls are not removed.

        Correct OCC implementation outline:

        1. ``BRepAlgoAPI_Fuse`` (or ``BRepAlgoAPI_BuilderAlgo`` GFA) of all
           input shapes.
        2. Iterate over all faces of the fused result. A face is an internal
           partition if it is shared by two distinct solids in the fused
           compound (i.e. a non-manifold / seam face that separates two
           previously independent volumes).
        3. Remove those internal faces using ``BRep_Builder.Remove`` and
           rebuild the shell/solid topology (``BRepBuilderAPI_MakeSolid``).
        4. Optionally: ``ShapeUpgrade_UnifySameDomain`` to merge now-coplanar
           faces for a cleaner result.

        Alternatively, look at ``BOPAlgo_MakerVolume`` which can build a set
        of non-intersecting solids from a compound and may expose the
        partition face list directly.
    """
    if not isinstance(shapes, list):
        raise TypeError(f"{shapes} is not a list.")
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    # TODO @bluemira: replace plain fuse with a proper connect that removes
    # internal walls (see docstring above for the required OCC implementation).
    # https://github.com/Fusion-Power-Plant-Framework/bluemira/issues
    result = shapes[0].fuse(*shapes[1:])
    if result is None or not is_valid(result):
        raise GeometryError("join_connect: boolean union failed")
    with contextlib.suppress(Exception):
        result = result.clean()
    return result


def make_bspline_g1_blend(*args, **kwargs):
    raise NotImplementedError(
        "_cadqueryapi: 'make_bspline_g1_blend' is not yet implemented."
    )


# ---------------------------------------------------------------------------
# CAD file I/O
# ---------------------------------------------------------------------------


class CADFileType(enum.Enum):
    """Minimal CAD file type enum (mirrors _freecadapi.CADFileType)."""

    STEP = "stp"
    STEP_ZIP = "stpz"
    IGES = "iges"
    BREP = "brep"
    FREECAD = "FCStd"

    @property
    def ext(self) -> str:
        """File extension (without leading dot)."""
        _exts = {
            "stp": "stp",
            "stpz": "stpz",
            "iges": "iges",
            "brep": "brep",
            "FCStd": "FCStd",
        }
        return _exts.get(self.value, self.value)

    @classmethod
    def _missing_(cls, value: str) -> CADFileType:
        # Allow "step" → STEP, "stp" → STEP, etc.
        _aliases = {
            "step": cls.STEP,
            "stp": cls.STEP,
            "iges": cls.IGES,
            "igs": cls.IGES,
            "brep": cls.BREP,
        }
        return _aliases.get(str(value).lower())

    @classmethod
    def unitless_formats(cls) -> tuple[CADFileType, ...]:
        return (cls.BREP, cls.FREECAD)

    @classmethod
    def mesh_import_formats(cls) -> tuple[CADFileType, ...]:
        return ()

    @classmethod
    def not_importable_formats(cls) -> tuple[CADFileType, ...]:
        return (cls.STEP_ZIP, cls.FREECAD)

    @classmethod
    def manual_mesh_formats(cls) -> tuple[CADFileType, ...]:
        return ()


def make_compound(shapes: list[apiShape]) -> apiCompound:
    """Make a compound of multiple shapes."""
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s.wrapped)
    return cq.Shape.cast(comp)


@contextlib.contextmanager
def _step_write_settings():
    """Force the OCCT STEP writer into the same schema + unit as FreeCAD.

    FreeCAD's initialisation sets the OCC globals ``write.step.unit = 'M'``
    (we want ``MM``) and the schema to ``AP242DIS`` (AP242 managed
    model-based 3D engineering). The defaults under OCP are different
    (unit ``M``, schema ``AP214IS`` → ``AUTOMOTIVE_DESIGN``), which
    produces byte-divergent STEP output compared with FreeCAD and breaks
    golden-file tests.

    Both settings are writer-scoped globals that are only registered once a
    ``STEPControl_Writer`` has been instantiated (OCCT lazy-inits the
    parameter table), so the override must wrap the entire writer creation
    + transfer + write sequence. We instantiate a throw-away writer up front
    to force param registration before reading the originals.
    """
    STEPControl_Writer()
    keys = ("write.step.unit", "write.step.schema")
    targets = {"write.step.unit": "MM", "write.step.schema": "AP242DIS"}
    originals = {k: Interface_Static.CVal_s(k) for k in keys}
    for k, v in targets.items():
        Interface_Static.SetCVal_s(k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            Interface_Static.SetCVal_s(k, v)


def save_as_STP(shapes: list[apiShape], filename: str = "test", **kwargs):
    """Save shapes as a STEP file (legacy single-file method)."""
    if not filename.lower().endswith((".stp", ".step")):
        filename += ".stp"

    if not isinstance(shapes, list):
        shapes = [shapes]

    with _step_write_settings():
        writer = STEPControl_Writer()
        for s in shapes:
            writer.Transfer(s.wrapped, STEPControl_AsIs)
        status = writer.Write(filename)

    if status != IFSelect_RetDone:
        raise FreeCADError(f"Failed to write STEP file: {filename}")


def save_cad(
    shapes: Iterable[apiShape],
    filename: str,
    cad_format: str | CADFileType = "stp",
    labels: Iterable[str] | None = None,
    **kwargs,
):
    """Save CAD shapes to a file."""
    if not isinstance(shapes, list):
        shapes = list(shapes)

    cad_format = (
        CADFileType(cad_format)
        if not isinstance(cad_format, CADFileType)
        else cad_format
    )
    ext = cad_format.ext
    p = Path(filename)
    current_ext = p.suffix.lower().lstrip(".")
    valid_exts = {ext.lower()}
    if ext.lower() == "stp":
        valid_exts.add("step")
    if current_ext not in valid_exts:
        filename = str(p) + f".{ext}"

    if cad_format == CADFileType.STEP:
        with _step_write_settings():
            writer = STEPControl_Writer()
            for s in shapes:
                writer.Transfer(s.wrapped, STEPControl_AsIs)
            status = writer.Write(str(filename))
        if status != IFSelect_RetDone:
            raise FreeCADError(f"Failed to write STEP file: {filename}")
    else:
        raise FreeCADError(f"CAD format not supported by CadQuery backend: {cad_format}")


def import_cad(
    file,
    filetype=None,
    unit_scale: str = "m",
    **kwargs,
) -> list[tuple[apiShape, str]]:
    """Import CAD from file. Returns list of (shape, label) tuples."""
    file = Path(file)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(file))
    if status != IFSelect_RetDone:
        raise FreeCADError(f"Failed to read STEP file: {file}")

    reader.TransferRoots()
    shape = reader.OneShape()
    result_shape = cq.Shape.cast(shape)

    # STEP reader often returns a Compound of raw edges — try to upgrade to wires.
    if isinstance(result_shape, cq.Compound):
        edges = result_shape.Edges()
        wires = result_shape.Wires()
        shells = result_shape.Shells()
        solids = result_shape.Solids()
        if edges and not wires and not shells and not solids:
            try:
                edge_seq = TopTools_HSequenceOfShape()
                for e in edges:
                    edge_seq.Append(e.wrapped)
                result_wires = TopTools_HSequenceOfShape()
                ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
                    edge_seq, 1e-6, False, result_wires
                )
                assembled = [
                    cq.Shape.cast(result_wires.Value(i))
                    for i in range(1, result_wires.Size() + 1)
                ]
                if len(assembled) == 1:
                    result_shape = assembled[0]
                elif assembled:
                    comp = TopoDS_Compound()
                    b = BRep_Builder()
                    b.MakeCompound(comp)
                    for w in assembled:
                        b.Add(comp, w.wrapped)
                    result_shape = cq.Shape.cast(comp)
            except Exception:  # noqa: BLE001, S110
                pass  # fall through with the original compound

    # CadQuery/OCC uses raw values without mm/m conversion — no scaling needed here.
    # (FreeCAD backend needs scaling because FreeCAD works in mm internally.)
    return [(result_shape, file.stem)]


def _placement_to_trsf(placement: _CQPlacement) -> gp_Trsf:
    """Build a gp_Trsf (rotation + translation) from a _CQPlacement."""
    trsf = gp_Trsf()
    axis = placement.Rotation.Axis
    angle = placement.Rotation.Angle
    if abs(angle) > _ANGLE_PARALLEL_TOL:
        ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(axis.x, axis.y, axis.z))
        trsf.SetRotation(ax1, angle)
    base = placement.Base
    trsf.SetTranslationPart(gp_Vec(base.x, base.y, base.z))
    return trsf


def change_placement(geo: apiShape, placement: _CQPlacement) -> None:
    """Compose *placement* onto *geo*'s current location in place.

    FreeCAD's homonym does a somewhat idiosyncratic composition on
    ``geo.Placement``; here we instead apply the placement's rigid transform as
    a relative location update on the underlying ``TopoDS_Shape`` — the natural
    OCCT composition ``new = current * placement``. This matches the semantic
    intent ("move this shape by that placement") used by every caller we've
    seen, without trying to reproduce the FreeCAD base-vs-rotation asymmetry.
    """
    trsf = _placement_to_trsf(placement)
    geo.wrapped.Move(TopLoc_Location(trsf))


def __getattr__(name: str):
    # Let Python handle dunder attributes normally (e.g. __path__, __spec__)
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise NotImplementedError(
        f"_cadqueryapi: '{name}' is not yet implemented in the CadQuery backend. "
        f"Add it to bluemira/codes/_cadqueryapi.py to continue."
    )


# ---------------------------------------------------------------------------
# Monkey-patch CadQuery shape classes with FreeCAD-compatible attributes.
# This allows code that accesses .Orientation, .reverse(), .Wires (property),
# .Faces, .Edges, .Shells, .Solids on raw CadQuery shapes to work correctly.
# ---------------------------------------------------------------------------


def _cq_orientation(self) -> str:
    """Return 'Forward' or 'Reversed' mirroring FreeCAD's Orientation property."""
    return "Reversed" if self.wrapped.Orientation() == TopAbs_REVERSED else "Forward"


def _cq_reverse(self) -> None:
    """Reverse the shape orientation in-place (mirrors FreeCAD's reverse() method)."""
    reversed_shape = self.wrapped.Reversed()
    # Cast back to the concrete OCCT type so type-specific OCC functions still work.
    self.wrapped = cq.Shape.cast(reversed_shape).wrapped


for _cls in (cq.Wire, cq.Face, cq.Edge, cq.Shell, cq.Solid, cq.Compound):
    if not hasattr(_cls, "Orientation") or not isinstance(
        _cls.__dict__.get("Orientation"), property
    ):
        _cls.Orientation = property(_cq_orientation)
    if not hasattr(_cls, "reverse"):
        _cls.reverse = _cq_reverse

# Area as a property on Face (FreeCAD: face.Area property; CadQuery: face.Area() method)
if not isinstance(cq.Face.__dict__.get("Area"), property):
    cq.Face.Area = property(_cq_area_prop)


class _CallableList(UserList):
    """A list that is also callable (returns itself).

    Lets ``obj.Wires`` and ``obj.Wires()`` both yield the same value, which the
    FreeCAD-flavoured callsites in bluemira rely on.
    """

    def __call__(self):
        return list(self)


def _make_shape_collection_prop(method_name: str, orig_method):
    """Return a property that wraps the original method in a _CallableList."""

    def _prop(self):
        return _CallableList(orig_method(self))

    _prop.__name__ = method_name
    return property(_prop)


# Patch collection accessors on Solid and Shell so that both `shape.Wires` and
# `shape.Wires()` work. Only patch classes that are returned by cadapi geometry
# functions (not Compound, which CadQuery uses internally with `.Wires()` calls).
for _cls in (cq.Solid, cq.Shell):
    for _name in ("Wires", "Faces", "Edges", "Shells", "Solids", "Vertices"):
        _orig = getattr(_cls, _name, None)
        if _orig is not None and callable(_orig) and not isinstance(_orig, property):
            setattr(_cls, _name, _make_shape_collection_prop(_name, _orig))
