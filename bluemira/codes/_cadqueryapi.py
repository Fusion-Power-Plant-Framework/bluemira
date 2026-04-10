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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.error import FreeCADError, InvalidCADInputsError
from bluemira.utilities.tools import ColourDescriptor
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
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet2d
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import Geom_BezierCurve, Geom_BSplineCurve, Geom_BSplineSurface
from OCP.GeomAbs import (
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Line,
)
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Compound

if TYPE_CHECKING:
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
    arr[np.abs(arr) < 1e-12] = 0.0
    return arr


# ---------------------------------------------------------------------------
# Geometry creation
# ---------------------------------------------------------------------------


def make_polygon(points: list | np.ndarray) -> apiWire:
    """Make a closed or open polygon wire from a sequence of points."""
    vecs = _to_cq_vectors(points)
    return cq.Wire.makePolygon(vecs)


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
        atol=EPS,
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
            return cq.Solid.revolve(outer_wire, inner_wires, degree, axis_start, axis_end)
        else:
            # Wire / Edge: use OCC directly to get a Shell without end-caps.
            n = np.asarray(direction, dtype=float)
            n = n / np.linalg.norm(n)
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

    # CadQuery sweep: outer wire + optional inner wires + path
    outer = profiles[0]
    inner = profiles[1:] if len(profiles) > 1 else []

    try:
        # transition: 0=transformed, 1=right, 2=round  (CadQuery string literals)
        transition_mode = ["transformed", "right", "round"][transition]
        result = cq.Solid.sweep(
            outer,
            inner,
            path,
            makeSolid=solid,
            isFrenet=frenet,
            transitionMode=transition_mode,
        )
    except Exception as exc:
        raise FreeCADError(f"CadQuery sweep failed: {exc}") from exc

    if solid:
        return result
    return result.Shells()[0]


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
    if thickness == 0.0:  # noqa: RUF069
        return wire

    if _wire_is_straight(wire):
        raise InvalidCADInputsError("Cannot offset a straight line.")

    if not _wire_is_planar(wire):
        raise InvalidCADInputsError("Cannot offset a non-planar wire.")

    # CadQuery offset2D kind: 'arc' | 'intersection' | 'tangent'
    _join_map = {"arc": "arc", "intersect": "intersection", "tangent": "tangent"}
    kind = _join_map[join.lower()]

    if join.lower() == "tangent":
        bluemira_warn(f"Join type: {join} may be unstable. Consider 'arc' or 'intersect'.")

    if wire.IsClosed() and open_wire:
        open_wire = False

    try:
        result_wires = wire.offset2D(thickness, kind=kind)
    except Exception as exc:
        raise FreeCADError(f"CadQuery was unable to make an offset of wire: {exc}") from exc

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
    if total == 0.0:
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
    """Total length of the shape (sum of all edge lengths)."""
    if isinstance(obj, (cq.Edge, cq.Wire)):
        return obj.Length()
    # For faces, shells, solids: sum up all edge lengths (matches FreeCAD Part.Shape.Length)
    return sum(e.Length() for e in obj.Edges())


def _occ_face_area(topoDS_face) -> float:
    """Compute the surface area of a TopoDS_Face via OCC mass properties."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(topoDS_face, props)
    return props.Mass()


def _cq_area_prop(self) -> float:
    """Area property for cq.Face (FreeCAD exposes this as a property, CadQuery as a method)."""
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
            hole_area = sum(_occ_face_area(cq.Face.makeFromWires(w).wrapped) for w in inner)
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
    Attempt to fix a shape in-place.

    CadQuery does not expose a direct fix() method. This is a no-op stub;
    shapes are expected to be valid after creation.

    TODO: wire in OCC BRepLib.buildCurves3d if needed.
    """


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
    except Exception:  # noqa: BLE001
        pass
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
    """
    Reorder edges of *new_wire* to match the orientation of *old_wire*.

    CadQuery's Wire.assembleEdges() already sorts edges by connectivity, so
    this is mostly a pass-through. Orientation flipping is a TODO.
    """
    try:
        return cq.Wire.assembleEdges(new_wire.Edges())
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


def collect_wires(solid: apiShape, deflection: float = 0.01, **_kwds) -> tuple[np.ndarray, np.ndarray]:
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
    from bluemira.codes import _polyscope as ps_backend

    # Temporarily patch the collect helpers polyscope uses so that it calls
    # our CadQuery-aware versions instead of the FreeCAD ones.
    import bluemira.codes._freecadapi as _orig_cadapi

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
        if hasattr(x, '__iter__'):
            x, y, z = x
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return f"_Vector({self.x}, {self.y}, {self.z})"

    def __array__(self, dtype=None):
        arr = np.array([self.x, self.y, self.z])
        return arr if dtype is None else arr.astype(dtype)


class _Rotation:
    def __init__(self, axis=(0, 0, 1), angle=0.0):
        self.Axis = _Vector(*axis)
        self.Angle = angle


class _HomogeneousMatrix:
    """Wraps a 4×4 numpy array; exposes `.A` as a flat list (FreeCAD Matrix.A API)."""

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
            _ax = _ax / _norm
        self.Rotation = _Rotation(tuple(_ax), angle_rad)

    @property
    def Matrix(self) -> _HomogeneousMatrix:
        """4×4 homogeneous transformation matrix (FreeCAD Placement.Matrix API)."""
        R = self._rot_matrix()
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = [self.Base.x, self.Base.y, self.Base.z]
        return _HomogeneousMatrix(m)

    def _rot_matrix(self) -> np.ndarray:
        """3x3 rotation matrix via Rodrigues' formula (angle in radians)."""
        k = np.array([self.Rotation.Axis.x, self.Rotation.Axis.y, self.Rotation.Axis.z], dtype=float)
        norm = np.linalg.norm(k)
        if norm < 1e-12:
            return np.eye(3)
        k = k / norm
        theta = self.Rotation.Angle  # already in radians
        c, s = math.cos(theta), math.sin(theta)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)

    def multVec(self, vec) -> np.ndarray:
        """Apply rotation + translation to a vector (returns numpy array)."""
        v = np.asarray(list(vec) if hasattr(vec, '__iter__') else [vec], dtype=float)
        result = self._rot_matrix() @ v + np.array([self.Base.x, self.Base.y, self.Base.z])
        return result

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
    """Extract a _CQPlacement from a 4×4 homogeneous transformation matrix.

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

    if abs(theta) < 1e-10:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = np.array([
            R_norm[2, 1] - R_norm[1, 2],
            R_norm[0, 2] - R_norm[2, 0],
            R_norm[1, 0] - R_norm[0, 1],
        ]) / (2.0 * math.sin(theta))
        n = np.linalg.norm(axis)
        if n > 0:
            axis = axis / n

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
        n = n / norm
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
    if norm < 1e-12:
        raise ValueError("Three points are collinear — cannot define a plane.")
    normal /= norm
    return _CQPlane(base=tuple(p1), axis=tuple(normal))


def vector_to_numpy(v) -> np.ndarray:
    """Convert a _Vector (or cq.Vector) to numpy array."""
    if hasattr(v, "x"):
        return np.array([v.x, v.y, v.z])
    return np.asarray(v)


def face_from_plane(plane: _CQPlane, width: float, height: float) -> cq.Face:
    """Create a rectangular face of size ``width x height`` centred at the plane base.

    Two orthogonal basis vectors in the plane are computed from the plane normal,
    then the four corners are wired into a face.
    """
    base = np.array([plane.Position.x, plane.Position.y, plane.Position.z])
    n = np.array([plane.Axis.x, plane.Axis.y, plane.Axis.z], dtype=float)
    n /= np.linalg.norm(n)

    # Build an orthonormal basis (u, v) in the plane
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])  # noqa: PLR2004
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
    verts = [cq.Vector(*c) for c in corners]
    edges = [
        cq.Edge.makeLine(verts[i], verts[(i + 1) % 4])
        for i in range(4)
    ]
    wire = cq.Wire.assembleEdges(edges)
    return cq.Face.makeFromWires(wire)


def _rotation_to_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (axis, angle_rad) rotation that aligns unit vector *src* with *dst*."""
    src = src / (np.linalg.norm(src) + 1e-16)
    dst = dst / (np.linalg.norm(dst) + 1e-16)
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < 1e-12:
        if dot > 0:
            return np.array([0.0, 0.0, 1.0]), 0.0
        # 180° rotation — pick arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])  # noqa: PLR2004
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
    """Passthrough decorator stub (no FreeCAD error translation needed)."""
    def decorator(func):
        return func
    return decorator


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


def optimal_bounding_box(obj: apiShape) -> tuple[float, float, float, float, float, float]:
    """Alias for bounding_box (CadQuery does not expose a tighter variant)."""
    return bounding_box(obj)


# ---------------------------------------------------------------------------
# Vertex / point accessors
# ---------------------------------------------------------------------------


def edge_tangent_at(edge: apiEdge, param: float) -> np.ndarray:
    """Return the unit tangent of *edge* at normalised parameter *param* in [0, 1]."""
    return _vector_to_numpy(edge.tangentAt(param))


def start_point(obj: apiShape) -> np.ndarray:
    """Start point of a wire (first vertex of first ordered edge)."""
    return _vector_to_numpy(ordered_edges(obj)[0].startPoint())


def end_point(obj: apiShape) -> np.ndarray:
    """End point of a wire (last vertex of last ordered edge)."""
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
        curve = Geom_BSplineCurve(
            tcol_poles, tcol_knots, tcol_mults, degree, periodic
        )

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
            tcol_poles.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

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
            tcol_poles, tcol_weights,
            _real_array(knot_vector_u), _real_array(knot_vector_v),
            _int_array(mults_u), _int_array(mults_v),
            degree_u, degree_v,
        )
    else:
        surface = Geom_BSplineSurface(
            tcol_poles,
            _real_array(knot_vector_u), _real_array(knot_vector_v),
            _int_array(mults_u), _int_array(mults_v),
            degree_u, degree_v,
        )
    return surface


def _freecad_ax2(center, axis):
    """
    Build a gp_Ax2 that matches FreeCAD's angle convention.

    FreeCAD's Part.Circle computes its X-axis by projecting (1,0,0) onto the
    plane perpendicular to *axis* (falls back to (0,1,0) when axis ∥ x).
    OCC's gp_Ax2(P, N) auto-picks a different X-axis, causing angle offsets.
    """

    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 1.0 - 1e-9:
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, n) * n
    x = x / np.linalg.norm(x)
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
) -> apiWire:
    """Create a circle or arc of circle with FreeCAD-compatible angle convention."""

    circ = gp_Circ(_freecad_ax2(center, axis), radius)
    if start_angle == end_angle:
        edge = cq.Edge(BRepBuilderAPI_MakeEdge(circ).Edge())
    else:
        arc = GC_MakeArcOfCircle(
            circ, math.radians(start_angle), math.radians(end_angle), True
        )
        edge = cq.Edge(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
    return cq.Wire.assembleEdges([edge])


def make_circle_arc_3P(  # noqa: N802
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

    start_angle = start_angle % 360.0
    end_angle = end_angle % 360.0
    if start_angle == end_angle:
        edge = cq.Edge.makeEllipse(major_radius, minor_radius, center_v, normal, major_axis_v)
    else:
        edge = cq.Edge.makeEllipse(
            major_radius, minor_radius, center_v, normal, major_axis_v,
            start_angle, end_angle,
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
    edge_list = wire.Edges()
    p_end = edge_list[-1].endPoint()
    p_start = edge_list[0].startPoint()
    return cq.Wire.assembleEdges([cq.Edge.makeLine(p_end, p_start)])


def close_wire(wire: apiWire) -> apiWire:
    """Return the wire closed with a straight line if not already closed."""
    if wire.IsClosed():
        return wire
    edge_list = wire.Edges()
    p_end = edge_list[-1].endPoint()
    p_start = edge_list[0].startPoint()
    closing = cq.Edge.makeLine(p_end, p_start)
    return cq.Wire.assembleEdges(edge_list + [closing])


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

    output = []
    last_pts = None
    for e in w.Edges():
        e_wire = cq.Wire.assembleEdges([e])
        if e_wire.Length() < 1e-6:
            continue
        pts = list(discretise(e_wire, dl=dl))
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


def wire_parameter_at(
    wire: apiWire, vertex, tolerance: float = 1e-8
) -> float:
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
    edges_1: list[cq.Edge] = []
    edges_2: list[cq.Edge] = []
    found = False

    for e in all_edges:
        if found:
            edges_2.append(e)
            continue
        e_wire = cq.Wire.assembleEdges([e])
        e_dist, _ = dist_to_shape(e_wire, vertex_shape)
        if e_dist <= tolerance:
            # Project the closest point exactly onto the edge curve

            adaptor = BRepAdaptor_Curve(e.wrapped)
            t0 = adaptor.FirstParameter()
            t1 = adaptor.LastParameter()
            curve = adaptor.Curve().Curve()
            pnt = gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            proj = GeomAPI_ProjectPointOnCurve(pnt, curve, t0, t1)
            if proj.NbPoints() > 0:
                t_split = proj.LowerDistanceParameter()
            else:
                t_split = t0
            if t_split - t0 > 1e-10:
                edges_1.append(cq.Edge(BRepBuilderAPI_MakeEdge(curve, t0, t_split).Edge()))
            if t1 - t_split > 1e-10:
                edges_2.append(cq.Edge(BRepBuilderAPI_MakeEdge(curve, t_split, t1).Edge()))
            found = True
        else:
            edges_1.append(e)

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
    n = n / np.linalg.norm(n)
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
    if remove_splitter:
        try:
            result = result.clean()
        except Exception:  # noqa: BLE001
            pass

    # For wire inputs: OCC fuse returns a compound; try to assemble a single wire.
    if all(isinstance(s, cq.Wire) for s in shapes):
        result_wires = result.Wires()
        if len(result_wires) == 1:
            return result_wires[0]
        if result_wires:
            try:
                combined = cq.Wire.combine(result_wires)
                return combined[0] if isinstance(combined, list) else combined
            except Exception:  # noqa: BLE001
                pass

    return result


def boolean_cut(
    shape: apiShape, tools: list, *, split: bool = True
) -> list[apiShape]:
    """Boolean subtraction — return list of result shapes."""
    if not isinstance(tools, list):
        tools = [tools]

    result = shape.cut(*tools)
    if isinstance(shape, apiSolid):
        return result.Solids()
    if isinstance(shape, apiFace):
        return result.Faces()
    if isinstance(shape, apiWire):
        return result.Wires()
    return [result]


def boolean_fragments(
    shapes: list, tolerance: float = 0.0
) -> tuple[apiCompound, list]:
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


def slice_shape(shape: apiShape, plane_origin, plane_axis):
    """Slice a shape with a plane.

    For wires returns a numpy array of intersection points (N, 3).
    For faces/solids/shells returns a list of intersection wires.
    """
    pln = gp_Pln(
        gp_Pnt(*[float(x) for x in plane_origin]),
        gp_Dir(*[float(x) for x in plane_axis]),
    )
    plane_face = BRepBuilderAPI_MakeFace(pln, -1e6, 1e6, -1e6, 1e6).Face()

    section = BRepAlgoAPI_Section(shape.wrapped, plane_face)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    section.Build()

    if not section.IsDone() or section.Shape().IsNull():
        if isinstance(shape, apiWire):
            return np.empty((0, 3))
        return []

    result = cq.Shape.cast(section.Shape())

    if isinstance(shape, apiWire):
        verts = result.Vertices()
        if not verts:
            return np.empty((0, 3))
        return np.array([_vector_to_numpy(v.Center()) for v in verts])

    wires = result.Wires()
    if wires:
        return wires

    # BRepAlgoAPI_Section returns edges in a compound; assemble them into wires.
    edges = result.Edges()
    if not edges:
        return []

    # Use ShapeAnalysis_FreeBounds to properly group edges into wires
    try:
        from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds  # noqa: PLC0415
        from OCP.TopTools import TopTools_HSequenceOfShape  # noqa: PLC0415

        edge_seq = TopTools_HSequenceOfShape()
        for e in edges:
            edge_seq.Append(e.wrapped)
        result_wires_seq = TopTools_HSequenceOfShape()
        ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edge_seq, 1e-4, False, result_wires_seq)
        assembled = [cq.Shape.cast(result_wires_seq.Value(i)) for i in range(1, result_wires_seq.Size() + 1)]
        if assembled:
            return assembled
    except Exception:  # noqa: BLE001
        pass

    try:
        return [cq.Wire.assembleEdges(edges)]
    except Exception:  # noqa: BLE001
        # Edges are disconnected — wrap each individually
        result_wires = []
        for e in edges:
            try:
                result_wires.append(cq.Wire.assembleEdges([e]))
            except Exception:  # noqa: BLE001
                pass
        return result_wires


def point_inside_shape(point, shape: apiShape) -> bool:
    """Return True if *point* is inside *shape*."""
    from OCP.BRepClass import BRepClass_FaceClassifier  # noqa: PLC0415
    from OCP.BRep import BRep_Tool as _BRep_Tool  # noqa: PLC0415
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf  # noqa: PLC0415
    from OCP.gp import gp_Pnt2d  # noqa: PLC0415

    pnt = gp_Pnt(*[float(x) for x in point])

    if isinstance(shape, apiFace):
        # Project point onto the face's surface, then classify in UV space
        surf = _BRep_Tool.Surface_s(shape.wrapped)
        projector = GeomAPI_ProjectPointOnSurf(pnt, surf)
        projector.Perform(pnt)
        if projector.NbPoints() == 0:
            return False
        u, v = projector.LowerDistanceParameters()
        classifier = BRepClass_FaceClassifier()
        classifier.Perform(shape.wrapped, gp_Pnt2d(u, v), 1e-6)
        return classifier.State() == TopAbs_IN

    classifier = BRepClass3d_SolidClassifier(shape.wrapped)
    classifier.Perform(pnt, 1e-6)
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
            t0 = adaptor.FirstParameter()
            t1 = adaptor.LastParameter()
            start_angle = math.degrees(t0)
            end_angle = math.degrees(t1)
            serialised.append({"ArcOfCircle": {
                "Radius": radius, "Center": center,
                "StartAngle": start_angle, "EndAngle": end_angle, "Axis": axis,
            }})
        elif ctype == GeomAbs_BSplineCurve:
            bsp = adaptor.BSpline()
            poles = [[bsp.Pole(i).X(), bsp.Pole(i).Y(), bsp.Pole(i).Z()] for i in range(1, bsp.NbPoles() + 1)]
            knots = [bsp.Knot(i) for i in range(1, bsp.NbKnots() + 1)]
            mults = [bsp.Multiplicity(i) for i in range(1, bsp.NbKnots() + 1)]
            weights = [bsp.Weight(i) for i in range(1, bsp.NbPoles() + 1)]
            serialised.append({"BSplineCurve": {
                "Poles": poles, "Knots": knots, "Mults": mults,
                "Degree": bsp.Degree(), "Weights": weights,
                "isPeriodic": bsp.IsPeriodic(), "checkRational": bsp.IsRational(),
            }})
        elif ctype == GeomAbs_BezierCurve:
            bez = adaptor.Bezier()
            poles = [[bez.Pole(i).X(), bez.Pole(i).Y(), bez.Pole(i).Z()] for i in range(1, bez.NbPoles() + 1)]
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
            f1 = [center[0] + c * major_ax[0], center[1] + c * major_ax[1], center[2] + c * major_ax[2]]
            serialised.append({"ArcOfEllipse": {
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
            }})
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
                v["Radius"], v["Center"], v["StartAngle"], v["EndAngle"], v["Axis"]
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
    from bluemira.geometry.error import GeometryError  # noqa: PLC0415

    is_closed = wire.IsClosed()
    work_wire = wire
    close_edge = None

    if not is_closed:
        # Add a dummy closing edge so BRepFilletAPI_MakeFillet2d can work on a face.
        wire_edges = ordered_edges(wire)
        start_pt = _vector_to_numpy(wire_edges[0].startPoint())
        end_pt = _vector_to_numpy(wire_edges[-1].endPoint())
        close_edge = cq.Edge.makeLine(cq.Vector(*end_pt), cq.Vector(*start_pt))
        work_wire = cq.Wire.assembleEdges(wire_edges + [close_edge])

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
            if np.linalg.norm(np.cross(t1, t2)) < 1e-6:
                continue
            builder.AddChamfer(e1_occ, e2_occ, radius, radius)
    else:
        # Build the set of endpoint-vertex keys to skip for open wires (endpoints can't be filleted)
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

    .. todo::
        The current implementation is a plain boolean fuse and therefore
        produces a **wrong volume** when the input shapes overlap: the internal
        partition walls are *not* removed.

        Correct OCC implementation outline:
        1. ``BRepAlgoAPI_Fuse`` (or ``BRepAlgoAPI_BuilderAlgo`` GFA) of all
           input shapes.
        2. Iterate over all faces of the fused result.  A face is an *internal
           partition* if it is shared by two distinct solids in the fused
           compound (i.e. it is a non-manifold / seam face that separates two
           previously independent volumes).
        3. Remove those internal faces using ``BRep_Builder.Remove`` and
           rebuild the shell/solid topology (``BRepBuilderAPI_MakeSolid``).
        4. Optionally: ``ShapeUpgrade_UnifySameDomain`` to merge now-coplanar
           faces for a cleaner result.

        Alternatively, look at ``BOPAlgo_MakerVolume`` which can build a set
        of non-intersecting solids from a compound and may expose the partition
        face list directly.
    """
    from bluemira.geometry.error import GeometryError  # noqa: PLC0415

    if not isinstance(shapes, list):
        raise TypeError(f"{shapes} is not a list.")
    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    # TODO: replace plain fuse with a proper connect that removes internal walls
    # (see docstring above for the required OCC implementation).
    result = shapes[0].fuse(*shapes[1:])
    if result is None or not is_valid(result):
        raise GeometryError("join_connect: boolean union failed")
    try:
        result = result.clean()
    except Exception:  # noqa: BLE001
        pass
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
def _step_unit_mm():
    """Force ``write.step.unit = MM`` and restore on exit.

    FreeCAD's initialisation sets the OCC global ``write.step.unit = 'M'``.
    ``STEPControl_Writer`` picks up this setting at construction time, so the
    override must wrap the entire writer creation + transfer + write sequence.
    """
    from OCP.Interface import Interface_Static  # noqa: PLC0415

    original = Interface_Static.CVal_s("write.step.unit")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    try:
        yield
    finally:
        Interface_Static.SetCVal_s("write.step.unit", original)


def save_as_STP(shapes: list[apiShape], filename: str = "test", **kwargs):
    """Save shapes as a STEP file (legacy single-file method)."""
    from OCP.IFSelect import IFSelect_RetDone  # noqa: PLC0415
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs  # noqa: PLC0415

    if not filename.lower().endswith((".stp", ".step")):
        filename = filename + ".stp"

    if not isinstance(shapes, list):
        shapes = [shapes]

    with _step_unit_mm():
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
    from OCP.IFSelect import IFSelect_RetDone  # noqa: PLC0415
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    if not isinstance(shapes, list):
        shapes = list(shapes)

    cad_format = CADFileType(cad_format) if not isinstance(cad_format, CADFileType) else cad_format
    ext = cad_format.ext
    p = _Path(filename)
    current_ext = p.suffix.lower().lstrip(".")
    valid_exts = {ext.lower()}
    if ext.lower() == "stp":
        valid_exts.add("step")
    if current_ext not in valid_exts:
        filename = str(p) + f".{ext}"

    if cad_format in (CADFileType.STEP,):
        with _step_unit_mm():
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
    from pathlib import Path as _Path  # noqa: PLC0415
    from OCP.STEPControl import STEPControl_Reader  # noqa: PLC0415
    from OCP.IFSelect import IFSelect_RetDone  # noqa: PLC0415

    file = _Path(file)
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
                from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds  # noqa: PLC0415
                from OCP.TopTools import TopTools_HSequenceOfShape  # noqa: PLC0415
                edge_seq = TopTools_HSequenceOfShape()
                for e in edges:
                    edge_seq.Append(e.wrapped)
                result_wires = TopTools_HSequenceOfShape()
                ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
                    edge_seq, 1e-6, False, result_wires
                )
                assembled = [cq.Shape.cast(result_wires.Value(i)) for i in range(1, result_wires.Size() + 1)]
                if len(assembled) == 1:
                    result_shape = assembled[0]
                elif assembled:
                    comp = TopoDS_Compound()
                    b = BRep_Builder()
                    b.MakeCompound(comp)
                    for w in assembled:
                        b.Add(comp, w.wrapped)
                    result_shape = cq.Shape.cast(comp)
            except Exception:  # noqa: BLE001
                pass

    # CadQuery/OCC uses raw values without mm/m conversion — no scaling needed here.
    # (FreeCAD backend needs scaling because FreeCAD works in mm internally.)
    return [(result_shape, file.stem)]


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
    if not hasattr(_cls, "Orientation") or not isinstance(_cls.__dict__.get("Orientation"), property):
        _cls.Orientation = property(_cq_orientation)
    if not hasattr(_cls, "reverse"):
        _cls.reverse = _cq_reverse

# Area as a property on Face (FreeCAD: face.Area property; CadQuery: face.Area() method)
if not isinstance(cq.Face.__dict__.get("Area"), property):
    cq.Face.Area = property(_cq_area_prop)


class _CallableList(list):
    """A list that is also callable (returns itself), so obj.Wires and obj.Wires() both work."""
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
