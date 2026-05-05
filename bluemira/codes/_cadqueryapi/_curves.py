# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Curve constructors for the CadQuery backend (Bezier, B-spline, circle,
ellipse, arcs).
"""

from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.ElCLib import ElCLib
from OCP.GC import GC_MakeArcOfCircle
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface, Geom_BezierCurve
from OCP.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec

from bluemira.codes._cadqueryapi._aliases import (
    _OCC_DEFAULT_TOL,
    _POINT_COINCIDENCE_TOL,
    apiEdge,
    apiWire,
)
from bluemira.codes.error import FreeCADError


def make_bezier(
    points: list | np.ndarray,
    first_parameter: float | None = None,
    last_parameter: float | None = None,
) -> apiWire:
    """Create a Bezier curve wire from a list of poles.

    *first_parameter* / *last_parameter* trim the resulting edge to a
    sub-range of the curve (used by deserialisation to round-trip trimmed
    edges).
    """
    pts = np.asarray(points)
    poles = TColgp_Array1OfPnt(1, len(pts))
    for i, p in enumerate(pts):
        poles.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    curve = Geom_BezierCurve(poles)
    if first_parameter is not None and last_parameter is not None:
        builder = BRepBuilderAPI_MakeEdge(curve, first_parameter, last_parameter)
    else:
        builder = BRepBuilderAPI_MakeEdge(curve)
    edge = cq.Edge(builder.Edge())
    return cq.Wire.assembleEdges([edge])


def make_bspline_g1_blend(
    edge1: apiEdge,
    edge2: apiEdge,
    scale: float = 0.2,
) -> apiWire:
    """Create a G1-continuous cubic Bézier blend wire between two edges.

    Port of ``_freecadapi.make_bspline_g1_blend``. Connects the end of
    ``edge1`` to the start of ``edge2`` with a cubic Bézier whose inner
    control points are placed along the edges' tangents at the join.

    Mirrors the FreeCAD version's orientation-disambiguation and its
    explicit end-tangent sign flip ("stupid fucking FreeCAD ... hopeless
    just override and hope they fix") so results are bit-comparable.

    Raises
    ------
    FreeCADError
        When the two edges' join points coincide (zero-length chord).
    """
    a1 = BRepAdaptor_Curve(edge1.wrapped)
    a2 = BRepAdaptor_Curve(edge2.wrapped)

    p0 = gp_Pnt()
    t0 = gp_Vec()
    a1.D1(a1.LastParameter(), p0, t0)
    p1 = gp_Pnt()
    t1 = gp_Vec()
    a2.D1(a2.FirstParameter(), p1, t1)

    chord = gp_Vec(p0, p1)
    chord_len = chord.Magnitude()
    if chord_len == 0:
        raise FreeCADError("Edges share identical endpoints")

    t0.Normalize()
    t1.Normalize()

    # Flip start tangent to always point toward p1.
    if t0.Dot(chord) < 0:
        t0.Reverse()
    # Flip end tangent to first point back toward p0 (FreeCAD-equivalent
    # orientation-disambiguation)...
    chord_rev = gp_Vec(chord.X(), chord.Y(), chord.Z())
    chord_rev.Reverse()
    if t1.Dot(chord_rev) < 0:
        t1.Reverse()
    # ...then override with a blanket sign flip, matching the FreeCAD
    # impl's explicit workaround for unreliable tangent signs.
    t1.Reverse()

    h = chord_len * scale
    poles = TColgp_Array1OfPnt(1, 4)
    poles.SetValue(1, p0)
    poles.SetValue(
        2, gp_Pnt(p0.X() + t0.X() * h, p0.Y() + t0.Y() * h, p0.Z() + t0.Z() * h)
    )
    poles.SetValue(
        3, gp_Pnt(p1.X() - t1.X() * h, p1.Y() - t1.Y() * h, p1.Z() - t1.Z() * h)
    )
    poles.SetValue(4, p1)

    curve = Geom_BezierCurve(poles)
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    maker = BRepBuilderAPI_MakeWire()
    maker.Add(edge)
    maker.Build()
    return cq.Wire(maker.Wire())


def make_bspline(
    poles: np.ndarray,
    mults: np.ndarray,
    knots: np.ndarray,
    *,
    periodic: bool,
    degree: int,
    weights: np.ndarray,
    check_rational: bool,
    first_parameter: float | None = None,
    last_parameter: float | None = None,
) -> apiWire:
    """Create a B-Spline wire from poles, multiplicities, and knots.

    *first_parameter* / *last_parameter* trim the resulting edge to a
    sub-range of the curve (used by deserialisation to round-trip trimmed
    edges).
    """
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

    if first_parameter is not None and last_parameter is not None:
        builder = BRepBuilderAPI_MakeEdge(curve, first_parameter, last_parameter)
    else:
        builder = BRepBuilderAPI_MakeEdge(curve)
    edge = cq.Edge(builder.Edge())
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
    # Wrap the parametric surface into a Face so callers receive an apiShape
    # (matches FreeCAD's bsplinesurface.toShape()). _OCC_DEFAULT_TOL is the
    # standard surface-builder confusion tolerance.
    builder = BRepBuilderAPI_MakeFace(surface, _OCC_DEFAULT_TOL)
    builder.Build()
    return cq.Face(builder.Face())


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
    """Create an arc of circle through three points.

    When *axis* is given, it overrides the natural plane normal derived from
    the three points to fix the angle-parameterisation convention (mirrors
    FreeCAD: build a circle on (radius, center, axis), then map the original
    start/end points to parameters on that circle and rebuild the arc).
    """
    try:
        nat_edge = cq.Edge.makeThreePointArc(
            cq.Vector(*p1), cq.Vector(*p2), cq.Vector(*p3)
        )
    except Exception as e:
        raise FreeCADError(str(e)) from e
    if axis is None:
        return cq.Wire.assembleEdges([nat_edge])

    nat_circ = BRepAdaptor_Curve(nat_edge.wrapped).Circle()
    centre = nat_circ.Location()
    new_circ = gp_Circ(
        _freecad_ax2((centre.X(), centre.Y(), centre.Z()), axis), nat_circ.Radius()
    )
    p_start = ElCLib.Parameter_s(new_circ, gp_Pnt(*[float(v) for v in p1]))
    p_end = ElCLib.Parameter_s(new_circ, gp_Pnt(*[float(v) for v in p3]))
    arc = GC_MakeArcOfCircle(new_circ, p_start, p_end, True)
    edge = cq.Edge(BRepBuilderAPI_MakeEdge(arc.Value()).Edge())
    return cq.Wire.assembleEdges([edge])


def make_ellipse(
    center: tuple = (0.0, 0.0, 0.0),
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: tuple = (1.0, 0.0, 0.0),
    minor_axis: tuple = (0.0, 1.0, 0.0),
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


__all__ = [
    "make_bezier",
    "make_bspline",
    "make_bspline_g1_blend",
    "make_bsplinesurface",
    "make_circle",
    "make_circle_arc_3P",
    "make_ellipse",
]
