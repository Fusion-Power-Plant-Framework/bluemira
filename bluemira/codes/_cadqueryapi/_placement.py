# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FreeCAD-parity Placement / Plane / Vector adapters for the CadQuery backend.

CadQuery has no public equivalent of FreeCAD's ``Base.Placement`` /
``Base.Vector`` / ``Part.Plane``. The minimal stand-ins below match the
attribute surface bluemira's geometry layer expects (``.Base``, ``.Rotation``,
``.Matrix``, ``.multVec``, ``.inverse`` …).
"""

from __future__ import annotations

import math
from itertools import starmap

import cadquery as cq
import numpy as np

from bluemira.codes._cadqueryapi._aliases import (
    _ANGLE_PARALLEL_TOL,
    _AXIS_DOMINANCE_TOL,
    _GEOM_NEAR_ZERO_TOL,
)

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


def make_vertex(x: float, y: float, z: float) -> cq.Vertex:
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


__all__ = [
    "Base",
    "apiPlacement",
    "apiPlane",
    "apiSurface",
    "face_from_plane",
    "make_placement",
    "make_placement_from_matrix",
    "make_plane",
    "make_plane_from_3_points",
    "make_vertex",
    "move_placement",
    "placement_from_plane",
    "vector_to_numpy",
    "vertex_to_numpy",
]
