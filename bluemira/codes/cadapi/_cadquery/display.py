# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Display helpers for the CadQuery backend.

* ``tessellate`` / ``collect_verts_faces`` / ``collect_wires`` — sample
  CadQuery shapes into vertex/index arrays for polyscope rendering.
* ``DefaultDisplayOptions`` / ``show_cad`` — entry points used by
  ``bluemira.display.displayer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_CompCurve
from OCP.GCPnts import GCPnts_QuasiUniformDeflection

from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.codes.cadapi._cadquery.aliases import apiShape
    from bluemira.display.palettes import ColorPalette


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
    solid: apiShape, tessellation: float = 0.1
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract tessellated vertices and face indices for polyscope display."""
    all_verts = []
    all_faces = []
    voffset = 0

    faces = solid.Faces()
    for face in faces:
        verts, tris = face.tessellate(tessellation)
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
        Maximum chord-height deviation, in metres; controls point density per
        wire. (Polyscope passes this as ``Deflection=`` — absorbed by
        ``**_kwds``.)
    """
    all_verts = []
    all_edges = []
    voffset = 0

    for wire in solid.Wires():
        pts = _discretise_wire(wire, deflection)
        n = len(pts)
        seg_idx = np.arange(voffset, voffset + n - 1)
        all_verts.append(pts)
        all_edges.append(np.column_stack([seg_idx, seg_idx + 1]))
        voffset += n

    return np.vstack(all_verts), np.vstack(all_edges)


_MIN_SEGMENT_POINTS = 2  # a drawable polyline needs at least two endpoints


def _discretise_wire(wire: apiShape, deflection: float) -> np.ndarray:
    """Sample a wire at a target chord-height *deflection*.

    Curvature-adaptive: straight runs get few points, tight bends get more.
    """
    adaptor = BRepAdaptor_CompCurve(wire.wrapped)
    sampler = GCPnts_QuasiUniformDeflection(adaptor, deflection)
    n = sampler.NbPoints() if sampler.IsDone() else 0
    if n < _MIN_SEGMENT_POINTS:
        # Degenerate wire: use the parameter-range endpoints.
        params = (adaptor.FirstParameter(), adaptor.LastParameter())
        pts = np.empty((2, 3))
        for i, t in enumerate(params):
            p = adaptor.Value(t)
            pts[i] = (p.X(), p.Y(), p.Z())
        return pts
    pts = np.empty((n, 3))
    for i in range(n):
        p = sampler.Value(i + 1)  # GCPnts points are 1-indexed
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


@dataclass
class DefaultDisplayOptions:
    """CadQuery backend display options (delegated to polyscope)."""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    material: str = "wax"
    tessellation: float = 0.05
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
    *,
    show_gui_panels: bool = False,
    **kwargs,
):
    """
    Display CadQuery shapes via polyscope.

    Inlines polyscope's setup-add-show flow so we can position the camera
    between ``add_features`` and ``ps.show()`` — polyscope's default camera
    sits at an isometric angle that misses the cross-section the user asked
    for when ``reactor.show_cad('xz')`` was called. See
    :func:`_compute_default_camera` for the heuristic.

    Parameters
    ----------
    show_gui_panels:
        If ``True``, show polyscope's right-side structure list and per-structure
        option panes (polyscope's native behaviour). Defaults to ``False`` for a
        cleaner FreeCAD-style window — there is no equivalent panel in the
        Coin3D viewer. Mouse navigation (pan/zoom/rotate) is unaffected either
        way.
    """
    from bluemira.codes import _polyscope as ps_backend  # noqa: PLC0415

    # Allows separate defaults to polyscope
    if None in part_options:
        part_options = [
            DefaultDisplayOptions() if o is None else o for o in part_options
        ]

    ps_backend.show_cad(
        parts, part_options, labels, show_gui_panels=show_gui_panels, **kwargs
    )


__all__ = [
    "DefaultDisplayOptions",
    "collect_verts_faces",
    "collect_wires",
    "show_cad",
    "tessellate",
]
