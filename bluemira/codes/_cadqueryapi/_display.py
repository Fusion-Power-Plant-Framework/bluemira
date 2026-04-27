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

from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.codes._cadqueryapi._aliases import apiShape
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
    # Local import to avoid a top-level _core ↔ _display cycle. ``_core`` may
    # also re-export from this module via __init__.
    from bluemira.codes._cadqueryapi._core import _vector_to_numpy  # noqa: PLC0415

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


__all__ = [
    "DefaultDisplayOptions",
    "collect_verts_faces",
    "collect_wires",
    "show_cad",
    "tessellate",
]
