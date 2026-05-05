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


def _compute_default_camera(parts: list[apiShape]) -> tuple[tuple, tuple]:
    """Pick a sensible polyscope camera position based on geometry extents.

    Auto-aligns to the flat axis when the geometry is planar (e.g. the wires
    produced by ``reactor.show_cad('xz')`` all live at y≈0); falls back to a
    FreeCAD-style xz side-view for genuinely 3D scenes. Returns
    ``(camera_position, look_at_target)`` as 3-tuples.
    """
    from bluemira.codes._cadqueryapi._core import bounding_box  # noqa: PLC0415

    bbs = [bounding_box(p) for p in parts]
    xmin = min(b[0] for b in bbs)
    ymin = min(b[1] for b in bbs)
    zmin = min(b[2] for b in bbs)
    xmax = max(b[3] for b in bbs)
    ymax = max(b[4] for b in bbs)
    zmax = max(b[5] for b in bbs)
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    ex, ey, ez = xmax - xmin, ymax - ymin, zmax - zmin
    m = max(ex, ey, ez) or 1.0
    d = 1.5 * m  # camera distance — 1.5x the dominant extent gives a comfortable framing
    flat = 0.01 * m  # near-zero extent threshold (1% of dominant)
    if ey < flat:  # flat in y → look at xz plane from -y (the dim='xz' case)
        cam = (center[0], center[1] - d, center[2])
    elif ex < flat:  # flat in x → look at yz plane from +x
        cam = (center[0] + d, center[1], center[2])
    elif ez < flat:  # flat in z → look at xy plane from +z (top-down)
        cam = (center[0], center[1], center[2] + d)
    else:  # 3D → xz side-view, mirrors _freecadapi.show_cad's (90,0,0) default
        cam = (center[0], center[1] - d, center[2])
    return cam, center


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
    import polyscope as ps  # noqa: PLC0415

    from bluemira.codes import _polyscope as ps_backend  # noqa: PLC0415

    parts_list = parts if isinstance(parts, list) else [parts]

    transparency = "none"
    for opt in part_options or []:
        if opt is not None and not np.isclose(opt["transparency"], 0):
            transparency = "pretty"
            break

    ps_backend.polyscope_setup(
        up_direction=kwargs.get("up_direction", "z_up"),
        fps=kwargs.get("fps", 60),
        aa=kwargs.get("aa", 1),
        transparency=transparency,
        render_passes=kwargs.get("render_passes", 3),
        gplane=kwargs.get("gplane", "none"),
    )
    ps_backend.add_features(labels, parts_list, part_options)

    cam, target = _compute_default_camera(parts_list)
    ps.look_at(cam, target)

    # Polyscope's panel toggle is a process-wide global; set it explicitly
    # both ways so a previous call with the opposite setting doesn't leak.
    ps.set_build_default_gui_panels(show_gui_panels)

    ps.show()


__all__ = [
    "DefaultDisplayOptions",
    "collect_verts_faces",
    "collect_wires",
    "show_cad",
    "tessellate",
]
