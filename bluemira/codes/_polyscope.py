# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import polyscope as ps
from matplotlib import colors

import bluemira.codes._geometryapi as cadapi
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette


@dataclass
class DefaultDisplayOptions:
    """Polyscope default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    material: str = "wax"
    tessellation: float = 0.05
    wires_on: bool = False
    wire_radius: float = 0.001
    smooth: bool = True

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour"""
        self.colour = value


def _compute_default_camera(parts: list[cadapi.apiShape]) -> tuple[tuple, tuple]:
    """Pick a sensible polyscope camera position based on geometry extents.

    Auto-aligns to the flat axis when the geometry is planar (e.g. the wires
    produced by ``reactor.show_cad('xz')`` all live at y≈0); falls back to a
    FreeCAD-style xz side-view for genuinely 3D scenes. Returns
    ``(camera_position, look_at_target)`` as 3-tuples.
    """  # noqa: DOC201
    bbs = [cadapi.bounding_box(p) for p in parts]
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
    else:  # 3D → xz side-view, mirrors _freecad.api.show_cad's (90,0,0) default
        cam = (center[0], center[1] - d, center[2])
    return cam, center


def show_cad(
    parts: cadapi.apiShape | list[cadapi.apiShape],
    part_options: list[dict],
    labels: list[str],
    *,
    show_gui_panels: bool = True,
    **kwargs,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts:
        The parts to display.
    part_options:
        The options to use to display the parts.
    labels:
        Labels to use for each part object
    **kwargs:
        options passed to polyscope
    """
    parts_list = parts if isinstance(parts, list) else [parts]

    if part_options is None:
        part_options = [None]

    if None in part_options:
        part_options = [
            asdict(DefaultDisplayOptions()) if o is None else o for o in part_options
        ]

    transparency = "none"
    for opt in part_options:
        if not np.isclose(opt["transparency"], 0):
            transparency = "pretty"
            break

    polyscope_setup(
        up_direction=kwargs.get("up_direction", "z_up"),
        fps=kwargs.get("fps", 60),
        aa=kwargs.get("aa", 1),
        transparency=transparency,
        render_passes=kwargs.get("render_passes", 3),
        gplane=kwargs.get("gplane", "none"),
    )

    add_features(labels, parts, part_options)

    cam, target = _compute_default_camera(parts_list)
    ps.look_at(cam, target)

    # Polyscope's panel toggle is a process-wide global; set it explicitly
    # both ways so a previous call with the opposite setting doesn't leak.
    ps.set_build_default_gui_panels(show_gui_panels)

    ps.show()


def polyscope_setup(
    up_direction: str = "z_up",
    fps: int = 60,
    aa: int = 1,
    transparency: str = "pretty",
    render_passes: int = 2,
    gplane: str = "none",
):
    """
    Setup Polyscope default scene

    Parameters
    ----------
    up_direction:
        'x_up' The positive X-axis is up.
        'neg_x_up' The negative X-axis is up.
        'y_up' The positive Y-axis is up.
        'neg_y_up' The negative Y-axis is up.
        'z_up' The positive Z-axis is up.
        'neg_z_up' The negative Z-axis is up.
    fps:
        maximum frames per second of viewer (-1 == infinite)
    aa:
        anti aliasing amount, 1 is off, 2 is usually enough
    transparency:
        the transparency mode (none, simple, pretty)
    render_passes:
        for transparent shapes how many render passes to undertake
    gplane:
        the ground plane mode (none, tile, tile_reflection, shadon_only)
    """
    _init_polyscope()

    ps.set_max_fps(fps)
    ps.set_SSAA_factor(aa)
    ps.set_transparency_mode(transparency)
    if transparency != "none":
        ps.set_transparency_render_passes(render_passes)
    ps.set_ground_plane_mode(gplane)
    ps.set_up_dir(up_direction)

    ps.remove_all_structures()


@functools.lru_cache(maxsize=1)
def _init_polyscope():
    """
    Initialise polyscope (just once)
    """
    bluemira_warn(
        "Polyscope is not a NURBS based viewer."
        " Some features may appear subtly different to their CAD representation"
    )
    ps.set_program_name("Bluemira Display")
    ps.init()


def add_features(
    labels: list[str],
    parts: cadapi.apiShape | list[cadapi.apiShape],
    options: dict | list[dict],
) -> tuple[list[ps.SurfaceMesh], list[ps.CurveNetwork]]:
    """
    Grab meshes of all parts to be displayed by Polyscope

    Parameters
    ----------
    parts:
        parts to be displayed
    options:
        display options

    Returns
    -------
    Registered Polyspline surface meshes

    """
    meshes = []
    curves = []

    # loop over every face adding their meshes to polyscope
    for shape_i, (label, part, option) in enumerate(
        zip(labels, parts, options, strict=False)
    ):
        verts, faces = cadapi.collect_verts_faces(part, option["tessellation"])

        if not (verts is None or faces is None):
            m = ps.register_surface_mesh(
                clean_name(label, str(shape_i)),
                verts,
                faces,
                smooth_shade=option["smooth"],
                color=colors.to_rgb(option["colour"]),
                transparency=1 - option["transparency"],
                material=option["material"],
            )
            meshes.append(m)

        if option["wires_on"] or (verts is None or faces is None):
            verts, edges = cadapi.collect_wires(part, Deflection=0.01)
            c = ps.register_curve_network(
                clean_name(label, f"{shape_i}_wire"),
                verts,
                edges,
                radius=option["wire_radius"],
                color=colors.to_rgb(option["colour"]),
                transparency=1 - option["transparency"],
                material=option["material"],
            )
            curves.append(c)

    return meshes, curves


def clean_name(label: str, index_label: str) -> str:
    """
    Cleans or creates name.
    Polyscope doesn't like hashes in names,
    repeat names overwrite existing component.

    Parameters
    ----------
    label:
        name to be cleaned
    index_label:
        if name is empty -> {index_label}: NO LABEL

    Returns
    -------
    name
    """
    label = label.replace("#", "_")
    index_label = index_label.replace("#", "_")
    if len(label) == 0 or label == "_":
        return f"{index_label}: NO LABEL"
    return f"{index_label}: {label}"
