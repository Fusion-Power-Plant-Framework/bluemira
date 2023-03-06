from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as colors
import numpy as np
import polyscope as ps

import bluemira.codes._freecadapi as cadapi
from bluemira.base.look_and_feel import bluemira_warn


@dataclass
class DefaultDisplayOptions:
    """Polyscope default display options"""

    colour: Union[Tuple, str]
    transparency: float = 0.0
    material: str = "wax"
    tesselation: float = 0.05
    wires_on: bool = False
    wire_radius: float = 0.001

    _colour: Union[Tuple, str] = field(
        init=False, repr=False, default_factory=lambda: colors.to_hex((0.5, 0.5, 0.5))
    )

    @property
    def colour(self):
        """Colour as rbg"""
        return colors.to_hex(self._colour)

    @colour.setter
    def colour(self, value):
        """Set colour"""
        self._colour = value

    @property
    def color(self):
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value):
        """See colour"""
        self.colour = value


def show_cad(parts, part_options, **kwargs):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts
        The parts to display.
    part_options
        The options to use to display the parts.
    **kwargs
        options passed to polyscope
    """
    if None in part_options:
        part_options = [
            DefaultDisplayOptions() if o is None else o for o in part_options
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
        render_passes=kwargs.get("render_passes", 2),
        gplane=kwargs.get("gplane", "none"),
    )

    add_features(parts, part_options)

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
    up_direction: str
        'x_up' The positive X-axis is up.
        'neg_x_up' The negative X-axis is up.
        'y_up' The positive Y-axis is up.
        'neg_y_up' The negative Y-axis is up.
        'z_up' The positive Z-axis is up.
        'neg_z_up' The negative Z-axis is up.
    fps: int
        maximum frames per second of viewer (-1 == infinite)
    aa: int
        anti aliasing amount, 1 is off, 2 is usually enough
    transparency: str
        the transparency mode (none, simple, pretty)
    render_passes: int
        for transparent shapes how many render passes to undertake
    gplane: str
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
        "Polyscope is a point based viewer."
        " Some features may appear different to their actual structure"
    )
    ps.set_program_name("Bluemira Display")
    ps.init()


def add_features(
    parts: Union[BluemiraGeo, List[BluemiraGeo]],  # noqa: F821
    options: Optional[Union[Dict, List[Dict]]] = None,
) -> Tuple[List[ps.SurfaceMesh]]:
    """
    Grab meshes of all parts to be displayed by Polyscope

    Parameters
    ----------
    parts
        parts to be displayed
    options
        display options

    Returns
    -------
    Registered Polyspline surface meshes

    """
    meshes = []
    curves = []

    # loop over every face adding their meshes to polyscope
    for shape_i, (part, option) in enumerate(zip(parts, options)):
        verts, faces = cadapi.collect_verts_faces(part._shape, option["tesselation"])

        if not (verts is None or faces is None):
            m = ps.register_surface_mesh(
                clean_name(part.label, shape_i),
                verts,
                faces,
            )
            m.set_color(colors.to_rgb(option["colour"]))
            m.set_transparency(1 - option["transparency"])
            m.set_material(option["material"])
            meshes.append(m)

        if option["wires_on"] or (verts is None or faces is None):
            verts, edges = cadapi.collect_wires(part._shape, Deflection=0.01)
            c = ps.register_curve_network(
                clean_name(part.label, f"{shape_i}_wire"),
                verts,
                edges,
                radius=option["wire_radius"],
            )
            c.set_color(option["colour"])
            c.set_transparency(1 - option["transparency"])
            c.set_material(option["material"])
            curves.append(c)

    return meshes, curves


def clean_name(name: str, number: int) -> str:
    """
    Cleans or creates name.
    Polyscope doesn't like hashes in names,
    repeat names overwrite existing component.

    Parameters
    ----------
    name
        name to be cleaned
    number
        if name is empty <NO LABEL num >

    Returns
    -------
    name

    """
    name = name.replace("#", "_")
    if len(name) == 0 or name == "_":
        return f"<NO LABEL {number}>"
    else:
        return name
