# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
api for plotting using freecad
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.colors as colors
import polyscope as ps

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.codes import _freecadapi as cadapi
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.plotter import DisplayOptions

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeo

DEFAULT_DISPLAY_OPTIONS = {
    "color": (0.5, 0.5, 0.5),
    "transparency": 1.0,
    "material": "wax",
    "tesselation": 0.05,
    "wires_on": False,
    "wire_radius": 0.001,
}


def get_default_options():
    """
    Returns the default display options.
    """
    return copy.deepcopy(DEFAULT_DISPLAY_OPTIONS)


class DisplayCADOptions(DisplayOptions):
    """
    The options that are available for displaying objects in 3D

    Parameters
    ----------
    color: Union[str, Tuple[float, float, float]]
        The colour to display the object, by default (0.5, 0.5, 0.5).
    transparency: float
        The transparency to display the object, by default 0.0.
    """

    __slots__ = ("_options",)

    def __init__(self, **kwargs):
        self._options = get_default_options()
        self.modify(**kwargs)

    def __setattr__(self, attr, val):
        """
        Set attributes in options dictionary
        """
        if (
            hasattr(self, "_options")
            and self._options is not None
            and attr in self._options
        ):
            self._options[attr] = val
        else:
            super().__setattr__(attr, val)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        dict_ = super().as_dict()
        if "color" in dict_:
            dict_["color"] = self.color
        return dict_

    @property
    def color(self) -> Tuple[float, float, float]:
        """
        The RBG colour to display the object.
        """
        # NOTE: We only convert to (R,G,B) at the last minute, so that the reprs are
        # legible.
        return colors.to_rgb(self._options["color"])


# =======================================================================================
# Visualisation
# =======================================================================================
def _get_displayer_class(part):
    """
    Get the displayer class for an object.
    """
    import bluemira.base.components

    if isinstance(part, bluemira.base.components.Component):
        plot_class = ComponentDisplayer
    else:
        raise DisplayError(
            f"{part} object cannot be displayed. No Displayer available for {type(part)}"
        )
    return plot_class


def _validate_display_inputs(parts, options):
    """
    Validate the lists of parts and options, applying some default options.
    """
    if parts is None:
        bluemira_debug("No new parts to display")
        return [], []

    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [get_default_options()] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )
    return parts, options


def show_cad(
    parts: Optional[Union[BluemiraGeo, List[BluemiraGeo]]] = None,
    options: Optional[Union[DisplayCADOptions, List[DisplayCADOptions]]] = None,
    **kwargs,
):
    """
    The CAD display API using Polyscope.

    Parameters
    ----------
    parts: Optional[Union[BluemiraGeo, List[BluemiraGeo]]]
        The parts to display.
    options: Optional[Union[_PlotCADOptions, List[_PlotCADOptions]]]
        The options to use to display the parts.
    kwargs: Dict
        Passed on to polyscope_setup
    """
    parts, options = _validate_display_inputs(parts, options)

    new_options = []
    for o in options:
        if isinstance(o, DisplayCADOptions):
            temp = DisplayCADOptions(**o.as_dict())
            temp.modify(**kwargs)
            new_options.append(temp)
        else:
            new_options.append(DisplayCADOptions(**kwargs))

    part_options = [o.as_dict() for o in new_options]

    transparency = "none"
    for opt in part_options:
        if opt["transparency"] < 1:
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
    ps.set_program_name("Bluemira Display")
    ps.set_max_fps(fps)
    ps.set_SSAA_factor(aa)
    ps.set_transparency_mode(transparency)
    if transparency != "none":
        ps.set_transparency_render_passes(render_passes)
    ps.set_ground_plane_mode(gplane)
    ps.set_up_dir(up_direction)

    # initialize
    ps.init()
    ps.remove_all_structures()


def add_features(
    parts: Union[BluemiraGeo, List[BluemiraGeo]],
    options: Optional[Union[Dict, List[Dict]]] = None,
) -> List[ps.SurfaceMesh]:
    """
    Grab meshes of all parts to be displayed by Polyscope

    Parameters
    ----------
    parts: Union[BluemiraGeo, List[BluemiraGeo]]
        parts to be displayed
    options: Optional[Union[Dict, List[Dict]]]
        display options

    Returns
    -------
    meshes: List[ps.SurfaceMesh]
        Registered Polyspline surface meshes

    """
    meshes = []
    curves = []
    if not isinstance(parts, list):
        parts = [parts]

    # loop over every face adding their meshes to polyscope
    for shape_i, (part, option) in enumerate(zip(parts, options)):
        verts, faces = cadapi.collect_verts_faces(part._shape, option["tesselation"])

        if not (verts is None or faces is None):
            m = ps.register_surface_mesh(
                clean_name(part.label, shape_i),
                verts,
                faces,
            )
            m.set_color(option["color"])
            m.set_transparency(option["transparency"])
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
            c.set_color(option["color"])
            c.set_transparency(option["transparency"])
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
    name: str
        name to be cleaned
    number: int
        if name is empty <NO LABEL num >

    Returns
    -------
    name: str

    """
    name = name.replace("#", "_")
    if len(name) == 0 or name == "_":
        return f"<NO LABEL {number}>"
    else:
        return name


class BaseDisplayer(ABC):
    """
    Displayer abstract class
    """

    _CLASS_DISPLAY_OPTIONS = {}

    def __init__(self, options: Optional[DisplayCADOptions] = None, **kwargs):
        self.options = (
            DisplayCADOptions(**self._CLASS_DISPLAY_OPTIONS)
            if options is None
            else options
        )
        self.options.modify(**kwargs)

    @abstractmethod
    def show_cad(self, objs, **kwargs):
        """
        Display a CAD object
        """
        pass


class ComponentDisplayer(BaseDisplayer):
    """
    CAD displayer for Components
    """

    def show_cad(
        self,
        comps,
        **kwargs,
    ):
        """
        Display the CAD of a component or iterable of components

        Parameters
        ----------
        comp: Union[Iterable[Component], Component]
            Component, or iterable of Components, to be displayed
        """
        import bluemira.base.components as bm_comp

        show_cad(
            *bm_comp.get_properties_from_components(
                comps, ("shape", "display_cad_options")
            ),
            **kwargs,
        )


class DisplayableCAD:
    """
    Mixin class to make a class displayable by imparting a show_cad method and options.
    """

    def __init__(self):
        super().__init__()
        self._display_cad_options: DisplayCADOptions = DisplayCADOptions()
        self._display_cad_options.color = next(BLUE_PALETTE)

    @property
    def display_cad_options(self) -> DisplayCADOptions:
        """
        The options that will be used to display the object.
        """
        return self._display_cad_options

    @display_cad_options.setter
    def display_cad_options(self, value: DisplayCADOptions):
        if not isinstance(value, DisplayCADOptions):
            raise DisplayError(
                "Display options must be set to a DisplayCADOptions instance."
            )
        self._display_cad_options = value

    @property
    def _displayer(self) -> BaseDisplayer:
        """
        The options that will be used to display the object.
        """
        return _get_displayer_class(self)(self._display_cad_options)

    def show_cad(self, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._displayer.show_cad(self, **kwargs)
