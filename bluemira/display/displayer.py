# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
api for plotting using CAD backend
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import suppress
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.tools import Options
from bluemira.utilities.tools import get_module

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeo


class ViewerBackend(Enum):
    """CAD viewer backends."""

    FREECAD = "bluemira.codes.cadapi._freecad.api"
    POLYSCOPE = "bluemira.codes._polyscope"
    CADQUERY = "bluemira.codes.cadapi._cadquery"
    JUPYTER = "bluemira.codes._jupytercad"

    @classmethod
    @property
    def DEFAULT(cls):  # noqa: N802
        """Default viewer based on backend availability"""
        chosen = os.environ.get("BLUEMIRA_GEOMETRY_BACKEND", "freecad").upper()
        with suppress(NameError):
            if (
                chosen != "freecad"
                and type(get_ipython()).__name__ == "ZMQInteractiveShell"
            ):
                return cls["jupyter".upper()]  # Jupyter notebook or qtconsole
        return cls[chosen]

    @lru_cache(2)
    def get_module(self):
        """Load viewer module

        Returns
        -------
        :
            The loaded module.

        Raises
        ------
        ModuleNotFoundError
            Cannot find backend
        FileNotFoundError
            Cannot find backend
        """
        try:
            return get_module(self.value)
        except (ModuleNotFoundError, FileNotFoundError):
            if self.name != self.DEFAULT.name:
                name = self.name.lower()
                bluemira_warn(
                    f"Unable to import {name.capitalize()} viewer\n"
                    f"Please 'pip install {name}' to use,"
                    f" falling back to {self.DEFAULT.name.lower()}."
                )
                return get_module(self.DEFAULT.value)
            raise

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            try:
                backend = cls[value.upper()]
            except KeyError:
                bluemira_warn(
                    f"Unknown viewer backend '{value}',"
                    f" defaulting to {cls.DEFAULT.name.lower()}"
                )
                backend = cls.DEFAULT
            return backend
        bluemira_debug(f"No viewer '{value}' selecting default")
        return cls.DEFAULT


def get_default_options(backend=ViewerBackend.DEFAULT):
    """
    Returns
    -------
    :
        The default display options for the specified backend.
    """
    return backend.get_module().DefaultDisplayOptions()


class DisplayCADOptions(Options):
    """
    The options that are available for displaying objects in 3D

    Parameters
    ----------
    backend
        the backend viewer being used
    """

    __slots__ = ()

    def __init__(self, backend=ViewerBackend.DEFAULT, **kwargs):
        self._options = get_default_options(backend)
        super().__init__(**kwargs)


# =======================================================================================
# Visualisation
# =======================================================================================


def _validate_display_inputs(parts, options, labels):
    """
    Validate the lists of parts and options, applying some default options.

    Returns
    -------
    :
        A tuple containing the list of parts, list of options and list of labels.

    Raises
    ------
    DisplayError
        Number of options not equal to number of parts
    """
    if parts is None:
        bluemira_debug("No new parts to display")
        return [], [], []

    if not isinstance(parts, list):
        parts = [parts]

    if not isinstance(options, list) or options is None:
        options = [options] * len(parts)

    if labels is None:
        labels = ""

    if isinstance(labels, str):
        labels = [labels] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )
    return parts, options, labels


def show_cad(
    parts: BluemiraGeo | list[BluemiraGeo] | None = None,  # avoiding circular deps
    options: DisplayCADOptions
    | list[DisplayCADOptions]
    | list[dict[str, float | str | None]]
    | None = None,
    labels: str | list[str] | None = None,
    backend: str | ViewerBackend | None = None,
    **kwargs,
):
    """
    The CAD display API.

    Parameters
    ----------
    parts
        The parts to display.
    options
        The options to use to display the parts.
    labels
        Labels to use for each part object
    backend
        Viewer backend. If ``None``, falls back to the
        ``BLUEMIRA_GEOMETRY_BACKEND`` environment variable, then to FreeCAD.
    kwargs
        Passed on to modifications to the plotting style options and backend
    """
    backend = ViewerBackend(backend)
    parts, options, labels = _validate_display_inputs(parts, options, labels)

    # Split kwargs by whether they correspond to a DisplayCADOptions field.
    # ``Options.__slots__ = ("_options",)`` rejects unknown attributes via
    # AttributeError, so any backend-specific kwarg (FreeCAD's camera_rotation,
    # polyscope's up_direction/fps/…, our cadquery backend's hide_gui_panels)
    # would crash if forwarded to DisplayCADOptions. Route them to the backend
    # only, and let the cosmetic ones still bind to the per-part options.
    opts_field_names = set(DisplayCADOptions(backend=backend).as_dict().keys())
    opts_kwargs = {k: v for k, v in kwargs.items() if k in opts_field_names}
    viewer_kwargs = {k: v for k, v in kwargs.items() if k not in opts_field_names}

    new_options = []
    for o in options:
        if isinstance(o, DisplayCADOptions):
            temp = DisplayCADOptions(**o.as_dict(), backend=backend)
            temp.modify(**opts_kwargs)
            new_options.append(temp)
        else:
            new_options.append(
                DisplayCADOptions(**{**opts_kwargs, **(o or {})}, backend=backend)
            )

    backend.get_module().show_cad(
        [part.shape for part in parts],
        [o.as_dict() for o in new_options],
        labels,
        **viewer_kwargs,
    )


class BaseDisplayer(ABC):
    """
    Displayer abstract class
    """

    _CLASS_DISPLAY_OPTIONS: ClassVar = {}

    def __init__(self, options: DisplayCADOptions | None = None, **kwargs):
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


def _get_displayer_class(part):
    """
    Returns
    -------
    :
        the displayer class for an object

    Raises
    ------
    DisplayError
        Cannot display specific type
    """
    import bluemira.base.components  # noqa: PLC0415

    if isinstance(part, bluemira.base.components.Component):
        plot_class = ComponentDisplayer
    else:
        raise DisplayError(
            f"{part} object cannot be displayed. No Displayer available for {type(part)}"
        )
    return plot_class


class DisplayableCAD:
    """
    Mixin class to make a class displayable by imparting a show_cad method and options.
    """

    def __init__(self):
        self._display_cad_options: DisplayCADOptions = DisplayCADOptions()
        self._display_cad_options.colour = next(BLUE_PALETTE)

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
        return _get_displayer_class(self)(self.display_cad_options)

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


class ComponentDisplayer(BaseDisplayer):
    """
    CAD displayer for Components
    """

    @staticmethod
    def show_cad(comps, **kwargs):
        """
        Display the CAD of a component or iterable of components

        Parameters
        ----------
        comp: Union[Iterable[Component], Component]
            Component, or iterable of Components, to be displayed
        """
        import bluemira.base.components as bm_comp  # noqa: PLC0415

        show_cad(
            *bm_comp.get_properties_from_components(
                comps, ("shape", "display_cad_options", "name")
            ),
            **kwargs,
        )
