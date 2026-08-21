# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from jupyter_cadquery import show

import bluemira.codes._geometryapi as cadapi
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette


@dataclass
class DefaultDisplayOptions:
    """Polyscope default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    wires_on: bool = False

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour"""
        self.colour = value


def show_cad(
    parts: cadapi.apiShape | list[cadapi.apiShape],
    part_options: list[dict],
    labels: list[str],
    *,
    show_gui_panels: bool = True,
    remove_names: bool = True,
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

    transparency = False
    wires = False
    for opt in part_options:
        if not transparency and not np.isclose(opt["transparency"], 0):
            transparency = True
            if wires:
                break
        if not wires and opt["wires_on"]:
            wires = True
            if transparency:
                break

    opts = {"colors": []}
    if transparency:
        opts["alphas"] = []
    for p in part_options:
        opts["colors"].append(p["colour"])
        if transparency:
            opts["alphas"].append(1 - p["transparency"])
    for op in list(opts.keys()):
        if len(opts[op]) != len(parts):
            opts.pop(op)
    if not remove_names and None not in labels and len(labels) == len(parts):
        opts["names"] = [clean_name(l, f"{i}") for i, l in enumerate(labels)]
    show(*parts, tools=show_gui_panels, **opts)


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
