# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from cadquery.assembly import Assembly
from cadquery.occ_impl import geom, shapes
from cadquery.occ_impl.assembly import Color
from cadquery.vis import show, style
from matplotlib import colors

from bluemira.codes.error import CadQueryError
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette

apiVertex = shapes.Vertex  # noqa: N816
apiVector = geom.Vector  # noqa: N816
apiEdge = shapes.Edge  # noqa: N816
apiWire = shapes.Wire  # noqa: N816
apiFace = shapes.Face  # noqa: N816
apiShell = shapes.Shell  # noqa: N816
apiSolid = shapes.Solid  # noqa: N816
apiShape = shapes.Shape  # noqa: N816
# apiSurface = Part.BSplineSurface
# apiPlacement = Base.Placement
apiPlane = shapes.Plane  # noqa: N816
apiCompound = shapes.Compound  # noqa: N816


# ======================================================================================
# Array, List, Vector, Point manipulation
# ======================================================================================


def vector_to_list(vectors: list[apiVector]) -> list[list[float]]:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a list"""  # noqa: DOC201
    return [list(v) for v in vectors]


def vector_to_numpy(vectors: list[apiVector]) -> np.ndarray:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a numpy array"""  # noqa: DOC201
    return np.array(vector_to_list(vectors))


# ======================================================================================
# Geometry visualisation
# ======================================================================================


@dataclass
class DefaultDisplayOptions:
    """CadQuery default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 1.0
    tolerance: float = 1e-2
    edges: bool = True
    mesh: bool = False
    specular: bool = True
    markersize: float = 5
    linewidth: float = 2
    spheres: bool = False
    tubes: bool = False
    edgecolor: str = "black"
    meshcolor: str = "lightgrey"
    vertexcolor: str = "cyan"

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour"""
        self.colour = value


def show_cad(
    parts: apiShape | list[apiShape],
    options: dict | list[dict | None] | None = None,
    labels: list[str] | None = None,
    **kwargs,
):
    if isinstance(parts, apiShape):
        parts = [parts]

    if options is None:
        options = [None] * len(parts)

    if labels is None:
        labels = [None] * len(parts)

    if len(options) != len(parts) != len(labels):
        raise CadQueryError(
            "If options for display or labels are provided then there must be as "
            "many as there are parts to display."
        )

    options = [{**asdict(DefaultDisplayOptions()), **(o or {})} for o in options]

    show(
        *(
            style(
                Assembly(
                    part,
                    name=label,
                    color=Color(
                        *colors.to_rgba(c=op.pop("colour"), alpha=op.pop("transparency"))
                    ),
                ),
                **op,
            )
            for part, op, label in zip(parts, options, labels, strict=False)
        ),
        title="Bluemira Display",
        **kwargs,
    )
