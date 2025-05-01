# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Utilities for converting CAD to DAGMC using PyMOAB.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from rich.progress import track

from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.imprint_solids import imprint_solids

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from bluemira.geometry.base import BluemiraGeoT

try:
    from pymoab import core, types

    pymoab_available = True
except ImportError:
    pymoab_available = False


# def moab_mesh():


def save_cad_to_dagmc_model(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    *,
    faceting_tolerance=0.001,
):
    """Converts the shapes with their associated names to a dagmc file using PyMOAB."""
    if not pymoab_available:
        raise ImportError("PyMOAB is required to convert CAD to DAGMC.")

    imprinted_solids = []

    # do a per compound imprint for now.
    # In the future, one should extract all solids then do the imprint on _all_ of them
    for shape in track(shapes):
        if isinstance(shape, BluemiraCompound):
            imprinted_solids.extend(imprint_solids(shape.solids))
        else:
            imprinted_solids.append(shape)

    # for each solid, extract the face
    # look up the face to see if it exists
    # if it does, associate the face with the solid (this one)
    # if it doesn't, create the face and associate it with the solid
    # it's mesh (faceting)


if __name__ == "__main__":
    # from bluemira.geometry.base import BluemiraGeoT
    # from bluemira.geometry.compound import BluemiraCompound
    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import (
        extrude_shape,
        make_polygon,
    )

    # Create a box
    box_a = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    )
    box_a = extrude_shape(box_a, [0, 0, 1])
    box_b = deepcopy(box_a)
    box_b.translate([-0.6, -0.6, 1])
    box_c = deepcopy(box_a)
    box_c.translate([0.6, 0.6, 1])

    pre_imps = [box_a, box_b, box_c]
    # show_cad(pre_imps)
    imps, imp_faces = imprint_solids(pre_imps)
