# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Despliner for any Component"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    make_polygon,
)

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid


def check_face_has_splines(bm_face: BluemiraFace) -> bool:
    """
    Inspect if the face contains spline/Bezier edges.

    Parameters
    ----------
    bm_face:
        BluemiraFace

    Returns
    -------
    bool
        True if the face has splines, else False.
    """
    face = bm_face.shape

    face_type = (
        face.geomType() if hasattr(face, "geomType") else face.Surface.__class__.__name__
    )

    if face_type in {"BSPLINE", "BEZIER", "BSplineSurface", "BezierSurface"}:
        return True

    edges = (
        face.Edges() if hasattr(face, "Edges") and callable(face.Edges) else face.Edges
    )

    for edge in edges:
        edge_type = (
            edge.geomType()
            if hasattr(edge, "geomType")
            else edge.Curve.__class__.__name__
        )

        if edge_type in {"BSPLINE", "BEZIER", "BSplineCurve", "BezierCurve"}:
            return True

    return False


def check_face_has_revolution(bm_face: BluemiraFace) -> bool:
    """
    Inspect if the face is of a revolution type.

    Parameters
    ----------
    bm_face:
        BluemiraFace

    Returns
    -------
    bool
        True if the face is of a revolution type, else False.
    """
    face = bm_face.shape
    face_type = (
        face.geomType() if hasattr(face, "geomType") else face.Surface.__class__.__name__
    )

    return bool(face_type == "REVOLUTION")


def check_solid_has_revolution(bm_solid: BluemiraSolid) -> bool:
    """
    Inspect if the solid has faces of revolution type.

    Parameters
    ----------
    bm_solid:
        BluemiraSolid

    Returns
    -------
    bool
        True if the solid has REVOLUTION faces, else False.
    """
    return any(check_face_has_revolution(face) for face in bm_solid.faces)


def check_solid_has_splines(bm_solid: BluemiraSolid) -> bool:
    """
    Inspect if the solid has faces containing spline/Bezier edges.

    Parameters
    ----------
    bm_solid:
        BluemiraSolid

    Returns
    -------
    bool
        True if the solid has splines, else False.
    """
    return any(check_face_has_splines(face) for face in bm_solid.faces)


def despline_xz_face(
    xz_face: BluemiraFace,
    discretisation: int = 100,
    name: str = "",
    parent_name: str = "",
    *,
    max_side_length: float = 0.1,
    min_side_length: float = 0.001,
) -> BluemiraFace:
    """
    Despline relevant splined edges in xz face.

    This is suitable for axisymmetric neutronics, where the geometry is
    assumed to be toroidally symmetric.

    Parameters
    ----------
    xz_face
        original 2D (xz) geometry.
    discretisation
        Discretisation for splined boundary (the total boundary of the
        face that has splined face is discretised).
    max_side_length
        Maximum length of wire edges, in metres, used to warn
        when the requested discretisation may be insufficient.
    min_side_length
        Minimum length of wire edges, in metres. A warning is
        issued if any side is shorter than this length.

    Returns
    -------
    BluemiraFace
        BluemiraFace with desplined boundaries.
    """
    boundaries = []

    for i, wire in enumerate(xz_face.boundary):
        if len(wire.vertexes.T) - 1 > discretisation:
            bluemira_warn(
                f"{name} for {parent_name},"
                f" xz boundary wire {i}:"
                f" The discretisation specified {discretisation}"
                f" is lower than the wire's current discretisation"
                f" {len(wire.vertexes.T) - 1}. With byedges=True,"
                " the provided discretisation may not determine"
                " the final number of edges in the desplined wire."
            )

        points = wire.discretise(
            ndiscr=discretisation,
            byedges=True,
        )

        side_lengths = np.linalg.norm(np.diff(points, axis=1), axis=0)

        max_edge_length = np.max(side_lengths)
        min_edge_length = np.min(side_lengths)

        if max_edge_length > max_side_length:
            bluemira_warn(
                f"{name} for {parent_name},"
                f" xz boundary wire {i}:"
                f" Maximum edge length is {max_edge_length:.4f} m,"
                f" exceeding the specified limit of {max_side_length} m."
                " Consider increasing the discretisation."
            )

        if min_edge_length < min_side_length:
            bluemira_warn(
                f"{name} for {parent_name},"
                f" xz boundary wire {i}:"
                f" Minimum edge length is {min_edge_length:.6f} m,"
                f" below the specified limit of {min_side_length} m."
                " Consider reducing the discretisation."
            )

        boundaries.append(
            make_polygon(
                points,
                closed=wire.is_closed(),
            )
        )

    return BluemiraFace(
        boundaries,
        label=xz_face.label,
    )
