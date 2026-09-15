# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Despliner for any Component"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.base.components import Component, PhysicalComponent
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


def despline_xz_component(
    component: Component,
    discretisation: int = 100,
    *,
    fallback_to_existing_discretisation: bool = False,
) -> Component:
    """
    Despline relevant splined edges in xz component.

    This is suitable for axisymmetric neutronics, where the geometry is
    assumed to be toroidally symmetric.

    **Does not consider more than one child under xz

    Parameters
    ----------
    component
        Component containing the original 2D (xz) geometry.
    discretisation
        Discretisation for splined boundary (the total boundary of the
        face that has splined face is discretised).

    Returns
    -------
    Component
        Component with xz component with desplined boundaries.

    Raises
    ------
    ComponentError
        If the component does not have both xz and xyz geometry, or if the
        xz component does not contain exactly one face.
    """
    desplined_comp = Component(component.name)

    # only consider one xz component
    xz_component = component.get_component("xz")
    # only consider one sub-component
    if len(xz_component.children) > 1:
        bluemira_warn(
            "create_desplined_comp_component() only supports "
            "one child. Considering first child only."
        )

    face = xz_component.children[0].shape
    if not check_face_has_splines(face):
        desplined_comp.add_child(
            Component("xz", children=[xz_component.children[0].copy()])
        )
        return desplined_comp

    # Else, Despline
    for i, wire in enumerate(face.boundary):
        # check if the discretisation is enough
        if len(wire.vertexes.T) - 1 > discretisation:
            # current discretisation {len(wire.vertexes.T)-2
            # excluding endpoints
            # parent.name is used as in general name of xz components are "xz"
            bluemira_warn(
                f"{xz_component.name} for {xz_component.parent.name},"
                f" xz boundary wire {i}:"
                f" The discretisation specified {discretisation}"
                " is lower than the wire's current discretisation"
                f" {len(wire.vertexes.T) - 1}."
            )
            if fallback_to_existing_discretisation:
                bluemira_warn(
                    f"falling back to the wire's current discretisation: "
                    f"{len(wire.vertexes.T) - 1}"
                )
                discretisation = len(wire.vertexes.T) - 1

    boundaries = [
        make_polygon(
            wire.discretise(
                ndiscr=discretisation,
                byedges=True,
            ),
            closed=wire.is_closed(),
        )
        for wire in face.boundary
    ]

    rebuilt_face = BluemiraFace(
        boundaries,
        label=face.label,
    )

    desplined_comp.add_child(
        Component(
            "xz",
            children=[
                PhysicalComponent(
                    name=xz_component.children[0].name,
                    shape=rebuilt_face,
                )
            ],
        )
    )
    return desplined_comp
