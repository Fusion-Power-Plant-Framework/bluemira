# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Despliner for any Component"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import ComponentError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon, revolve_shape

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid


def has_splines(bm_solid: BluemiraSolid) -> bool:
    """
    Inspect if the solid has faces
    containing spline/Bezier edges.

    Parameters
    ----------
    bm_solid:
        BluemiraSolid

    Returns
    -------
    bool
        True if the solid has splines, else False
    """
    # access the cadquery/freecad solid shape
    solid = bm_solid.shape
    planar_faces = []
    revolution_faces = []

    faces = (
        solid.Faces()
        if hasattr(solid, "Faces") and callable(solid.Faces)
        else solid.Faces
    )

    for i, face in enumerate(faces):
        face_type = (
            face.geomType()
            if hasattr(face, "geomType")
            else face.Surface.__class__.__name__
        )

        if face_type in {"BSPLINE", "BEZIER", "BSplineSurface", "BezierSurface"}:
            return True

        edges = (
            face.Edges()
            if hasattr(face, "Edges") and callable(face.Edges)
            else face.Edges
        )

        spline_edges = [
            (
                edge.geomType()
                if hasattr(edge, "geomType")
                else edge.Curve.__class__.__name__
            )
            for edge in edges
            if (
                edge.geomType()
                if hasattr(edge, "geomType")
                else edge.Curve.__class__.__name__
            )
            in {"BSPLINE", "BEZIER", "BSplineCurve", "BezierCurve"}
        ]

        if not spline_edges:
            continue

        if face_type in {"PLANE", "Plane"}:
            planar_faces.append(i)
        elif face_type in {"REVOLUTION", "Revolution"}:
            revolution_faces.append(i)

    return bool(planar_faces or revolution_faces)


def create_desplined_component_360(
    inp_component: Component,
    discretisation: int = 100,
) -> Component:
    """
    Despline relevant splined edges and create a fully revolved 360° solid.

    This is suitable for axisymmetric neutronics, where the geometry is
    assumed to be toroidally symmetric.

    **Does not consider more than one child under xz or xyz components

    Parameters
    ----------
    inp_component
        Component containing the original 2D (xz) and 3D (xyz) geometry.
    discretisation
        Discretisation for splined boundary (the total boundary of the
        face that has splined face is discretised).

    Returns
    -------
    Component
        Component with desplined xz and fully revolved 360° xyz components.

    Raises
    ------
    ComponentError
        If the component does not have both xz and xyz geometry, or if the
        xz component does not contain exactly one face.
    """
    # only consider one sub-component
    xz_component = inp_component.get_component("xz")
    xyz_component = inp_component.get_component("xyz")

    if not (xz_component and xyz_component):
        raise ComponentError(
            "Input component should have both of xz and xyz"
            " components to use the create_desplined_component_360()"
            " function"
        )

    desplined_xz = Component(xz_component.name)
    desplined_xyz = Component(xyz_component.name)

    for xz_child, xyz_child in zip(
        xz_component.children, xyz_component.children, strict=False
    ):
        xz_faces = xz_child.leaves

        if len(xz_faces) != 1:
            raise ComponentError(
                f"{xz_child.name} should contain one face, "
                f"but {len(xz_faces)} were found."
            )

        face = xz_faces[0].shape

        if not has_splines(xyz_child.get_component_properties("shape")):
            desplined_xz.add_child(xz_child.copy())
            desplined_xyz.add_child(xyz_child.copy())
            continue

        for i, wire in enumerate(face.boundary):
            # check if the discretisation is enough
            if len(wire.vertexes.T) > discretisation:
                # current discretisation {len(wire.vertexes.T)-2
                # excluding endpoints
                bluemira_warn(
                    f"{inp_component.name}, xz boundary wire {i}:"
                    f" The discretisation specified {discretisation}"
                    " is lower than the wire's current discretisation"
                    f" {len(wire.vertexes.T) - 2}"
                )

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

        desplined_xz.add_child(
            PhysicalComponent(
                name=xz_child.name,
                shape=rebuilt_face,
            )
        )

        desplined_xyz.add_child(
            PhysicalComponent(
                name=xyz_child.name,
                shape=revolve_shape(
                    rebuilt_face,
                    base=(0, 0, 0),
                    direction=(0, 0, 1),
                    degree=360.0,
                ),
                material=xyz_child.get_component_properties("material"),
            )
        )

    return Component(
        inp_component.name,
        children=[desplined_xz, desplined_xyz],
    )


def despline_component_tree(
    component: Component,
    discretisation: int,
) -> Component:
    """
    Despline all components in the tree

    Returns
    -------
    :
        The underlying components, all desplined,
        with all descendants.

    Notes
    -----
    * returned components would have only xz
    and xyz views
    """
    children = {child.name: child for child in component.children}

    if "xz" in children and "xyz" in children:
        return create_desplined_component_360(
            inp_component=component,
            discretisation=discretisation,
        )

    return Component(
        component.name,
        children=[
            despline_component_tree(child, discretisation)
            for child in component.children
        ],
    )
