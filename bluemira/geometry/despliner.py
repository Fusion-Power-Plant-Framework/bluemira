# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Despliner for any Component"""


def has_splines(solid) -> bool:
    """
    Inspect if the solid has faces
    containing spline/Bezier edges.

    Parameters
    ----------
    solid:
        CadQuery solid or FreeCAD shape.

    Returns
    -------
    bool
        True if the solid has splines, else False

    Raises
    ------
    NotImplementedError
        If a face itself is a BSpline or Bezier surface.
    """
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
            raise NotImplementedError(
                f"Face {i} has unsupported {face_type} surface geometry."
            )

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
