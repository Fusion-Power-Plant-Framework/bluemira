# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Contains functions to efficiently convert to
CGAL geometry and perform meshed-based collision detections.
"""

from __future__ import annotations

import numpy as np

from bluemira.codes.cgal_ext._guard import guard_cgal_available

try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polygon_mesh_processing import (
        Int_Vector,
        Point_3_Vector,
        Polygon_Vector,
        do_intersect,
        polygon_soup_to_polygon_mesh,
    )
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
except ImportError:
    pass


def _scale_points_from_centroid(points, scale_factor):
    """
    Scale a set of points from their centroid by a given scale factor.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (n, 3) representing the x, y, z coordinates of the points.
    scale_factor : float
        The scale factor by which to scale the points.

    Returns
    -------
    scaled_points : np.ndarray
        An array of shape (n, 3) representing the scaled x, y, z
        coordinates of the points.
    """
    # Calculate the centroid
    centroid = np.mean(points, axis=0)

    # Translate points to the origin
    translated_points = points - centroid

    # Scale the points
    scaled_points = translated_points * scale_factor

    # Translate points back to the original position
    scaled_points += centroid

    return scaled_points


@guard_cgal_available
def tri_mesh_to_cgal_mesh(points: np.ndarray, tris: np.ndarray, scale: float = 1):
    """
    Convert a triangle mesh to a CGAL Polyhedron_3 object.
    This function is used to create a CGAL mesh from a set of points and triangles.
    It scales the points from their centroid by a given scale factor.

    Parameters
    ----------
    points
        An array of shape (n, 3) representing the x, y, z coordinates of the points.
    tris
        An array of shape (m, 3) representing the indices of the points that form
        the triangles.
    scale
        The scale factor by which to scale the points from their centroid.

    Returns
    -------
    Polyhedron_3
        A CGAL Polyhedron_3 object representing the mesh.

    Raises
    ------
    ImportError
        If CGAL is not available, an ImportError is raised.
    """
    points = np.asarray(points)
    points = _scale_points_from_centroid(points, scale)
    pt_3_vec = Point_3_Vector()
    pt_3_vec.reserve(3)
    for p in points:
        pt_3_vec.append(Point_3(p[0], p[1], p[2]))
    poly_vec = Polygon_Vector()
    poly_vec.reserve(len(tris))
    for t in tris:
        poly = Int_Vector()
        poly.reserve(3)
        poly.append(int(t[0]))
        poly.append(int(t[1]))
        poly.append(int(t[2]))
        poly_vec.append(poly)
    p = Polyhedron_3()
    polygon_soup_to_polygon_mesh(pt_3_vec, poly_vec, p)
    return p


@guard_cgal_available
def polys_collide(
    mesh_a: Polyhedron_3,
    mesh_b: Polyhedron_3,
) -> bool:
    """
    Check if two CGAL Polyhedron_3 objects collide.

    Parameters
    ----------
    mesh_a
        The first CGAL Polyhedron_3 object.
    mesh_b
        The second CGAL Polyhedron_3 object.

    Returns
    -------
    bool
        True if the two meshes collide, False otherwise.
    """
    return do_intersect(mesh_a, mesh_b)
