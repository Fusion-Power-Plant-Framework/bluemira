from collections.abc import Iterable

import numpy as np

from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.solid import BluemiraSolid

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

    cgal_available = True
except ImportError:
    cgal_available = False


def scale_points_from_centroid(points, scale_factor):
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
        An array of shape (n, 3) representing the scaled x, y, z coordinates of the points.
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


def tri_mesh_to_cgal_mesh(points, tris, scale=1):
    if not cgal_available:
        raise ImportError(
            "CGAL is not available. Please install it to use this function."
        )
    points = np.asarray(points)
    points = scale_points_from_centroid(points, scale)
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


def find_approx_overlapping_pairs(solids: Iterable[BluemiraSolid]):
    """Finds the pairs of solids that are approximately overlapping."""

    def to_bb_matrix(bb: BoundingBox):
        return np.array([
            [bb.x_min, bb.x_max],
            [bb.y_min, bb.y_max],
            [bb.z_min, bb.z_max],
        ])

    def bb_of_tri_coords(pt_a, pt_b, pt_c):
        points = np.array([pt_a, pt_b, pt_c])
        tri_bb = np.array([
            [np.min(points[:, 0]), np.max(points[:, 0])],
            [np.min(points[:, 1]), np.max(points[:, 1])],
            [np.min(points[:, 2]), np.max(points[:, 2])],
        ])
        tri_bb[np.isclose(tri_bb, 0, atol=D_TOLERANCE)] = 0
        return tri_bb

    aabbs = []
    approx_geometry = []

    for solid in solids:
        aabbs.append(to_bb_matrix(solid.bounding_box))

        tsl = solid._tessellate(1)
        points = tsl[0]
        tris = tsl[1]
        if cgal_available:
            # we scale the geometry by 1.1 to slightly over-approximate
            # intersections later (need to error on the side of caution).
            approx_geometry.append(tri_mesh_to_cgal_mesh(points, tris, scale=1.1))
        else:
            # we build bbs around each triangle
            # which creates an approximation of the geometry.
            # bbs can quickly be checked for intersection.
            approx_geometry.append(
                np.array([
                    bb_of_tri_coords(
                        points[tri[0]],
                        points[tri[1]],
                        points[tri[2]],
                    )
                    for tri in tris
                ])
            )

    aabbs = np.array(aabbs)
