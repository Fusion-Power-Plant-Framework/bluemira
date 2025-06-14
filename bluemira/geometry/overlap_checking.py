# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Contains functions to efficiently check for overlaps between solids."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

import bluemira.codes.cgal_ext as cgal
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.geometry.constants import D_TOLERANCE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.geometry.bound_box import BoundingBox
    from bluemira.geometry.solid import BluemiraSolid


def two_set_mutually_exclusive(
    min_a: np.ndarray[float],
    max_a: np.ndarray[float],
    min_b: np.ndarray[float],
    max_b: np.ndarray[float],
) -> np.ndarray:
    """
    Given TWO lists of bounds, (e.g. x-bounds, i.e. x-min and x-max for each cell),
    find whether each cell is mutually exclusive (i.e. does NOT overlap) with other
    cells. This forms a 2D exclusivity matrix.

    Parameters
    ----------
    min_a:
        lower bound for each cell in set A, a 1D array.
    max_a:
        upper bound for each cell in set A, a 1D array.
    min_b:
        lower bound for each cell in set B, a 1D array.
    max_b:
        upper bound for each cell in set B, a 1D array.


    Returns
    -------
    :
        A 2D exclusivity matrix showing True where they're NOT overlapping, False if
        overlapping. The main-diagonal of this matrix can be ignored.

    Note
    ----
    Must have the property that all(min_a<=max_a) and all(min_b<=max_b).
    """
    len_a = len(min_a)
    len_b = len(min_b)
    matrix_min_a = np.broadcast_to(min_a, (len_b, len_a)).T
    matrix_max_a = np.broadcast_to(max_a, (len_b, len_a)).T
    matrix_min_b = np.broadcast_to(min_b, (len_a, len_b))
    matrix_max_b = np.broadcast_to(max_b, (len_a, len_b))
    return np.logical_or(matrix_max_a < matrix_min_b, matrix_min_a > matrix_max_b)


def check_two_sets_bb_non_interference(
    set_a_3d_tensor: np.ndarray, set_b_3d_tensor: np.ndarray
) -> np.ndarray:
    """
    Check which bounding box do not interfere/overlap with which other bounding box.

    Parameters
    ----------
    set_a_3d_tensor:
        An array of 2D arrays (each with shape = (3,2)), each row of the 2D array is
        the x, y, z bounds (min, max) for that set A cell.
    set_b_3d_tensor:
        An array of 2D arrays (each with shape = (3,2)), each row of the 2D array is
        the x, y, z bounds (min, max) for that set B cell.

    Returns
    -------
    exclusivity_matrix:
        A matrix of booleans showing whether the bounding boxes overlap.
    """
    x_bounds_a, y_bounds_a, z_bounds_a = set_a_3d_tensor.transpose([1, 2, 0])
    x_bounds_b, y_bounds_b, z_bounds_b = set_b_3d_tensor.transpose([1, 2, 0])

    return np.array([
        two_set_mutually_exclusive(*x_bounds_a, *x_bounds_b),
        two_set_mutually_exclusive(*y_bounds_a, *y_bounds_b),
        two_set_mutually_exclusive(*z_bounds_a, *z_bounds_b),
    ]).any(axis=0)


def get_overlaps_asymmetric(exclusivity_matrix) -> np.ndarray:
    """
    Get the indices of the bounding boxes that are overlapping. The overlap matrix is the
    element-wise negation of the exclusivity matrix. This function returns the 2-D
    indices of non-zero elements.

    Parameters
    ----------
    exclusivity_matrix:
        The matrix denoting whether each bounding box overlap with other bounding boxes,
        generated by :func:`~check_two_sets_bb_non_interference`.

    Returns
    -------
    indices:
        2D array of integers, each row is a pair of indices of i<j
    """
    i, j = np.where(~exclusivity_matrix)
    return np.array([i, j]).T


def is_mutually_exclusive(
    min_: np.ndarray[float], max_: np.ndarray[float]
) -> np.ndarray:
    """
    Given a list of bounds, (e.g. x-bounds, showing .xmin() and .xmax() for each cell),
    find whether each cell is mutually exclusive (i.e. does NOT overlap) with other
    cells. This forms a 2D exclusivity matrix.

    Parameters
    ----------
    min_:
        lower bound for each cell, a 1D array.
    max_:
        upper bound for each cell, a 1D array.

    Returns
    -------
    :
        A 2D exclusivity matrix showing True where they're NOT overlapping, False if
        overlapping. The main-diagonal of this matrix can be ignored.

    Note
    ----
    Must have the property that all(min_<=max_)==True.
    """
    len_ = len(min_)
    matrix_min = np.broadcast_to(min_, (len_, len_)).T
    matrix_max = np.broadcast_to(max_, (len_, len_)).T
    return np.logical_or(matrix_max < min_, matrix_min > max_)


# Brancheless implementation
def check_bb_non_interference(tensor_3d: np.ndarray) -> np.ndarray:
    """
    Check which bounding box do not interfere/overlap with which other bounding box.

    Parameters
    ----------
    tensor_3d:
        An array of 2D arrays (each with shape = (3,2)), each row of the 2D array is
        the x, y, z bounds (min, max) for that cell.

    Returns
    -------
    exclusivity_matrix:
        A matrix of booleans showing whether the bounding boxes overlap.
    """
    x_bounds, y_bounds, z_bounds = tensor_3d.transpose([1, 2, 0])

    return np.array([
        is_mutually_exclusive(*x_bounds),
        is_mutually_exclusive(*y_bounds),
        is_mutually_exclusive(*z_bounds),
    ]).any(axis=0)


def get_overlaps_arr(exclusivity_matrix) -> np.ndarray:
    """
    Get the indices of the bounding boxes that are overlapping. The overlap matrix is the
    element-wise negation of the exclusivity matrix. This function returns the 2-D
    indices of non-zero elements on the upper-right triangle of this matrix.

    Parameters
    ----------
    exclusivity_matrix:
        The matrix denoting whether each bounding box overlap with other bounding boxes,
        generated by :func:`~check_bb_non_interference`.

    Returns
    -------
    indices:
        2D array of integers, each row is a pair of indices of i<j
    """
    i, j = np.where(~exclusivity_matrix)
    # only return the upper-triangle part of the matrix.
    duplicates = i >= j
    i, j = i[~duplicates], j[~duplicates]
    return np.array([i, j]).T


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


def find_approx_overlapping_pairs(
    solids: Iterable[BluemiraSolid], *, use_cgal: bool = True
):
    """Finds the pairs of solids that are approximately overlapping.

    This function uses bounding boxes to quickly eliminate non-overlapping pairs,
    and then refines the results using a numpy (or CGAL, if installed) based apporach
    to refind the number of overlapping pairs taking into the geometry,
    in a confirmal manner.

    Parameters
    ----------
    solids:
        An iterable of BluemiraSolid objects to check for overlaps between.

    use_cgal:
        If True, use CGAL to check for overlaps. Otherwise, use numpy.
        If CGAL is not available, this will default to numpy (even if True).

    Returns
    -------
        A list of tuples of indices of overlapping pairs.

    """

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

    if use_cgal and not cgal.cgal_available():
        use_cgal = False
        bluemira_debug(
            "CGAL is not available. Falling back to numpy based overlap checking."
            "`pip install cgal` for a faster more accurate result."
        )

    for solid in solids:
        aabbs.append(to_bb_matrix(solid.bounding_box))

        tsl = solid._tessellate(1)
        points = tsl[0]
        tris = tsl[1]
        if use_cgal:
            # we scale the geometry by 1.1 to slightly over-approximate
            # intersections later (need to error on the side of caution).
            approx_geometry.append(cgal.tri_mesh_to_cgal_mesh(points, tris, scale=1.1))
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
    ex_mat = check_bb_non_interference(aabbs)
    overlap_idxs = get_overlaps_arr(ex_mat)

    def refine_overlap(i, j):
        geo_i = approx_geometry[i]
        geo_j = approx_geometry[j]
        itc = (
            cgal.do_polys_collide(geo_i, geo_j)
            if use_cgal
            else np.any(check_two_sets_bb_non_interference(geo_i, geo_j))
        )
        return (i, j) if itc else None

    with ThreadPoolExecutor() as executor:
        refined_overlap_idxs = list(
            filter(None, executor.map(lambda pair: refine_overlap(*pair), overlap_idxs))
        )

    return refined_overlap_idxs  # noqa: RET504
