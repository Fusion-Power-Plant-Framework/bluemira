import numpy as np
import numba as nb

from bluemira.base.constants import EPS
from bluemira.codes.error import FreeCADError


def coords_triang_mutually_exclusive(
    val_a: np.ndarray[float],
    min_b: np.ndarray[float],
    max_b: np.ndarray[float],
) -> np.ndarray:
    """
    Given some coordinates (A) and the bounds of some triangles (B) (e.g. x-bounds, i.e.
    x-min and x-max for each triangle), find whether each point lies outside the bounding
    box of the triangle. This forms a 2D exclusivity matrix.

    Parameters
    ----------
    val_a:
        actaul value for each element in set A, a 1D array.
    min_b:
        lower bound for each triangle in set B, a 1D array.
    max_b:
        upper bound for each triangle in set B, a 1D array.


    Returns
    -------
    :
        A 2D exclusivity matrix showing True where they're NOT overlapping, False if
        overlapping. The main-diagonal of this matrix can be ignored.

    Note
    ----
    Must have the property that all(min_b<=max_b).
    """
    len_a = len(val_a)
    len_b = len(min_b)
    matrix_val_a = np.broadcast_to(val_a, (len_b, len_a)).T
    matrix_min_b = np.broadcast_to(min_b, (len_a, len_b))
    matrix_max_b = np.broadcast_to(max_b, (len_a, len_b))
    return np.logical_or(matrix_val_a < matrix_min_b, matrix_val_a > matrix_max_b)


@nb.jit(nopython=True, cache=True)
def check_if_point_below_triangle(point: np.ndarray[float], triangle: np.ndarray) -> int:
    """
    Checks if a point is situated below the triangle or not.

    Parameters
    ----------
    point:
        The 3D point that you'd like to check for overlaps.
    triangle:
        The triangle's vertices stored as a 3x3 (3 vertices, xyz coordinates) 2D matrix.

    Returns
    -------
    :
        An integer that has the following meanings:
        -1 for it's ON the triangle.
        0 for it's not directly below the triangle.
        1 for it's below the triangle's face
        2 for it's below one of the triangle's edge
        3 for it's below one of the triangle's vertices
    """
    t0 = triangle[0]
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    (ax, ay, az), (bx, by, bz) = a, b
    # array = np.array([[ax, bx], [ay, by]])  # linalg
    # det = np.linalg.det(array)
    det = ax * by - bx * ay
    # For shapes that aren't triangle:
    # retry this step with different choices of indices in a and b until det>0.
    if det > EPS:
        d = point[:2] - t0[:2]
        # u, v = uv = np.linalg.inv(array) @ d
        u = (by * d[0] - ay * d[1]) / det
        v = (-bx * d[0] + ax * d[1]) / det

        if 0 <= u <= 1 and 0 <= v <= 1:
            height = t0[2] + az * u + bz * v
            # height = np.array([az, bz]) @ uv  # linalg
            # if (uv==1).any(axis=-1) or (uv==0).all(axis=-1):  # linalg
            if u == 1 or v == 1 or (u == 0 and v == 0):
                if point[2] == height:
                    return -1
                if point[2] < height:
                    return 3
            # elif uv.sum(axis=-1)==1 or (uv==0).any(axis=-1):  # linalg
            elif ((u + v) == 1) or u == 0 or v == 0:
                if point[2] == height:
                    return -1
                if point[2] < height:
                    return 2
            # elif uv.sum(axis=-1)<1:  # linalg
            elif (u + v) < 1:
                if point[2] == height:
                    return -1
                if point[2] < height:
                    return 1
    return 0


@nb.jit(nopython=True)
def point_in_triangles_mesh(point: np.ndarray[float], mesh: np.ndarray) -> bool:
    """
    Checks if a point is inside a triangular surface mesh or not.

    Parameters
    ----------
    point:
        The 3D point that you'd like to check for overlaps.
    mesh:
        A list of triangles, stored as a Nx3x3 (N triangles, 3 vertices each, xyz
        coordinates for each vertex) matrix.


    Returns
    -------
    :
        A boolean indicating whether the point is inside the mesh or not.

    Raises
    ------
    FreeCADError
        Raised when the number of crossings to edges and vertices does not match
        expectation.

    Notes
    -----
    Method used:
        Create a parametric plane (anchored by the 0th vertex of the triangle, + two
        parametric vectors u*a + v * b) for each triangle. Draw a line from the point
        upwards towards +z infinity. If this line intersect the triangle, then we count
        intersection +=1. If the total number of intersection with all triangles is odd,
        then the point exists inside the solid. Otherwise, it exists outside the solid.
    """
    in_face, on_edge, at_vertex = 0, 0, 0

    for triangle in mesh:
        result = check_if_point_below_triangle(point, triangle)
        if result == -1:
            return True
        if result == 1:
            in_face += 1
        if result == 2:  # noqa: PLR2004
            on_edge += 1
        if result == 3:  # noqa: PLR2004
            at_vertex += 1
    if on_edge % 2 != 0:
        raise FreeCADError(
            f"{on_edge} triangles have their edges struck by the line projected to +z "
            "infinity, which should've been even, but is odd!"
        )
    if at_vertex % 3 != 0:
        raise FreeCADError(
            "We assumed each vertex hit by the projection point is"
            f"shared by exactly 3 triangles, but instead we have {at_vertex} triangles "
            f"where one of their vertices was hit, suggesting that {at_vertex / 3} "
            "vertices were struck by the imaginary line projected towards +z infinity."
        )
    num_surfaces_passed_through = in_face + on_edge / 2 + at_vertex / 3

    return num_surfaces_passed_through % 2 == 1.0


def point_in_solid(
    set_of_points: np.ndarray, triangle_bb: np.ndarray, cad_solid: np.ndarray
) -> np.ndarray[bool] | bool:
    """
    Parameters
    ----------
    set_of_points:
        Set of points (2D array of shape Nx3) each of which we want to check if it exists
        inside cad_solid or not.
    triangle_bb:
        An array of shape N, 3, 2. For each of the N 2D arrays (each with shape = (3,2)),
        each row is the x, y, z bounds (min, max) for that cell.
    cad_solid:
        Triangular mesh that together completely covers the surface of the CAD solid.
        Shape == N, 3, 3.
        Each bounding box in triangle_bb corresponds to each triangle in cad_solid.

    Returns
    -------
    :
        a boolean array, denoting if each point is a thing/
        a boolean, denoting if ANY point is in the cad_solid.
        This one requires further consulting with Oli.
    """
    bb_x_bounds, bb_y_bounds, _ = triangle_bb.transpose([1, 2, 0])

    mutually_exclusive = np.array([
        coords_triang_mutually_exclusive(set_of_points[:, 0], *bb_x_bounds),
        coords_triang_mutually_exclusive(set_of_points[:, 1], *bb_y_bounds),
    ]).any(axis=0)
    boolean_result = []
    for point, within_reach_of_point in zip(
        set_of_points, ~mutually_exclusive, strict=False
    ):
        # if point_in_triangles_mesh(point, cad_solid[within_reach_of_point]):
        #     return True
        boolean_result.append(
            point_in_triangles_mesh(point, cad_solid[within_reach_of_point])
        )
    # else:
    #     return False
    return boolean_result
