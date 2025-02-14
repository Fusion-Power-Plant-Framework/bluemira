import numpy as np


def is_mutually_exclusive(
    min_: np.ndarray[float], max_: np.ndarray[float]
) -> np.ndarray:
    """
    Given a list of bounds, (e.g. x-bounds, i.e. x-min and x-max for each cell),
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
    l = len(min_)
    matrix_min = np.broadcast_to(min_, (l, l)).T
    matrix_max = np.broadcast_to(max_, (l, l)).T
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
    x_bounds = tensor_3d[:, 0, :]
    y_bounds = tensor_3d[:, 1, :]
    z_bounds = tensor_3d[:, 2, :]

    return np.array([
        is_mutually_exclusive(*x_bounds),
        is_mutually_exclusive(*y_bounds),
        is_mutually_exclusive(*z_bounds),
    ]).any(axis=0)


def get_overlaps(exclusivity_matrix) -> np.ndarray:
    """
    Get the indices of the bounding boxes that are overlapping. The overlap matrix is the
    element-wise negation of the exclusivity matrix. This function returns the 2-D
    indices of non-zero elements on the upper-right triangle of this matrix.

    Parameters
    ----------
    exclusivity_matrix:
        The matrix denoting whether each bounding box overlap with other bounding boxes,
        generated by check_bb_non_interference.

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
