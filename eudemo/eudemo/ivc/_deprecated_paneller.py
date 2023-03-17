from itertools import count
from typing import List, Tuple

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
)
from bluemira.utilities.optimiser import Optimiser, approx_derivative

DEG_TO_RAD = np.pi / 180


class Paneller:
    def __init__(
        self, x: np.ndarray, z: np.ndarray, angle: float, dx_min: float, dx_max: float
    ):
        self.x = x
        self.z = z

        tx, tz = tangent(self.x, self.z)
        length_norm = lengthnorm(self.x, self.z)

        self.loop = {
            "x": InterpolatedUnivariateSpline(length_norm, x),
            "z": InterpolatedUnivariateSpline(length_norm, z),
        }
        self.tangent = {
            "x": InterpolatedUnivariateSpline(length_norm, tx),
            "z": InterpolatedUnivariateSpline(length_norm, tz),
        }
        points = np.array([x, z]).T
        string, index = make_pivoted_string(
            points, max_angle=angle, dx_min=dx_min, dx_max=50
        )
        self.string = string
        print(string)

        self.n_opt = string.shape[0] - 2
        self.n_constraints = self.n_opt - 1 + 2 * (self.n_opt + 2)
        self.x_opt = length_norm[index][1:-1]
        self.dl_limit = {"min": dx_min, "max": dx_max}
        self.d2 = None
        self.bounds = np.vstack([np.zeros(self.n_opt), np.ones(self.n_opt)])

        self.it = 1

    def length(self, x: np.ndarray, index: int) -> float:
        p_corner = self.corners(x)[0]
        d_l = np.sqrt(np.diff(p_corner[:, 0]) ** 2 + np.diff(p_corner[:, 1]) ** 2)
        length = np.sum(d_l)
        data = [length, d_l]
        return data[index]

    def corners(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the corner points and indices
        """
        x = np.sort(x)
        x = np.append(np.append(0, x), 1)
        p_corner = np.zeros((len(x) - 1, 2))  # corner points
        p_o = np.array([self.loop["x"](x), self.loop["z"](x)]).T
        t_o = np.array([self.tangent["x"](x), self.tangent["z"](x)]).T
        for i in range(self.n_opt + 1):
            p_corner[i] = vector_intersect(
                p_o[i], p_o[i] + t_o[i], p_o[i + 1], p_o[i + 1] + t_o[i + 1]
            )
        p_corner = np.append(p_o[0].reshape(1, 2), p_corner, axis=0)
        p_corner = np.append(p_corner, p_o[-1].reshape(1, 2), axis=0)
        return p_corner, p_o

    def f_objective(self, x: np.ndarray, grad: np.ndarray) -> float:
        length = self.length(x, 0)
        if grad.size > 0:
            grad[:] = approx_derivative(self.length, x, bounds=self.bounds, args=(0,))
        self.it += 1
        return length

    def constrain_length(self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray):
        d_l_space = 1e-5  # minimum inter-point spacing
        if grad.size > 0:
            grad[:] = np.zeros((self.n_constraints, self.n_opt))  # initalise
            for i in range(self.n_opt - 1):  # order points
                grad[i, i] = -1
                grad[i, i + 1] = 1

            for i in range(2 * self.n_opt + 4):
                grad[self.n_opt - 1 + i, :] = approx_derivative(
                    self.set_min_max_constraints,
                    x,
                    bounds=self.bounds,
                    args=(np.zeros(2 * self.n_opt + 4), i),
                )

        # Where is the constraint on the angles?
        constraint[: self.n_opt - 1] = (
            x[: self.n_opt - 1] - x[1 : self.n_opt]  # + d_l_space
        )
        self.set_min_max_constraints(x, constraint[self.n_opt - 1 :], 0)
        return constraint

    def set_min_max_constraints(
        self, x: np.ndarray, cmm: np.ndarray, index: int
    ) -> float:
        d_l = self.length(x, 1)
        cmm[: self.n_opt + 2] = self.dl_limit["min"] - d_l
        cmm[self.n_opt + 2 :] = d_l - self.dl_limit["max"]
        return cmm[index]


def tangent(x, z):
    """
    Returns tangent vectors along an anticlockwise X, Z loop
    """
    d_x, d_z = np.gradient(x), np.gradient(z)
    mag = np.sqrt(d_x**2 + d_z**2)
    index = mag > 0
    d_x, d_z, mag = d_x[index], d_z[index], mag[index]  # clear duplicates
    t_x, t_z = d_x / mag, d_z / mag
    return t_x, t_z


def lengthnorm(x, z):
    """
    Return a normalised 1-D parameterisation of an X, Z loop.

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    total_length: np.array(N)
        The cumulative normalised length of each individual segment in the loop
    """
    total_length = length(x, z)
    return total_length / total_length[-1]


def length(x, z):
    """
    Return a 1-D parameterisation of an X, Z loop.

    Parameters
    ----------
    x: array_like
        x coordinates of the loop [m]
    z: array_like
        z coordinates of the loop [m]

    Returns
    -------
    lengt: np.array(N)
        The cumulative length of each individual segment in the loop
    """
    lengt = np.append(0, np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    return lengt


def vector_intersect(p1, p2, p3, p4):
    """

    Parameters
    ----------
    p1: np.array(2)
        The first point on the first vector
    p2: np.array(2)
        The second point on the first vector
    p3: np.array(2)
        The first point on the second vector
    p4: np.array(2)
        The second point on the second vector

    Returns
    -------
    p_inter: np.array(2)
        The point of the intersection between the two vectors
    """
    da = p2 - p1
    db = p4 - p3

    if np.isclose(np.cross(da, db), 0):  # vectors parallel
        # NOTE: careful modifying this, different behaviour required...
        point = p2
    else:
        dp = p1 - p3
        dap = normal_vector(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        point = num / denom.astype(float) * db + p3
    return point


def normal_vector(side_vectors):
    """
    Anti-clockwise

    Parameters
    ----------
    side_vectors: np.array(N, 2)
        The side vectors of a polygon

    Returns
    -------
    a: np.array(2, N)
        The array of 2-D normal vectors of each side of a polygon
    """
    a = -np.array([-side_vectors[1], side_vectors[0]]) / np.sqrt(
        side_vectors[0] ** 2 + side_vectors[1] ** 2
    )
    a[np.isnan(a)] = 0
    return a


class PanellingOptProblem(OptimisationProblem):
    def __init__(self, paneller: Paneller, optimiser: Optimiser):
        self.paneller = paneller
        objective = OptimisationObjective(self.paneller.f_objective)
        constraint = OptimisationConstraint(
            self.paneller.constrain_length,
            f_constraint_args={},
            tolerance=np.full(self.paneller.n_constraints, 1e-8),
        )

        super().__init__(self.paneller.x_opt, optimiser, objective, [constraint])

        n_variables = self.paneller.n_opt
        self.set_up_optimiser(n_variables, bounds=paneller.bounds)

    def optimise(self):
        self.paneller.x_opt = self.opt.optimise(self.paneller.x_opt)
        return self.paneller.x_opt


def make_pivoted_string(
    boundary_points: np.ndarray,
    max_angle: float = 10,
    dx_min: float = 0,
    dx_max: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a set of pivot points along the given boundary.

    Given a set of boundary points, some maximum angle, and minimum and
    maximum segment length, this function derives a set of pivot points
    along the boundary, that define a 'string'. You might picture a
    'string' as a thread wrapped around some nails (pivot points) on a
    board.

    Parameters
    ----------
    points
        The coordinates (in 3D) of the pivot points. Must have shape
        (N, 3) where N is the number of boundary points.
    max_angle
        The maximum angle between neighbouring pivot points.
    dx_min
        The minimum distance between pivot points.
    dx_max
        The maximum distance between pivot points.

    Returns
    -------
    new_points
        The pivot points' coordinates. Has shape (M, 3), where M is the
        number of pivot points.
    index
        The indices of the pivot points into the input points.
    """
    if dx_min > dx_max:
        raise ValueError(
            f"'dx_min' cannot be greater than 'dx_max': '{dx_min} > {dx_max}'"
        )
    tangent_vec = boundary_points[1:] - boundary_points[:-1]
    tangent_vec_norm = np.linalg.norm(tangent_vec, axis=1)
    # Protect against dividing by zero
    tangent_vec_norm[tangent_vec_norm == 0] = 1e-32
    average_step_length = np.median(tangent_vec_norm)
    tangent_vec /= tangent_vec_norm.reshape(-1, 1) * np.ones(
        (1, np.shape(tangent_vec)[1])
    )

    new_points = np.zeros_like(boundary_points)
    index = np.zeros(boundary_points.shape[0], dtype=int)
    delta_x = np.zeros_like(boundary_points)
    delta_turn = np.zeros_like(boundary_points)

    new_points[0] = boundary_points[0]
    to, po = tangent_vec[0], boundary_points[0]

    k = count(1)
    for i, (p, t) in enumerate(zip(boundary_points[1:], tangent_vec)):
        c = np.cross(to, t)
        c_mag = np.linalg.norm(c)
        dx = np.linalg.norm(p - po)  # segment length
        if c_mag > np.sin(max_angle * DEG_TO_RAD) and dx > dx_min:
            j = next(k)
            new_points[j] = boundary_points[i]  # pivot point
            index[j] = i + 1  # pivot index
            delta_x[j - 1] = dx  # panel length
            delta_turn[j - 1] = np.arcsin(c_mag) / DEG_TO_RAD
            to, po = t, p  # update
    if dx > dx_min:
        j = next(k)
        delta_x[j - 1] = dx  # last segment length
    else:
        delta_x[j - 1] += dx  # last segment length
    new_points[j] = p  # replace/append last point
    index[j] = i + 1  # replace/append last point index
    new_points = new_points[: j + 1]  # trim
    index = index[: j + 1]  # trim
    delta_x = delta_x[:j]  # trim
    return new_points, index


if __name__ == "__main__":
    theta = np.linspace(0, np.pi, 10)
    x = np.cos(theta)
    z = np.sin(theta)

    import matplotlib.pyplot as plt

    paneller = Paneller(x, z, 20, 0.5, 10)
    print(paneller.x_opt)
    print(paneller)

    # paneller.corners(paneller)
