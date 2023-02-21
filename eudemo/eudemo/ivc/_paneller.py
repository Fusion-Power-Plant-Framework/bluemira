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
from eudemo.ivc.panelling import make_pivoted_string


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
            points, max_angle=angle, dx_min=dx_min, dx_max=dx_max
        )

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
            # grad[:] = approx_fprime(x, self.length, 1e-6, self.bounds, 0)
        print(f"\n{self.it}.\n   x: {x}\n   f: {length}\n   g: {grad}")
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
                # grad[self.n_opt - 1 + i, :] = approx_fprime(
                #     x,
                #     self.set_min_max_constraints,
                #     1e-6,
                #     self.bounds,
                #     np.zeros(2 * self.n_opt + 4),
                #     i,
                # )
                grad[self.n_opt - 1 + i, :] = approx_derivative(
                    self.set_min_max_constraints,
                    x,
                    bounds=self.bounds,
                    args=(np.zeros(2 * self.n_opt + 4), i),
                )

        constraint[: self.n_opt - 1] = (
            x[: self.n_opt - 1] - x[1 : self.n_opt] + d_l_space
        )
        print(f"   c: {constraint}\n  cg: {grad}")
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
        paneller.x_opt = self.opt.optimise(paneller.x_opt)
        return paneller.x_opt


def approx_fprime(xk, func, epsilon, bounds, *args, f0=None):
    """
    An altered version of a scipy function, but with the added feature
    of clipping the perturbed variables to be within their prescribed bounds

    Parameters
    ----------
    xk: array_like
        The state vector at which to compute the Jacobian matrix.
    func: callable f(x,*args)
        The vector-valued function.
    epsilon: float
        The perturbation used to determine the partial derivatives.
    bounds: array_like(len(xk), 2)
        The bounds the variables to respect
    args: sequence
        Additional arguments passed to func.
    f0: Union[float, None]
        The initial value of the function at x=xk. If None, will be calculated

    Returns
    -------
    grad: array_like(len(func), len(xk))
        The gradient of the func w.r.t to the perturbed variables

    Notes
    -----
    The approximation is done using forward differences.
    """
    if f0 is None:
        f0 = func(*((xk,) + args))

    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for i in range(len(xk)):
        ei[i] = 1.0
        # The delta value to add the the variable vector
        d = epsilon * ei
        # Clip the perturbed variable vector with the variable bounds
        xk_d = np.clip(xk + d, bounds[0], bounds[1])

        # Get the clipped length of the perturbation
        delta = xk_d[i] - xk[i]

        if delta == 0:
            df = 0
        else:
            df = (func(*((xk_d,) + args)) - f0) / delta

        if not np.isscalar(df):
            try:
                df = df.item()
            except (ValueError, AttributeError):
                raise ValueError(
                    "The user-provided objective function must return a scalar value."
                )
        grad[i] = df
        ei[i] = 0.0
    return grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.display import plot_2d
    from bluemira.geometry.tools import make_polygon

    x = np.load("/home/bf2936/bluemira/code/BLUEPRINT/hull_x.npy")
    z = np.load("/home/bf2936/bluemira/code/BLUEPRINT/hull_z.npy")
    res_x = np.load("/home/bf2936/bluemira/code/BLUEPRINT/result_x.npy")
    res_z = np.load("/home/bf2936/bluemira/code/BLUEPRINT/result_z.npy")

    # paneller = Paneller(x, z, angle=20, dx_min=0.5, dx_max=2.5)

    coords = np.array([x, np.zeros_like(x), z])
    wire = make_polygon(coords)
    boundary = wire.discretize(400, byedges=True)
    paneller = Paneller(boundary.x, boundary.z, 20, 0.5, 2.5)

    initial_points = paneller.corners(paneller.x_opt)[0].T

    optimiser = Optimiser(
        "COBYLA",
        n_variables=paneller.n_opt,
        opt_conditions={"max_eval": 400, "ftol_rel": 1e-4},  # , "xtol_rel": 1e-4},
    )
    opt_problem = PanellingOptProblem(paneller, optimiser)
    x_opt = opt_problem.optimise()

    points = paneller.corners(x_opt)[0].T

    print("Initial and optimised equal:", np.allclose(initial_points, points))
    print("Equal to BLUEPRINT:", np.allclose(points, np.vstack([res_x, res_z])))
    print("BLUEPRINT len:", length(res_x, res_z)[-1])
    print(" bluemira len:", length(points[0], points[1])[-1])

    _, ax = plt.subplots()
    # plot_2d(wire, ax=ax, show=False)
    ax.plot(x, z, color="k")
    # ax.plot(initial_points[0], initial_points[1], "-o", color="r")
    ax.plot(points[0], points[1], "-o", color="g")
    ax.set_aspect("equal", adjustable="box")

    _, ax2 = plt.subplots()
    ax2.set_title("BLUEPRINT reference")
    ax2.plot(x, z, color="k")
    ax2.plot(res_x, res_z, "-o", color="g")
    ax2.set_aspect("equal", adjustable="box")

    plt.show()
