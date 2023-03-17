# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
from itertools import count
from typing import Tuple, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from bluemira.utilities.optimiser import approx_derivative

DEG_TO_RAD = np.pi / 180


class Paneller:
    """
    Provides functions for generating panelling along the outside of a boundary

    Parameters
    ----------
    boundary_points
        The points defining the boundary along which to build the panels.
        This should have shape (2, N), where N is the number of points.
    """

    def __init__(self, boundary_points: np.ndarray, max_angle: float, dx_min: float):
        self.max_angle = max_angle
        self.dx_min = dx_min

        length_norm = norm_lengths(boundary_points)
        self._x_boundary_spline = InterpolatedUnivariateSpline(
            length_norm, boundary_points[0]
        )
        self._z_boundary_spline = InterpolatedUnivariateSpline(
            length_norm, boundary_points[1]
        )

        tangent_norm = norm_tangents(boundary_points)
        self._x_tangent_spline = InterpolatedUnivariateSpline(
            length_norm, tangent_norm[0]
        )
        self._z_tangent_spline = InterpolatedUnivariateSpline(
            length_norm, tangent_norm[1]
        )

        # Build the initial guess of our panels, these points are the
        # coordinates of where the panels tangent the boundary
        _, idx = make_pivoted_string(
            boundary_points.T,
            max_angle=max_angle,
            dx_min=dx_min,
        )
        self.n_points = len(idx)
        self.x0: np.ndarray = length_norm[idx][1:-1]

    def x_boundary(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find the x-coordinate at a given normalised distance along the boundary."""
        return self._x_boundary_spline(dist)

    def z_boundary(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find the z-coordinate at a given normalised distance along the boundary."""
        return self._z_boundary_spline(dist)

    def x_boundary_tangent(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find the x-coordinate of the tangent vector at the given distance along the boundary."""
        return self._x_tangent_spline(dist)

    def z_boundary_tangent(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find the z-coordinate of the tangent vector at the given distance along the boundary."""
        return self._z_tangent_spline(dist)

    @property
    def n_opts(self) -> int:
        """
        The number of optimisation parameters.

        The optimisation parameters are how far along the boundary's
        length each panel tangents the boundary. We exclude the start
        and end points which are fixed.
        """
        # exclude start and end points; hence 'N - 2'
        return self.n_points - 2

    @property
    def n_constraints(self) -> int:
        """
        The number of optimisation constraints.

        We constrain:

            - the minimum length of each panel
              (no. of panels = no. of touch points + 2)
            - the angle between each panel
              (no. of angles = no. of touch points + 1)
        """
        return 2 * self.n_opts + 4

    def joints(self, dists: np.ndarray) -> np.ndarray:
        """
        Calculate panel joint coordinates from panel-boundary tangent points.

        Parameters
        ----------
        dists
            The normalised distances along the boundary at which there
            are panel-boundary tangent points.
        """
        # Add the start and end panel joints at distances 0 & 1
        dists = np.sort(np.hstack((0, dists, 1)))
        points = np.vstack((self.x_boundary(dists), self.z_boundary(dists)))
        tangents = np.vstack(
            (self.x_boundary_tangent(dists), self.z_boundary_tangent(dists))
        )
        # TODO(hsaunders1904): vectorize
        #  https://stackoverflow.com/a/40637858
        joints = np.zeros((2, len(dists) + 1))
        joints[:, 0] = points[:, 0]
        joints[:, -1] = points[:, -1]
        for i in range(joints.shape[1] - 2):
            joints[:, i + 1] = vector_intersect(
                points[:, i],
                points[:, i] + tangents[:, i],
                points[:, i + 1],
                points[:, i + 1] + tangents[:, i + 1],
            )
        return joints

    # TODO(hsaunders1904): sort out caching of the joints so we're not
    #  calculating them 3 times for every opt loop
    def length(self, dists: np.ndarray) -> float:
        return self.panel_lengths(dists).sum()

    def angles(self, dists: np.ndarray) -> np.ndarray:
        joints = self.joints(dists)
        line_vectors: np.ndarray = joints[:, 1:] - joints[:, :-1]
        dots = (line_vectors[:, :-1] * line_vectors[:, 1:]).sum(axis=0)
        magnitudes = np.linalg.norm(line_vectors, axis=0)
        dots /= magnitudes[:-1] * magnitudes[1:]
        return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)), out=dots)

    def panel_lengths(self, dists: np.ndarray) -> np.ndarray:
        joints = self.joints(dists)
        return np.hypot(joints[0], joints[1])


def norm_lengths(points: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative normalized lengths between each 2D point.

    Parameters
    ----------
    points
        A numpy array of points, shape should be (2, N).
    """
    dists = np.diff(points, axis=1)
    sq_dists = np.square(dists, out=dists)
    summed_dists = np.sum(sq_dists, axis=0)
    sqrt_dists = np.sqrt(summed_dists, out=summed_dists)
    cumulative_sum = np.cumsum(summed_dists, out=sqrt_dists)
    return np.hstack((0, cumulative_sum / cumulative_sum[-1]))


def norm_tangents(points: np.ndarray) -> np.ndarray:
    """
    Calculate the normalised tangent vector at each of the given points.

    Parameters
    ----------
    points
        Array of coordinates. This must have shape (2, N), where N is
        the number of points.

    Returns
    -------
    tangents
        The normalised vector of tangents.
    """
    grad = np.gradient(points, axis=1)
    magnitudes = np.hypot(grad[0], grad[1])
    return np.divide(grad, magnitudes, out=grad)


def test_norm_lengths_gives_expected_lengths():
    points = np.array([[0, 0], [2, 1], [3, 3], [5, 1], [3, 0]], dtype=float)

    lengths = norm_lengths(points)

    expected = np.array(
        [
            np.sqrt(5),
            np.sqrt(5) + np.sqrt(5),
            np.sqrt(5) + np.sqrt(5) + np.sqrt(8),
            np.sqrt(5) + np.sqrt(5) + np.sqrt(8) + np.sqrt(5),
        ]
    )
    expected /= expected[-1]
    np.testing.assert_allclose(lengths, expected)


def test_tangent_returns_tanget_vectors():
    xz = np.array(
        [
            [
                0.34558419,
                0.82161814,
                0.33043708,
                -1.30315723,
                0.90535587,
                0.44637457,
                -0.53695324,
                0.5811181,
                0.3645724,
                0.2941325,
                0.02842224,
                0.54671299,
            ],
            [
                -0.73645409,
                -0.16290995,
                -0.48211931,
                0.59884621,
                0.03972211,
                -0.29245675,
                -0.78190846,
                -0.25719224,
                0.00814218,
                -0.27560291,
                1.29406381,
                1.00672432,
            ],
        ]
    )

    tngnt = tangent(xz)

    expected = np.array(
        [
            [
                0.63866332,
                -0.05945048,
                -0.94133318,
                0.74046041,
                0.8910329,
                -0.86890311,
                0.96741701,
                0.75207387,
                -0.9979486,
                -0.25290957,
                0.19325713,
                0.87458661,
            ],
            [
                0.7694863,
                0.99823126,
                0.33747866,
                0.67209998,
                -0.45393874,
                -0.49498221,
                0.25318831,
                0.65907882,
                -0.06402027,
                0.96748992,
                0.98114814,
                -0.48486932,
            ],
        ]
    )
    np.testing.assert_allclose(tngnt, expected)


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
        if (
            c_mag > np.sin(max_angle * DEG_TO_RAD) and dx > dx_min
        ) or dx + average_step_length > dx_max:
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


from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
    Optimiser,
)


class PanellingOptProblem(OptimisationProblem):
    def __init__(self, paneller: Paneller, optimiser: Optimiser):
        self.paneller = paneller
        self.bounds = (np.zeros_like(self.paneller.x0), np.ones_like(self.paneller.x0))
        objective = OptimisationObjective(self.objective)
        constraint = OptimisationConstraint(
            self.constrain_min_length_and_angles,
            f_constraint_args={},
            tolerance=np.full(self.paneller.n_constraints, 1e-5),
        )
        super().__init__(self.paneller.x0, optimiser, objective, [constraint])
        self.set_up_optimiser(self.paneller.n_opts, bounds=self.bounds)

    def optimise(self):
        self.paneller.x0 = self.opt.optimise(self.paneller.x0)
        return self.paneller.x0

    def objective(self, x: np.ndarray, grad: np.ndarray) -> float:
        length = self.paneller.length(x)
        if grad.size > 0:
            grad[:] = approx_derivative(
                self.paneller.length, x, bounds=self.bounds, f0=length
            )
        return length

    def constrain_min_length_and_angles(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        # Constrain minimum length
        lengths = self.paneller.panel_lengths(x)
        constraint[: len(lengths)] = self.paneller.dx_min - lengths
        if grad.size > 0:
            # TODO(hsaunders1904): work out what BLUEPRINT was doing to
            #  get this gradient
            grad[: len(lengths)] = approx_derivative(
                lambda x_opt: -self.paneller.panel_lengths(x_opt),
                x0=x,
                f0=constraint[: len(lengths)],
                bounds=self.bounds,
            )

        # Constrain joint angles
        constraint[len(lengths) :] = self.paneller.angles(x) - self.paneller.max_angle
        if grad.size > 0:
            # TODO(hsaunders): I'm sure we can be smarter about this gradient
            grad[len(lengths) :, :] = approx_derivative(
                lambda x_opt: self.paneller.angles(x_opt),
                x0=x,
                f0=constraint[len(lengths) :],
                bounds=self.bounds,
            )
        return constraint


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.geometry.tools import (
        boolean_cut,
        find_clockwise_angle_2d,
        make_polygon,
    )
    from bluemira.geometry.wire import BluemiraWire

    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    z = np.sin(theta)

    def cut_wire_below_z(wire: BluemiraWire, proportion: float) -> BluemiraWire:
        """Cut a wire below the z-coordinate that is 'proportion' of the height of the wire."""
        bbox = wire.bounding_box
        z_cut_coord = proportion * (bbox.z_max - bbox.z_min) + bbox.z_min
        cutting_box = np.array(
            [
                [bbox.x_min - 1, 0, bbox.z_min - 1],
                [bbox.x_min - 1, 0, z_cut_coord],
                [bbox.x_max + 1, 0, z_cut_coord],
                [bbox.x_max + 1, 0, bbox.z_min - 1],
                [bbox.x_min - 1, 0, bbox.z_min - 1],
            ]
        )
        pieces = boolean_cut(wire, [make_polygon(cutting_box, closed=True)])
        return pieces[np.argmax([p.center_of_mass[2] for p in pieces])]

    def make_cut_johner():
        """
        Make a wall shape and cut it below a (fictional) x-point.

        As this is for testing, we just use a JohnerLCFS with a slightly
        larger radius than default, then cut it below a z-coordinate that
        might be the x-point in an equilibrium.
        """
        johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
        return cut_wire_below_z(johner_wire, 1 / 4)

    shape = make_cut_johner()
    coords = shape.discretize(byedges=True)
    x, z = coords.x, coords.z

    paneller = Paneller(np.array([x, z]), 30, 0.05)
    initial_joints = paneller.joints(paneller.x0)

    print("Initial:")
    # print(f"joints: {initial_joints}")
    print(f"length: {paneller.length(paneller.x0)}")
    print(f"angles: {paneller.angles(paneller.x0)}")

    optimiser = Optimiser(
        "SLSQP",
        n_variables=paneller.n_opts,
        opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    )
    opt = PanellingOptProblem(paneller, optimiser)

    import time

    start = time.time()
    x_opt = opt.optimise()
    print(f"optimisation took {time.time() - start} seconds")

    opt_joints = paneller.joints(x_opt)

    print("\nOptimised:")
    # print(f"joints: {opt_joints}")
    print(f"length: {paneller.length(x_opt)}")
    print(f"angles: {paneller.angles(x_opt)}")

    _, ax = plt.subplots()
    ax.plot(x, z, linewidth=0.1)
    ax.plot(initial_joints[0], initial_joints[1], "--x", label="initial", linewidth=0.65)
    ax.plot(opt_joints[0], opt_joints[1], "-x", label="optimised", linewidth=0.75)
    ax.set_aspect("equal")
    ax.legend()
    plt.show()
