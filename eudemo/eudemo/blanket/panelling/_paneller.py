# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from bluemira.geometry.coordinates import (
    Coordinates,
    vector_intersect,
    vector_lengthnorm,
)


class Paneller:
    """
    Provides functions for generating panelling along the outside of a boundary.

    Parameters
    ----------
    boundary_points:
        The points defining the boundary along which to build the panels.
        This should have shape (2, N), where N is the number of points.
    max_angle:
        Maximum turning angle of the panels
    dx_min:
        Minimum panel length
    fix_num_panels:
        If specified, fix the number of panels to an integer value
    """

    def __init__(
        self,
        boundary_points: np.ndarray,
        max_angle: float,
        dx_min: float,
        fix_num_panels: int | None = None,
    ):
        self.max_angle = max_angle
        self.dx_min = dx_min
        self.boundary = LengthNormBoundary(boundary_points)

        if fix_num_panels:
            self.x0 = np.linspace(0, 1, fix_num_panels)[1:-1]
        else:
            # Build the initial guess of our panels, these points are the
            # coordinates of where the panels tangent the boundary
            _, idx = Coordinates(boundary_points).simplify(
                max_angle=max_angle, dx_min=dx_min
            )
            self.x0: np.ndarray = self.boundary.length_norm[idx][1:-1]

    @property
    def n_panels(self) -> int:
        """The number of panels defined by this paneller."""
        return len(self.x0) + 2

    def joints(self, dists: np.ndarray) -> np.ndarray:
        """
        Calculate panel joint coordinates from panel-boundary tangent points.

        Parameters
        ----------
        dists:
            The normalised distances along the boundary at which there
            are panel-boundary tangent points.
        """
        # Add the start and end panel joints at distances 0 & 1
        dists = np.hstack((0, dists, 1))
        points = np.vstack((self.boundary.x(dists), self.boundary.z(dists)))
        tangents = np.vstack((
            self.boundary.x_tangent(dists),
            self.boundary.z_tangent(dists),
        ))
        # This could potentially be vectorized, if we need a speed up.
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

    # Typically, in a panelling optimisation loop, we'll call, at least,
    # 'length' and 'angles', which both call out to 'joints'.
    # For a potential optimisation we could cache the result of 'joints'.
    def length(self, dists: np.ndarray) -> float:
        """The cumulative length of the panels."""
        return self.panel_lengths(dists).sum()

    def angles(self, dists: np.ndarray) -> np.ndarray:
        """
        Return the angles of rotation between each set of adjacent panels.

        Note that this is the tail-tail angle between the panel vectors,
        not the head-tail angles.
        """
        joints = self.joints(dists)
        line_vectors = np.diff(joints, axis=1)
        magnitudes = np.linalg.norm(line_vectors, axis=0)
        # avoid division by zero later
        magnitudes[np.isclose(magnitudes, 0)] += 1e-8
        dots = (line_vectors[:, :-1] * line_vectors[:, 1:]).sum(axis=0)
        dots /= magnitudes[:-1] * magnitudes[1:]
        return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)), out=dots)

    def panel_lengths(self, dists: np.ndarray) -> np.ndarray:
        """Return the lengths of each panel."""
        panel_vecs = np.diff(self.joints(dists))
        return np.hypot(panel_vecs[0], panel_vecs[1])


class LengthNormBoundary:
    """Class to represent a wire interpolated over the normalised distance along it."""

    def __init__(self, boundary_points: np.ndarray):
        self.length_norm = vector_lengthnorm(boundary_points[0], boundary_points[1])
        self._x_spline = InterpolatedUnivariateSpline(
            self.length_norm, boundary_points[0]
        )
        self._z_spline = InterpolatedUnivariateSpline(
            self.length_norm, boundary_points[1]
        )

        self.tangent_norm = norm_tangents(boundary_points)
        self._x_tangent_spline = InterpolatedUnivariateSpline(
            self.length_norm, self.tangent_norm[0]
        )
        self._z_tangent_spline = InterpolatedUnivariateSpline(
            self.length_norm, self.tangent_norm[1]
        )

    def x(self, dist: float | np.ndarray) -> np.ndarray:
        """Find x at the given normalised distance along the boundary."""
        return self._x_spline(dist)

    def z(self, dist: float | np.ndarray) -> np.ndarray:
        """Find z at the given normalised distance along the boundary."""
        return self._z_spline(dist)

    def x_tangent(self, dist: float | np.ndarray) -> np.ndarray:
        """Find x at the tangent vector a given distance along the boundary."""
        return self._x_tangent_spline(dist)

    def z_tangent(self, dist: float | np.ndarray) -> np.ndarray:
        """Find z at the tangent vector a given distance along the boundary."""
        return self._z_tangent_spline(dist)


def norm_tangents(points: np.ndarray) -> np.ndarray:
    """
    Calculate the normalised tangent vector at each of the given points.

    Parameters
    ----------
    points:
        Array of coordinates. This must have shape (2, N), where N is
        the number of points.

    Returns
    -------
    The normalised vector of tangents.
    """
    grad = np.gradient(points, axis=1)
    magnitudes = np.hypot(grad[0], grad[1])
    return np.divide(grad, magnitudes)
