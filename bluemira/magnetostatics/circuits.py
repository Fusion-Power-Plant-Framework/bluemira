# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

"""
Three-dimensional current source terms.
"""

import numpy as np

from bluemira.geometry._deprecated_tools import distance_between_points, rotation_matrix
from bluemira.geometry.coordinates import get_normal_vector
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.tools import process_loop_array, process_xyz_array
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource

__all__ = ["ArbitraryPlanarRectangularXSCircuit", "HelmholtzCage"]


class ArbitraryPlanarRectangularXSCircuit(SourceGroup):
    """
    An arbitrary, planar current loop of constant rectangular cross-section
    and uniform current density.

    Parameters
    ----------
    shape: Union[np.array, Loop]
        The geometry from which to form an ArbitraryPlanarCurrentLoop
    breadth: float
        The breadth of the current source (half-width) [m]
    depth: float
        The depth of the current source (half-height) [m]
    current: float
        The current flowing through the source [A]
    """

    shape: np.array
    breadth: float
    depth: float
    current: float

    def __init__(self, shape, breadth, depth, current):
        self.shape = process_loop_array(shape)
        normal = get_normal_vector(*self.shape.T)
        closed = np.allclose(self.shape[-1], self.shape[0])

        # Set up geometry, calculating all trapezoidal prism sources
        self.d_l = np.diff(self.shape, axis=0)
        self.midpoints = self.shape[:-1, :] + 0.5 * self.d_l
        sources = []

        if closed:
            beta = self._get_half_angle(
                self.midpoints[-1], self.shape[0], self.midpoints[0]
            )
        else:
            beta = 0.0

        for i, (midpoint, d_l) in enumerate(zip(self.midpoints, self.d_l)):

            if i != len(self.midpoints) - 1:
                alpha = self._get_half_angle(
                    midpoint, self.shape[i + 1], self.midpoints[i + 1]
                )
            else:
                if closed:
                    alpha = self._get_half_angle(
                        midpoint, self.shape[-1], self.midpoints[0]
                    )
                else:
                    alpha = 0.0

            d_l_norm = d_l / np.linalg.norm(d_l)
            t_vec = np.cross(d_l_norm, normal)

            source = TrapezoidalPrismCurrentSource(
                midpoint,
                d_l,
                normal,
                t_vec,
                breadth,
                depth,
                alpha,
                beta,
                current,
            )
            sources.append(source)
            beta = alpha
        super().__init__(sources)

    def _get_half_angle(self, p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p1
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        cos_angle = np.dot(v1, v2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        d = distance_between_points(p0, p1)
        v_norm = np.linalg.norm(-v1 + v2)
        if np.isclose(v_norm, 0):
            return 0.5 * angle

        r1 = p1 + 0.1 * d * (-v1 + v2) / v_norm

        if self._point_in_triangle(r1, p0, p1, p2):
            if np.isclose(angle, np.pi / 2):
                angle += 2 * np.pi
            else:
                angle = -(angle - 2 * np.pi)
        return 0.5 * angle

    @staticmethod
    def _point_in_triangle(point, p0, p1, p2):
        """
        Determine whether a point lies inside a 3-D triangle.
        """
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        alpha = np.linalg.norm(np.cross(p1 - point, p2 - point)) / (2 * area)
        beta = np.linalg.norm(np.cross(p2 - point, p0 - point)) / (2 * area)
        gamma = 1.0 - alpha - beta
        return (
            (0 < alpha < 1)
            and (0 < beta < 1)
            and (0 < gamma < 1)
            and (np.isclose(alpha + beta + gamma, 1.0))
        )


class HelmholtzCage(SourceGroup):
    """
    Axisymmetric arrangement of current sources about the z-axis.

    Notes
    -----
    The plane at 0 degrees is set to be between two circuits.
    """

    def __init__(self, circuit, n_TF):
        self.n_TF = n_TF
        sources = self._pattern(circuit)

        super().__init__(sources)

    def _pattern(self, circuit):
        """
        Pattern the CurrentSource axisymmetrically.
        """
        angles = np.linspace(0, 360, int(self.n_TF), endpoint=False)
        sources = []
        for angle in angles:
            source = circuit.copy()
            source.rotate(angle, axis="z")
            sources.append(source)
        return sources

    @process_xyz_array
    def ripple(self, x, y, z):
        """
        Get the toroidal field ripple at a point.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the ripple
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the ripple
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the ripple

        Returns
        -------
        ripple: float
            The value of the TF ripple at the point(s) [%]
        """
        point = np.array([x, y, z])
        ripple_field = np.zeros(2)
        n = np.array([0, 1, 0])
        planes = [0, np.pi / self.n_TF]  # rotate (inline, ingap)

        for i, theta in enumerate(planes):
            r_matrix = rotation_matrix(theta).T
            sr = np.dot(point, r_matrix)
            nr = np.dot(n, r_matrix)
            field = self.field(*sr)
            ripple_field[i] = np.dot(nr, field)

        ripple = 1e2 * (ripple_field[0] - ripple_field[1]) / np.sum(ripple_field)
        return ripple
