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
from bluemira.geometry._deprecated_tools import (
    get_angle_between_vectors,
    rotation_matrix,
    get_normal_vector,
)
from bluemira.magnetostatics.tools import process_loop_array, process_xyz_array
from bluemira.magnetostatics.baseclass import SourceGroup
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

        # Set up geometry, calculating all trapezoial prism sources
        self.d_l = np.diff(self.shape, axis=0)
        self.midpoints = self.shape[:-1, :] + self.d_l / 2
        sources = []
        beta = get_angle_between_vectors(self.d_l[-1], self.d_l[0]) / 2

        for i, (midpoint, d_l) in enumerate(zip(self.midpoints, self.d_l)):
            angle = get_angle_between_vectors(self.d_l[i - 1], d_l, signed=True)

            alpha = angle / 2
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
        angles = np.pi / self.n_TF + np.linspace(
            0, 2 * np.pi, int(self.n_TF), endpoint=False
        )
        sources = []
        for angle in angles:
            source = circuit.copy()
            source.rotate(angle, axis="z")
            sources.append(source)
        return sources

    def set_current(self, current):
        """
        Set the current inside each of the circuits.

        Parameters
        ----------
        current: float
            The current of each circuit [A]
        """
        for source in self.sources:
            source.current = current

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
        planes = [np.pi / self.n_TF, 0]  # rotate (inline, ingap)

        for i, theta in enumerate(planes):
            r_matrix = rotation_matrix(theta)
            sr = np.dot(point, r_matrix)
            nr = np.dot(n, r_matrix)
            field = self.field(*sr)
            ripple_field[i] = np.dot(nr, field)

        ripple = 1e2 * (ripple_field[0] - ripple_field[1]) / np.sum(ripple_field)
        return ripple
