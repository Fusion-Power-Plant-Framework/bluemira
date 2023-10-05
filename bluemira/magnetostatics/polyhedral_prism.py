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

"""
Analytical expressions for the field inside an arbitrarily shaped winding pack
with arbitrarily shaped cross-section, following equations as described in:


"""
import math

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.tools import make_polygon, point_inside_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.baseclass import (
    ArbitraryCrossSectionCurrentSource,
)
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array

__all__ = ["PolyhedralPrismCurrentSource"]


def h_field_x(j, nprime, nprime2, d, xyz, lam, gam, psi):
    """
    Function to calculate magnetic field strength in x direction from working parameters
    """
    return (
        j
        / (4 * np.pi)
        * (
            (nprime2 * d - nprime * (nprime * xyz[0] + nprime2 * xyz[1])) * lam
            + (nprime * d - nprime * (-nprime2 * xyz[0] + nprime * xyz[1])) * gam
            - nprime * nprime2 * psi
        )
    )


def h_field_y(j, nprime, nprime2, d, xyz, lam, gam, psi):
    """
    Function to calculate magnetic field strength in y direction from working parameters
    """
    return (
        j
        / (4 * np.pi)
        * (
            -(nprime * d - nprime * (-nprime2 * xyz[0] + nprime * xyz[1])) * lam
            + (nprime2 * d - nprime * (nprime * xyz[0] + nprime2 * xyz[1])) * gam
            + nprime * nprime * psi
        )
    )


def h_field_z(j, nprime, d, eta, zeta):
    """
    Function to calculate magnetic field strength in z direction from working parameters
    """
    return j / (4 * np.pi) * (nprime * d * eta - nprime**2 * zeta)


def trap_dist(theta, pos, min_pos, vec):
    """
    Function to calculate distance betwen squared end and trapezoidal
    end at the position pos

    Parameters
    ----------
    theta:
        the angle of the trapezoidal end (either upper or lower) [rad]
    pos:
        the (x, y, z) coordinates the function is evaluated at
    min_pos:
        the starting point of the sloped end (minimum point along trapezoidal vector)
    vec:
        the trapezoidal vector

    """
    dy = np.dot((pos - min_pos), vec)
    return dy * np.tan(theta)


class PolyhedralPrismCurrentSource(ArbitraryCrossSectionCurrentSource):
    """
    3-D polyhedral prism current source with an arbitrary cross-section and
    uniform current distribution.

    Current is acting in the local z direction.

    Parameters
    ----------
    origin:
        The origin of the current source in global coordinates [m]
    ds:
        The direction vector of the current source in global coordinates [m]
    normal:
        The normalised normal vector of the current source in global coordinates [m]
    t_vec:
        The normalised tangent vector of the current source in global coordinates [m]
    trap_vec:
        The normalised vector to apply the trapezoidal slope along [m]
    n:
        The number of sides to the polyhedral (when coords=None)
    length:
        The length of the current source (excluding the trapezoidal ends) [m]
    width:
        The distance between the origin and the vertices (when coords=None) [m]
    alpha:
        The first angle of the trapezoidal prism [°] [0, 90)
    beta:
        The second angle of the trapezoidal prism [°] [0, 90)
    current:
        The current flowing through the source [A]
    nrows:
        The number of rows used for segmentation when calculating the magnetic field
        for the current source.
    coords:
        The input coordinates for the current source as a cross section slice centred
        at the origin point for z.
        By default is None so the vertices of the cross section are created using n
        and width.
        When used input needs to be a closed bluemira wire.

    Notes
    -----
    Negative angles are allowed, but both angles must be either 0 or negative.
    """

    def __init__(
        self,
        origin: np.ndarray,
        ds: np.ndarray,
        normal: np.ndarray,
        t_vec: np.ndarray,
        trap_vec: np.ndarray,
        n: int,
        length: float,
        width: float,
        alpha: float,
        beta: float,
        current: float,
        wire: BluemiraWire = None,
    ):
        self.origin = origin
        if wire is None:
            self.n = n
        else:
            self.n = np.shape(wire.vertexes)[1]
        # ensure normal vector is normalised
        self.normal = normal / np.linalg.norm(normal)
        self.length = np.linalg.norm(ds)
        self.dcm = np.array([t_vec, ds / self.length, normal])
        self.length = length
        self.width = width
        self.theta = 2 * np.pi / self.n
        # ensure trapezoidal vector is normalised
        self.trap_vec = trap_vec / np.linalg.norm(trap_vec)
        # ensure vector perpendicular to trapezoidal is normalised
        self.perp_vec = np.cross(normal, self.trap_vec) / np.linalg.norm(
            np.cross(normal, self.trap_vec)
        )
        self.theta_l = np.deg2rad(beta)
        self.theta_u = np.deg2rad(alpha)
        self._check_angle_values(self.theta_u, self.theta_l)
        self.tolerance = 1e-8
        self.shell = self._shell_creation(wire)
        self.XSface = BluemiraFace(boundary=self.wire)
        self.area = self.XSface.area
        self._check_raise_self_intersection(self.length, self.theta_u, self.theta_l)
        # current density
        self.J = current / self.area
        self.current = current
        # direction vector for current
        # this is along direction vector
        self.j_hat = ds / np.linalg.norm(ds)
        # direction vector for magnetisation
        # perp to J (set to be perp to tvec
        # but anything perp to J would work)
        j_cross_tvec = np.cross(self.j_hat, t_vec)
        self.mc_hat = j_cross_tvec / -np.linalg.norm(j_cross_tvec)
        # direction vector for magnetisation value
        # perp to J and Mc
        j_cross_m = np.cross(self.j_hat, self.mc_hat)
        self.d_hat = j_cross_m / np.linalg.norm(j_cross_m)

    def _check_angle_values(self, alpha, beta):
        """
        Check that end-cap angles are acceptable.
        """
        sign_alpha = np.sign(alpha)
        sign_beta = np.sign(beta)
        one_zero = np.any(np.array([sign_alpha, sign_beta]) == 0.0)  # noqa: PLR2004
        if not one_zero and sign_alpha != sign_beta:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: end-cap angles "
                f"must have the same sign {alpha=:.3f}, {beta=:.3f}."
            )
        if not (0 <= abs(alpha) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} is "
                "outside bounds of [0, 90°)."
            )
        if not (0 <= abs(beta) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {beta=:.3f} is "
                "outside bounds of [0, 90°)."
            )

    def _check_raise_self_intersection(self, length: float, alpha: float, beta: float):
        """
        Check for bad combinations of source length and end-cap angles.
        """
        points = self.wire.vertexes.T
        pmin, pmax = self._shape_min_max(points, self.trap_vec)
        dist = round(np.dot(pmax - pmin, self.trap_vec), 10)
        a = np.tan(np.abs(alpha)) * dist
        b = np.tan(np.abs(beta)) * dist
        if (a + b) > length:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: source length and "
                "angles imply a self-intersecting prism."
            )

    @staticmethod
    def _shape_min_max(points, vector):
        """
        Function to calculate min and max points of prism cross section
        along vector.
        """
        vals = []
        for p in points:
            vals += [np.dot(p, vector)]
        pmin = points[vals.index(min(vals)), :]
        pmax = points[vals.index(max(vals)), :]
        return pmin, pmax

    def _shell_creation(self, wire):
        """
        Function to calculate all the points of the prism in local coords
        and return in global.
        """
        # no coordinated provided so calculates central cross section using
        # width, n and theta
        if wire is None:
            c_points = []
            for i in range(self.n + 1):
                c_points += [
                    np.array(
                        [
                            round(self.width * np.sin(i * self.theta), 10),
                            round(self.width * np.cos(i * self.theta), 10),
                            0,
                        ]
                    )
                ]
            c_points = np.vstack([np.array(c_points)])
            x = c_points[:, 0]
            y = c_points[:, 1]
            z = c_points[:, 2]
            coords = Coordinates({"x": x, "y": y, "z": z})
            self.wire = make_polygon(coords, closed=True)
        # coordinates provided as a closed wire so need to extract and format as desired
        else:
            self.wire = wire
            points = wire.vertexes.T
            c_points = []
            for p in points:
                c_points += [np.array(p)]
            c_points += [np.array(points[0, :])]
            c_points = np.vstack([np.array(c_points)])
        boundl, _ = self._shape_min_max(c_points, self.trap_vec)

        lower_points = []
        upper_points = []

        for p in c_points:
            dz_l = trap_dist(self.theta_l, p, boundl, self.trap_vec)
            lower_points += [
                np.array([p[0], p[1], round(p[2] - 0.5 * self.length - dz_l, 10)])
            ]
            dz_u = trap_dist(self.theta_u, p, boundl, self.trap_vec)
            upper_points += [
                np.array([p[0], p[1], round(p[2] + 0.5 * self.length + dz_u, 10)])
            ]
        lower_points = np.vstack([np.array(lower_points)])
        upper_points = np.vstack([np.array(upper_points)])
        points = [lower_points, upper_points]
        for i in range(self.n):
            points += [np.vstack([lower_points[i], upper_points[i]])]
        self.points = np.array([self._local_to_global(p) for p in points], dtype=object)
        sumation = np.array([0.0, 0.0, 0.0])
        for k in range(self.n):
            x = lower_points[k, 0]
            y = lower_points[k, 1]
            z = lower_points[k, 2]
            sumation += np.array([x, y, z])
        self.x = sumation / self.n
        faces = []
        for i in range(self.n):
            x = [
                lower_points[i, 0],
                upper_points[i, 0],
                upper_points[i + 1, 0],
                lower_points[i + 1, 0],
            ]
            y = [
                lower_points[i, 1],
                upper_points[i, 1],
                upper_points[i + 1, 1],
                lower_points[i + 1, 1],
            ]
            z = [
                lower_points[i, 2],
                upper_points[i, 2],
                upper_points[i + 1, 2],
                lower_points[i + 1, 2],
            ]
            coords = Coordinates({"x": x, "y": y, "z": z})
            wire = make_polygon(coords, closed=True)
            faces += [BluemiraFace(boundary=wire)]
        coords = Coordinates(
            {"x": lower_points[:, 0], "y": lower_points[:, 1], "z": lower_points[:, 2]}
        )
        wire = make_polygon(coords, closed=True)
        faces += [BluemiraFace(boundary=wire)]
        coords = Coordinates(
            {"x": upper_points[:, 0], "y": upper_points[:, 1], "z": upper_points[:, 2]}
        )
        wire = make_polygon(coords, closed=True)
        faces += [BluemiraFace(boundary=wire)]
        return BluemiraShell(boundary=faces)

    def _calculate_mc(self, point):
        """
        Provides value of Mc at point
        """
        if point_inside_shape(point, self.shell) is True:
            mc = self.current * self._calculate_vector_distance(point, self.d_hat)
        else:
            mc = 0.0
        return mc * self.mc_hat

    @staticmethod
    def _calculate_normal(face):
        """
        calculate the normal of a face
        """
        points = face.vertexes.T
        u = np.array(points[3, :]) - np.array(points[0, :])
        v = np.array(points[1, :]) - np.array(points[0, :])
        u_x_v = np.cross(u, v)
        return u_x_v / np.linalg.norm(u_x_v)

    def _calculate_angles(self, normal):
        """
        Calculate the cosine of the angles between the normal with Mc and Mc cross J
        (n' and n'' respectively)
        """
        o = np.array([0, 0, 0])
        # cosine of the angle between side normal and magnetisation direction
        n_prime = np.dot(normal, self.mc_hat) / (
            math.dist(normal, o) * math.dist(self.mc_hat, o)
        )
        # sets value to zero if small
        if np.abs(n_prime) < self.tolerance:
            n_prime = 0.0
        # cosine of angle between side normal and magnetisation value direction
        n_prime2 = np.dot(normal, self.d_hat) / (
            math.dist(normal, o) * math.dist(self.d_hat, o)
        )
        # sets value to zero if small
        if np.abs(n_prime2) < self.tolerance:
            n_prime2 = 0.0

        return n_prime, n_prime2

    def _calculate_vector_distance(self, p, v):
        """
        Calculate distance along vector v between point X and p
        """
        # new point fpr p that is on plane of X
        p_prime = p - (np.dot(np.dot(p - self.x, v), v))
        # calculates distance between p' and X along vector Dhat
        return np.dot(p_prime - self.x, self.d_hat)

    def _face_cutting(self, p):
        """
        Splits shape made up of points p into triangles and trapezoids
        """
        shapes = []
        for i in range(self.n - 4):
            if i == 0:
                # makes the first shape and loops back to start point
                cut = np.append(p[:3, :], p[:1, :], axis=0)
            else:
                # makes remaining shapes and loops back to start point
                cut = np.append(p[(0, i + 1, i + 2), :], p[:1, :], axis=0)
            # adds new shapes to list
            shapes += [np.vstack([cut])]
        # creates final shape
        cut = np.append(p[:1, :], p[-3:, :], axis=0)
        # adds start point to final shape
        cut = np.append(cut, p[:1, :], axis=0)
        # adds final shape to list
        shapes += [np.vstack([cut])]
        # rearranges shapes into array
        shape_arr = []
        for s in shapes:
            shape_arr += [s]
        return shape_arr

    @staticmethod
    def _position_vector(point, fpoint):
        """
        Creates a vector R from the shape vertices to the fieldpoint
        """
        r1 = fpoint - point[0, :]
        r2 = fpoint - point[1, :]
        r3 = fpoint - point[2, :]
        r4 = fpoint - point[3, :]
        return np.array([r1, r2, r3, r4])

    @staticmethod
    def _length_vector(p, q, r):
        """
        Creates a set of vectors that go from point p to q on an existing shape
        """
        # position vector for point p
        rp = r[p - 1, :]
        # position vector for point q
        rq = r[q - 1, :]
        # return length vector from p to q using position vectors
        return rp - rq

    def _points_reordering(self, points, normal):
        """
        Reorders the points of a side k so that it is correct for magnetostatic
        calculations with the ordering based upon theory.
        """
        # remove repeated inital point
        # point = points[:-1, :]
        # surface current vector direction
        j_sc = np.cross(normal, self.mc_hat)
        # if surface current is non zero
        if np.linalg.norm(j_sc) > 0:
            # dot product of points with mc_hat and j_sc
            # rounding needed so that when minimising values match
            # which allows for intersect to be taken correctly
            m_out = np.round(np.dot(points, self.mc_hat), 15)
            j_out = np.round(np.dot(points, j_sc), 15)
            # minimising j_sc
            (j0,) = np.nonzero(j_out == j_out.min())
            # minimising mc_hat
            if np.dot(j_sc, self.j_hat) > 0:
                (m0,) = np.nonzero(m_out == m_out.min())
            # maximising mc_hat
            else:
                (m0,) = np.nonzero(m_out == m_out.max())
            if np.size(j0) == 1:
                idx0 = j0[0]
            elif np.size(m0) == 1:
                idx0 = m0[0]
            else:
                idx0 = np.intersect1d(m0, j0)[0]
            # reverse direction if needed
            try:
                value = np.linalg.norm(
                    np.cross(points[idx0 + 1, :] - points[idx0, :], j_sc)
                )
            except IndexError:
                value = np.linalg.norm(np.cross(points[0, :] - points[idx0, :], j_sc))
            if value == 0:
                p = np.append(points[idx0:, :], points[:idx0, :], axis=0)
                p = np.append(p, p[:1, :], axis=0)
                p = p[::-1, :]
            else:
                # shift points array to start at idx0
                p = np.append(points[idx0:, :], points[:idx0, :], axis=0)
                p = np.append(p, p[:1, :], axis=0)
        else:
            # find starting point by minimising d_hat and j_hat dotted with points
            d_out = np.dot(points, self.d_hat)
            j_out = np.dot(points, self.j_hat)
            (d0,) = np.nonzero(d_out == d_out.min())
            (j0,) = np.nonzero(j_out == j_out.min())
            if np.size(d0) == 1:
                idx0 = d0[0]
            elif np.size(j0) == 1:
                idx0 = j0[0]
            else:
                idx0 = np.intersect1d(d0, j0)[0]
            # reverse direction if needed
            try:
                value = np.linalg.norm(
                    np.cross(points[idx0 + 1, :] - points[idx0, :], self.j_hat)
                )
            except IndexError:
                value = np.linalg.norm(
                    np.cross(points[0, :] - points[idx0, :], self.j_hat)
                )
            if value == 0:
                p = np.append(points[idx0:, :], points[:idx0, :], axis=0)
                p = np.append(p, p[:1, :], axis=0)
                p = p[::-1, :]
            else:
                # shift points array to start at idx0
                p = np.append(points[idx0:, :], points[:idx0, :], axis=0)
                p = np.append(p, p[:1, :], axis=0)
        # return reorganised points
        return p

    def _vector_coordinates(self, point, fpoint):
        """
        Creates a set of vector working coordinates that are used to calculate the
        magnetic field for the prism. Does this for a side of the prism at a time.
        """
        # vector R between vertices and fieldpoint
        r = self._position_vector(point, fpoint)
        # vector that sets z direction
        zhat = self._length_vector(1, 4, r) / np.linalg.norm(
            self._length_vector(1, 4, r)
        )
        # value of z change between vertex 1 and p (=2,3,4)
        zp = np.array(
            [
                [np.dot(zhat, self._length_vector(1, 2, r))],
                [np.dot(zhat, self._length_vector(1, 3, r))],
                [np.dot(zhat, self._length_vector(1, 4, r))],
            ]
        )
        # value d which is width of shape
        d = np.sqrt(np.linalg.norm(self._length_vector(1, 2, r)) ** 2 - zp[0] ** 2)
        # vector that sets x direction
        xhat = np.cross(self._length_vector(1, 2, r), zhat) / d
        # vector that sets y direction
        yhat = np.cross(zhat, xhat)
        # x value (between vertex and fieldpoint)
        x = np.dot(xhat, r[0, :])
        # y value (between vertex and fieldpoint)
        y = np.dot(yhat, r[0, :])
        # z value (between vertex and fieldpoint)
        z = np.dot(zhat, r[0, :])
        # projections of area vectors along normal to trapezoid side
        p12 = np.dot(xhat, np.cross(r[0, :], r[1, :]))
        p34 = np.dot(-xhat, np.cross(r[2, :], r[3, :]))
        # scalar products of area vector
        q12 = np.dot(r[0, :], self._length_vector(1, 2, r))
        q34 = -np.dot(r[3, :], self._length_vector(3, 4, r))
        # calculates some parameters to simplify later equations ie lam, psi, eta...
        lambda_12 = np.log(
            (
                np.linalg.norm(r[0, :]) * np.linalg.norm(self._length_vector(1, 2, r))
                - q12
            )
            / (
                (np.linalg.norm(r[1, :]) + np.linalg.norm(self._length_vector(1, 2, r)))
                * np.linalg.norm(self._length_vector(1, 2, r))
                - q12
            )
        )
        lambda_34 = np.log(
            (
                (np.linalg.norm(r[2, :]) + np.linalg.norm(self._length_vector(3, 4, r)))
                * np.linalg.norm(self._length_vector(3, 4, r))
                - q34
            )
            / (
                np.linalg.norm(r[3, :]) * np.linalg.norm(self._length_vector(3, 4, r))
                - q34
            )
        )
        # major term used in calculation of h field
        lam = (
            np.log(
                ((np.linalg.norm(r[0, :]) + z) * (np.linalg.norm(r[2, :]) + z - zp[1]))
                / (
                    (np.linalg.norm(r[1, :]) + z - zp[0])
                    * (np.linalg.norm(r[3, :]) + z - zp[2])
                )
            )
            + zp[0] * lambda_12 / np.linalg.norm(self._length_vector(1, 2, r))
            + (zp[1] - zp[2]) * lambda_34 / np.linalg.norm(self._length_vector(3, 4, r))
        )
        if np.abs(lam) < self.tolerance:
            lam = 0
        # components of gamma equation to simplify equation
        a1 = z * q12 - zp[0] * np.linalg.norm(r[0, :]) ** 2
        a2 = x * np.linalg.norm(r[0, :]) * d
        a_sgn = np.sign(a1 * a2)
        b1 = (z - zp[0]) * (
            q12 - np.linalg.norm(self._length_vector(1, 2, r)) ** 2
        ) - zp[0] * np.linalg.norm(r[1, :]) ** 2
        b2 = x * np.linalg.norm(r[1, :]) * d
        b_sgn = np.sign(b1 * b2)
        c1 = (z - zp[1]) * (q34 - np.linalg.norm(self._length_vector(3, 4, r)) ** 2) - (
            zp[1] - zp[2]
        ) * np.linalg.norm(r[2, :]) ** 2
        c2 = x * np.linalg.norm(r[2, :]) * d
        c_sgn = np.sign(c1 * c2)
        d1 = (z - zp[2]) * q34 - (zp[1] - zp[2]) * np.linalg.norm(r[3, :]) ** 2
        d2 = x * np.linalg.norm(r[3, :]) * d
        d_sgn = np.sign(d1 * d2)
        # major term used in calculation of h field
        gam = (
            a_sgn * np.arctan2(np.abs(a1), np.abs(a2))
            - b_sgn * np.arctan2(np.abs(b1), np.abs(b2))
            + c_sgn * np.arctan2(np.abs(c1), np.abs(c2))
            - d_sgn * np.arctan2(np.abs(d1), np.abs(d2))
        )
        # major term used in calculation of h field
        psi = (
            d
            * zp[0]
            * (np.linalg.norm(r[0, :]) - np.linalg.norm(r[1, :]))
            / (np.linalg.norm(self._length_vector(1, 2, r)) ** 2)
            + d
            * (zp[1] - zp[2])
            * (np.linalg.norm(r[2, :]) - np.linalg.norm(r[3, :]))
            / (np.linalg.norm(self._length_vector(3, 4, r)) ** 2)
            - d
            * d
            * (
                p12 * lambda_12 / (np.linalg.norm(self._length_vector(1, 2, r)) ** 3)
                + p34 * lambda_34 / (np.linalg.norm(self._length_vector(3, 4, r)) ** 3)
            )
        )
        # major term used in calculation of h field
        eta = d * (
            lambda_12 / np.linalg.norm(self._length_vector(1, 2, r))
            + lambda_34 / np.linalg.norm(self._length_vector(3, 4, r))
        )
        # major term used in calculation of h field
        zeta = (
            d
            * d
            * (
                (np.linalg.norm(r[0, :]) - np.linalg.norm(r[1, :]))
                / (np.linalg.norm(self._length_vector(1, 2, r)) ** 2)
                + (np.linalg.norm(r[2, :]) - np.linalg.norm(r[3, :]))
                / (np.linalg.norm(self._length_vector(3, 4, r)) ** 2)
                + q12 * lambda_12 / (np.linalg.norm(self._length_vector(1, 2, r)) ** 3)
                + q34 * lambda_34 / (np.linalg.norm(self._length_vector(3, 4, r)) ** 3)
            )
        )
        # coordinates for x,y,z from perspective of shape
        coords = np.array([x, y, z])
        # returns parameters needed in h field calculation
        return coords, lam, gam, psi, eta, zeta

    def _hxhyhz(self, fpoint):
        """
        Produces h field at fieldpoint using h field functions and vector coordinates
        """
        h_array = []
        # h field contribution from sides
        for f in self.shell.faces:
            normal = self._calculate_normal(f)
            points = f.vertexes.T
            nprime, nprime2 = self._calculate_angles(normal)
            f_vertices = 4
            if len(points[:, 0]) == f_vertices:
                # get vector values from functions
                d = -(self._calculate_vector_distance(points[0, :], self.mc_hat))
                points = self._points_reordering(points, normal)
                coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                    points, fpoint
                )
                # calculate h field in all directions
                hx = h_field_x(self.J, nprime, nprime2, d, coords, lam, gam, psi)
                hy = h_field_y(self.J, nprime, nprime2, d, coords, lam, gam, psi)
                hz = h_field_z(self.J, nprime, d, eta, zeta)
                # add outputs to array
                h_array.append(np.hstack([hx, hy, hz]))

            # h field contribution from faces
            else:
                # splits face into smaller shapes
                shapes = self._face_cutting(points)
                # cycle through shapes to calculate field
                for i in range(self.n - 4):
                    shape = shapes[i]
                    dist = []
                    for p in shape:
                        dist += [self._calculate_vector_distance(p, normal)]
                    # takes distance as the maximum from all the points
                    d = np.abs(np.max(dist))
                    # value always zero for face
                    nprime2 = 0
                    # get working vector coordinates
                    coords, lam, gam, psi, eta, zeta = self._vector_coordinates(
                        points, fpoint
                    )
                    # calcaulte h field
                    hx = h_field_x(self.J, nprime, nprime2, d, coords, lam, gam, psi)
                    hy = h_field_y(self.J, nprime, nprime2, d, coords, lam, gam, psi)
                    hz = h_field_z(self.J, nprime, d, eta, zeta)
                    # add outputs to array
                    h_array.append(np.hstack([hx, hy, hz]))

        # reorganise array and include magnetisation before returning
        hx = 0
        hy = 0
        hz = 0
        for h in h_array:
            hx += h[0]
            hy += h[1]
            hz += h[2]
        return [hx, hy, hz, *self._calculate_mc(fpoint)]

    def _bxbybz(self, point):
        """
        Calculate the b field at a fieldpoint using h field calculator and
        converting to b field
        """
        h = self._hxhyhz(point)
        b = (h + self._calculate_mc(point)) * MU_0
        b_field = []
        # convert negligible values to 0
        for b_xyz in b:
            if np.abs(b_xyz) < self.tolerance:
                b_xyz = 0  # noqa: PLW2901
            b_field += [b_xyz]
        return np.array([b_field[0], b_field[1], b_field[2]])

    @process_xyz_array
    def field(self, x, y, z):
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        return self._bxbybz(point)
