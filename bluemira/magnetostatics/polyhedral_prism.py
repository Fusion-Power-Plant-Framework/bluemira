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
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import distance_to, make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.baseclass import (
    ArbitraryCrossSectionCurrentSource,
    SourceGroup,
)
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_xyz_array
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource

__all__ = ["PolyhedralPrismCurrentSource"]


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
        nrows: int,
        wire: BluemiraWire = None,
    ):
        self.origin = origin
        self.wire = wire
        if wire is None:
            self.n = n
        else:
            self.n = np.shape(self.wire.vertexes)[1]
        # ensure normal vector is normalised
        self.normal = normal / np.linalg.norm(normal)
        self.length = np.linalg.norm(ds)
        self.dcm = np.array([t_vec, ds / self.length, normal])
        self.length = length
        self.width = width
        self.theta = 2 * np.pi / self.n
        self.nrows = nrows
        vec = trap_vec
        # ensure trapezoidal vector is normalised
        self.trap_vec = vec / np.linalg.norm(vec)
        perp_vec = np.cross(normal, self.trap_vec)
        # ensure vector perpendicular to trapezoidal is normalised
        self.perp_vec = perp_vec / np.linalg.norm(perp_vec)
        self.theta_l = np.deg2rad(beta)
        self.theta_u = np.deg2rad(alpha)
        self._check_angle_values(self.theta_u, self.theta_l)
        self.points = self._calc_points(self.wire)
        self.area = self._cross_section_area()
        self._check_raise_self_intersection(self.length, self.theta_u, self.theta_l)
        # current density
        self.J = current / self.area
        # trapezoidal sources for field calculation
        self.sources = self._segmentation_setup(self.nrows)
        if round(self.area, 4) != round(self.seg_area, 4):
            bluemira_warn(
                "Difference between prism area and total segment area at 4dp."
                f"Prism area = {self.area} and Segment area = {self.seg_area}."
                "Try using more segments by increasing nrows."
            )

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
        points = self.points[0] if self.wire is None else self.wire.vertexes.T
        pmin, pmax = self._shape_min_max(points, self.trap_vec)
        dist = round(np.dot(pmax - pmin, self.trap_vec), 10)
        a = np.tan(np.abs(alpha)) * dist
        b = np.tan(np.abs(beta)) * dist
        if (a + b) > length:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: source length and "
                "angles imply a self-intersecting prism."
            )

    def _cross_section_area(self):
        """
        Function to calculate cross sectional area of prism.
        """
        points = self.points[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        coords = Coordinates({"x": x, "y": y, "z": z})
        wire = make_polygon(coords, closed=True)
        face = BluemiraFace(boundary=wire)
        return face.area

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

    def _calc_points(self, wire):
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
        # coordinates provided as a closed wire so need to extract and format as desired
        else:
            points = wire.vertexes.T
            c_points = []
            for p in points:
                c_points += [np.array(p)]
            c_points += [np.array(points[0, :])]
            c_points = np.vstack([np.array(c_points)])
        boundl, _ = self._shape_min_max(c_points, self.trap_vec)

        l_points = []
        u_points = []

        for p in c_points:
            dz_l = trap_dist(self.theta_l, p, boundl, self.trap_vec)
            l_points += [
                np.array([p[0], p[1], round(p[2] - 0.5 * self.length - dz_l, 10)])
            ]
            dz_u = trap_dist(self.theta_u, p, boundl, self.trap_vec)
            u_points += [
                np.array([p[0], p[1], round(p[2] + 0.5 * self.length + dz_u, 10)])
            ]
        l_points = np.vstack([np.array(l_points)])
        u_points = np.vstack([np.array(u_points)])
        points = [c_points, l_points, u_points]
        # add lines between cuts
        for i in range(self.n):
            points += [np.vstack([l_points[i], u_points[i]])]

        return np.array([self._local_to_global(p) for p in points], dtype=object)

    def _segmentation_setup(self, nrows):
        """
        Function to break up current source into a series of trapezoidal prism segments.
        Method of segmentation is to bound the central line of each segment
        with the edge of the prism, with top of first segment (and bot of last
        segment) matching the top (and bot) vertex.
        nrows is number of segments.
        """
        if self.wire is None:
            points = self.points[0]
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            coords = Coordinates({"x": x, "y": y, "z": z})
            main_wire = make_polygon(coords, closed=True)
        else:
            main_wire = self.wire
            points = self.wire.vertexes.T
        par_min, par_max = self._shape_min_max(points, self.trap_vec)
        b = round(np.dot(par_max - par_min, self.trap_vec), 10) / nrows
        perp_min, perp_max = self._shape_min_max(points, self.perp_vec)
        perp_dist = round(np.dot(perp_max - perp_min, self.perp_vec), 10)
        sources = SourceGroup([])
        c_area = 0
        for i in range(nrows):
            d = i * b + b / 2
            c = par_min + d * self.trap_vec
            up = c + perp_dist * self.perp_vec
            low = c - perp_dist * self.perp_vec
            x = np.array([low[0], up[0]])
            y = np.array([low[1], up[1]])
            z = np.array([low[2], up[2]])
            coords = Coordinates({"x": x, "y": y, "z": z})
            wire = make_polygon(coords, closed=False)
            dist, vectors = distance_to(main_wire, wire)
            if np.round(dist, 4) > 0:
                print("no intersect between line and wire")
            else:
                p1 = np.array(vectors[0][0])
                p2 = np.array(vectors[1][0])
                o = np.multiply(0.5, (p1 + p2))
                width = np.linalg.norm(p2 - p1)
                area = width * b
                c_area += area
                current = self.J * area
                dz_l = trap_dist(self.theta_l, o, par_min, self.trap_vec)
                dz_u = trap_dist(self.theta_u, o, par_min, self.trap_vec)
                length = self.length + dz_l + dz_u
                source = TrapezoidalPrismCurrentSource(
                    o,
                    length * self.normal,
                    self.perp_vec,
                    self.trap_vec,
                    b / 2,
                    width / 2,
                    self.theta_u,
                    self.theta_l,
                    current,
                )
                sources.add_to_group([source])
        self.seg_area = c_area
        return sources

    @process_xyz_array
    def field(self, x, y, z):
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array([x, y, z])
        Bx, By, Bz = self.sources.field(*point)
        return np.array([Bx, By, Bz])
