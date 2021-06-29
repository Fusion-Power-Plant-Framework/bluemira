# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Analytical expressions for the field inside an arbitrarily shaped winding packs
of rectangular cross-section.
"""
import numpy as np
import abc
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.geometry.geomtools import (
    get_angle_between_vectors,
    bounding_box,
    circle_seg,
)
from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.magnetostatics.trapezoidal_prism import (
    Bx_analytical_prism,
    Bz_analytical_prism,
)
from BLUEPRINT.magnetostatics.circular_arc import (
    Bx_analytical_circular,
    Bz_analytical_circular,
)

__all__ = [
    "TrapezoidalPrismCurrentSource",
    "CircularArcCurrentSource",
    "ArbitraryPlanarCurrentLoop",
    "AnalyticalMagnetostaticSolver",
]


class CurrentSource(abc.ABC):
    origin: np.array
    dcm: np.array
    points: np.array
    breadth: float
    depth: float
    length: float

    def _local_to_global(self, points):
        """
        Convert local x', y', z' point coordinates to global x, y, z point coordinates.
        """
        return np.array([self.origin + self.dcm.T @ p for p in points])

    def _global_to_local(self, points):
        """
        Convert global x, y, z point coordinates to local x', y', z' point coordinates.
        """
        return np.array([(self.dcm @ (p - self.origin)) for p in points])

    def plot(self, ax=None):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        if ax is None:
            ax = Plot3D()
            # If no ax provided, we assume that we want to plot only this source,
            # and thus set aspect ratio equality on this term only
            edge_points = np.concatenate(self.points)
            xbox, ybox, zbox = bounding_box(*edge_points.T)
            ax.scatter(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, color="w", marker=None)

        ax.scatter(*self.origin, color="k")
        for points in self.points:
            ax.plot(*points.T, color="b", linewidth=1)

        # Plot local coordinate system
        ax.quiver(*self.origin, *self.dcm[0], length=self.breadth, color="r")
        ax.quiver(*self.origin, *self.dcm[1], length=self.length, color="r")
        ax.quiver(*self.origin, *self.dcm[2], length=self.depth, color="r")


class TrapezoidalPrismCurrentSource(CurrentSource):
    """
    3-D trapezoidal prism current source with a retangular cross-section and
    uniform current distribution.

    The current direction is along the local y coordinate.

    Parameters
    ----------
    origin: np.array(3)
        The origin of the current source in global coordinates [m]
    ds: np.array(3)
        The direction vector of the current source in global coordinates [m]
    normal: np.array(3)
        The normalised normal vector of the current source in global coordinates [m]
    t_vec: np.array(3)
        The normalised tangent vector of the current source in global coordinates [m]
    breadth: float
        The breadth of the current source (half-width) [m]
    depth: float
        The depth of the current source (half-height) [m]
    alpha: float
        The first angle of the trapezoidal prism [rad]
    beta: float
        The second angle of the trapezoidal prism [rad]
    current: float
        The current flowing through the source [A]
    """

    def __init__(self, origin, ds, normal, t_vec, breadth, depth, alpha, beta, current):
        self.origin = origin

        length = np.linalg.norm(ds)
        # Normalised direction cosine matrix
        self.dcm = np.array([t_vec, ds / length, normal])
        self.length = (
            length - breadth * np.tan(alpha) * 1 - breadth * np.tan(beta) * 1
        ) / 2
        self.breadth = breadth
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        # Current density
        self.rho = current / (4 * breadth * depth)
        self.points = self._calculate_points()

    def _xyzlocal_to_rql(self, x_local, y_local, z_local):
        """
        Convert local x, y, z coordinates to working coordinates.
        """
        b = self.length
        c = self.depth
        d = self.breadth

        l1 = -d - x_local
        l2 = d - x_local
        q1 = -c - z_local
        q2 = c - z_local
        r1 = (d + x_local) * np.tan(self.alpha) + b - y_local
        r2 = (d + x_local) * np.tan(self.beta) + b + y_local
        return l1, l2, q1, q2, r1, r2

    def _BxByBz(self, point):
        """
        Calculate the field at a point in local coordinates.
        """
        l1, l2, q1, q2, r1, r2 = self._xyzlocal_to_rql(*point)
        bx = Bx_analytical_prism(self.alpha, self.beta, l1, l2, q1, q2, r1, r2)
        bz = Bz_analytical_prism(self.alpha, self.beta, l1, l2, q1, q2, r1, r2)
        return np.array([bx, 0, bz])

    def field(self, point):
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        point: np.array(3)
            The target point in global coordinates [m]

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array(point)
        # Convert to local coordinates
        point = self._global_to_local([point])[0]
        # Evaluate field in local coordinates
        b_local = 1e-7 * self.rho * self._BxByBz(point)
        # Convert vector back to global coordinates
        return self.dcm.T @ b_local

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        b = self.length
        c = self.depth
        d = self.breadth
        # Lower rectangle
        p1 = np.array([-d, -b, -c])
        p2 = np.array([d, -b - 2 * d * np.tan(self.beta), -c])
        p3 = np.array([d, -b - 2 * d * np.tan(self.beta), c])
        p4 = np.array([-d, -b, c])

        # Upper rectangle
        p5 = np.array([-d, b, -c])
        p6 = np.array([d, b + 2 * d * np.tan(self.alpha), -c])
        p7 = np.array([d, b + 2 * d * np.tan(self.alpha), c])
        p8 = np.array([-d, b, c])

        points_array = []
        points = [
            np.vstack([p1, p2, p3, p4, p1]),
            np.vstack([p5, p6, p7, p8, p5]),
            # Lines between rectangle corners
            np.vstack([p1, p5]),
            np.vstack([p2, p6]),
            np.vstack([p3, p7]),
            np.vstack([p4, p8]),
        ]

        for p in points:
            points_array.append(self._local_to_global(p))

        return np.array(points_array, dtype=object)


class CircularArcCurrentSource(CurrentSource):
    """
    3-D circular arc prism current source with a retangular cross-section and
    uniform current distribution.

    Parameters
    ----------
    origin: np.array(3)
        The origin of the current source in global coordinates [m]
    ds: np.array(3)
        The direction vector of the current source in global coordinates [m]
    normal: np.array(3)
        The normalised normal vector of the current source in global coordinates [m]
    t_vec: np.array(3)
        The normalised tangent vector of the current source in global coordinates [m]
    breadth: float
        The breadth of the current source (half-width) [m]
    depth: float
        The depth of the current source (half-height) [m]
    radius: float
        The radius of the circular arec from the origin [m]
    dtheta: float
        The azimuthal width of the arc [rad]
    current: float
        The current flowing through the source [A]

    Notes
    -----
    The origin is at the centre of the circular arc, with the ds vector pointing
    towards the start of the circular arc.

    Cylindrical coordinates are used for calculations under the hood.
    """

    def __init__(
        self, origin, ds, normal, t_vec, breadth, depth, radius, dtheta, current
    ):
        self.origin = origin
        self.breadth = breadth
        self.depth = depth
        self.length = 0.5 * (breadth + depth)  # For plotting only
        self.radius = radius
        self.dtheta = dtheta
        self.rho = current / (4 * breadth * depth)
        self.dcm = np.array([ds, normal, t_vec])
        self.points = self._calculate_points()

    @staticmethod
    def _local_to_cylindrical(point):
        """
        Convert from local to cylindrical coordinates.
        """
        x, y, z = point
        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return np.array([rho, theta, z])

    def _cylindrical_to_working(self, zp):
        """
        Convert from local cylindrical coordinates to working coordinates.
        """
        r1 = self.radius - self.breadth
        r2 = self.radius + self.breadth
        z1 = zp + self.depth
        z2 = zp - self.depth
        return r1, r2, z1, z2

    def _BxByBz(self, rp, tp, zp):
        """
        Calculate the field at a point in local coordinates.
        """
        r1, r2, z1, z2 = self._cylindrical_to_working(zp)
        bx = Bx_analytical_circular(r1, r2, z1, z2, self.dtheta, rp, tp)
        bz = Bz_analytical_circular(r1, r2, z1, z2, self.dtheta, rp, tp)
        return np.array([bx, 0, bz])

    def field(self, point):
        """
        Calculate the magnetic field at a point due to the current source.

        Parameters
        ----------
        point: np.array(3)
            The target point in global coordinates [m]

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        point = np.array(point)
        # Convert to local cylindrical coordinates
        point = self._global_to_local([point])[0]
        rp, tp, zp = self._local_to_cylindrical(point)
        # Calculate field in local coordindates
        b_local = 1e-7 * self.rho * self._BxByBz(rp, tp, zp)
        # Convert field to global coordinates
        return self.dcm.T @ b_local

    def _calculate_points(self):
        """
        Calculate extrema points of the current source for plotting and debugging.
        """
        r = self.radius
        a = self.breadth
        b = self.depth

        # Circle arcs
        n = 200
        theta = np.rad2deg(self.dtheta)
        ones = np.ones(n)
        arc_1x, arc_1y = circle_seg(r - a, (0, 0), angle=theta, start=0, npoints=n)
        arc_2x, arc_2y = circle_seg(r + a, (0, 0), angle=theta, start=0, npoints=n)
        arc_3x, arc_3y = circle_seg(r + a, (0, 0), angle=theta, start=0, npoints=n)
        arc_4x, arc_4y = circle_seg(r - a, (0, 0), angle=theta, start=0, npoints=n)
        arc_1 = np.array([arc_1x, arc_1y, -b * ones]).T
        arc_2 = np.array([arc_2x, arc_2y, -b * ones]).T
        arc_3 = np.array([arc_3x, arc_3y, b * ones]).T
        arc_4 = np.array([arc_4x, arc_4y, b * ones]).T

        slices = np.linspace(0, n - 1, 5, endpoint=True, dtype=np.int)
        points = [arc_1, arc_2, arc_3, arc_4]

        # Rectangles
        for s in slices:
            points.append(np.vstack([arc_1[s], arc_2[s], arc_3[s], arc_4[s], arc_1[s]]))

        points_array = []
        for p in points:
            points_array.append(self._local_to_global(p))

        return np.array(points_array, dtype=object)

    def plot(self, ax=None):
        """
        Plot the CircularArcCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        super().plot(ax=ax)
        ax = plt.gca()
        theta = np.rad2deg(self.dtheta)
        x, y = circle_seg(
            self.radius, (0, 0), angle=theta / 2, start=theta / 4, npoints=200
        )
        centre_arc = np.array([x, y, np.zeros(200)]).T
        points = self._local_to_global(centre_arc)
        ax.plot(*points.T, color="r")
        ax.plot([points[-1][0]], [points[-1][1]], [points[-1][2]], marker="^", color="r")


class MultiCurrentSource(abc.ABC):
    """
    Abstract base class for multiple current sources.
    """

    sources: list

    def field(self, point):
        """
        Calculate the magnetic field at a point.

        Parameters
        ----------
        point: np.array(3)
            The target point in global coordinates [m]

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        return np.sum([source.field(point) for source in self.sources], axis=0)

    def plot(self, ax=None):
        """
        Plot the MultiCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        if ax is None:
            ax = Plot3D()

        # Bounding box to set equal aspect ratio plot
        all_points = np.vstack([np.vstack(s.points) for s in self.sources])
        xbox, ybox, zbox = bounding_box(*all_points.T)
        ax.scatter(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, color="w", marker=None)

        for source in self.sources:
            source.plot(ax=ax)


class ArbitraryPlanarCurrentLoop(MultiCurrentSource):
    """
    An arbitrary, planar, closed current loop of constant rectangular cross-section
    and uniform current density.

    Parameters
    ----------
    loop: Loop
        The Loop object from which to form an ArbitraryPlanarCurrentLoop
    breadth: float
        The breadth of the current source (half-width) [m]
    depth: float
        The depth of the current source (half-height) [m]
    current: float
        The current flowing through the source [A]
    """

    def __init__(self, loop, breadth, depth, current):
        super().__init__()
        self.loop = loop.copy()

        if not self.loop.closed:
            bluemira_warn("Closed current loop required.")
            self.loop.close()

        # Set up geometry, calculating all trapezoial prism sources
        self.d_l = np.diff(loop.xyz).T
        self.midpoints = loop.xyz[:, :-1].T + self.d_l / 2
        self.sources = []
        normal = self.loop.n_hat
        beta = np.deg2rad(get_angle_between_vectors(self.d_l[-1], self.d_l[0])) / 2

        for i, (midpoint, d_l) in enumerate(zip(self.midpoints, self.d_l)):
            angle = np.deg2rad(
                get_angle_between_vectors(self.d_l[i - 1], d_l, signed=True)
            )
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
            self.sources.append(source)
            beta = alpha

    def plot(self, ax=None):
        """
        Plot the ArbitraryPlanarCurrentLoop.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        super().plot(ax=ax)
        self.loop.plot(ax=ax, fill=False, edgecolor="r")


class AnalyticalMagnetostaticSolver:
    """
    3-D magnetostatic solver for multiple arbitrary source terms.

    Parameters
    ----------
    sources: List[MultipleCurrentSource]
        The list of current source terms
    """

    def __init__(self, sources):
        self.sources = sources

    def field(self, point):
        """
        Calculate the magnetic field at a point.

        Parameters
        ----------
        point: np.array(3)
            The target point in global coordinates [m]

        Returns
        -------
        field: np.array(3)
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        return np.sum([source.field(point) for source in self.sources], axis=0)

    def plot(self, ax=None):
        """
        Plot the AnalyticalMagnetostaticSolver.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        if ax is None:
            ax = Plot3D()

        # Bounding box to set equal aspect ratio plot
        all_points = []
        for source in self.sources:
            if isinstance(source, MultiCurrentSource):
                for sub_source in source.sources:
                    all_points.extend(sub_source.points)
            elif isinstance(source, CurrentSource):
                all_points.extend(source.points)
        all_points = np.vstack(all_points)
        xbox, ybox, zbox = bounding_box(*all_points.T)
        ax.scatter(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, color="w", marker=None)

        for source in self.sources:
            source.plot(ax=ax)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
