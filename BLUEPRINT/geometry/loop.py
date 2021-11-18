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
A coordinate-series object class
"""
import numpy as np

# Plotting imports
import matplotlib.pyplot as plt
from random import randint
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from shapely.geometry import Polygon, LineString
from matplotlib.patches import PathPatch
from sectionproperties.pre.sections import CustomSection
from sectionproperties.analysis.cross_section import CrossSection
from BLUEPRINT.geometry.geombase import (
    GeomBase,
    Plane,
    point_dict_to_array,
    _check_other,
)
from BLUEPRINT.geometry.geomtools import (
    get_centroid,
    get_centroid_3d,
    length,
    lengthnorm,
    distance_between_points,
    vector_intersect,
    qrotate,
    circle_seg,
    in_polygon,
    line_crossing,
    loop_plane_intersect,
    vector_lengthnorm,
    bounding_box,
    get_control_point,
    clean_loop_points,
)
from BLUEPRINT.geometry.constants import VERY_BIG
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.base.error import GeometryError
from BLUEPRINT.utilities.plottools import pathify, BPPathPatch3D, Plot3D
from BLUEPRINT.utilities.tools import furthest_perp_point
from bluemira.utilities.tools import is_num


class Loop(GeomBase):
    """
    The BLUEPRINT Loop object, which holds a set of connected 2D/3D coordinates
    and provides methods to manipulate them.

    Loops must be comprised of planar coordinates for some methods to work.

    Loops cannot be self-intersecting.

    Loops are by default anti-clockwise. Closed loops will automatically be
    made anti-clockwise.

    Loops can be geometrically open or closed, but some methods only make sense
    for closed loops.

    Parameters
    ----------
    x: iterable(N) or None
        The set of x coordinates [m]
    y: iterable(N) or None
        The set of y coordinates [m]
    z: iterable(N) or None
        The set of z coordinates [m]

    If only two coordinate sets are provided, will automatically backfill
    the third coordinate with np.zeros(N)
    """

    warning_printed = False

    def __init__(self, x=None, y=None, z=None, enforce_ccw=True):
        self._print_warning()
        self.x = x
        self.y = y
        self.z = z
        self._check_lengths()

        self.plan_dims  # call property because it messes around with stuff :(
        self.closed = self._check_closed()
        self.ccw = self._check_ccw()
        if not self.ccw:
            if enforce_ccw:
                self.reverse()
        self._remove_duplicates(enforce_ccw)

        self.inner = None
        self.outer = None

    def _print_warning(self):
        if not Loop.warning_printed:
            bluemira_warn(
                type(self).__module__ + "." + type(self).__name__ + " is deprecated."
            )
            Loop.warning_printed = True

    @classmethod
    def from_dict(cls, xyz_dict):
        """
        Initialises a Loop object from a dictionary

        Parameters
        ----------
        xyz_dict: dict
            Dictionary with {'x': [], 'y': [], 'z': []}
        """
        # NOTE: Stabler than **xyz_dict
        x = xyz_dict.get("x", 0)
        y = xyz_dict.get("y", 0)
        z = xyz_dict.get("z", 0)
        return cls(x=x, y=y, z=z)

    @classmethod
    def from_array(cls, xyz_array):
        """
        Initialises a Loop object from a numpy array

        Parameters
        ----------
        xyz_array: np.array(3, N)
            The numpy array of Loop coordinates
        """
        if xyz_array.shape[0] != 3:
            raise GeometryError("Need a (3, N) shape coordinate array.")
        return cls(*xyz_array)

    # =========================================================================
    #      Conversions
    # =========================================================================

    def as_dict(self):
        """
        Casts the Loop as a dictionary

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self["x"], "y": self["y"], "z": self["z"]}

    def as_shpoly(self):
        """
        Casts the Loop as a shapely Polygon object

        Returns
        -------
        polygon: shapely Polygon object
        """
        if self.closed:
            return Polygon(self.d2.T[:-1])
        else:
            # TODO: Fix how shapely treats open Loops... tricky
            # If end points are open, shapely closes them and throws a
            # self-intersection error
            return Polygon(self.d2.T)

    # =========================================================================
    #     User methods
    # =========================================================================

    def point_inside(self, point, include_edges=False):
        """
        Determines whether or not a point is within in the Loop

        Parameters
        ----------
        point: iterable(2-3)
            The 2-D or 3-D coordinates of the point (coord conversion handled)
        include_edges: bool
            Whether or not to return True if a point is on the perimeter of the
            Loop

        Returns
        -------
        in_polygon: bool
            Whether or not the point is within the Loop
        """
        point = self._point32d(point)
        return in_polygon(*point, self.d2.T, include_edges=include_edges)

    def trim(self, p1, p2, method="biggest"):
        """
        Trims a loop by a line between two points. Keeps biggest unless
        told otherwise
        Now preserves 3rd dimension
        """
        bluemira_warn("Geometry::Loop: using trim")
        l1, l2 = [dict(zip(self.plan_dims, i.T)) for i in self.split_by_line(p1, p2)]
        c = self._get_3rd_dim()
        v = self[c][0]
        l1[c] = v
        l2[c] = v
        l1 = Loop(**l1)
        l2 = Loop(**l2)
        l1.close()
        l2.close()
        if method == "furthest":
            return (
                l1
                if sum(np.array(l1.centroid) ** 2) >= sum(np.array(l2.centroid) ** 2)
                else l2
            )
        elif method == "closest":
            return (
                l1
                if sum(np.array(l1.centroid) ** 2) < sum(np.array(l2.centroid) ** 2)
                else l2
            )
        else:
            return l1 if l1.area >= l2.area else l2

    def split_by_line(self, p1, p2):
        """
        Splits a Loop along a vector p2-p1

        Parameters
        ----------
        p1: np.array(2)
            The first point in the split line
        p2: np.array(2)
            The second point in the split line

        Returns
        -------
        coords1: np.array(N, 2)
            The coordinates of the first half of the split
        coords2: np.array(M, 2)
            The coordinates of the second half of the split
        """
        if type(p1) != np.ndarray or type(p2) != np.ndarray:
            p1, p2 = np.array(p1), np.array(p2)
        self.intersect(p1, p2, join=True)
        v1 = p2 - p1
        v2 = self.d2.T - p2
        cp = np.cross(v1, v2)
        return self.d2.T[cp <= 0], self.d2.T[cp >= 0]

    def chop_by_line(self, p1, angle):
        """
        Chops a Loop up based on first two intersections.
        Differs from split_by_line as this takes all intersections
        if n_inter = 2, the result is the same

        Parameters
        ----------
        p1: np.array(2)
            The point from which to project the chop line
        angle: float
            The angle at which to project from the point

        Returns
        -------
        l1: Geometry::Loop
            The largest loop resulting from the chop
        l1: Geometry::Loop
            The smaller loop resulting from the chop
        """
        if type(p1) != np.ndarray:
            p1 = np.array(p1)
        p2 = self._project(p1, angle)
        inter = self.intersect(p1, p2, join=True, get_arg=True)
        inter, args = inter[0], inter[1]
        i1, i2 = [self.get_nth_inter(p1, inter, i) for i in range(2)]
        a, b = args[i1], args[i2]
        a, b = max([a, b]), min([a, b])
        a += 1
        b += 1
        l1 = Loop(*self[b : a + 1])
        l1.close()
        l2 = Loop(*np.concatenate((self[a:], self[: b + 1]), axis=1))
        l2.close()
        return (l1, l2) if l1.centroid[0] <= l2.centroid[0] else (l2, l1)

    def get_min_length(self):
        """
        Calculates the minimum segment length of the Loop

        Returns
        -------
        min_length: float
            The minimum length segment
        """
        return np.min(np.diff(length(*self.d2)))

    def get_max_length(self):
        """
        Calculates the maximum segment length of the Loop

        Returns
        -------
        max_length: float
            The maximum length segment
        """
        return np.max(np.diff(length(*self.d2)))

    def get_visible_width(self, angle):
        """
        Get the visible width of the Loop as seen from an angle

        Parameters
        ----------
        angle: float
            The angle in degrees

        Returns
        -------
        width: float
            The width of the Loop as seen from the angle in 2-D
        """
        return self._get_visible(angle)[0]

    def get_visible_extrema(self, angle):
        """
        Get the visible extrema of the Loop as seen from an angle

        Parameters
        ----------
        angle: float
            The angle in degrees

        Returns
        -------
        extrema: list
            The 2-D coordinates of the extrema points as seen from the angle
        """
        return self._get_visible(angle)[1]

    def section(self, plane):
        """
        Calculates the intersection of the Loop with a plane

        Parameters
        ----------
        plane: Geometry::Plane object
            The plane with which to calculate the intersection
        """
        inter = loop_plane_intersect(self, plane)
        if inter is None:
            bluemira_warn(
                "Geometry::Loop::section: No intersection with plane detected."
            )
        return inter

    def distance_to(self, point):
        """
        Calculates the distances from each point in the loop to the point

        Parameters
        ----------
        point: iterable(2 or 3)
            The point to which to calculate the distances

        Returns
        -------
        distances: np.array(N)
            The vector of distances to the point
        """
        if len(point) == 2:
            point = self._point_23d(point)
        point = np.array(point)
        point = point.reshape(3, 1).T
        return cdist(self.xyz.T, point, "euclidean")

    def argmin(self, point2d):
        """
        Parameters
        ----------
        point2d: iterable(2)
            The point to which to calculate the distances

        Returns
        -------
        arg: int
            The index of the closest point
        """
        return np.argmin(self.distance_to(point2d))

    def interpolate(self, n_points):
        """
        Repurposed from S. McIntosh geom.py
        """
        ll = vector_lengthnorm(self.xyz.T)
        linterp = np.linspace(0, 1, int(n_points))
        self.x = interp1d(ll, self.x)(linterp)
        self.y = interp1d(ll, self.y)(linterp)
        self.z = interp1d(ll, self.z)(linterp)

    def interpolate_midpoints(self):
        """
        Interpolate the Loop adding the midpoint of each segment to the Loop
        """
        xyz_new = self.xyz[:, :-1] + np.diff(self.xyz) / 2
        xyz_new = np.insert(xyz_new, np.arange(len(self) - 1), self.xyz[:, :-1], axis=1)
        xyz_new = np.append(xyz_new, self.xyz[:, -1].reshape(3, 1), axis=1)
        self.x = xyz_new[0]
        self.y = xyz_new[1]
        self.z = xyz_new[2]

    def interpolator(self):
        """
        Returns interpolation spline
        """
        x, z = self.plan_dims

        if len(self) == 3:
            self.interpolate(21)  # resample (close enough)

        ln = lengthnorm(self[x], self[z])
        func = {
            "x": InterpolatedUnivariateSpline(ln, self[x]),
            "z": InterpolatedUnivariateSpline(ln, self[z]),
            "L": self.length,
        }
        func["dx"] = func["x"].derivative()
        func["dz"] = func["z"].derivative()
        return func

    def intersect(self, p1, p2, join=False, get_arg=False, first=False):
        """
        Returns np.array of intersections with a line between p1 and p2

        Parameters
        ----------
        p1: interable(2)
            First 2-D point in a line along which intersections are to be
            returned
        p2: interable(2)
            Second 2-D point in a line along which intersections are to be
            returned
        join: bool
            Joins the intersection points to the loop
        get_arg: bool
            Returns the indices of the intersection points, as well as the
            intersections
        first: bool
            Only join / arg / intersect for the first point, starting from p1

        Returns
        -------
        intersections: np.array(N)
            The 2-D intersection points on the Loop
        args: ints(N)
            The indices of the intersection points
        """
        # NOTE: use join_intersect, and get_intersect for intersections with
        # other Loops

        p1, p2 = np.array(p1), np.array(p2)
        c = np.zeros(len(self) - 1)
        for i, p in enumerate(self.d2.T[:-1]):
            c[i] = line_crossing(p, self.d2.T[i + 1], p1, p2)
        args = np.nonzero(c)[0]  # As posicoes onde tem uma interseção
        if len(args) == 0:
            f, ax = plt.subplots()
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker="o")
            self.plot(ax=ax)
            a, b = self.plan_dims
            s = 1.5
            ax.set_xlim([min(self[a]) / s, s * max(self[a])])
            ax.set_ylim([min(self[b]) / s, s * max(self[b])])
            raise GeometryError(
                "No intersection was found between Loop and "
                f"the intersection line between {p1} and {p2},"
                " this will likely cause issues."
            )

        ints = []
        for arg in args:
            ints.append(vector_intersect(self.d2.T[arg], self.d2.T[arg + 1], p1, p2))
        if join and first is False:
            # De Morgan's law is butt ugly here
            # Joins all points
            for j, (i, p) in enumerate(zip(args, ints)):
                # Eh oui, faut indexer ceux que l'on vient d'ajouter
                self.insert(self._point_23d(p), pos=i + j + 1)

            if get_arg:
                args = list(np.array(args) + np.array(list(range(len(args)))))
                return np.array(ints), args

        if first:
            # Get and join only first point
            i = self.get_nth_inter(p1, ints, n=0)
            if not join and not get_arg:
                return np.array(ints[i])
            if join:
                if not self._check_already_in(ints[i]):
                    arg = args[i] + 1
                    self.insert(self._point_23d(ints[i]), pos=arg)
                else:
                    arg = self.argmin(ints[i])
            if get_arg:
                return np.array(ints[i]), arg
        return np.array(ints)

    def receive_projection(self, point, angle, get_arg=False):
        """
        Joins a projection from a point onto a Loop, and returns either the
        intersection location or index

        Parameters
        ----------
        point: iterable(2)
            2-D point from which to project
        angle: float
            The angle from the point along which to project onto the Loop
        get_arg: bool
            Whether or not to return the index of the intersection

        Returns
        -------
        inter: iterable(2)
            The 2-D intersection point that was joined to the Loop
        OR
        arg: int
            The index of the intersection point in the Loop
        """
        p2 = self._project(point, angle)
        inter, arg = self.intersect(point, p2, join=True, get_arg=True, first=True)
        if get_arg:
            return arg
        else:
            return inter

    @staticmethod
    def _project(point, angle):
        """
        Calculates a projection point
        Returns second point (Far away from first) along a vector of angle

        Parameters
        ----------
        point: iterable(2)
            2-D point from which to project
        angle: float
            The angle from the point along which to project

        Returns
        -------
        point2: list(2)
            The 2-D projected point
        """
        if len(point) != 2:
            raise GeometryError("Can only handle 2D plane projections.")

        angle = np.radians(angle)
        point2 = [
            point[0] + VERY_BIG * np.cos(angle),
            point[1] + VERY_BIG * np.sin(angle),
        ]
        return point2

    def get_nth_inter(self, point1, inter, n=0):
        """
        Returns the nth intersection. Defaults to closest
        """
        point1 = self._point32d(point1)
        ll = []
        for p in inter:
            ll.append(distance_between_points(point1, p))
        ll = np.array(ll)
        return np.where(ll == np.partition(ll, n)[n])[0][0]

    def offset(self, delta):
        """
        Returns a new loop offset by `delta` from existing Loop
        3rd coordinate is also matched in the new Loop

        Parameters
        ----------
        delta: float
            The offset to take from the Loop. Negative numbers mean a smaller
            Loop

        Returns
        -------
        delta_loop: Loop
            A new Loop object offset from this Loop

        See Also
        --------
        BLUEPRINT.geometry.offset
        """
        if delta == 0.0:
            return self.copy()

        # Circular import handling
        from BLUEPRINT.geometry.offset import offset

        o = offset(self[self.plan_dims[0]], self[self.plan_dims[1]], delta)
        new = {self.plan_dims[0]: o[0], self.plan_dims[1]: o[1]}
        c = list({"x", "y", "z"} - set(self.plan_dims))[0]
        new[c] = [self[c][0]]  # Third coordinate must be all equal (flat)
        return Loop(**new)

    def offset_clipper(self, delta, method="square", miter_limit=2.0):
        """
        Loop offset using the pycliper methods

        Parameters
        ----------
        delta: float
            Value to be offset
        method: str from ['square', 'round', 'miter'] (default = 'square')
            The type of offset to perform
        miter_limit: float (default = 2.0)
            The ratio of delta to used when mitering acute corners. Only used if
            method == 'miter'

        Return
        ------
        Loop
            loop with the indicated offest applied
        """
        # Setup here to avoid circular import
        from BLUEPRINT.geometry.offset import offset_clipper

        return offset_clipper(self, delta, method, miter_limit)

    def rotate(self, theta, update=True, enforce_ccw=True, **kwargs):
        """
        Rotates the Loop by an angle theta

        Parameters
        ----------
        theta: float
            The angle of rotation [degrees]
        update: bool (default = True)
            if True: will update the Loop object
            if False: will return a new Loop object, and leave this one alone
        enforce_ccw: bool (default = True)
            if True: will enforce ccw direction in new Loop
            if False: will not enforce ccw direction in new Loop

        Other Parameters
        ----------------
        theta: float
            Rotation angle [radians]
        p1: [float, float, float]
            Origin of rotation vector
        p2: [float, float, float]
            Second point defining rotation axis
        OR
        theta: float
            Rotation angle [radians]
        xo: [float, float, float]
            Origin of rotation vector
        dx: [float, float, float] or one of 'x', 'y', 'z'
            Direction vector definition rotation axis from origin.
            If a string is specified the dx vector is automatically
            calculated, e.g. 'z': (0, 0, 1)
        OR
        quart: Quarternion object
            The rotation quarternion
        xo: [float, float, float]
            Origin of rotation vector
        """
        rotated = qrotate(self.as_dict(), theta=np.radians(theta), **kwargs)
        if update:
            for k in ["x", "y", "z"]:
                self.__setattr__(k, rotated[k])
        else:
            return Loop(
                rotated["x"], rotated["y"], rotated["z"], enforce_ccw=enforce_ccw
            )

    def rotate_dcm(self, dcm, update=True):
        """
        Rotates the loop based on a direction cosine matrix

        Parameters
        ----------
        dcm: np.array((3, 3))
            The direction cosine matrix array
        update: bool (default = True)
            if True: will update the Loop object
            if False: will return a new Loop object, and leave this one alone
        """
        xyz = dcm @ self.xyz
        if update:
            for i, k in enumerate(["x", "y", "z"]):
                self.__setattr__(k, xyz[i])
        else:
            return Loop(*xyz)

    def translate(self, vector, update=True):
        """
        Translates the Loop

        Parameters
        ----------
        vector: iterable(3)
            The [dx, dy, dz] vector to translate the Loop by
        update: bool (default = True)
            Whether or not to update the Loop object. If False, will return a
            new Loop object
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        v = vector * np.ones([len(self), 1])
        t = self.xyz + v.T
        if update:
            for i, k in enumerate(["x", "y", "z"]):
                self.__setattr__(k, t[i])
        else:
            return Loop(**dict(zip(["x", "y", "z"], t)))

    def reorder(self, arg, pos=0):
        """
        Reorders a closed loop taking index [arg] and setting it to [pos].
        """
        if not self.closed:
            raise GeometryError(
                "Du darfst eine ungeschlossene Schleife nicht " "neu ordenen!"
            )
        self.remove(-1)
        roll_index = arg - pos
        for k in ["x", "y", "z"]:
            c = self.__getattribute__(k)
            if c is not None:
                r = np.roll(c, roll_index)
                self.__setattr__(k, r)
        self.close()

    def stitch(self, other_loop):
        """
        Stitch the Loop to another Loop object.

        Parameters
        ----------
        other_loop: Loop
            The Loop to stitch this one to

        Returns
        -------
        loop: Loop
            The resulting single stitched Loop
        """
        other = _check_other(other_loop, "Loop")
        x = np.concatenate((self.x[:-2], other.x[:-1]))
        y = np.concatenate((self.y[:-2], other.y[:-1]))
        z = np.concatenate((self.z[:-2], other.z[:-1]))
        loop = Loop(x, y, z)
        return loop

    def split(self, pos):
        """
        Splits into inner/outer sub-Loops. Smallest A is inner
        """
        bluemira_warn("Geometry::Loop: using split")
        l1, l2 = Loop.from_array(self[:pos]), Loop.from_array(self[pos:])
        a1, a2 = l1.area, l2.area
        self.inner = l1 if a1 < a2 else l2
        self.outer = l2 if a2 > a1 else l1

    def insert(self, point, pos=0):
        """
        Inserts a point into the Loop

        Parameters
        ----------
        point: iterable(3)
            The 3-D point to insert into the Loop
        pos: int > 0
            The position of the point in the Loop (order index)
        """
        if len(point) != 3:
            point = self._point_23d(point)
        if not self._check_already_in(point):
            if self._check_plane(point):
                for p, k in zip(point, ["x", "y", "z"]):
                    c = self.__getattribute__(k)
                    if pos == -1:  # ⴽⴰⵔⴻⴼⵓⵍ ⵏⴲⵡ
                        self.__setattr__(k, np.append(c, [p]))
                    else:
                        self.__setattr__(k, np.concatenate((c[:pos], [p], c[pos:])))
            else:
                raise GeometryError("Inserted point is not on Loop plane.")

    def remove(self, pos=0):
        """
        Removes a point of index `pos` from Loop.
        """
        for k in ["x", "y", "z"]:
            c = self.__getattribute__(k)
            self.__setattr__(k, np.delete(c, pos, axis=0))

    def sort_bottom(self):
        """
        Re-orders the loop so that it is indexed at 0 at its lowest point.

        Notes
        -----
        Useful for comparing polygons with identical coordinates but different
        starting points. Will only work for closed polygons.
        """
        if not self.closed:
            bluemira_warn("On ne peut pas faire cela avec des Loops qui sont ouverts")
            return

        arg = np.argmin(self.z)
        self.reorder(arg, 0)

    def open_(self):
        """
        Opens a closed Loop to make an open Loop
        """
        if self.closed:
            for k in ["x", "y", "z"]:
                c = self.__getattribute__(k)
                if c is not None:
                    self.__setattr__(k, c[:-1])
            self.closed = False

    def close(self):
        """
        Closes an open Loop to make a closed Loop
        """
        if not self._check_closed():
            for k in ["x", "y", "z"]:
                c = self.__getattribute__(k)
                if c is not None:
                    self.__setattr__(k, np.append(c, c[0]))
            self.closed = True

    def plot(self, ax=None, points=False, **kwargs):
        """
        Only deals with unique colors - no cycles here please
        Handle all kwargs explicitly here, no kwargs-magic beyond this point

        Parameters
        ----------
        ax: Axes object
            The matplotlib axes on which to plot the Loop
        points: bool
            Whether or not to plot individual points (with numbering)

        Other Parameters
        ----------------
        edgecolor: str
            The edgecolor to plot the Loop with
        facecolor: str
            The facecolor to plot the Loop fill with
        alpha: float
            The transparency to plot the Loop fill with
        """
        if self.ndim == 2 and ax is None:
            ax = kwargs.get("ax", plt.gca())
        fc = kwargs.get("facecolor", "royalblue")
        lw = kwargs.get("linewidth", 2)
        ls = kwargs.get("linestyle", "-")
        alpha = kwargs.get("alpha", 1)

        if self.closed:
            fill = kwargs.get("fill", True)
            ec = kwargs.get("edgecolor", "k")
        else:
            fill = kwargs.get("fill", False)
            ec = kwargs.get("edgecolor", "r")

        if self.ndim == 2 and not hasattr(ax, "zaxis"):
            a, b = self.plan_dims
            marker = "o" if points else None
            ax.set_xlabel(a + " [m]")
            ax.set_ylabel(b + " [m]")
            if fill:
                poly = pathify(self.as_shpoly())
                p = PathPatch(poly, color=fc, alpha=alpha)
                ax.add_patch(p)
            ax.plot(*self.d2, color=ec, marker=marker, linewidth=lw, linestyle=ls)
            if points:
                for i, p in enumerate(self.d2.T):
                    ax.annotate(i, xy=(p[0], p[1]))
            if not hasattr(ax, "zaxis"):
                ax.set_aspect("equal")
        else:
            kwargs = {
                "edgecolor": ec,
                "facecolor": fc,
                "linewidth": lw,
                "linestyle": ls,
                "alpha": alpha,
                "fill": fill,
            }
            self._plot_3d(ax, **kwargs)

    def _plot_3d(self, ax=None, **kwargs):
        if ax is None:
            ax = Plot3D()
            # Maintenant on re-arrange un peu pour que matplotlib puisse nous
            # montrer qqchose un peu plus correct
            x_bb, y_bb, z_bb = bounding_box(*self.xyz)
            for x, y, z in zip(x_bb, y_bb, z_bb):
                ax.plot([x], [y], [z], color="w")

        ax.plot(*self.xyz, color=kwargs["edgecolor"], lw=kwargs["linewidth"])
        if kwargs["fill"]:
            dcm = self.rotation_matrix(-self.n_hat, np.array([0.0, 0.0, 1.0]))

            loop = self.rotate_dcm(dcm.T, update=False)

            c = np.array(loop._point_23d(loop.centroid))
            loop.translate(-c, update=True)

            # Pour en faire un objet que matplotlib puisse comprendre
            poly = pathify(loop.as_shpoly())

            # En suite en re-transforme l'objet matplotlib en 3-D!
            c = self._point_23d(self.centroid)

            p = BPPathPatch3D(
                poly, -self.n_hat, c, color=kwargs["facecolor"], alpha=kwargs["alpha"]
            )
            ax.add_patch(p)

        if not hasattr(ax, "zaxis"):
            ax.set_aspect("equal")

    def get_points(self):
        """
        Get the [x, z] points corresponding to this loop

        If the loop is closed then skips the last (closing) point.

        Returns
        -------
        points : List[float, float]
            The [x, z] points corresponding to this loop.
        """
        if self.closed:
            return self.d2.T[:-1].tolist()
        else:
            return self.d2.T.tolist()

    def get_closed_facets(self, start=0):
        """
        Get the closed facets corresponding to this loop

        The facets are closed by linking the last point back to the first.

        Parameters
        ----------
        start : int, optional
            The index to assign to the first point, by default 0

        Returns
        -------
        facets : List[int, int]
            The closed facets corresponding to this loop.
        """
        num_points = len(self.get_points())
        facets = [[i, i + 1] for i in range(start, start + num_points - 1)]
        facets.append([start + num_points - 1, start])
        return facets

    def get_control_point(self):
        """
        Get the control point correponding to this loop.

        Returns
        -------
        control_point : (float, float)
            The control point corresponding to this loop.
        """
        closed_loop = Loop(*self.d2)
        closed_loop.close()
        return list(get_control_point(closed_loop))

    def get_hole(self):
        """
        Get the hole corresponding to this loop

        For loops the hole will always be an empty list.

        Returns
        -------
        hole : (float, float)
            The hole corresponding to this loop.
        """
        return []

    def generate_cross_section(
        self, mesh_sizes=None, min_length=None, min_angle=None, verbose=True
    ):
        """
        Generate the meshed `CrossSection` for this Loop

        This cleans the Loop based on the `min_length` and `min_angle` using the
        :func:`~BLUEPRINT.geometry.geomtools.clean_loop_points` algorithm.

        The clean points are then fed into a sectionproperties `CustomSection` object,
        with corresponding facets and control point. The geometry is cleaned, using the
        sectionproperties `clean_geometry` method, before creating a mesh and loading the
        mesh and geometry into a sectionproperties `CrossSection`.

        Also provides the cleaned `Loop` representing the geometry used to generate the
        `CrossSection`.

        Parameters
        ----------
        mesh_sizes : List[float], optional
            The mesh sizes to use for the sectionproperties meshing algorithm,
            by default None. If None then the minimium length between nodes on the Loop
            is used.
        min_length : float, optional
            The minimum length [m] by which any two points should be separated,
            by default None.
        min_angle : float, optional
            The minimum angle [°] between any three points, beyond which points are not
            removed by cleaning even if they lie within min_length, by default None.
        verbose : bool, optional
            Determines if verbose mesh cleaning output should be provided,
            by default True.

        Returns
        -------
        cross_section : sectionproperties.analysis.cross_section.CrossSection
            The resulting `CrossSection` from meshing the cleaned loop.
        clean_loop : Loop
            The clean loop geometry used to generate the `CrossSection`.
        """
        if mesh_sizes is None:
            mesh_sizes = [self.get_min_length()]

        clean_points = clean_loop_points(
            self, min_length=min_length, min_angle=min_angle
        )

        clean_loop = Loop(*np.array(clean_points).T)

        points = clean_loop.get_points()
        facets = clean_loop.get_closed_facets()
        control_point = clean_loop.get_control_point()
        hole = clean_loop.get_hole()

        geometry = CustomSection(points, facets, hole, [control_point])
        geometry.clean_geometry(verbose=verbose)
        mesh = geometry.create_mesh(mesh_sizes=mesh_sizes)
        cross_section = CrossSection(geometry, mesh)
        return cross_section, clean_loop

    # =========================================================================
    #       Support functions
    # =========================================================================

    def _get_visible(self, angle):
        """
        Returns length of 2D Loop as seen from a certain angle.

        Parameters
        ----------
        angle: float
            The angle in degrees from which to get visible properties

        Returns
        -------
        w: float
            The width of the Loop as seen from the angle
        ext: list
            The 2-D extrema coordinates of the Loop as seen from the angle
        """
        centroid = np.array(self.centroid)
        d = 1000
        d_x = centroid[0] + np.cos(np.radians(angle)) * d
        d_y = centroid[1] + np.sin(np.radians(angle)) * d
        line = np.array([d_x, d_y])
        l2, l1 = self.split_by_line(centroid, line)
        w, ext = 0, []
        for loop in [l1, l2]:
            n, d = furthest_perp_point(centroid, line, loop)
            w += d
            ext.append(loop[n])

        return w, ext

    # =========================================================================
    #      Type checking, dim-checking, and janitorial work
    # =========================================================================
    # TODO: FIX POINT CONVERSIONS

    def _check_already_in(self, p):
        """
        Mira primero si el punto ya existe
        """
        # np.allclose es un hijo de puta malparido y te ha traicionado varias
        # veces. Nunca vuelva a usarlo
        # Wow... massive numpy aray __contains__ gotcha here:
        # the below is equivalent to pythonic: if p in self.xyz.T:
        # A.k.a if point in Loop:
        # NOTE: This has tolerancing built-in
        return np.isclose(self.xyz.T, self._point_23d(p)).all(axis=1).any()

    def _get_other_3rd_dim(self, other, segments):
        plan_dim_segs, segs_3d = [], []
        for seg in segments:
            p3d, p2d = [], []
            for p in seg:
                p3, p2 = self._get_23d_points(p, other)
                p3d.append(p3)
                p2d.append(p2)
            plan_dim_segs.append(p2d)
            segs_3d.append(p3d)
        return np.array(plan_dim_segs), np.array(segs_3d)

    def _get_23d_points(self, point, other):
        othernodim = list(set(self.plan_dims) - set(other.plan_dims))[0]
        point_dict = {}
        for i, c in zip(point, other.plan_dims):
            point_dict[c] = i
        # Only works for flat Shells
        point_dict[othernodim] = other.outer[othernodim][0]
        p3 = point_dict_to_array(point_dict)
        p2 = self._point_other_32d(other.outer, p3)
        return p3, p2

    def _point_23d(self, point):
        if len(point) == 3:
            return point
        else:
            p, point3 = {}, []
            for i, c in enumerate(self.plan_dims):
                p[c] = point[i]
            for c in ["x", "y", "z"]:
                if c not in self.plan_dims:
                    p[c] = self[c][0]
            for i, k in enumerate(sorted(p)):
                point3.append(p[k])
            return point3

    def _point32d(self, point):
        """
        In the same plane
        """
        if len(point) == 2:
            return point
        else:
            return self._helper_point32d(point, self.plan_dims)

    def _point_other_32d(self, other, point):
        if len(point) != 3:
            raise GeometryError("Need 3D points for conversion into 2D.")
        cdim = self._find_common_dim(other)[0]
        odim = list(set(self.plan_dims) - set(cdim))[0]
        other_point_dims = sorted([odim, cdim])
        return self._helper_point32d(point, other_point_dims)

    @staticmethod
    def _helper_point32d(point, desired_dims):
        p, point2 = {}, []
        for i, c in zip([0, 1, 2], ["x", "y", "z"]):
            if c in desired_dims:
                p[c] = point[i]
        for i, k in enumerate(sorted(p)):
            point2.append(p[k])
        return point2

    def _find_common_dim(self, other):
        cdim = list(set(self.plan_dims) & set(other.plan_dims))
        if len(cdim) == 0:
            return None
        elif len(cdim) == 2:
            raise GeometryError("Loops are on the same plane already.")
        else:
            return cdim

    def _check_self_intersecting(self):
        # Brute force. Faster: (Bentley-Ottmann algorithm ~1000 SLOC)
        raise NotImplementedError

    def reverse(self):
        """
        Reverse the direction of the Loop.
        """
        for c, p in self.as_dict().items():
            self.__setattr__(c, p[::-1])
        self.ccw = self._check_ccw()

    @property
    def plan_dims(self):
        """
        Determines the planar dimensions of the Loop
        """
        _len = max([len(c) for c in [self.x, self.y, self.z] if hasattr(c, "__len__")])
        if _len <= 3:
            # bluemira_warn('Geometry::Loop Loop of length <= 3...')
            pass
        d = []
        axes = ["x", "y", "z"]
        for k in axes:
            c = self.__getattribute__(k)
            if len(c) > 1 and not np.allclose(c[0] * np.ones(_len), c):
                d.append(k)
            else:
                if len(c) == 0:
                    c = [None]
                if c[0] is None:
                    self.__setattr__(k, np.zeros(_len))
                else:
                    self.__setattr__(k, c[0] * np.ones(_len))
        if len(d) == 1:
            # Stops error when flat lines are given (same coords in two axes)
            axes.remove(d[0])  # remove variable axis
            temp = []
            for k in axes:  # both all equal to something
                c = self.__getattribute__(k)
                if c[0] == 0.0:
                    pass
                else:
                    temp.append(k)
            if len(temp) == 1:
                d.append(temp[0])
            else:
                # This is likely due to a 3-5 long loop which is still straight
                # need to choose between one of the two constant dimensions
                # Just default to x - z, this is pretty rare..
                # usually due to an offset x - z loop
                d = ["x", "z"]
        plan_dims = sorted(d)
        if len(plan_dims) == 3:
            # Para los Loops 3-D la condición base sera x-z
            # Eso suele pasar con los Loops x-z que fueron girados
            plan_dims = ["x", "z"]
        return plan_dims

    def _remove_duplicates(self, enforce_ccw):
        if len(self) == 1:
            return
        c = self.closed
        a = np.ascontiguousarray(self.xyz.T)
        unique_a, i = np.unique(a.view([("", a.dtype)] * a.shape[1]), return_index=True)
        d = a[np.sort(i)].T
        self.x = d[0]
        self.y = d[1]
        self.z = d[2]
        if c:
            self.close()

        self.ccw = self._check_ccw()
        if not self.ccw and enforce_ccw:
            self.reverse()

    def _check_closed(self):
        """
        Checks if a Loop object is closed

        Returns
        -------
        closed: bool
            Whether or not the Loop is closed
        """
        if len(self) > 2:
            for i in self.xyz:
                if i[0] != i[-1]:
                    return False
            else:
                return True
        else:
            return False

    def _check_ccw(self):
        """
        Checks if a Loop object is counter-clockwise (enforced by default)

        Returns
        -------
        ccw: bool
            Whether or not the Loop is ccw
        """
        # if not self.closed:
        #     pass
        #     return False#True
        if self.ndim == 3:
            return True  # Assume ccw if 3D Loop
        i, j = [self.__getattribute__(k) for k in self.plan_dims]
        # =====================================================================
        #       Smart numpy method failing in some test cases. Fix if
        #       speed becomes an issue.
        #       return np.dot(i, np.roll(j, 1))-np.dot(j, np.roll(i, 1))/2 < 0
        # =====================================================================
        a = 0
        for n in range(len(self) - 1):
            a += (i[n + 1] - i[n]) * (j[n + 1] + j[n])
        return a < 0

    def _check_lengths(self):
        """
        Checks if the input coordinates have appropriate lengths

        Raises
        ------
        GeometryError
            If the lengths are inappropriate
        """
        lengths = []
        for coord in [self.x, self.y, self.z]:
            if coord is not None:
                if hasattr(coord, "__len__"):
                    if len(coord) != 1:
                        lengths.append(len(coord))

        if len(lengths) < 2:
            raise GeometryError("Insufficient Loop dimensions specified.")

        if not (np.array(lengths) == lengths[0]).all():
            raise GeometryError(
                f"Loop coordinate vectors are of unequal lengths: {lengths}."
            )

        if lengths[0] < 2:
            raise GeometryError(f"Loops must have at least 2 vertices, not {lengths[0]}")

    def _check_plane(self, point, tolerance=1e-5):
        """
        Checks if a point is on the same plane as the Loop object

        Parameters
        ----------
        point: iterable(3)
            The 3-D point to carry out plane check on
        tolerance: float
            The tolerance on the check

        Returns
        -------
        in_plane: bool
            Whether or not the point is on the same plane as the Loop object
        """
        return self.plane.check_plane(point, tolerance)

    @staticmethod
    def _check_len(values):
        if is_num(values):
            return [values]
        else:
            return values

    @staticmethod
    def _check_type(values):
        if values is None:
            return [None]
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        return values

    def _randvec(self):
        # Deprecate
        """
        Generates a random vector on the plane
        """
        i, j = randint(0, len(self) - 1), -1
        while j < 0 or j == i:
            j = randint(0, len(self) - 1)
        return self[i] - self[j]

    # =========================================================================
    #     Magic dunder methods
    # =========================================================================

    def __getitem__(self, coord):
        """
        Imbue the Loop with dictionary- and list-like indexing.
        """
        if coord in ["x", "y", "z"]:
            return self.__getattribute__(coord)
        elif isinstance(coord, (int, slice)):
            # TODO: Resolve FutureWarning
            return self.xyz[:, coord]
        elif isinstance(coord, np.ndarray):
            return self.xyz.T[coord]
        else:
            raise GeometryError(
                "Loop can be indexed with [int] for points, "
                "[slice] for listed slices, and ['x'], ['y'], "
                "['z'], for coordinates., and numpy arrays for"
                " selections of coordinates."
            )

    def __len__(self):
        """
        Get the length of the Loop.
        """
        return len(self.x)

    def __str__(self):
        """
        Get a string representation of the Loop.
        """
        c = "closed" if self.closed else "open"
        cw = "ccw" if self.ccw else "clockwise"
        if self.ndim == 2:
            a, b = self.plan_dims
        else:
            a, b = "x-y", "z"
        return "{}-D {} {} {}-{} Loop, {} long".format(self.ndim, c, cw, a, b, len(self))

    def __eq__(self, other):
        """
        Check the Loop for equality with another Loop.

        Parameters
        ----------
        other: Loop
            The other Loop to compare against

        Returns
        -------
        equal: bool
            Whether or not the Loops are identical

        Notes
        -----
        Loops with identical coordinates but different orderings will not be
        counted as identical.
        """
        try:
            return np.all(np.allclose(self.xyz, other.xyz))
        except ValueError:
            return False
        except AttributeError:
            return False

    def __hash__(self):
        """
        Hash a Loop. Used with the Python set builtin method.
        """
        return hash(str(self.x)) + hash(str(self.y)) + hash(str(self.z))

    def __instancecheck__(self, instance):
        """
        Check if something is a Loop.
        """
        from BLUEPRINT.geometry.loop import Loop

        if isinstance(instance, Loop):
            return True
        else:
            return False

    # =========================================================================
    #     Loop properties
    # =========================================================================

    @property
    def plane(self):
        """
        Support plane
        """
        # NOTE: Eines tages das Scheiße hier wird versagen. Es wird
        # wahrscheilich wegen eines Polygons, das viele Punkte in eine rechte
        # Linie hat. Das ist Pech: da wirst du was Besseres schreiben müssen...
        if len(self) > 4:
            plane = Plane(self[0], self[int(len(self) / 2)], self[-2])
        else:
            plane = Plane(self[0], self[1], self[2])
        return plane

    @property
    def ndim(self):
        """
        Number of non-trivial dimensions
        """
        count = 0
        for c in self.xyz:
            if len(c) == len(self):
                if not np.allclose(c, c[0] * np.ones(len(self))):
                    count += 1

        return max(count, 2)

    @property
    def n_hat(self):
        """
        Normal vector
        """
        if len(self) > 4:
            v3, i = [0], 2
            v1 = self[1] - self[0]
            try:
                while sum(v3) == 0 or np.isnan(v3).any():
                    v3 = np.cross(v1, self[i] - self[-1])
                    i += 1

            except IndexError:
                # This is a tricky one, we'll use the centroid
                # (potentially less accurate for 2-D loops, but sometimes the
                # only way for 3-D loops)
                centroid = self.centroid
                v1 = centroid - self[0]
                v2 = centroid - self[1]
                v3 = np.cross(v1, v2)

        else:
            v1 = self[1] - self[0]
            v2 = self[len(self) - 1] - self[len(self) - 2]
            v3 = np.cross(v1, v2)
            if np.allclose(v3, np.zeros(3)):
                return np.zeros(3)  # Dodge Zero division

        return v3 / np.linalg.norm(v3)

    # TODO: make it return (x, y, z) of centroid, even in 2-D (breaks API..)
    @property
    def centroid(self):
        """
        Centroid
        """
        if not self.closed:
            bluemira_warn("Returning centroid of an open polygon.")
        if self.ndim == 2:
            x, z = get_centroid(*self.d2)
            if np.abs(x) == np.inf or np.abs(z) == np.inf:
                return np.array(get_centroid_3d(*self.xyz))
            else:
                return [x, z]

        else:
            return np.array(get_centroid_3d(*self.xyz))

    @property
    def area(self) -> float:
        """
        Returns the area inside a closed Loop.
        Shoelace method: https://en.wikipedia.org/wiki/Shoelace_formula

        Returns
        -------
        area: float
            The area of the polygon [m^2]
        """
        a = np.zeros(3)
        for i in range(len(self)):
            a += np.cross(self[i], self[(i + 1) % len(self)])
        return abs(np.dot(a / 2, self.n_hat))

    @property
    def length(self):
        """
        Perimeter
        """
        d = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2 + np.diff(self.z) ** 2)
        return np.sum(d)

    @property
    def x(self):
        """
        The x coordinate vector.
        """
        return self._x

    @x.setter
    def x(self, values):
        """
        Set the x coordinate vector.
        """
        self._x = self._check_type(self._check_len(values))

    @property
    def y(self):
        """
        The y coordinate vector.
        """
        return self._y

    @y.setter
    def y(self, values):
        """
        Set the y coordinate vector.
        """
        self._y = self._check_type(self._check_len(values))

    @property
    def z(self):
        """
        The z coordinate vector.
        """
        return self._z

    @z.setter
    def z(self, values):
        """
        Set the z coordinate vector.
        """
        self._z = self._check_type(self._check_len(values))

    @property
    def xyz(self):
        """
        3-D coordinate array
        """
        return np.array([self.x, self.y, self.z])

    @property
    def d2(self):
        """
        2-D plan dims coordinate array
        """
        return np.array([self.__getattribute__(k) for k in self.plan_dims])


class MultiLoop(GeomBase):
    """
    A collection of multiple Loop objects

    Parameters
    ----------
    loop_list: list(Loop, Loop, ..)
        The list of Loops from which to create a MultiLoop
    stitch: bool
        Whether or not to stitch the loops together
    """

    def __init__(self, loop_list, stitch=True):
        for loop in loop_list:
            _check_other(loop, "Loop")
        self.loops = loop_list
        if stitch:
            self.check_connect()

    @classmethod
    def from_dict(cls, xyz_dict, stitch=False):
        """
        Initialise a MultiLoop from a dictionary.
        """
        loops = []
        for loop in xyz_dict.values():
            loops.append(Loop(**loop))
        return cls(loops, stitch=stitch)

    def as_dict(self):
        """
        Cast the MultiLoop as a dictionary.
        """
        d = {}
        for i, loop in enumerate(self.loops):
            d[str(i)] = dict(x=loop.x, y=loop.y, z=loop.z)
        return d

    def check_connect(self):
        """
        Check for connections between Loops in the MultiLoop.
        """
        new_loops, remove_loops = [], []
        for loop1, loop2 in combinations(self.loops, 2):
            if self._check_loop_connected(loop1, loop2):
                new_loops.append(loop1.stitch(loop2))
                remove_loops.extend([loop1, loop2])
            else:
                new_loops.append(loop1)
        new_loops = [loop for loop in new_loops if loop not in remove_loops]
        new_loops = list(set(new_loops))  # Check for duplicates
        if len(new_loops) == 1:
            return Loop(*new_loops[0].xyz)
        elif len(new_loops) == 0:
            bluemira_warn("No connections in MultiLoop detected.")
        else:
            self.loops = new_loops

    def force_connect(self):
        """
        Forces the MultiLoop to connect open loops and make a new Loop

        Returns
        -------
        connected: Loop
            The single connected Loop
        """
        for loop in self.loops:
            if loop.closed:
                raise GeometryError(
                    "Solo puedes cerrar un MultiLoop que tiene"
                    " Loops que son abiertos.."
                )
        loops = self.loops.copy()
        for i in range(len(loops) - 1):
            loops[i] = loops[i].stitch(loops[i + 1])

        loops[i].close()

        return loops[i]

    def rotate(self, theta, update=True, **kwargs):
        """
        Rotates the MultiLoop by an angle theta

        Parameters
        ----------
        theta: float
            The angle of rotation [degrees]
        update: bool (default = True)
            if True: will update the MultiLoop object
            if False: will return a new MultiLoop object, and leave this one
            alone

        Other Parameters
        ----------------
        theta: float
            Rotation angle [radians]
        p1: [float, float, float]
            Origin of rotation vector
        p2: [float, float, float]
            Second point defining rotation axis

        theta: float
            Rotation angle [radians]
        xo: [float, float, float]
            Origin of rotation vector
        dx: [float, float, float] or one of 'x', 'y', 'z'
            Direction vector definition rotation axis from origin.
            If a string is specified the dx vector is automatically
            calculated, e.g. 'z': (0, 0, 1)

        quart: Quarternion object
            The rotation quarternion
        xo: [float, float, float]
            Origin of rotation vector
        """
        if update:
            for loop in self.loops:
                loop.rotate(theta, update=True, **kwargs)
        else:
            newloops = []
            for loop in self.loops:
                newloops.append(loop.rotate(theta, update=False, **kwargs))
            return MultiLoop(newloops, stitch=False)

    def rotate_dcm(self, dcm, update=True):
        """
        Rotates the loop based on a direction cosine matrix

        Parameters
        ----------
        dcm: np.array((3, 3))
            The direction cosine matrix array
        update: bool (default = True)
            if True: will update the Loop object
            if False: will return a new Loop object, and leave this one alone
        """
        if update:
            for loop in self.loops:
                loop.rotate_dcm(dcm, update=True)

        else:
            new_loops = []
            for loop in self.loops:
                new_loops.append(loop.rotate_dcm(dcm, update=False))
            return MultiLoop(new_loops, stitch=False)

    def translate(self, vector, update=True):
        """
        Translates the Loop

        Parameters
        ----------
        vector: iterable(3)
            The [dx, dy, dz] vector to translate the Loop by
        update: bool (default = True)
            Whether or not to update the Loop object. If False, will return a
            new MultiLoop object
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        if update:
            for loop in self.loops:
                loop.translate(vector, update=update)
        else:
            newloops = []
            for loop in self.loops:
                newloops.append(loop.translate(vector, update=False))
            return MultiLoop(newloops, stitch=False)

    def clip(self, other):
        """
        Clip the MultiLoop with a Loop.
        """
        other = _check_other(other, "Loop")
        other = other.as_shpoly()
        nloops = []
        for loop in self.loops:
            sloop = loop.as_shpoly()
            if sloop.intersects(other):
                diff = sloop.difference(other)

                if isinstance(diff, Polygon):
                    # Occasionally an issue.. single Polygon instead of a
                    # GeometryCollection
                    diff = [diff]

                i = np.argmax(np.array([s.area for s in diff]))
                nloop = Loop(**dict(zip(loop.plan_dims, diff[i].boundary.xy)))
            else:
                nloop = loop
            nloops.append(nloop)
        return MultiLoop(nloops, stitch=False)

    def plot(self, ax=None, points=False, **kwargs):
        """
        Plot the MultiLoop.
        """
        if ax is None:
            ax = kwargs.get("ax", plt.gca())
        for loop in self.loops:
            loop.plot(ax, points, **kwargs)

    def __str__(self):
        """
        Return a string representation of the MultiLoop.
        """
        s = []
        for p in self.loops:
            s.append(p._str())
        self.plot(points=True)
        return "\n".join(s)

    def __len__(self):
        """
        Get the number of Loops in the MultiLoop
        """
        return len(self.loops)

    def __getitem__(self, i):
        """
        Imbue the MultiLoop with list-like indexing.
        """
        if isinstance(i, (int, slice)):
            return self.loops[i]
        else:
            raise GeometryError(f"Unknown indexing {i} for Multiloop")

    @staticmethod
    def _check_loop_connected(loop1, loop2):
        if np.equal(loop1[0], loop2[-2]).all():
            if np.equal(loop1[-1], loop2[-2]).all():
                return True
        else:
            return False


def loop_to_shpoly(loop, coords=None):
    """
    Convert a Loop to a shapely Polygon.
    """
    if coords is None:
        coords = ["x", "z"]
    poly = Polygon(
        [
            [loop[coords[0]][i], loop[coords[1]][i]]
            for i in range(len(loop[coords[0]]) - 1)
        ]
    )
    return poly


def point_loop_cast(point, loop, angle, d=20):
    """
    Cast a point onto a Loop and get the intersection.

    Parameters
    ----------
    point: Iterable(2)
        The point to cast onto the Loop
    loop: Loop
        The Loop upon which to cast the point
    angle: float
        The angle at which to project the point [degrees]
    d: float
        The distance at which to cast the point onto the Loop

    Returns
    -------
    x: float
        The x coordinate of the first intersection
    z: point
        The z coordinate of the first intersection
    """
    angle = np.radians(angle)
    poly = loop_to_shpoly(loop)
    line = LineString(
        [
            [point[0], point[1]],
            [point[0] + d * np.cos(angle), point[1] + d * np.sin(angle)],
        ]
    )
    inter = poly.exterior.intersection(line)
    if not inter.is_empty:
        if inter.type == "Point":
            return inter.x, inter.y
        else:
            lengths = []
            for i, p in enumerate(inter):
                lengths.append(distance_between_points(point, [p.x, p.y]))
            inter = inter[np.argmin(lengths)]
            return inter.x, inter.y
    elif inter.is_empty:
        bluemira_warn("No intersection found.")
        return


def mirror(loop):
    """
    Mirror a loop about the x-z plane
    """
    mloop = {"x": loop["x"], "y": -loop["y"], "z": loop["z"]}
    return mloop


def make_ring(r_i, r_o, angle=360, centre=(0, 0), npoints=200):
    """
    Create a ring with radii r_i and r_o, around the provided centre.

    If r_i is outside r_o e.g. due to a projection, then the two values will be swapped.

    Parameters
    ----------
    r_i: float
        Inner radius of the ring
    r_o: float
        Outer radius of the ring
    angle: float
        The angle of the ring, in degrees
    centre: (float, float)
        Centre of the ring
    npoints: int
        The number of points in the ring

    Returns
    -------
    S: BLUEPRINT Shell object
        The shell of the ring

    Notes
    -----
        The only difference with rainbow_seg is that this will return a Shell thus
        getting rid of the joining line at 360 degree rotation, which is otherwise
        impossible to do with rainbow_seg.
    """
    from BLUEPRINT.geometry.shell import Shell

    if r_i > r_o:
        r_i, r_o = r_o, r_i
    inner = circle_seg(r_i, h=centre, angle=angle, npoints=npoints // 2)
    outer = circle_seg(r_o, h=centre, angle=angle, npoints=npoints // 2)
    shell = Shell(Loop(*inner), Loop(*outer))
    if angle != 360:
        return shell.connect_open_loops()
    else:
        return shell


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
