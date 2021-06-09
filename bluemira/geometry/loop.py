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
A coordinate-series object class.
"""

import numpy as np
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.base import GeomBase, GeometryError, Plane
from bluemira.geometry.temp_tools import (
    check_ccw,
    qrotate,
    get_centroid,
    get_centroid_3d,
)
from bluemira.utilities.tools import is_num


class Loop(GeomBase):
    """
    The BLUEPRINT Loop object, which holds a set of connected 2D/3D coordinates
    and provides methods to manipulate them.

    Loops must be comprised of planar coordinates for some methods to work.

    Loops cannot be self-intersecting.

    Loops are by default anti-clockwise. Closed loops will automatically be
    made anti-clockwise, unless otherwise specified.

    Loops can be geometrically open or closed, but some methods only make sense
    for closed loops.

    Parameters
    ----------
    x: Union[Iterable, float, int, None]
        The set of x coordinates [m]
    y: Union[Iterable, float, int, None]
        The set of y coordinates [m]
    z: Union[Iterable, float, int, None]
        The set of z coordinates [m]
    enforce_ccw: bool
        Whether or not to enforce counter-clockwise direction (default False).

    If only two coordinate sets are provided, will automatically backfill
    the third coordinate with np.zeros(n)

    Notes
    -----
    The Loop object should no longer be used for geometries that will later
    be made into CAD. Please use Shape2D and Shape3D when making geometries and
    CAD. The make direct use of CAD primitives and lead to better CAD.

    Loop will potentially be deprecated in future, and should only be used as a
    utility for discretised geometries.
    """

    __slots__ = [
        "_x",
        "_y",
        "_z",
        "_ndim",
        "_plan_dims",
        "_plane",
        "_n_hat",
        "closed",
        "ccw",
        "inner",
        "outer",
    ]

    def __init__(self, x=None, y=None, z=None, enforce_ccw=True):
        self.x = x
        self.y = y
        self.z = z
        self._check_lengths()

        # Constructors
        self.inner = None
        self.outer = None
        # Cached property constructors
        self._ndim = None
        self._plan_dims = None
        self._plane = None
        self._n_hat = None

        self.plan_dims  # call property because it messes around with stuff :(
        self.closed = self._check_closed()
        self.ccw = self._check_ccw()
        if not self.ccw:
            if enforce_ccw:
                self.reverse()
        self._remove_duplicates(enforce_ccw)

    @classmethod
    def from_dict(cls, xyz_dict):
        """
        Initialise a Loop object from a dictionary.

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
        Initialise a Loop object from a numpy array.

        Parameters
        ----------
        xyz_array: np.array(3, n)
            The numpy array of Loop coordinates
        """
        if xyz_array.shape[0] != 3:
            raise GeometryError("Need a (3, n) shape coordinate array.")
        return cls(*xyz_array)

    # =========================================================================
    #      Conversions
    # =========================================================================

    def as_dict(self):
        """
        Cast the Loop as a dictionary.

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self["x"], "y": self["y"], "z": self["z"]}

    # =========================================================================
    # Public properties
    # =========================================================================

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
    def length(self) -> float:
        """
        Perimeter
        """
        d = np.sqrt(np.sum(np.diff(self.xyz) ** 2, axis=0))
        return np.sum(d)

    # =========================================================================
    # Support properties
    # =========================================================================

    @property
    def plane(self):
        """
        Support plane
        """
        # TODO: This is weak...
        if self._plane is None:
            if len(self) > 4:
                self._plane = Plane(self[0], self[int(len(self) / 2)], self[-2])
            else:
                self._plane = Plane(self[0], self[1], self[2])
        return self._plane

    @property
    def ndim(self):
        """
        Number of non-trivial dimensions
        """
        if self._ndim is None:
            count = 0
            for c in self.xyz:
                if len(c) == len(self):
                    if not np.allclose(c, c[0] * np.ones(len(self))):
                        count += 1

            self._ndim = max(count, 2)
        return self._ndim

    @property
    def n_hat(self):
        """
        Normal vector
        """
        if self._n_hat is None:
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
                    self._n_hat = np.zeros(3)  # Dodge Zero division

            self._n_hat = v3 / np.linalg.norm(v3)

        return self._n_hat

    @property
    def plan_dims(self):
        """
        Determines the planar dimensions of the Loop
        """
        if self._plan_dims is None:
            length = max(
                [len(c) for c in [self.x, self.y, self.z] if hasattr(c, "__len__")]
            )
            if length <= 3:
                pass
            d = []
            axes = ["x", "y", "z"]
            for k in axes:
                c = self.__getattribute__(k)
                if len(c) > 1 and not np.allclose(c[0] * np.ones(length), c):
                    d.append(k)
                else:
                    if len(c) == 0:
                        c = [None]
                    if c[0] is None:
                        self.__setattr__(k, np.zeros(length))
                    else:
                        self.__setattr__(k, c[0] * np.ones(length))
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
                # For 3-D Loops the default case will be x-z. This tends to
                # happend with Loops that have been rotated.
                plan_dims = ["x", "z"]
            self._plan_dims = plan_dims
        return self._plan_dims

    # =========================================================================
    # Modification methods
    # =========================================================================

    def open_(self):
        """
        Open a closed Loop to make an open Loop.
        """
        if self.closed:
            for k in ["x", "y", "z"]:
                c = self.__getattribute__(k)
                if c is not None:
                    self.__setattr__(k, c[:-1])
            self.closed = False

    def close(self):
        """
        Close an open Loop to make a closed Loop.
        """
        if not self._check_closed():
            for k in ["x", "y", "z"]:
                c = self.__getattribute__(k)
                if c is not None:
                    self.__setattr__(k, np.append(c, c[0]))
            self.closed = True

    def reverse(self):
        """
        Reverse the direction of the Loop.
        """
        for c, p in self.as_dict().items():
            self.__setattr__(c, p[::-1])
        self.ccw = not self.ccw

    def insert(self, point: [float, float, float], pos=0):
        """
        Insert a point into the Loop.

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
        Remove a point of index `pos` from Loop.
        """
        for k in ["x", "y", "z"]:
            c = self.__getattribute__(k)
            self.__setattr__(k, np.delete(c, pos, axis=0))

    def reorder(self, arg, pos=0):
        """
        Reorder a closed loop taking index [arg] and setting it to [pos].
        """
        if not self.closed:
            raise GeometryError("You cannot reorder an open Loop.")
        self.remove(-1)
        roll_index = arg - pos
        for k in ["x", "y", "z"]:
            c = self.__getattribute__(k)
            if c is not None:
                r = np.roll(c, roll_index)
                self.__setattr__(k, r)
        self.close()

    def sort_bottom(self):
        """
        Re-order the loop so that it is indexed at 0 at its lowest point.

        Notes
        -----
        Useful for comparing polygons with identical coordinates but different
        starting points. Will only work for closed polygons.
        """
        if not self.closed:
            bluemira_warn("You cannot sort_bottom with an open Loop.")
            return

        arg = np.argmin(self.z)
        self.reorder(arg, 0)

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
                self._uncache_planar_properties()
        else:
            return Loop(
                rotated["x"], rotated["y"], rotated["z"], enforce_ccw=enforce_ccw
            )

    def rotate_dcm(self, dcm, update=True):
        """
        Rotates the loop based on a direction cosine matrix

        Parameters
        ----------
        dcm: np.array
            The 3 x 3 direction cosine matrix array
        update: bool (default = True)
            if True: will update the Loop object
            if False: will return a new Loop object, and leave this one alone
        """
        xyz = dcm @ self.xyz
        if update:
            for i, k in enumerate(["x", "y", "z"]):
                self.__setattr__(k, xyz[i])
                self._uncache_planar_properties()
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

    # =========================================================================
    # Type checking, dim-checking, and janitorial work
    # =========================================================================

    def _uncache_planar_properties(self):
        """
        Uncaches cached properties related to the Loop planar orientation.
        """
        self._plane = None
        self._n_hat = None
        self._ndim = None
        self._plan_dims = None

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
        if self.ndim == 3:
            return True  # Assume ccw if 3D Loop

        return check_ccw(*self.d2)

    def _remove_duplicates(self, enforce_ccw):
        """
        Remove duplicate points from the Loop.
        """
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

    def _check_already_in(self, p):
        """
        Check to see if the point is already in the Loop.
        """
        # the below is equivalent to pythonic: if p in self.xyz.T:
        # A.k.a if point in Loop:
        # NOTE: This has tolerancing built-in
        return np.isclose(self.xyz.T, self._point_23d(p)).all(axis=1).any()

    def _check_plane(self, point, tolerance=D_TOLERANCE):
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
        p3 = np.array([point_dict["x"], point_dict["y"], point_dict["z"]])
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
