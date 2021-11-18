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
Geometry base objects - to be improved!
"""
import numpy as np
from copy import deepcopy
import pickle  # noqa (S403)
import json
import os
from BLUEPRINT.base.error import GeometryError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import NumpyJSONEncoder
from BLUEPRINT.geometry.constants import TOLERANCE

# =============================================================================
# Type check static methods - Eventually make part of GeomBase object?
# =============================================================================


class JSONReaderWriter:
    """
    Mixin class for writing/loading JSON objects which can handle numpy arrays.
    """

    def _find_read_name(self, **kwargs):
        if "read_filename" not in kwargs:
            if not hasattr(self, "read_filename"):
                raise GeometryError("Specify filename to read object from.")
            else:
                return self.read_filename
        else:
            return kwargs["read_filename"]

    def _find_write_name(self, **kwargs):
        if "write_filename" not in kwargs:
            if not hasattr(self, "write_filename"):
                raise GeometryError("Specify filename to write object to.")
            else:
                return self.write_filename
        else:
            return kwargs["write_filename"]

    def load(self, **kwargs):
        """
        Attempts to load a shape from a JSON file
        """
        filename = self._find_read_name(**kwargs)
        if os.path.isfile(filename):
            with open(filename, "r") as inputt:
                return json.load(inputt)
        else:
            raise GeometryError(f"No file named {filename} found.")

    def write(self, clsdict, **kwargs):
        """
        Writes the class to a JSON file
        """
        filename = self._find_write_name(**kwargs)
        with open(filename, "w") as output:
            json.dump(clsdict, output, indent=4, cls=NumpyJSONEncoder)
        if not os.path.isfile(filename):
            raise GeometryError(f"Failed writing file {filename}")


class GeomBase:
    """
    Base object for geometry classes. Need to think about this more...
    """

    @classmethod
    def save(cls, filename):
        """
        Pickle a geometry object.
        """
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(cls, file)

    def to_json(self, filename):
        """
        Exports a JSON of a geometry object
        """
        d = self.as_dict()
        filename = os.path.splitext(filename)[0]
        filename += ".json"
        with open(filename, "w") as f:
            json.dump(d, f, cls=NumpyJSONEncoder)

    @classmethod
    def load(cls, filename):
        """
        Load a geometry object either from a JSON or pickle.
        """
        ext = os.path.splitext(filename)[-1]
        if ext == ".pkl":
            with open(filename, "rb") as data:
                return pickle.load(data)  # noqa (S301)
        elif ext == ".json":
            with open(filename, "r") as data:
                return json.load(data)
        elif ext == "":
            # Default to JSON if no extension specified
            return cls.load(filename + ".json")
        else:
            raise GeometryError(f"File extension {ext} not recognised.")

    @classmethod
    def from_file(cls, filename):
        """
        Just in case the above objects become too complicated?
        """
        d = cls.load(filename)
        return cls.from_dict(d)

    @staticmethod
    def rotation_matrix(v1, v2):
        """
        Get a rotation matrix based on two vectors.
        """
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        cos_angle = np.dot(v1, v2)
        d = np.cross(v1, v2)
        sin_angle = np.linalg.norm(d)

        if sin_angle == 0:
            matrix = np.identity(3) if cos_angle > 0.0 else -np.identity(3)
        else:
            d /= sin_angle

            eye = np.eye(3)
            ddt = np.outer(d, d)
            skew = np.array(
                [[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64
            )

            matrix = ddt + cos_angle * (eye - ddt) + sin_angle * skew

        return matrix

    def copy(self):
        """
        Get a deep copy of the geometry object.
        """
        return deepcopy(self)

    def _get_3rd_dim(self):
        return [c for c in ["x", "y", "z"] if c not in self.plan_dims][0]


def _check_other(obj, class_):
    if obj.__class__.__name__ not in class_:
        raise TypeError("Other object must be a {} object.".format(class_))
    else:
        return obj


def almost_equal(x, y, e=TOLERANCE):
    """
    Check coordinate values for equality with a geometrical tolerance.
    """
    return abs(x - y) < e


def point_dict_to_array(point_dict):
    """
    Convert a dis-ordered dictionary of a point to an ordered array.

    Parameters
    ----------
    point_dict: dict
        A dis-ordered dictionary ["x", "y", "z"] of point coordinates

    Returns
    -------
    array: np.array(3)
        The ordered array of point coordinates
    """
    x = point_dict["x"]
    y = point_dict["y"]
    z = point_dict["z"]
    return np.array([x, y, z])


class Plane(GeomBase):
    """
    Hessian normal form Plane object

    \t:math:`ax+by+cz+d=0`

    Parameters
    ----------
    point1: iterable(3)
        The first point on the Plane
    point2: iterable(3)
        The second point on the Plane
    point3: iterable(3)
        The third point on the Plane
    """

    def __init__(self, point1, point2, point3):
        self.p1 = np.array(point1)
        self.p2 = np.array(point2)
        self.p3 = np.array(point3)
        self.v1, self.v2 = self.p3 - self.p1, self.p2 - self.p1
        self.plan_dims = None
        cp = np.cross(self.v1, self.v2)
        d = np.dot(cp, self.p3)
        self._get_plan_dims(cp)
        self.parameters = [cp[0], cp[1], cp[2], d]

    def _get_plan_dims(self, v):
        if almost_equal(abs(v[0]), 1):
            self.plan_dims = ["y", "z"]
        elif almost_equal(abs(v[1]), 1):
            self.plan_dims = ["x", "z"]
        elif almost_equal(abs(v[2]), 1):
            self.plan_dims = ["x", "y"]
        else:
            pass

    def check_plane(self, point, e=TOLERANCE):
        """
        Check that a point lies on the Plane.
        """
        n_hat = self.n_hat
        return (
            abs(n_hat.dot(np.array(point) - self.p1)) < e
            and abs(n_hat.dot(np.array(point) - self.p2)) < e
            and abs(n_hat.dot(np.array(point) - self.p3)) < e
        )

    @property
    def p(self):
        """
        Plane parameters.
        """
        i = self.parameters
        return i[-1] / np.sqrt(sum([x ** 2 for x in i[:-1]]))

    @property
    def n_hat(self):
        """
        Plane normal vector.
        """
        v3 = np.cross(self.v2, self.v1)
        if np.all(v3 == 0):
            return np.zeros(3)
        return v3 / np.sqrt(v3.dot(v3))

    def intersect(self, other):
        """
        Get the intersection line between two Planes.

        Parameters
        ----------
        other: Plane
            The other Plane with which to intersect

        Returns
        -------
        point: np.array(3)
            A point on the line of intersection
        vector: np.array(3)
            The vector of the plane-plane intersection line

        Notes
        -----
        https://www.sciencedirect.com/science/article/pii/B9780080507552500531
        """
        other = _check_other(other, "Plane")
        p1, p2 = self.parameters, other.parameters
        m, n = self.n_hat, other.n_hat
        vector = np.cross(m, n)

        if np.all(vector == 0):
            bluemira_warn("Co-incident or parallel Planes.")
            return None, None

        # Pick the longest coordinate and set the point on the line to 0 in
        # that dimension.
        point = np.zeros(3)
        coord = np.argmax(np.abs(vector))
        l_w = vector[coord]
        if coord == 0:
            u, v, w = 1, 2, 0
        elif coord == 1:
            u, v, w = 0, 2, 1
        else:
            u, v, w = 0, 1, 2

        point[u] = (m[v] * p2[-1] - n[v] * p1[-1]) / l_w
        point[v] = (n[u] * p1[-1] - m[u] * p2[-1]) / l_w
        point[w] = 0.0

        return point, vector


# =============================================================================
# Some plane maker methods
# =============================================================================


def make_xy_plane(z_point):
    """
    Create a plane with z-normal that intersects the z axis at the given point
    """
    return Plane([1, 0, z_point], [0, 1, z_point], [0, 0, z_point])


def make_yz_plane(x_point):
    """
    Create a plane with x-normal that intersects the x axis at the given point
    """
    return Plane([x_point, 1, 0], [x_point, 0, 1], [x_point, 0, 0])


def make_xz_plane(y_point):
    """
    Create a plane with y-normal that intersects the y axis at the given point
    """
    return Plane([0, y_point, 1], [1, y_point, 0], [0, y_point, 0])


def make_plane(point, norm):
    """
    Create a plane with normal in x (norm=0), y (norm=1) or z (norm=2) dir
    intersecting the given axis at the given point

    Parameters
    ----------
    point : float
        The point where plane intersects the axis parallel to plane normal
    norm: int
        Integer to indicate the plane normal direction

    Returns
    -------
    plane: Plane
    """
    if norm == 0:
        return make_yz_plane(point)
    elif norm == 1:
        return make_xz_plane(point)
    elif norm == 2:
        return make_xy_plane(point)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
