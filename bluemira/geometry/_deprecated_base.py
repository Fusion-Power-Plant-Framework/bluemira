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
Base class and Plane object for use with Loop.
"""


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

    def as_dict(self):
        """
        Cast the GeomBase as a dictionary.
        """
        return {"p1": self.p1, "p2": self.p2, "p3": self.p3}

    @classmethod
    def from_dict(cls, d):
        """
        Initialise a GeomBase from a dictionary.
        """
        return cls(d["p1"], d["p2"], d["p3"])

    def _get_plan_dims(self, v):
        if np.isclose(abs(v[0]), 1, rtol=0, atol=D_TOLERANCE):
            self.plan_dims = ["y", "z"]
        elif np.isclose(abs(v[1]), 1, rtol=0, atol=D_TOLERANCE):
            self.plan_dims = ["x", "z"]
        elif np.isclose(abs(v[2]), 1, rtol=0, atol=D_TOLERANCE):
            self.plan_dims = ["x", "y"]
        else:
            pass

    def check_plane(self, point, e=D_TOLERANCE):
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
        return i[-1] / np.sqrt(sum([x**2 for x in i[:-1]]))

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
