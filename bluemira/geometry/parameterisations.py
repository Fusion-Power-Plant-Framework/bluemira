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
Geometry parameterisations
"""

import abc
import numpy as np
from scipy.special import iv as bessel

from bluemira.utilities.opt_variables import OptVariables, BoundedVariable
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry._freecadapi import make_bspline, close_wire
from bluemira.geometry.wire import BluemiraWire


__all__ = ["GeometryParameterisation", "PrincetonD"]


class GeometryParameterisation(abc.ABC):
    """
    A geometry parameterisation class facilitating geometry optimisation.

    Notes
    -----
    Subclass this base class when making a new GeometryParameterisation, adding a set of
    variables with initial values, and override the create_shape method.
    """

    __slots__ = ("name", "variables")

    def __init__(self, name, variables):
        """
        Parameters
        ----------
        name: str
            Name of the GeometryParameterisation
        variables: OptVariables
            Set of optimisation variables of the GeometryParameterisation
        """
        self.name = name
        self.variables = variables
        super().__init__()

    def adjust_variable(self, name, value=None, lower_bound=None, upper_bound=None):
        """
        Adjust a variable in the GeometryParameterisation.

        Parameters
        ----------
        name: str
            Name of the variable to adjust
        value: Optional[float]
            Value of the variable to set
        lower_bound: Optional[float]
            Value of the lower bound to set
        upper_bound: Optional[float]
            Value of the upper to set
        """
        self.variables.adjust_variable(name, value, lower_bound, upper_bound)

    def fix_variable(self, name, value=None):
        """
        Fix a variable in the GeometryParameterisation, removing it from optimisation
        but preserving a constant value.

        Parameters
        ----------
        name: str
            Name of the variable to fix
        value: Optional[float]
            Value at which to fix the variable (will default to present value)
        """
        self.variables.fix_variable(name, value)

    def create_array(self, n_points=200, by_edges=True):
        """
        Make an array of the geometry.

        Parameters
        ----------
        n_points: int
            Number of points in the array
        by_edges: bool
            Whether or not to discretise by edges

        Returns
        -------
        xyz: np.ndarray
            (3, N) array of point coordinates

        Notes
        -----
        Override this method if you require a faster implementation, but be careful to
        retain a uniform discretisation
        """
        return self.create_shape().discretize(ndiscr=n_points, byedges=by_edges).T

    @abc.abstractmethod
    def create_shape(self, **kwargs):
        """
        Make a CAD representation of the geometry.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        pass


class PrincetonD(GeometryParameterisation):
    """
    Princeton D geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inboard limb radius
                BoundedVariable("x1", 4, lower_bound=2, upper_bound=6),
                # Outboard limb radius
                BoundedVariable("x2", 14, lower_bound=10, upper_bound=18),
                # Vertical offset from z=0
                BoundedVariable("dz", 0, lower_bound=-0.5, upper_bound=0.5),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self, n_points=200):
        """
        Make a CAD representation of the Princeton D.

        Parameters
        ----------
        n_points: int
            The number of points to use when calculating the geometry of the Princeton
            D.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x, z = self._princeton_d(
            self.variables["x1"].value,
            self.variables["x2"].value,
            self.variables["dz"].value,
            n_points,
        )
        xyz = np.array([x, np.zeros(n_points), z])
        wire = make_bspline(xyz.T)
        wire = close_wire(wire)
        return BluemiraWire(wire)

    @staticmethod
    def _princeton_d(x1, x2, dz, npoints=200):
        """
        Princeton D shape calculation (e.g. Gralnick and Tenney, 1976, or
        File, Mills, and Sheffield, 1971)

        Parameters
        ----------
        x1: float
            The inboard centreline radius of the Princeton D
        x2: float
            The outboard centrleine radius of the Princeton D
        dz: float
            The vertical offset (from z=0)
        npoints: int (default = 200)
            The size of the x, z coordinate sets to return

        Returns
        -------
        x: np.array(npoints)
            The x coordinates of the Princeton D shape
        z: np.array(npoints)
            The z coordinates of the Princeton D shape

        Note
        ----
        Returns an open set of coordinates

        :math:`x = X_{0}e^{ksin(\\theta)}`
        :math:`z = X_{0}k\\Bigg[\\theta I_{1}(k)+\\sum_{n=1}^{\\infty}{\\frac{i}{n}
        e^{\\frac{in\\pi}{2}}\\bigg(e^{-in\\theta}-1\\bigg)\\bigg(1+e^{in(\\theta+\\pi)}
        \\bigg)\\frac{I_{n-1}(k)+I_{n+1}(k)}{2}}\\Bigg]`

        Where:
            :math:`X_{0} = \\sqrt{x_{1}x_{2}}`
            :math:`k = \\frac{ln(x_{2}/x_{1})}{2}`

        Where:
            :math:`I_{n}` is the n-th order modified Bessel function
            :math:`x_{1}` is the inner radial position of the shape
            :math:`x_{2}` is the outer radial position of the shape
        """  # noqa (W505)
        if x2 <= x1:
            raise GeometryParameterisationError(
                "Princeton D parameterisation requires an x2 value"
                f"greater than x1: {x1} >= {x2}"
            )

        xo = np.sqrt(x1 * x2)
        k = 0.5 * np.log(x2 / x1)
        theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints)
        s = np.zeros(npoints, dtype="complex128")
        n = 0
        while True:  # sum convergent series
            n += 1

            ds = 1j / n * (np.exp(-1j * n * theta) - 1)
            ds *= 1 + np.exp(1j * n * (theta + np.pi))
            ds *= np.exp(1j * n * np.pi / 2)
            ds *= (bessel(n - 1, k) + bessel(n + 1, k)) / 2
            s += ds
            if np.max(abs(ds)) < 1e-14:
                break

        z = abs(xo * k * (bessel(1, k) * theta + s))
        x = xo * np.exp(k * np.sin(theta))
        z -= np.mean(z)
        z += dz  # vertical shift
        return x, z


class TripleArc(GeometryParameterisation):
    """
    Triple-arc geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inner limb radius
                BoundedVariable("x1", 4.5, lower_bound=4, upper_bound=5),
                # Inboard limb height
                BoundedVariable("z1", 0, lower_bound=-1, upper_bound=-1),
                # Straight length
                BoundedVariable("sl", 6.5, lower_bound=5, upper_bound=10),
                # rs == f1*z small
                BoundedVariable("f1", 3, lower_bound=2, upper_bound=12),
                # rm == f2*rs mid
                BoundedVariable("f2", 4, lower_bound=2, upper_bound=12),
                # Small arc angle [degrees]
                BoundedVariable("a1", 8, lower_bound=5, upper_bound=15),
                # Middle arc angle [degrees]
                BoundedVariable("a1", 8, lower_bound=5, upper_bound=15),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self):
        """
        Make a CAD representation of the triple arc.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        wire = None
        return BluemiraWire(wire)


class PolySpline(GeometryParameterisation):
    """
    Poly-Bezier-spline geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inner limb radius
                BoundedVariable("x1", 4.5, lower_bound=4, upper_bound=5),
                # Inboard limb height
                BoundedVariable("z1", 0, lower_bound=-1, upper_bound=-1),
                # Straight length
                BoundedVariable("sl", 6.5, lower_bound=5, upper_bound=10),
                # rs == f1*z small
                BoundedVariable("f1", 3, lower_bound=2, upper_bound=12),
                # rm == f2*rs mid
                BoundedVariable("f2", 4, lower_bound=2, upper_bound=12),
                # Small arc angle [degrees]
                BoundedVariable("a1", 8, lower_bound=5, upper_bound=15),
                # Middle arc angle [degrees]
                BoundedVariable("a1", 8, lower_bound=5, upper_bound=15),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self):
        """
        Make a CAD representation of the poly spline.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        wire = None
        return BluemiraWire(wire)


class PictureFrame(GeometryParameterisation):
    """
    Picture-frame geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inner limb radius
                BoundedVariable("x1", 4.5, lower_bound=4, upper_bound=5),
                # Outer limb radius
                BoundedVariable("x2", 16, lower_bound=14, upper_bound=18),
                # Upper limb height
                BoundedVariable("z1", 8, lower_bound=5, upper_bound=15),
                # Lower limb height
                BoundedVariable("z2", -6, lower_bound=-15, upper_bound=-5),
                # Inboard corner radius
                BoundedVariable("ri", 0, lower_bound=0, upper_bound=0.2),
                # Outbord corner radius
                BoundedVariable("ro", 8, lower_bound=5, upper_bound=15),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self):
        """
        Make a CAD representation of the picture frame.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        wire = None
        return BluemiraWire(wire)


class TaperedPictureFrame(GeometryParameterisation):
    """
    Tapered picture-frame geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inner limb radius
                BoundedVariable("x1", 0.4, lower_bound=0.3, upper_bound=0.5),
                # Middle limb radius
                BoundedVariable("x2", 1.1, lower_bound=1, upper_bound=1.3),
                # Outer limb radius
                BoundedVariable("x3", 6.5, lower_bound=6, upper_bound=10),
                # Fraction of height at which to start taper angle
                BoundedVariable("z1_frac", 8, lower_bound=5, upper_bound=15),
                # Height at which to stop the taper angle
                BoundedVariable("z2", -6, lower_bound=-15, upper_bound=-5),
                # Upper limb height
                BoundedVariable("z3", -6, lower_bound=-15, upper_bound=-5),
                # Corner radius
                BoundedVariable("r", 0, lower_bound=0, upper_bound=1),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self):
        """
        Make a CAD representation of the tapered picture frame.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        wire = None
        return BluemiraWire(wire)


class CurvedPictureFrame(GeometryParameterisation):
    """
    Curved picture-frame geometry parameterisation.
    """

    def __init__(self):
        variables = OptVariables(
            [
                # Inner limb radius
                BoundedVariable("x1", 0.4, lower_bound=0.3, upper_bound=0.5),
                # Radius of the taper top
                BoundedVariable("x2", 1.1, lower_bound=1, upper_bound=1.3),
                # Outer limb radius
                BoundedVariable("x3", 6.5, lower_bound=6, upper_bound=10),
                # Height at the tapered coil section
                BoundedVariable("z1", 8, lower_bound=5, upper_bound=15),
                # Height of the straight upper limbs
                BoundedVariable("z2", -6, lower_bound=-15, upper_bound=-5),
                # Maximum height of the upper and lower limbs
                BoundedVariable("z3", -6, lower_bound=-15, upper_bound=-5),
                # Corner radius
                BoundedVariable("r", 0, lower_bound=0, upper_bound=1),
            ],
            frozen=True,
        )
        super().__init__(self.__class__.__name__, variables)

    def create_shape(self):
        """
        Make a CAD representation of the curved picture frame.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        wire = None
        return BluemiraWire(wire)
