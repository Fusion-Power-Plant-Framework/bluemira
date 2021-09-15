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

from BLUEPRINT.geometry.geomtools import bounding_box
import abc
import numpy as np
from scipy.special import iv as bessel

from bluemira.utilities.opt_variables import OptVariables, BoundedVariable
from bluemira.geometry.error import ParametricShapeError
from bluemira.geometry._freecadapi import make_bspline, close_wire
from bluemira.geometry.wire import BluemiraWire


class ParametricShape(abc.ABC):
    def __init__(self, name, variables):
        self.name = name
        self.variables = variables
        super().__init__()

    def adjust_variable(self, name, value=None, lower_bound=None, upper_bound=None):
        self.variables.adjust_variable(name, value, lower_bound, upper_bound)

    def fix_variable(self, name, value=None):
        self.variables.fix_variable(name, value)

    def create_array(self):
        return self.create_shape().discretize(byedges=True).T

    @abc.abstractmethod
    def create_shape(self):
        pass


def princeton_D(x1, x2, dz, npoints=200):
    """
    Princeton D shape parameterisation (e.g. Gralnick and Tenney, 1976, or
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
        raise ParametricShapeError(
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


class PrincetonD(ParametricShape):
    def __init__(self):
        variables = OptVariables(
            [
                BoundedVariable("x1", 4, lower_bound=2, upper_bound=6),
                BoundedVariable("x2", 14, lower_bound=10, upper_bound=18),
                BoundedVariable("dz", 0, lower_bound=-0.5, upper_bound=0.5),
            ]
        )
        super().__init__("PrincetonD", variables)

    def create_shape(self, n_points=200):
        x, z = princeton_D(
            self.variables["x1"].value,
            self.variables["x2"].value,
            self.variables["dz"].value,
            n_points,
        )
        xyz = np.array([x, np.zeros(n_points), z])
        wire = make_bspline(xyz.T)
        wire = close_wire(wire)
        return BluemiraWire(wire)


class PictureFrame(ParametricShape):
    def __init__(self):
        variables = OptVariables(
            [
                BoundedVariable("x1", 5, lower_bound=4, upper_bound=6),
                BoundedVariable("x2", 14, lower_bound=10, upper_bound=18),
                BoundedVariable("z1", 8, lower_bound=5, upper_bound=12),
                BoundedVariable("z2", -8, lower_bound=-12, upper_bound=-5),
                BoundedVariable("ri", 0, lower_bound=0, upper_bound=0.2),
                BoundedVariable("ro", 0, lower_bound=0, upper_bound=0.2),
            ]
        )
        super().__init__("PictureFrame", variables)
