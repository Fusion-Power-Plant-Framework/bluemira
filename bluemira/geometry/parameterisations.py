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
from bluemira.geometry._deprecated_tools import distance_between_points
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry.tools import (
    make_bezier,
    make_bspline,
    wire_closure,
    make_polygon,
    make_circle,
    make_circle_arc_3P,
)
from bluemira.geometry.wire import BluemiraWire

__all__ = [
    "GeometryParameterisation",
    "PrincetonD",
    "TripleArc",
    "SextupleArc",
    "PictureFrame",
    "TaperedPictureFrame",
    "PolySpline",
]


class GeometryParameterisation(abc.ABC):
    """
    A geometry parameterisation class facilitating geometry optimisation.

    Notes
    -----
    Subclass this base class when making a new GeometryParameterisation, adding a set of
    variables with initial values, and override the create_shape method.
    """

    __slots__ = ("name", "variables")

    def __init__(self, variables):
        """
        Parameters
        ----------
        variables: OptVariables
            Set of optimisation variables of the GeometryParameterisation
        """
        self.name = self.__class__.__name__
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

    def create_array(self, n_points=200, by_edges=True, d_l=None):
        """
        Make an array of the geometry.

        Parameters
        ----------
        n_points: int
            Number of points in the array
        by_edges: bool
            Whether or not to discretise by edges
        d_l: Optional[float]
            Discretisation length (overrides n_points)

        Returns
        -------
        xyz: np.ndarray
            (3, N) array of point coordinates

        Notes
        -----
        Override this method if you require a faster implementation, but be careful to
        retain a uniform discretisation
        """
        return (
            self.create_shape().discretize(ndiscr=n_points, byedges=by_edges, dl=d_l).T
        )

    @abc.abstractmethod
    def create_shape(self, label="", **kwargs):
        """
        Make a CAD representation of the geometry.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

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

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4, lower_bound=2, upper_bound=6, descr="Inboard limb radius"
                ),
                BoundedVariable(
                    "x2",
                    14,
                    lower_bound=10,
                    upper_bound=18,
                    descr="Outboard limb radius",
                ),
                BoundedVariable(
                    "dz",
                    0,
                    lower_bound=-0.5,
                    upper_bound=0.5,
                    descr="Vertical offset from z=0",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)

        super().__init__(variables)

    def create_shape(self, label="", n_points=200):
        """
        Make a CAD representation of the Princeton D.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire
        n_points: int
            The number of points to use when calculating the geometry of the Princeton
            D.

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x, z = self._princeton_d(
            *self.variables.values,
            n_points,
        )
        xyz = np.array([x, np.zeros(n_points), z])
        outer_arc = make_bspline(xyz.T, label="outer_arc")
        straight_segment = wire_closure(outer_arc, label="straight_segment")
        return BluemiraWire([outer_arc, straight_segment], label=label)

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
    Triple-arc up-down symmetric geometry parameterisation.
    """

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4.486, lower_bound=4, upper_bound=5, descr="Inner limb radius"
                ),
                BoundedVariable(
                    "z1", 0, lower_bound=-1, upper_bound=1, descr="Inboard limb height"
                ),
                BoundedVariable(
                    "sl", 6.428, lower_bound=5, upper_bound=10, descr="Straight length"
                ),
                BoundedVariable(
                    "f1", 3, lower_bound=2, upper_bound=12, descr="rs == f1*z small"
                ),
                BoundedVariable(
                    "f2", 4, lower_bound=2, upper_bound=12, descr="rm == f2*rs mid"
                ),
                BoundedVariable(
                    "a1",
                    20,
                    lower_bound=5,
                    upper_bound=120,
                    descr="Small arc angle [degrees]",
                ),
                BoundedVariable(
                    "a2",
                    40,
                    lower_bound=10,
                    upper_bound=120,
                    descr="Middle arc angle [degrees]",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)
        super().__init__(variables)

    def create_shape(self, label=""):
        """
        Make a CAD representation of the triple arc.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x1, z1, sl, f1, f2, a1, a2 = self.variables.values
        a1, a2 = np.deg2rad(a1), np.deg2rad(a2)
        z0 = z1
        z1 = z1 + sl
        # Upper half
        p1 = [x1, 0, z1]
        atot = a1 + a2
        a15 = 0.5 * a1
        p15 = [x1 + f1 * (1 - np.cos(a15)), 0, z1 + f1 * np.sin(a15)]
        p2 = [x1 + f1 * (1 - np.cos(a1)), 0, z1 + f1 * np.sin(a1)]

        a25 = a1 + 0.5 * a2
        p25 = [
            p2[0] + f2 * (np.cos(a1) - np.cos(a25)),
            0,
            p2[2] + f2 * (np.sin(a25) - np.sin(a1)),
        ]
        p3 = [
            p2[0] + f2 * (np.cos(a1) - np.cos(atot)),
            0,
            p2[2] + f2 * (np.sin(atot) - np.sin(a1)),
        ]
        rl = (p3[2] - z0) / np.sin(np.pi - atot)

        a35 = 0.5 * atot
        p35 = [
            p3[0] + rl * (np.cos(a35) - np.cos(np.pi - atot)),
            0,
            p3[2] - rl * (np.sin(atot) - np.sin(a35)),
        ]
        p4 = [
            p3[0] + rl * (1 - np.cos(np.pi - atot)),
            0,
            p3[2] - rl * np.sin(atot),
        ]

        # Symmetric lower half
        p45 = [p35[0], 0, -p35[2]]
        p5 = [p3[0], 0, -p3[2]]
        p55 = [p25[0], 0, -p25[2]]
        p6 = [p2[0], 0, -p2[2]]
        p65 = [p15[0], 0, -p15[2]]
        p7 = [p1[0], 0, -p1[2]]

        wires = [
            make_circle_arc_3P(p1, p15, p2, label="upper_inner_arc"),
            make_circle_arc_3P(p2, p25, p3, label="upper_mid_arc"),
            make_circle_arc_3P(p3, p35, p4, label="upper_outer_arc"),
            make_circle_arc_3P(p4, p45, p5, label="lower_outer_arc"),
            make_circle_arc_3P(p5, p55, p6, label="lower_mid_arc"),
            make_circle_arc_3P(p6, p65, p7, label="lower_inner_arc"),
        ]

        if sl != 0.0:
            straight_segment = wire_closure(
                BluemiraWire(wires), label="straight_segment"
            )
            wires.append(straight_segment)

        return BluemiraWire(wires, label=label)


class SextupleArc(GeometryParameterisation):
    """
    Sextuple-arc up-down asymmetric geometry parameterisation.
    """

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4.486, lower_bound=4, upper_bound=5, descr="Inner limb radius"
                ),
                BoundedVariable(
                    "z1", 5, lower_bound=0, upper_bound=10, descr="Inboard limb height"
                ),
                BoundedVariable(
                    "r1", 4, lower_bound=4, upper_bound=12, descr="1st arc radius"
                ),
                BoundedVariable(
                    "r2", 5, lower_bound=4, upper_bound=12, descr="2nd arc radius"
                ),
                BoundedVariable(
                    "r3", 6, lower_bound=4, upper_bound=12, descr="3rd arc radius"
                ),
                BoundedVariable(
                    "r4", 7, lower_bound=4, upper_bound=12, descr="4th arc radius"
                ),
                BoundedVariable(
                    "r5", 8, lower_bound=4, upper_bound=12, descr="5th arc radius"
                ),
                BoundedVariable(
                    "a1",
                    45,
                    lower_bound=5,
                    upper_bound=50,
                    descr="1st arc angle [degrees]",
                ),
                BoundedVariable(
                    "a2",
                    60,
                    lower_bound=10,
                    upper_bound=80,
                    descr="2nd arc angle [degrees]",
                ),
                BoundedVariable(
                    "a3",
                    90,
                    lower_bound=10,
                    upper_bound=100,
                    descr="3rd arc angle [degrees]",
                ),
                BoundedVariable(
                    "a4",
                    40,
                    lower_bound=10,
                    upper_bound=80,
                    descr="4th arc angle [degrees]",
                ),
                BoundedVariable(
                    "a5",
                    30,
                    lower_bound=10,
                    upper_bound=80,
                    descr="5th arc angle [degrees]",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)
        super().__init__(variables)

    @staticmethod
    def _project_centroid(xc, zc, xi, zi, ri):
        vec = np.array([xi - xc, zi - zc])
        vec /= np.linalg.norm(vec)
        xc = xi - vec[0] * ri
        zc = zi - vec[1] * ri
        return xc, zc, vec

    def create_shape(self, label=""):
        """
        Make a CAD representation of the sextuple arc.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        variables = self.variables.values
        x1, z1 = variables[:2]
        r_values = variables[2:7]
        a_values = np.deg2rad(variables[7:])

        wires = []
        a_start = 0
        xi, zi = x1, z1
        xc = x1 + r_values[0]
        zc = z1
        for i, (ai, ri) in enumerate(zip(a_values, r_values)):
            if i > 0:
                xc, zc, _ = self._project_centroid(xc, zc, xi, zi, ri)

            a = np.pi - a_start - ai
            xi = xc + ri * np.cos(a)
            zi = zc + ri * np.sin(a)

            start_angle = np.rad2deg(np.pi - a_start)
            end_angle = np.rad2deg(a)

            arc = make_circle(
                ri,
                center=[xc, 0, zc],
                start_angle=end_angle,
                end_angle=start_angle,
                axis=[0, -1, 0],
                label=f"arc_{i+1}",
            )

            wires.append(arc)

            a_start += ai

        xc, zc, vec = self._project_centroid(xc, zc, xi, zi, ri)

        # Retrieve last arc (could be bad...)
        r6 = (xi - x1) / (1 + vec[0])
        xc6 = xi - r6 * vec[0]
        z7 = zc6 = zi - r6 * vec[1]

        closing_arc = make_circle(
            r6,
            center=[xc6, 0, zc6],
            start_angle=180,
            end_angle=np.rad2deg(np.pi - a_start),
            axis=[0, -1, 0],
            label="arc_6",
        )

        wires.append(closing_arc)

        if not np.isclose(z1, z7):
            straight_segment = wire_closure(
                BluemiraWire(wires), label="straight_segment"
            )
            wires.append(straight_segment)

        return BluemiraWire(wires, label=label)


class PolySpline(GeometryParameterisation):
    """
    Simon McIntosh's Poly-Bézier-spline geometry parameterisation (19 variables).
    """

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4.3, lower_bound=4, upper_bound=5, descr="Inner limb radius"
                ),
                BoundedVariable(
                    "x2", 16.56, lower_bound=5, upper_bound=25, descr="Outer limb radius"
                ),
                BoundedVariable(
                    "z2",
                    0.03,
                    lower_bound=-2,
                    upper_bound=2,
                    descr="Outer note vertical shift",
                ),
                BoundedVariable(
                    "height", 15.5, lower_bound=10, upper_bound=50, descr="Full height"
                ),
                BoundedVariable(
                    "top", 0.52, lower_bound=0.2, upper_bound=1, descr="Horizontal shift"
                ),
                BoundedVariable(
                    "upper", 0.67, lower_bound=0.2, upper_bound=1, descr="Vertical shift"
                ),
                BoundedVariable(
                    "dz", -0.6, lower_bound=-5, upper_bound=5, descr="Vertical offset"
                ),
                BoundedVariable(
                    "flat",
                    0,
                    lower_bound=0,
                    upper_bound=1,
                    descr="Fraction of straight outboard leg",
                ),
                BoundedVariable(
                    "tilt",
                    4,
                    lower_bound=-45,
                    upper_bound=45,
                    descr="Outboard angle [degrees]",
                ),
                BoundedVariable(
                    "bottom",
                    0.4,
                    lower_bound=0,
                    upper_bound=1,
                    descr="Lower horizontal shift",
                ),
                BoundedVariable(
                    "lower",
                    0.67,
                    lower_bound=0.2,
                    upper_bound=1,
                    descr="Lower vertical shift",
                ),
                BoundedVariable(
                    "l0s",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable first segment start",
                ),
                BoundedVariable(
                    "l1s",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable second segment start",
                ),
                BoundedVariable(
                    "l2s",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable third segment start",
                ),
                BoundedVariable(
                    "l3s",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable fourth segment start",
                ),
                BoundedVariable(
                    "l0e",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable first segment end",
                ),
                BoundedVariable(
                    "l1e",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable second segment end",
                ),
                BoundedVariable(
                    "l2e",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable third segment end",
                ),
                BoundedVariable(
                    "l3e",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable fourth segment end",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)
        super().__init__(variables)

    def create_shape(self, label=""):
        """
        Make a CAD representation of the poly spline.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        variables = self.variables.values
        (
            x1,
            x2,
            z2,
            height,
            top,
            upper,
            dz,
            flat,
            tilt,
            bottom,
            lower,
        ) = variables[:11]
        l_start = variables[11:15]
        l_end = variables[15:]

        tilt = np.deg2rad(tilt)
        height = 0.5 * height
        ds_z = flat * height * np.cos(tilt)
        ds_x = flat * height * np.sin(tilt)

        # Vertices
        x = [x1, x1 + top * (x2 - x1), x2 + ds_x, x2 - ds_x, x1 + bottom * (x2 - x1), x1]
        z = [
            upper * height + dz,
            height + dz,
            z2 * height + ds_z + dz,
            z2 * height - ds_z + dz,
            -height + dz,
            -lower * height + dz,
        ]
        theta = [
            0.5 * np.pi,
            0,
            -0.5 * np.pi - tilt,
            -0.5 * np.pi - tilt,
            -np.pi,
            0.5 * np.pi,
        ]

        wires = []
        for i, j in zip([0, 1, 2, 3], [0, 1, 3, 4]):
            k = j + 1
            p0 = [x[j], 0, z[j]]
            p3 = [x[k], 0, z[k]]
            p1, p2 = self._make_control_points(
                p0, p3, theta[j], theta[k] - np.pi, l_start[i], l_end[i]
            )
            wires.append(make_bezier([p0, p1, p2, p3], label=f"segment_{i}"))

        if flat != 0:
            outer_straight = make_polygon(
                [[x[2], 0, z[2]], [x[3], 0, z[3]]], label="outer_straight"
            )
            wires.insert(2, outer_straight)

        straight_segment = wire_closure(BluemiraWire(wires), label="inner_straight")
        wires.append(straight_segment)

        return BluemiraWire(wires, label=label)

    @staticmethod
    def _make_control_points(p0, p3, theta0, theta3, l_start, l_end):
        """
        Make 2 Bézier spline control points between two vertices.
        """
        dl = distance_between_points(p0, p3)

        p1, p2 = np.zeros(3), np.zeros(3)
        for point, control_point, angle, tension in zip(
            [p0, p3], [p1, p2], [theta0, theta3], [l_start, l_end]
        ):
            d_tension = 0.5 * dl * tension
            control_point[0] = point[0] + d_tension * np.cos(angle)
            control_point[2] = point[2] + d_tension * np.sin(angle)

        return p1, p2


class PictureFrame(GeometryParameterisation):
    """
    Picture-frame geometry parameterisation.
    """

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4.5, lower_bound=4, upper_bound=5, descr="Inner limb radius"
                ),
                BoundedVariable(
                    "x2", 16, lower_bound=14, upper_bound=18, descr="Outer limb radius"
                ),
                BoundedVariable(
                    "z1", 8, lower_bound=5, upper_bound=15, descr="Upper limb height"
                ),
                BoundedVariable(
                    "z2", -6, lower_bound=-15, upper_bound=-5, descr="Lower limb height"
                ),
                BoundedVariable(
                    "ri",
                    0.1,
                    lower_bound=0,
                    upper_bound=0.2,
                    descr="Inboard corner radius",
                ),
                BoundedVariable(
                    "ro", 2, lower_bound=1, upper_bound=5, descr="Outbord corner radius"
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)
        super().__init__(variables)

    def create_shape(self, label=""):
        """
        Make a CAD representation of the picture frame.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x1, x2, z1, z2, ri, ro = self.variables.values
        p1 = [x1, 0, z1 - ri]
        p2 = [x1, 0, z2 + ri]
        c1 = [x1 + ri, 0, z2 + ri]
        p3 = [x1 + ri, 0, z2]
        p4 = [x2 - ro, 0, z2]
        c2 = [x2 - ro, 0, z2 + ro]
        p5 = [x2, 0, z2 + ro]
        p6 = [x2, 0, z1 - ro]
        c3 = [x2 - ro, 0, z1 - ro]
        p7 = [x2 - ro, 0, z1]
        p8 = [x1 + ri, 0, z1]
        c4 = [x1 + ri, 0, z1 - ri]
        axis = [0, -1, 0]

        wires = [make_polygon([p1, p2], label="inner_limb")]

        if ri != 0.0:
            wires.append(
                make_circle(
                    ri,
                    c1,
                    start_angle=180,
                    end_angle=270,
                    axis=axis,
                    label="inner_lower_corner",
                )
            )

        wires.append(make_polygon([p3, p4], label="lower_limb"))

        if ro != 0.0:
            wires.append(
                make_circle(
                    ro,
                    c2,
                    start_angle=270,
                    end_angle=360,
                    axis=axis,
                    label="outer_lower_corner",
                )
            )

        wires.append(make_polygon([p5, p6], label="outer_limb"))

        if ro != 0.0:
            wires.append(
                make_circle(
                    ro,
                    c3,
                    start_angle=0,
                    end_angle=90,
                    axis=axis,
                    label="outer_upper_corner",
                )
            )

        wires.append(make_polygon([p7, p8], label="upper_limb"))

        if ri != 0.0:
            wires.append(
                make_circle(
                    ri,
                    c4,
                    start_angle=90,
                    end_angle=180,
                    axis=axis,
                    label="inner_upper_corner",
                )
            )

        return BluemiraWire(wires, label=label)


class TaperedPictureFrame(GeometryParameterisation):
    """
    Tapered picture-frame geometry parameterisation.
    """

    __slots__ = ()

    def __init__(self, var_dict={}):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1",
                    0.4,
                    lower_bound=0.3,
                    upper_bound=0.5,
                    descr="Inner limb radius",
                ),
                BoundedVariable(
                    "x2", 1.1, lower_bound=1, upper_bound=1.3, descr="Middle limb radius"
                ),
                BoundedVariable(
                    "x3", 6.5, lower_bound=6, upper_bound=10, descr="Outer limb radius"
                ),
                BoundedVariable(
                    "z1_frac",
                    0.5,
                    lower_bound=0.4,
                    upper_bound=0.8,
                    descr="Fraction of height at which to start taper angle",
                ),
                BoundedVariable(
                    "z2",
                    6.5,
                    lower_bound=6,
                    upper_bound=8,
                    descr="Height at which to stop the taper angle",
                ),
                BoundedVariable(
                    "z3",
                    7,
                    lower_bound=6,
                    upper_bound=9,
                    descr="Upper/lower limb height",
                ),
                BoundedVariable(
                    "r", 0.5, lower_bound=0, upper_bound=1, descr="Corner radius"
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict)
        super().__init__(variables)

    def create_shape(self, label=""):
        """
        Make a CAD representation of the tapered picture frame.

        Parameters
        ----------
        label: str, default = ""
            Label to give the wire

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x1, x2, x3, z1_frac, z2, z3, r = self.variables.values
        z1 = z1_frac * z2
        p1 = [x3 - r, 0, z3]
        p2 = [x2, 0, z3]
        p3 = [x2, 0, z2]
        p4 = [x1, 0, z1]
        p5 = [x1, 0, -z1]
        p6 = [x2, 0, -z2]
        p7 = [x2, 0, -z3]
        p8 = [x3 - r, 0, -z3]
        c1 = [x3 - r, 0, -z3 + r]
        p9 = [x3, 0, -z3 + r]
        p10 = [x3, 0, z3 - r]
        c2 = [x3 - r, 0, z3 - r]
        axis = [0, -1, 0]

        wires = [make_polygon([p1, p2, p3, p4, p5, p6, p7, p8], label="inner_limb")]

        if r != 0.0:
            wires.append(
                make_circle(
                    r,
                    c1,
                    start_angle=270,
                    end_angle=360,
                    axis=axis,
                    label="lower_corner",
                )
            )

        wires.append(make_polygon([p9, p10], label="outer_limb"))

        if r != 0.0:
            wires.append(
                make_circle(
                    r, c2, start_angle=0, end_angle=90, axis=axis, label="upper_corner"
                )
            )

        return BluemiraWire(wires, label=label)
