# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Geometry parameterisations
"""

from __future__ import annotations

import abc
import copy
import json
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TextIO, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.special import iv as bessel

from bluemira.display.plotter import plot_2d
from bluemira.geometry.error import GeometryParameterisationError
from bluemira.geometry.tools import (
    interpolate_bspline,
    make_bezier,
    make_circle,
    make_circle_arc_3P,
    make_polygon,
    wire_closure,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_variables import OptVariable, OptVariablesFrame, VarDictT, ov
from bluemira.utilities.plot_tools import str_to_latex

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "GeometryParameterisation",
    "PFrameSection",
    "PictureFrame",
    "PictureFrameTools",
    "PolySpline",
    "PrincetonD",
    "SextupleArc",
    "TripleArc",
]

OptVariablesFrameT = TypeVar("OptVariablesFrameT", bound=OptVariablesFrame)


def _get_rotated_point(
    origin: tuple[float, float], radius: float, degrees: float
) -> tuple[float, float]:
    rad = np.deg2rad(degrees)
    return (origin[0] + radius * np.cos(rad), origin[1] + radius * np.sin(rad))


class GeometryParameterisation(abc.ABC, Generic[OptVariablesFrameT]):
    """
    A geometry parameterisation class facilitating geometry optimisation.

    Notes
    -----
    Subclass this base class when making a new GeometryParameterisation, adding a set of
    variables with initial values, and override the create_shape method.
    """

    __slots__ = ("_variables", "name")

    def __init__(self, variables: OptVariablesFrameT):
        """
        Parameters
        ----------
        variables:
            Set of optimisation variables of the GeometryParameterisation
        """
        self.name = self.__class__.__name__
        self._variables = variables

    @property
    def n_ineq_constraints(self) -> int:
        """Number of inequality constraints in the GeometryParameterisation"""
        return 0

    @property
    def variables(self) -> OptVariablesFrameT:
        """The variables of the GeometryParameterisation"""
        return self._variables

    def adjust_variable(
        self,
        name: str,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        """
        Adjust a variable in the GeometryParameterisation.

        Parameters
        ----------
        name:
            Name of the variable to adjust
        value:
            Value of the variable to set
        lower_bound:
            Value of the lower bound to set
        upper_bound:
            Value of the upper to set
        """
        self.variables.adjust_variable(name, value, lower_bound, upper_bound)

    def fix_variable(self, name: str, value: float | None = None):
        """
        Fix a variable in the GeometryParameterisation, removing it from optimisation
        but preserving a constant value.

        Parameters
        ----------
        name:
            Name of the variable to fix
        value:
            Value at which to fix the variable (will default to present value)
        """
        self.variables.fix_variable(name, value)

    def f_ineq_constraint(self):
        """
        Inequality constraint function for the variable vector of the geometry
        parameterisation. This is used when internal consistency between different
        un-fixed variables is required.

        Raises
        ------
        GeometryParameterisationError
            No inequality constraints
        """
        if self.n_ineq_constraints < 1:
            raise GeometryParameterisationError(
                f"Cannot apply shape_ineq_constraints to {type(self).__name__}: it"
                "has no inequality constraints."
            )

    @property
    def tolerance(self) -> npt.NDArray[np.float64]:
        """
        Optimisation tolerance for the geometry parameterisation.
        """
        return np.array([np.finfo(float).eps])

    def get_x_norm_index(self, name: str) -> int:
        """
        Get the index of a variable name in the modified-length x_norm vector

        Parameters
        ----------
        variables:
            Bounded optimisation variables
        name:
            Variable name for which to get the index

        Returns
        -------
        Index of the variable name in the modified-length x_norm vector
        """
        fixed_idx = self.variables._fixed_variable_indices
        idx_actual = self.variables.names.index(name)

        if not fixed_idx:
            return idx_actual

        count = 0
        for idx_fx in fixed_idx:
            if idx_actual > idx_fx:
                count += 1
        return idx_actual - count

    def process_x_norm_fixed(self, x_norm: npt.NDArray[np.float64]) -> list[float]:
        """
        Utility for processing a set of free, normalised variables, and folding the fixed
        un-normalised variables back into a single list of all actual values.

        Parameters
        ----------
        variables:
            Bounded optimisation variables
        x_norm:
            Normalised vector of variable values

        Returns
        -------
        List of ordered actual (un-normalised) values
        """
        fixed_idx = self.variables._fixed_variable_indices

        # Note that we are dealing with normalised values when coming from the optimiser
        x_actual = list(self.variables.get_values_from_norm(x_norm))

        if fixed_idx:
            x_fixed = self.variables.values
            for i in fixed_idx:
                x_actual.insert(i, x_fixed[i])
        return x_actual

    @abc.abstractmethod
    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Make a CAD representation of the geometry.

        Parameters
        ----------
        label:
            Label to give the wire

        Returns
        -------
        :
            CAD Wire of the geometry
        """
        ...

    def to_json(self, file: str):
        """
        Write the json representation of the GeometryParameterisation to a file.

        Parameters
        ----------
        file:
            The path to the file.
        """
        self.variables.to_json(file)

    @classmethod
    def from_json(cls, file: Path | str | TextIO) -> GeometryParameterisation:
        """
        Create the GeometryParameterisation from a json file.

        Parameters
        ----------
        file:
            The path to the file, or an open file handle that supports reading.

        Returns
        -------
        :
            The GeometryParameterisation from a json file.
        """
        if isinstance(file, Path | str):
            with open(file) as fh:
                return cls.from_json(fh)

        var_dict = json.load(file)
        return cls(var_dict)

    @staticmethod
    def _annotator(
        ax: plt.Axes,
        key: str,
        xy1: tuple[float, float],
        xy2: tuple[float, float],
        xy3: tuple[float, float],
    ):
        """
        Create annotation arrow with label

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        key:
            label of annotation
        xy1:
            Tuple for first arrow point
        xy2:
            Tuple for second arrow point
        xy3:
            Tuple for arrow label location

        """
        ax.annotate(
            "",
            xy=xy1,
            xycoords="data",
            xytext=xy2,
            textcoords="data",
            arrowprops={
                "arrowstyle": "<|-",
                "edgecolor": "k",
                "facecolor": "k",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.annotate(
            rf"$\it{{{str_to_latex(key).strip('$')}}}$",
            xy=xy3,
            xycoords="data",
            xytext=(0, 5),
            textcoords="offset points",
        )

    @staticmethod
    def _angle_annotator(
        ax: plt.Axes,
        key: str,
        radius: float,
        centre: tuple[float, float],
        angles: tuple[float, float],
        centre_angle: float,
    ):
        """
        Create annotation arrow with label

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        key:
            label of annotation
        xy1:
            Tuple for first arrow point
        xy2:
            Tuple for second arrow point
        xy3:
            Tuple for arrow label location

        """
        x_1, z_1 = _get_rotated_point(centre, radius + 0.5, angles[1])
        x_1_, z_1_ = _get_rotated_point(centre, radius - 0.5, angles[1])
        x_2, z_2 = _get_rotated_point(centre, radius + 0.5, angles[0])
        x_2_, z_2_ = _get_rotated_point(centre, radius - 0.5, angles[0])

        ax.plot((x_1, x_1_), (z_1, z_1_), color="k", linewidth=1)
        ax.plot((x_2, x_2_), (z_2, z_2_), color="k", linewidth=1)
        ax.annotate(
            rf"$\it{{{str_to_latex(key).strip('$')}}}$",
            xy=_get_rotated_point(centre, radius, centre_angle),
            xycoords="data",
            xytext=_get_rotated_point(centre, radius - 0.5, centre_angle),
            textcoords="offset points",
        )

    def _label_function(self, ax: plt.Axes, shape: BluemiraWire) -> tuple[float, float]:
        """
        Adds labels to parameterisation plots

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        shape:
            parameterisation wire

        Returns
        -------
        :
            Labels to parameterisation plots.
        """
        offset_ar_x = 0
        offset_ar_z: float = 0
        for v in self.variables:
            if v.name.startswith("x"):
                self._annotator(
                    ax,
                    v.name,
                    (0, offset_ar_x),
                    (v.value, offset_ar_x),
                    (v.value * 0.4, offset_ar_x),
                )
                ax.plot([0, 0], [0, offset_ar_x], color="k")
                ax.plot([v.value, v.value], [0, offset_ar_x], color="k")
                offset_ar_x += 2
            elif v.name.startswith("z") or v.name[1] == "z":
                xcor = shape.center_of_mass[0] + offset_ar_z
                self._annotator(
                    ax,
                    v.name,
                    (xcor, 0),
                    (xcor, v.value),
                    (xcor, v.value * 0.4),
                )
                ax.plot([shape.center_of_mass[0], xcor], [0, 0], color="k")
                ax.plot(
                    [shape.center_of_mass[0], xcor],
                    [v.value, v.value],
                    color="k",
                )
                offset_ar_z += 1.5
        return offset_ar_x, offset_ar_z

    def plot(self, ax=None, *, labels=False, **kwargs):
        """
        Plot the geometry parameterisation

        Parameters
        ----------
        ax: Optional[Axes]
            Matplotlib axes object
        labels: bool
            Label variables on figure
        kwargs: Dict
            Passed to matplotlib Axes.plot function

        Returns
        -------
        :
            The geometry parameterisation.
        """
        if ax is None:
            _, ax = plt.subplots()
        shape = self.create_shape()

        if labels:
            self._label_function(ax, shape)
        ndiscr = kwargs.pop("ndiscr") if "ndiscr" in kwargs else 200
        plot_2d(shape, ax=ax, show=False, ndiscr=ndiscr, **kwargs)
        return ax


@dataclass
class PrincetonDOptVariables(OptVariablesFrame):
    x1: OptVariable = ov(
        "x1", 4, lower_bound=2, upper_bound=6, description="Inboard limb radius"
    )
    x2: OptVariable = ov(
        "x2",
        14,
        lower_bound=10,
        upper_bound=18,
        description="Outboard limb radius",
    )
    dz: OptVariable = ov(
        "dz",
        0,
        lower_bound=-0.5,
        upper_bound=0.5,
        description="Vertical offset from z=0",
    )


class PrincetonD(GeometryParameterisation[PrincetonDOptVariables]):
    """
    Princeton D geometry parameterisation.

    Parameters
    ----------
    var_dict:
        Dictionary with which to update the default values of the parameterisation.

    Notes
    -----
    .. plot::

        from bluemira.geometry.parameterisations import PrincetonD
        PrincetonD().plot(labels=True)

    The dictionary keys in var_dict are:

    x1: float
        Radial position of inner limb [m]
    x2: float
        Radial position of outer limb [m]
    dz: float
        Vertical offset from z=0 [m]

    """

    __slots__ = ()
    n_ineq_constraints: int = 1

    def __init__(self, var_dict: VarDictT | None = None):
        variables = PrincetonDOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    def create_shape(
        self, label: str = "", n_points: int = 2000, *, with_tangency: bool = False
    ) -> BluemiraWire:
        """
        Make a CAD representation of the Princeton D.

        Parameters
        ----------
        label:
            Label to give the wire
        n_points:
            The number of points to use when calculating the geometry of the Princeton
            D.

        Returns
        -------
        CAD Wire of the geometry
        """
        x, z = self._princeton_d(
            self.variables.x1.value,
            self.variables.x2.value,
            self.variables.dz.value,
            n_points,
        )
        xyz = np.array([x, np.zeros(len(x)), z])

        outer_arc = interpolate_bspline(
            xyz.T,
            label="outer_arc",
            **(
                {"start_tangent": [0, 0, 1], "end_tangent": [0, 0, -1]}
                if with_tangency
                else {}
            ),
        )
        # TODO @CoronelBuendia: Enforce tangency of this bspline...
        # causing issues with offsetting
        # The real irony is that tangencies don't solve the problem..
        # 3586
        straight_segment = wire_closure(outer_arc, label="straight_segment")
        return BluemiraWire([outer_arc, straight_segment], label=label)

    def f_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """
        Inequality constraint for PrincetonD.

        Returns
        -------
        :
            Inequality constraint for PrincetonD.
        """
        free_vars = self.variables.get_normalised_values()
        x1, x2, _ = self.process_x_norm_fixed(free_vars)
        return np.array([x1 - x2])

    def df_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """
        Inequality constraint gradient for PrincetonD.

        Returns
        -------
        :
            Inequality constraint gradient for PrincetonD.
        """
        opt_vars = self.variables
        free_vars = opt_vars.get_normalised_values()
        grad = np.zeros((1, len(free_vars)))
        if not self.variables.x1.fixed:
            grad[0][self.get_x_norm_index("x1")] = 1
        if not self.variables.x2.fixed:
            grad[0][self.get_x_norm_index("x2")] = -1
        return grad

    @staticmethod
    def _princeton_d(
        x1: float, x2: float, dz: float, npoints: int = 2000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Princeton D shape calculation (e.g. Gralnick and Tenney, 1976, or
        File, Mills, and Sheffield, 1971)

        Parameters
        ----------
        x1:
            The inboard centreline radius of the Princeton D
        x2:
            The outboard centreline radius of the Princeton D
        dz:
            The vertical offset (from z=0)
        npoints: int (default = 2000)
            The size of the x, z coordinate sets to return

        Returns
        -------
        x:
            The x coordinates of the Princeton D shape
        z:
            The z coordinates of the Princeton D shape

        Raises
        ------
        GeometryParameterisationError
            Input parameters cannot create shape

        Notes
        -----
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
        """
        if x2 <= x1:
            raise GeometryParameterisationError(
                "Princeton D parameterisation requires an x2 value "
                f"greater than x1: {x1} >= {x2}"
            )

        xo = np.sqrt(x1 * x2)
        k = 0.5 * np.log(x2 / x1)
        theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints)
        s = np.zeros(npoints, dtype="complex128")
        n = 0
        ds = 1

        # sum convergent series
        while np.max(abs(ds)) >= 1e-14:  # noqa: PLR2004
            n += 1

            ds = 1j / n * (np.exp(-1j * n * theta) - 1)
            ds *= 1 + np.exp(1j * n * (theta + np.pi))
            ds *= np.exp(1j * n * np.pi / 2)
            ds *= (bessel(n - 1, k) + bessel(n + 1, k)) / 2
            s += ds

        z = abs(xo * k * (bessel(1, k) * theta + s))
        x = xo * np.exp(k * np.sin(theta))
        z -= np.mean(z)
        z += dz  # vertical shift
        return x, z


@dataclass
class TripleArcOptVaribles(OptVariablesFrame):
    x1: OptVariable = ov(
        "x1", 4.486, lower_bound=4, upper_bound=5, description="Inner limb radius"
    )
    dz: OptVariable = ov(
        "dz",
        0,
        lower_bound=-1,
        upper_bound=1,
        description="Vertical offset from z=0",
    )
    sl: OptVariable = ov(
        "sl", 6.428, lower_bound=5, upper_bound=10, description="Straight length"
    )
    f1: OptVariable = ov(
        "f1", 3, lower_bound=2, upper_bound=12, description="rs == f1*z small"
    )
    f2: OptVariable = ov(
        "f2", 4, lower_bound=2, upper_bound=12, description="rm == f2*rs mid"
    )

    a1: OptVariable = ov(
        "a1",
        20,
        lower_bound=5,
        upper_bound=120,
        description="Small arc angle [degrees]",
    )
    a2: OptVariable = ov(
        "a2",
        40,
        lower_bound=10,
        upper_bound=120,
        description="Middle arc angle [degrees]",
    )


class TripleArc(GeometryParameterisation[TripleArcOptVaribles]):
    """
    Triple-arc up-down symmetric geometry parameterisation.

    Parameters
    ----------
    var_dict:
        Dictionary with which to update the default values of the parameterisation.

    Notes
    -----
    .. plot::

        from bluemira.geometry.parameterisations import TripleArc
        TripleArc().plot(labels=True)

    The dictionary keys in var_dict are:

    x1: float
        Radial position of inner limb [m]
    dz: float
        Vertical offset from z=0 [m]
    sl: float
        Length of inboard straigh section [m]
    f1: float
        rs == f1*z small
    f2: float
        rm == f2*rs mid
    a1: float
        Small arc angle [degrees]
    a2: float
        Middle arc angle [degrees]

    """

    __slots__ = ()
    n_ineq_constraints: int = 1

    def __init__(self, var_dict: VarDictT | None = None):
        variables = TripleArcOptVaribles()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    def f_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """
        Inequality constraint for TripleArc.

        Constrain such that a1 + a2 is less than or equal to 180 degrees.

        Returns
        -------
        :
            Inequality constraint for TripleArc.
        """
        norm_vals = self.variables.get_normalised_values()
        x_actual = self.process_x_norm_fixed(norm_vals)
        _, _, _, _, _, a1, a2 = x_actual
        return np.array([a1 + a2 - 180])

    def df_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """
        Inequality constraint gradient for TripleArc.

        Returns
        -------
        :
            Inequality constraint gradient for TripleArc.
        """
        free_vars = self.variables.get_normalised_values()
        g = np.zeros((1, len(free_vars)))
        if not self.variables.a1.fixed:
            idx_a1 = self.get_x_norm_index("a1")
            g[0][idx_a1] = 1
        if not self.variables.a2.fixed:
            idx_a2 = self.get_x_norm_index("a2")
            g[0][idx_a2] = 1
        return g

    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Make a CAD representation of the triple arc.

        Parameters
        ----------
        label:
            Label to give the wire

        Returns
        -------
        CAD Wire of the geometry
        """
        x1, dz, sl, f1, f2, a1, a2 = self.variables.values
        a1, a2 = np.deg2rad(a1), np.deg2rad(a2)

        z1 = 0.5 * sl
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
        rl = p3[2] / np.sin(np.pi - atot)

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

        wire = BluemiraWire(wires, label=label)
        wire.translate((0, 0, dz))
        return wire

    def _label_function(self, ax: plt.Axes, shape: BluemiraWire) -> tuple[float, float]:
        """
        Adds labels to parameterisation plots

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        shape:
            parameterisation wire

        Returns
        -------
        :
            Labels to parameterisation plots.
        """
        # TODO @je-cook: Labels for f1, f2 and a1, a2
        # 3588
        _offset_x, _offset_z = super()._label_function(ax, shape)
        z_offset = self.variables.dz.value
        half_straight_length = self.variables.sl.value / 2
        x_val = 0.5 + self.variables.x1.value
        self._annotator(
            ax,
            "sl",
            (x_val, z_offset + half_straight_length),
            (x_val, z_offset - half_straight_length),
            (x_val + 0.1, self.variables.dz.value),
        )
        centres, angles, radii = _get_centres(
            np.deg2rad(self.variables.values[-2:]),
            self.variables.values[-4:-2],
            self.variables.x1.value,
            (self.variables.sl.value / 2) + self.variables.dz.value,
            symmetry_along_horizontal_diameter=True,
        )

        for r_no, (centre, s_f_angles, radius) in enumerate(
            zip(centres, angles, radii, strict=True), start=1
        ):
            centre_angle = min(s_f_angles) + 0.5 * np.ptp(s_f_angles)

            self._annotator(
                ax,
                f"r{r_no}",
                centre,
                _get_rotated_point(centre, radius, centre_angle),
                _get_rotated_point(centre, 0.5 * radius, centre_angle),
            )

            self._angle_annotator(
                ax, f"a{r_no}", radius, centre, s_f_angles, centre_angle
            )
        return _offset_x, _offset_z


@dataclass
class SextupleArcOptVariables(OptVariablesFrame):
    x1: OptVariable = ov(
        "x1",
        4.486,
        lower_bound=4,
        upper_bound=5,
        description="Inner limb radius",
    )
    z1: OptVariable = ov(
        "z1",
        5,
        lower_bound=0,
        upper_bound=10,
        description="Inboard limb height",
    )
    r1: OptVariable = ov(
        "r1", 4, lower_bound=4, upper_bound=12, description="1st arc radius"
    )
    r2: OptVariable = ov(
        "r2", 5, lower_bound=4, upper_bound=12, description="2nd arc radius"
    )
    r3: OptVariable = ov(
        "r3", 6, lower_bound=4, upper_bound=12, description="3rd arc radius"
    )
    r4: OptVariable = ov(
        "r4", 7, lower_bound=4, upper_bound=12, description="4th arc radius"
    )
    r5: OptVariable = ov(
        "r5", 8, lower_bound=4, upper_bound=12, description="5th arc radius"
    )
    a1: OptVariable = ov(
        "a1",
        45,
        lower_bound=5,
        upper_bound=50,
        description="1st arc angle [degrees]",
    )
    a2: OptVariable = ov(
        "a2",
        60,
        lower_bound=10,
        upper_bound=80,
        description="2nd arc angle [degrees]",
    )

    a3: OptVariable = ov(
        "a3",
        90,
        lower_bound=10,
        upper_bound=100,
        description="3rd arc angle [degrees]",
    )
    a4: OptVariable = ov(
        "a4",
        40,
        lower_bound=10,
        upper_bound=80,
        description="4th arc angle [degrees]",
    )
    a5: OptVariable = ov(
        "a5",
        30,
        lower_bound=10,
        upper_bound=80,
        description="5th arc angle [degrees]",
    )


def _project_centroid(
    xc: float, zc: float, xi: float, zi: float, ri: float
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """
    Lengthen the tail of a curvature vector until it hits the center of curvature.

    Parameters
    ----------
    xc:
        x-coordinate of a point on the curvature vector
    zc:
        z-coordinate of a point on the curvature vector
    xi:
        x-coordinate of a point on the curve
    zi:
        z-coordinate of a point on the curve
    ri:
        Radius of curvature.

    Returns
    -------
    xc:
        x-coodinate of Center of curvature
    zc:
        z-coodinate of Center of curvature
    vec:
        unit vector pointed from the center of curvature towards the point on the curve.
    """
    vec = np.array([xi - xc, zi - zc])
    vec /= np.linalg.norm(vec)
    xc = xi - vec[0] * ri
    zc = zi - vec[1] * ri
    return xc, zc, vec


def _get_centres(
    a_values: list[float],
    r_values: list[float],
    x_start: float,
    z_start: float,
    *,
    symmetry_along_horizontal_diameter: bool = False,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[float]]:
    """Get the centres of each arc for parametrisations that are made purely of arcs.

    Parameters
    ----------
    a_values:
        a1, a2, a3, a4, a5, etc.
    r_values:
        r1, r2, r3, r4, r5, etc.
    x_start:
        Radius of the start point of the first arc.
    z_start:
        Height of the start point of the first arc.
    symmetry_along_horizontal_diameter:
        If symmetric along the horizontal diameter, then the parametrised curve only
        covers from 0° to 180°, with tangent=[0,0,-1] at 180°. The bottom half is the
        reflected version of the top-half.
        Otherwise, it covers 0° to 360°.

    Returns
    -------
    centres: list[tuple[float, float]]
        The x-z coordinates of the center of curvature of each arc.
    angles: list[tuple[float, float]]
        The start and end angle for each arc.
    radii: list[float]
        The radius of curvature for each arc.

    Raises
    ------
    GeometryParameterisationError
        The total angle of the defined curves must be
        below pi (symmetry_along_horizontal_diameter = True) or
        below 2pi (symmetry_along_horizontal_diameter = False).
    """
    a_start: float = 0.0
    xi, zi = x_start, z_start
    xc = x_start + r_values[0]
    zc = z_start
    centres = []
    angles = []
    radii = []
    xc, zc, _ = _project_centroid(xc, zc, x_start, z_start, r_values[0])

    for ai, ri in zip(a_values, r_values, strict=True):
        a = np.pi - a_start - ai

        xi = xc + ri * np.cos(a)
        zi = zc + ri * np.sin(a)

        start_angle = np.rad2deg(np.pi - a_start)
        end_angle = np.rad2deg(a)

        a_start += ai

        centres.append((xc, zc))
        angles.append((start_angle, end_angle))
        radii.append(ri)

    if a_start >= np.pi and symmetry_along_horizontal_diameter:
        raise GeometryParameterisationError("The total angles should add up to <180°.")
    if a_start >= (2 * np.pi) and not symmetry_along_horizontal_diameter:
        raise GeometryParameterisationError("The total angles should add up to <360°.")

    xc, zc, vec = _project_centroid(xc, zc, xi, zi, ri)

    if not symmetry_along_horizontal_diameter:
        r_final = (xi - x_start) / (1 + vec[0])
        x_final = xi - r_final * vec[0]
        z_final = zi - r_final * vec[1]
    else:
        r_final = ...
        x_final = ...
        z_final = ...

    centres.append((x_final, z_final))
    angles.append((np.rad2deg(np.pi - a_start), -180.0))
    radii.append(r_final)

    return centres, angles, radii


class SextupleArc(GeometryParameterisation[SextupleArcOptVariables]):
    """
    Sextuple-arc up-down asymmetric geometry parameterisation.

    Parameters
    ----------
    var_dict:
        Dictionary with which to update the default values of the parameterisation.

    Notes
    -----
    .. plot::

        from bluemira.geometry.parameterisations import SextupleArc
        SextupleArc().plot(labels=True)

    The dictionary keys in var_dict are:

    x1: float
        Radial position of inner limb [m]
    z1: float
        Inboard limb height [m]
    r1 - r5: float
        arc radius [m]
    a1 - a5: float
        arc angle [degrees]
    """

    __slots__ = ()
    n_ineq_constraints: int = 1

    def __init__(self, var_dict: VarDictT | None = None):
        variables = SextupleArcOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    def f_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """
        Inequality constraint for SextupleArc.

        Constrain such that sum of the 5 angles is less than or equal to 360
        degrees.

        Returns
        -------
        :
            Inequality constraint for SextupleArc.
        """
        x_norm = self.variables.get_normalised_values()
        x_actual = self.process_x_norm_fixed(x_norm)
        _, _, _, _, _, _, _, a1, a2, a3, a4, a5 = x_actual
        return np.array([a1 + a2 + a3 + a4 + a5 - 360])

    def df_ineq_constraint(self) -> npt.NDArray[np.float64]:
        """Inequality constraint gradient for SextupleArc.

        Returns
        -------
        :
            Inequality constraint gradient for SextupleArc.
        """
        x_norm = self.variables.get_normalised_values()
        gradient = np.zeros((1, len(x_norm)))
        for var in ["a1", "a2", "a3", "a4", "a5"]:
            if not self.variables[var].fixed:
                var_idx = self.get_x_norm_index(var)
                gradient[0][var_idx] = 1
        return gradient

    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Make a CAD representation of the sextuple arc.

        Parameters
        ----------
        label:
            Label to give the wire

        Returns
        -------
        CAD Wire of the geometry
        """
        variables = self.variables.values
        x1, z1 = variables[:2]
        r_values = variables[2:7]
        a_values = np.deg2rad(variables[7:])

        wires = []
        for i, ((xc, zc), (start_angle, end_angle), ri) in enumerate(
            zip(*_get_centres(a_values, r_values, x1, z1), strict=True), start=1
        ):
            arc = make_circle(
                ri,
                center=(xc, 0, zc),
                start_angle=end_angle,
                end_angle=start_angle,
                axis=(0, -1, 0),
                label=f"arc_{i}",
            )

            wires.append(arc)

        if not np.isclose(z1, zc):
            straight_segment = wire_closure(
                BluemiraWire(wires), label="straight_segment"
            )
            wires.append(straight_segment)

        return BluemiraWire(wires, label=label)

    def _label_function(self, ax: plt.Axes, shape: BluemiraWire):
        """
        Adds labels to parameterisation plots

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        shape:
            parameterisation wire

        """
        _offset_x, _offset_z = super()._label_function(ax, shape)
        variables = self.variables.values
        centres, angles, radii = _get_centres(
            np.deg2rad(variables[7:]), variables[2:7], *variables[:2]
        )

        for r_no, (centre, s_f_angles, radius) in enumerate(
            zip(centres, angles, radii, strict=True), start=1
        ):
            centre_angle = min(s_f_angles) + 0.5 * np.ptp(s_f_angles)
            self._annotator(
                ax,
                f"r{r_no}",
                centre,
                _get_rotated_point(centre, radius, centre_angle),
                _get_rotated_point(centre, 0.5 * radius, centre_angle),
            )
            self._angle_annotator(
                ax, f"a{r_no}", radius, centre, s_f_angles, centre_angle
            )


@dataclass
class PolySplineOptVariables(OptVariablesFrame):
    x1: OptVariable = ov(
        "x1",
        4.3,
        lower_bound=4,
        upper_bound=5,
        description="Inner limb radius",
    )
    x2: OptVariable = ov(
        "x2",
        16.56,
        lower_bound=5,
        upper_bound=25,
        description="Outer limb radius",
    )
    z2: OptVariable = ov(
        "z2",
        0.03,
        lower_bound=-2,
        upper_bound=2,
        description="Outer note vertical shift",
    )
    height: OptVariable = ov(
        "height",
        15.5,
        lower_bound=10,
        upper_bound=50,
        description="Full height",
    )
    top: OptVariable = ov(
        "top",
        0.52,
        lower_bound=0.2,
        upper_bound=1,
        description="Horizontal shift",
    )
    upper: OptVariable = ov(
        "upper",
        0.67,
        lower_bound=0.2,
        upper_bound=1,
        description="Vertical shift",
    )
    dz: OptVariable = ov(
        "dz",
        -0.6,
        lower_bound=-5,
        upper_bound=5,
        description="Vertical offset",
    )
    flat: OptVariable = ov(
        "flat",
        0,
        lower_bound=0,
        upper_bound=1,
        description="Fraction of straight outboard leg",
    )
    tilt: OptVariable = ov(
        "tilt",
        4,
        lower_bound=-45,
        upper_bound=45,
        description="Outboard angle [degrees]",
    )
    bottom: OptVariable = ov(
        "bottom",
        0.4,
        lower_bound=0,
        upper_bound=1,
        description="Lower horizontal shift",
    )
    lower: OptVariable = ov(
        "lower",
        0.67,
        lower_bound=0.2,
        upper_bound=1,
        description="Lower vertical shift",
    )
    l0s: OptVariable = ov(
        "l0s",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable first segment start",
    )
    l1s: OptVariable = ov(
        "l1s",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable second segment start",
    )
    l2s: OptVariable = ov(
        "l2s",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable third segment start",
    )
    l3s: OptVariable = ov(
        "l3s",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable fourth segment start",
    )
    l0e: OptVariable = ov(
        "l0e",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable first segment end",
    )
    l1e: OptVariable = ov(
        "l1e",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable second segment end",
    )
    l2e: OptVariable = ov(
        "l2e",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable third segment end",
    )
    l3e: OptVariable = ov(
        "l3e",
        0.8,
        lower_bound=0.1,
        upper_bound=1.9,
        description="Tension variable fourth segment end",
    )


class PolySpline(GeometryParameterisation[PolySplineOptVariables]):
    """
    Simon McIntosh's Poly-Bézier-spline geometry parameterisation (19 variables).

    Parameters
    ----------
    var_dict:
        Dictionary with which to update the default values of the parameterisation.

    Notes
    -----
    .. plot::

        from bluemira.geometry.parameterisations import PolySpline
        PolySpline().plot(labels=True)

    The dictionary keys in var_dict are:

    x1: float
        Radial position of inner limb [m]
    x2: float
        Radial position of outer limb [m]
    z2: float
        Outer note vertical shift [m]
    height: float
        Full height [m]
    top: float
        Horizontal shift [m]
    upper: float
        Vertical shift [m]
    dz: float
        Vertical offset [m]
    flat: float
        Fraction of straight outboard leg []
    tilt: float
        Outboard angle [degrees]
    bottom: float
        Lower horizontal shift [m]
    lower: float
        Lower vertical shift [m]
    l0s - l3s: float
        Tension variable segment start
    l0e - l3e: float
        Tension variable segment end

    """

    __slots__ = ()

    def __init__(self, var_dict: VarDictT | None = None):
        variables = PolySplineOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Make a CAD representation of the poly spline.

        Parameters
        ----------
        label:
            Label to give the wire

        Returns
        -------
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
        height *= 0.5
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
        for i, j in zip([0, 1, 2, 3], [0, 1, 3, 4], strict=False):
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
    def _make_control_points(
        p0: list[float],
        p3: list[float],
        theta0: list[float],
        theta3: list[float],
        l_start: float,
        l_end: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Make 2 Bézier spline control points between two vertices.

        Returns
        -------
        :
            Two Bézier spline control points.
        """
        dl = np.sqrt(np.sum((np.array(p3) - np.array(p0)) ** 2))

        p1, p2 = np.zeros(3), np.zeros(3)
        for point, control_point, angle, tension in zip(
            [p0, p3], [p1, p2], [theta0, theta3], [l_start, l_end], strict=False
        ):
            d_tension = 0.5 * dl * tension
            control_point[0] = point[0] + d_tension * np.cos(angle)
            control_point[2] = point[2] + d_tension * np.sin(angle)

        return p1, p2

    def _label_function(self, ax: plt.Axes, shape: BluemiraWire):
        """
        Adds labels to parameterisation plots

        Parameters
        ----------
        ax:
            Matplotlib axis instance
        shape:
            parameterisation wire

        """
        # TODO @je-cook: add labels for height top upper dz flat tilt bottom lower
        # l0s - l3s l0e - l3e
        # 3587
        _offset_x, _offset_z = super()._label_function(ax, shape)


class PictureFrameTools:
    """
    Tools Class containing methods to produce various PictureFrame variant limbs.

    """

    @staticmethod
    def _make_domed_leg(
        x_out: float,
        x_curve_start: float,
        x_mid: float,
        z_top: float,
        z_mid: float,
        ri: float,
        axis: Iterable[float] = (0, -1, 0),
        *,
        flip: bool = False,
    ) -> BluemiraWire:
        """
        Makes smooth dome for CP coils. This includes a initial straight section
        and a main curved dome section, with a transitioning 'joint' between them,
        producing smooth tangent curves.

        Parameters
        ----------
        x_out:
            Radial position of outer edge of limb [m]
        x_curve_start:
            Radial position of straight-curve transition of limb [m]
        x_mid:
            Radial position of inner edge of  upper/lower limb [m]
        z_top:
            Vertical position of top of limb dome [m]
        z_mid:
            Vertical position of flat section [m]
        ri:
            Radius of inner corner transition. Nominally 0 [m]
        axis:
            [x,y,z] vector normal to plane of parameterisation
        flip:
            True if limb is lower limb of section, False if upper

        Returns
        -------
        CAD Wire of the geometry
        """
        # Define the basic main curve (with no joint or transitions curves)
        alpha = np.arctan(0.5 * (x_out - x_curve_start) / abs(z_top - z_mid))
        theta_leg_basic = 2 * (np.pi - 2 * alpha)
        r_leg = 0.5 * (x_out - x_curve_start) / np.sin(theta_leg_basic * 0.5)

        # Transitioning Curves
        sin_a = np.sin(theta_leg_basic * 0.5)
        cos_a = np.cos(theta_leg_basic * 0.5)

        # Joint Curve
        r_j = min(x_curve_start - x_mid, 0.8)
        theta_j = np.arccos((r_leg * cos_a + r_j) / (r_leg + r_j))
        deg_theta_j = np.rad2deg(theta_j)

        # Corner Transitioning Curve
        theta_trans = np.arccos((r_j - r_leg * sin_a) / (r_j - r_leg))
        deg_theta_trans = np.rad2deg(theta_trans)

        # Main leg curve angle
        leg_angle = 90 + deg_theta_j

        # Labels
        if flip:
            label = "bottom"
            z_top_r_leg = z_top + r_leg
            z_mid_r_j = z_mid - r_j
            z_trans_diff = -(r_leg - r_j)
            z_corner = z_mid + ri
            corner_angle_s = 90
            corner_angle_e = 180
            joint_angle_s = 90 - deg_theta_j
            joint_angle_e = 90
            leg_angle_s = tc_angle_e = deg_theta_trans
            leg_angle_e = leg_angle
            tc_angle_s = 0
            ind = slice(None, None, -1)
        else:
            label = "top"
            z_top_r_leg = z_top - r_leg
            z_mid_r_j = z_mid + r_j
            z_trans_diff = r_leg - r_j
            z_corner = z_mid - ri
            corner_angle_s = 180
            corner_angle_e = 270
            joint_angle_s = -90
            joint_angle_e = deg_theta_j - 90
            leg_angle_s = -leg_angle
            leg_angle_e = tc_angle_s = -deg_theta_trans
            tc_angle_e = 0
            ind = slice(None)

        # Basic main curve centre
        leg_centre = (x_out - 0.5 * (x_out - x_curve_start), 0, z_top_r_leg)

        # Joint curve centre
        joint_curve_centre = (
            leg_centre[0] - (r_leg + r_j) * np.sin(theta_j),
            0,
            z_mid_r_j,
        )

        # Transition curve centre
        x_trans = leg_centre[0] + (r_leg - r_j) * np.cos(theta_trans)
        z_trans = leg_centre[2] + z_trans_diff * np.sin(theta_trans)

        # Inner Corner
        corner_in = make_circle(
            ri,
            (x_mid + ri, 0.0, z_corner),
            start_angle=corner_angle_s,
            end_angle=corner_angle_e,
            axis=(0, 1, 0),
            label=f"inner_{label}_corner",
        )

        # Build straight section of leg
        p1 = [x_mid + ri, 0, z_mid]
        p2 = [leg_centre[0] - (r_leg + r_j) * np.sin(theta_j), 0, z_mid]
        straight_section = make_polygon([p2, p1] if flip else [p1, p2])

        # Dome-inboard section transition curve
        joint_curve = make_circle(
            radius=r_j,
            center=joint_curve_centre,
            start_angle=joint_angle_s,
            end_angle=joint_angle_e,
            axis=axis,
            label=f"{label}_limb_joint",
        )

        # Main leg curve
        leg_curve = make_circle(
            radius=r_leg,
            center=leg_centre,
            start_angle=leg_angle_s,
            end_angle=leg_angle_e,
            axis=(0, 1, 0),
            label=f"{label}_limb_dome",
        )

        # Outboard corner transition curve
        transition_curve = make_circle(
            radius=r_j,
            center=(x_trans, 0, z_trans),
            start_angle=tc_angle_s,
            end_angle=tc_angle_e,
            axis=(0, 1, 0),
            label=f"{label}_limb_corner",
        )

        return BluemiraWire(
            [corner_in, straight_section, joint_curve, leg_curve, transition_curve][ind],
            label=f"{label}_limb",
        )

    @staticmethod
    def _make_flat_leg(
        x_mid: float,
        x_out: float,
        z: float,
        r_i: float,
        r_o: float,
        axis: Iterable[float] = (0, 1, 0),
        *,
        flip: bool = False,
    ) -> BluemiraWire:
        """
        Makes a flat leg (top/bottom limb) with the option of one end rounded.

        Parameters
        ----------
        x_mid:
            Radial position of inner edge of limb [m]
        x_out:
            Radial position of outer edge of limb [m]
        z:
            Vertical position of limb [m]
        r_i:
            Radius of inner corner [m]
        r_o:
            Radius of outer corner [m]
        axis:
            [x,y,z] vector normal to plane of parameterisation
        flip:
            True if limb is lower limb of section, False if upper

        Returns
        -------
        CAD Wire of the geometry
        """
        wires = []
        label = "bottom" if flip else "top"

        # Set corner radius centres
        c_i = (x_mid + r_i, 0.0, z + r_i if flip else z - r_i)
        c_o = (x_out - r_o, 0.0, z + r_o if flip else z - r_o)

        # Inner Corner
        if r_i != 0.0:
            wires.append(
                make_circle(
                    r_i,
                    c_i,
                    start_angle=90 if flip else 180,
                    end_angle=180 if flip else 270,
                    axis=axis,
                    label=f"inner_{label}_corner",
                )
            )
        # Straight Section
        p1 = [x_mid + r_i, 0.0, z]
        p2 = [x_out - r_o, 0.0, z]
        wires.append(make_polygon([p2, p1] if flip else [p1, p2], label=f"{label}_limb"))

        # Outer corner
        if r_o != 0.0:
            wires.append(
                make_circle(
                    r_o,
                    c_o,
                    start_angle=0 if flip else 270,
                    end_angle=90 if flip else 0,
                    axis=axis,
                    label=f"outer_{label}_corner",
                )
            )

        if flip:
            wires.reverse()

        return BluemiraWire(wires, label=f"{label}_limb")

    @staticmethod
    def _make_tapered_inner_leg(
        x_in: float,
        x_mid: float,
        z_bot: float,
        z_taper: float,
        z_top: float,
        r_min: float,
        axis: tuple[float, float, float] = (0, 1, 0),
    ) -> BluemiraWire:
        """
        Makes a tapered inboard leg using a circle arc taper, symmetric about the
        midplane with the tapering beginning at a certain height and reaching a
        maximum taper at the midplane.

        Parameters
        ----------
        x_in:
            Radial position of innermost point of limb [m]
        x_mid:
            Radial position of outer edge of limb [m]
        z_bot:
            Vertical position of bottom of limb [m]
        z_taper:
            Vertical position of start of tapering [m]
        z_top:
            Vertical position of top of limb [m]
        r_min:
            Minimum radius of curvature [m]
        axis:
            [x,y,z] vector normal to plane of parameterisation

        Returns
        -------
        CAD Wire of the geometry

        Raises
        ------
        GeometryParameterisationError
            Input parameters cannot create shape
        """
        # Pre-calculations necessary
        if not (z_bot < -z_taper < 0 < z_taper < z_top):
            raise GeometryParameterisationError(
                "the straight-curve transition point z_taper must lie between"
                "z_top and z_bot."
            )
        theta = 2 * np.arctan2(x_mid - x_in, z_taper)
        r_taper = z_taper / np.sin(theta) - r_min

        if r_taper < r_min or theta >= (np.pi / 2):
            raise GeometryParameterisationError(
                f"Cannot achieve radius of curvature <= {r_min=}"
                "as the taper (x_mid - x_in) is too deep for this given value of r_min."
            )

        theta_deg = np.rad2deg(theta)

        # bottom straight line
        p1 = [x_mid, 0, z_bot]
        p2 = [x_mid, 0, -z_taper]
        bot_straight = make_polygon([p1, p2], label="inner_limb_mid_down")

        # curve into taper
        bot_curve = make_circle(
            radius=r_min,
            center=(x_mid - r_min, 0, -z_taper),
            start_angle=0,
            end_angle=theta_deg,
            axis=(0, -1, 0),
            label="inner_limb_lower_curve",
        )

        taper = make_circle(
            radius=r_taper,
            center=(x_in + r_taper, 0, 0),
            start_angle=180.0 - theta_deg,
            end_angle=180.0 + theta_deg,
            axis=axis,
            label="inner_limb_taper_curve",
        )

        # curve out of taper
        top_curve = make_circle(
            radius=r_min,
            center=(x_mid - r_min, 0, z_taper),
            start_angle=360.0 - theta_deg,
            end_angle=0.0,
            axis=(0, -1, 0),
            label="inner_limb_upper_curve",
        )

        # top straight line.
        p6 = [x_mid, 0, z_taper]
        p7 = [x_mid, 0, z_top]
        top_straight = make_polygon([p6, p7], label="inner_limb_mid_up")

        return BluemiraWire(
            [bot_straight, bot_curve, taper, top_curve, top_straight], label="inner_limb"
        )

    def _connect_to_outer_limb(
        self, top, bottom, *, top_curve: bool = False, bot_curve: bool = False
    ) -> BluemiraWire:
        return self._outer_limb(
            top.discretise(100, byedges=True)[:, -1] if top_curve else top,
            bottom.discretise(100, byedges=True)[:, 0] if bot_curve else bottom,
        )

    def _connect_straight_to_inner_limb(
        self, top: npt.NDArray[np.float64], bottom: npt.NDArray[np.float64]
    ) -> BluemiraWire:
        return self._inner_limb(top, bottom)

    @staticmethod
    def _inner_limb(
        p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
    ) -> BluemiraWire:
        return make_polygon([p1, p2], label="inner_limb")

    @staticmethod
    def _outer_limb(
        p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
    ) -> BluemiraWire:
        return make_polygon([p1, p2], label="outer_limb")


class PFrameSection(Enum):
    """
    Picture Frame sections
    """

    CURVED = partial(PictureFrameTools._make_domed_leg)
    FLAT = partial(PictureFrameTools._make_flat_leg)
    TAPERED_INNER = partial(PictureFrameTools._make_tapered_inner_leg)

    def __call__(self, *args, **kwargs):
        """
        Call linked function on access

        Returns
        -------
        :
            Linked function.
        """
        return self.value(*args, **kwargs)


@dataclass
class PictureFrameOptVariables(OptVariablesFrame):
    x1: OptVariable = ov(
        "x1",
        0.4,
        lower_bound=0.3,
        upper_bound=0.5,
        description="Inner limb radius",
    )
    x2: OptVariable = ov(
        "x2",
        9.5,
        lower_bound=9.4,
        upper_bound=9.8,
        description="Outer limb radius",
    )
    z1: OptVariable = ov(
        "z1",
        9.5,
        lower_bound=8,
        upper_bound=10.5,
        description="Upper limb height",
    )
    z2: OptVariable = ov(
        "z2",
        -9.5,
        lower_bound=-10.5,
        upper_bound=-8,
        description="Lower limb height",
    )
    ri: OptVariable = ov(
        "ri",
        0.1,
        lower_bound=0,
        upper_bound=2,
        description="Inboard corner radius",
    )
    ro: OptVariable = ov(
        "ro",
        2,
        lower_bound=1,
        upper_bound=5,
        description="Outboard corner radius",
    )
    x3: OptVariable = ov(
        "x3",
        2.5,
        lower_bound=2.4,
        upper_bound=2.6,
        description="Curve start radius",
    )
    z1_peak: OptVariable = ov(
        "z1_peak",
        11,
        lower_bound=6,
        upper_bound=12,
        description="Upper limb curve height",
    )
    z2_peak: OptVariable = ov(
        "z2_peak",
        -11,
        lower_bound=-12,
        upper_bound=-6,
        description="Lower limb curve height",
    )
    x4: OptVariable = ov(
        "x4",
        1.1,
        lower_bound=1,
        upper_bound=1.3,
        description="Middle limb radius",
    )
    z3: OptVariable = ov(
        "z3",
        6.5,
        lower_bound=6,
        upper_bound=8,
        description="Taper angle stop height",
    )

    def configure(
        self,
        upper: str | PFrameSection,
        lower: str | PFrameSection,
        inner: str | PFrameSection | None,
    ):
        """Fix variables based on the upper, lower and inner limbs."""
        if upper is PFrameSection.CURVED and lower is PFrameSection.CURVED:
            self.ro.fixed = True
        elif upper is PFrameSection.FLAT and lower is PFrameSection.FLAT:
            self.z1_peak.fixed = True
            self.z2_peak.fixed = True
            self.x3.fixed = True
        if inner is not PFrameSection.TAPERED_INNER:
            self.x4.fixed = True
            self.z3.fixed = True


class PictureFrame(
    GeometryParameterisation[PictureFrameOptVariables], PictureFrameTools
):
    """
    Picture-frame geometry parameterisation.

    Parameters
    ----------
    var_dict:
        Dictionary with which to update the default values of the parameterisation.

    Notes
    -----
    .. plot::

        from bluemira.geometry.parameterisations import PictureFrame
        PictureFrame(
                     inner="TAPERED_INNER",
                     upper="FLAT",
                     lower="CURVED",
                     var_dict={'ri': {'value': 1}}
        ).plot(labels=True)

    The base dictionary keys in var_dict are:

    x1: float
        Radial position of inner limb [m]
    x2: float
        Radial position of outer limb [m]
    z1: float
        Vertical position of top limb [m]
    z2: float
        Vertical position of top limb [m]
    ri: float
        Radius of inner corners [m]
    ro: float
        Radius of outer corners [m]

    For curved pictures frames 'ro' is ignored on curved sections but there
    are additional keys:

    z1_peak: float
        Vertical position of top of limb dome [m]
    z2_peak: float
        Vertical position of top of limb dome [m]
    x3: float
        The radius to start the dome curve [m]

    For tapered inner leg the additional keys are:

    x4: float
        Radial position of outer limb [m]
    z3: float
        Vertical position of top of tapered section [m]

    """

    __slots__ = tuple(
        f"{leg}{var}"
        for leg in ["inner", "upper", "lower", "outer"]
        for var in ["", "_vars"]
    )

    def __init__(
        self,
        var_dict: VarDictT | None = None,
        *,
        upper: str | PFrameSection = PFrameSection.FLAT,
        lower: str | PFrameSection = PFrameSection.FLAT,
        inner: str | PFrameSection | None = None,
    ):
        self.upper = upper if isinstance(upper, PFrameSection) else PFrameSection[upper]
        self.lower = lower if isinstance(lower, PFrameSection) else PFrameSection[lower]

        if isinstance(inner, str):
            inner = PFrameSection[inner]
        self.inner = inner

        variables = PictureFrameOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        variables.configure(self.upper, self.lower, self.inner)
        super().__init__(variables)

    def __deepcopy__(self, memo) -> PictureFrame:
        """Picture Frame deepcopy

        Returns
        -------
        :
            Deepcopy of Picture Frame.
        """
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k in (*self.__slots__, *super().__slots__):
            with suppress(AttributeError):
                v = getattr(self, k)
                setattr(
                    result,
                    k,
                    v if isinstance(v, PFrameSection) else copy.deepcopy(v, memo),
                )
        return result

    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Make a CAD representation of the picture frame.

        Parameters
        ----------
        label:
            Label to give the wire

        Returns
        -------
        CAD Wire of the Picture Frame geometry
        """
        inb_leg = self._make_inb_leg()
        top_leg = self._make_upper_lower_leg(make_upper_section=True, flip=False)
        bot_leg = self._make_upper_lower_leg(make_upper_section=False, flip=True)
        out_leg = self._make_out_leg(top_leg, bot_leg)

        return BluemiraWire([inb_leg, top_leg, out_leg, bot_leg], label=label)

    def _make_inb_leg(self) -> BluemiraWire:
        v = self.variables
        if isinstance(self.inner, PFrameSection):
            if self.inner is not PFrameSection.TAPERED_INNER:
                raise ValueError(f"The inner leg cannot be {self.inner}")
            return self.inner(
                v.x1.value,
                v.x4.value,
                v.z2 + v.ri,
                v.z3.value,
                v.z1 - v.ri,
                v.ri.value,
            )
        if self.inner is None:
            return self._connect_straight_to_inner_limb(
                [v.x1.value, 0, v.z2 + v.ri],
                [v.x1.value, 0, v.z1 - v.ri],
            )
        return None

    def _make_upper_lower_leg(
        self, *, make_upper_section: bool, flip: bool
    ) -> PFrameSection:
        v = self.variables
        section_func: PFrameSection = self.upper if make_upper_section else self.lower
        if section_func == PFrameSection.CURVED:
            return section_func(
                v.x2.value,
                v.x3.value,
                v.x4.value if self.inner is PFrameSection.TAPERED_INNER else v.x1.value,
                v.z1_peak.value if make_upper_section else v.z2_peak.value,
                v.z1.value if make_upper_section else v.z2.value,
                v.ri.value,
                flip=flip,
            )
        if section_func == PFrameSection.FLAT:
            return section_func(
                v.x4.value if self.inner is PFrameSection.TAPERED_INNER else v.x1.value,
                v.x2.value,
                v.z1.value if make_upper_section else v.z2.value,
                v.ri.value,
                v.ro.value,
                flip=flip,
            )
        raise ValueError(f"The leg cannot be {section_func}")

    def _make_out_leg(
        self, top_leg: PFrameSection, bot_leg: PFrameSection
    ) -> BluemiraWire:
        v = self.variables
        return self._connect_to_outer_limb(
            (
                top_leg
                if self.upper is PFrameSection.CURVED
                else [v.x2.value, 0, v.z1 - v.ro]
            ),
            (
                bot_leg
                if self.lower is PFrameSection.CURVED
                else [v.x2.value, 0, v.z2 + v.ro]
            ),
            top_curve=self.upper is PFrameSection.CURVED,
            bot_curve=self.lower is PFrameSection.CURVED,
        )

    def _label_function(self, ax, shape):
        super()._label_function(ax, shape)
        ro = self.variables.ro
        ri = self.variables.ri
        z = self.variables.z1
        x_in = (
            self.variables.x4
            if self.inner is PFrameSection.TAPERED_INNER
            else self.variables.x1
        )

        x_out = self.variables.x2
        _r1 = ri * (1 - np.sqrt(0.5))
        _r2 = ro * (1 - np.sqrt(0.5))
        self._annotator(
            ax,
            "ri",
            (x_in + ri, z - ri),
            (x_in + _r1, z - _r1),
            ((x_in + ri) * 0.8, z - 6 * _r1),
        )
        self._annotator(
            ax,
            "ro",
            (x_out - ro, z - ro),
            (x_out - _r2, z - _r2),
            ((x_out + ro) * 0.6, z - 3 * _r2),
        )

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.1)
