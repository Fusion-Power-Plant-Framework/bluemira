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
"""Designer, builder, and tools for wall panelling."""

from itertools import count
from typing import Dict, Optional, TypedDict

import numpy as np
from typing_extensions import NotRequired

from bluemira.geometry.error import GeometryError
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables

DEG_TO_RAD = np.pi / 180


class VarDictT(TypedDict):
    """Typing for a geometry parameterisation's 'var_dict'."""

    value: float
    lower_bound: NotRequired[float]
    upper_bound: NotRequired[float]
    descr: NotRequired[str]


class WrappedString(GeometryParameterisation):
    """
    Parameterisation of a 'string' wrapped around fixed points.

    Given a boundary, this parameterisation defines a wire that, given
    some maximum angle and minimum and maximum segment length, selects
    a set of pivot points on the boundary and draws straight lines
    between those points. You might imagine a set of pins with a string
    wrapped tightly around them.

    Parameters
    ----------
    boundary
        A wire defining the line the pivot points must lay on.
    var_dict
        Dictionary defining all, or some of, the shape's parameters.
        Allowed keys are:

        * max_angle: the maximum angle between neighbouring pivot points.
        * min_segment_len: the minimum distance between pivot points.
        * max_segment_len: the maximum distance between pivot points.

        Each value must be a dictionary with optional keys

        * value: float
        * lower_bound: float
        * upper_bound: float
        * fixed: bool
    n_boundary_points
        The number of points to use when discretizing the boundary.
    """

    def __init__(
        self,
        boundary: BluemiraWire,
        var_dict: Optional[Dict[str, VarDictT]] = None,
        n_boundary_points: int = 100,
    ):
        self.boundary = boundary
        self.n_boundary_points = n_boundary_points
        variables = OptVariables(
            [
                BoundedVariable(
                    "max_angle",
                    10,
                    lower_bound=0,
                    upper_bound=80,
                    descr="Maximum turning angle [degree]",
                ),
                BoundedVariable(
                    "min_segment_len",
                    0.5,
                    lower_bound=0,
                    upper_bound=10,
                    descr="Minimum segment length [m]",
                ),
                BoundedVariable(
                    "max_segment_len",
                    2.5,
                    lower_bound=0,
                    upper_bound=10,
                    descr="Minimum segment length [m]",
                ),
            ]
        )
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)
        self.n_ineq_constraints = 1

    def create_shape(self, label="") -> BluemiraWire:
        """Build the wrapped string."""
        dx_min = self.variables["min_segment_len"].value
        dx_max = self.variables["max_segment_len"].value
        if dx_min > dx_max:
            raise GeometryError(
                "The minimum_segment_len must be less than or equal to "
                f"the max_segment_length: {dx_min} > {dx_max}."
            )
        boundary_points = self.boundary.discretize(self.n_boundary_points)
        points = self._make_string(
            points=boundary_points.T,
            max_angle=self.variables["max_angle"].value,
            dx_min=self.variables["min_segment_len"].value,
            dx_max=self.variables["max_segment_len"].value,
        )
        return make_polygon(points, closed=True, label=label)

    def shape_ineq_constraints(
        self, constraint: np.ndarray, x: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """
        The inequality constraints for this parameterisation.

        The only constraint is that the minimum segment length must not
        be greater than the maximum segment length.
        """
        x_actual = self._process_x_norm_fixed(x)
        _, min_segment_len, max_segment_len = x_actual

        idx_min_len = self._get_x_norm_index("min_segment_len")
        idx_max_len = self._get_x_norm_index("max_segment_len")

        constraint[0] = min_segment_len - max_segment_len
        if grad.size > 0:
            if not self.variables["min_segment_len"].fixed:
                grad[0, idx_min_len] = 1
            if not self.variables["max_segment_len"].fixed:
                grad[0, idx_max_len] = -1
        return constraint

    @staticmethod
    def _make_string(
        points: np.ndarray, max_angle: float, dx_min: float, dx_max: float
    ) -> np.ndarray:
        """
        Derive the coordinates of the pivot points of the string.

        Parameters
        ----------
        points
            The coordinates (in 3D) of the pivot points. Must have shape
            (3, N).
        max_angle
            The maximum angle between neighbouring pivot points.
        dx_min
            The minimum distance between pivot points.
        dx_max
            The maximum distance between pivot points.

        Returns
        -------
        new_points
            The pivot points' coordinates. Has shape (3, N).
        """
        tangent_vec = points[1:] - points[:-1]
        tangent_vec_norm = np.linalg.norm(tangent_vec, axis=1)
        # Protect against dividing by zero
        tangent_vec_norm[tangent_vec_norm == 0] = 1e-32
        average_step_length = np.median(tangent_vec_norm)
        tangent_vec /= tangent_vec_norm.reshape(-1, 1) * np.ones(
            (1, np.shape(tangent_vec)[1])
        )

        new_points = np.zeros_like(points)
        index = np.zeros(points.shape[0], dtype=int)
        delta_x = np.zeros_like(points)
        delta_turn = np.zeros_like(points)

        new_points[0] = points[0]
        to, po = tangent_vec[0], points[0]

        k = count(1)
        for i, (p, t) in enumerate(zip(points[1:], tangent_vec)):
            c = np.cross(to, t)
            c_mag = np.linalg.norm(c)
            dx = np.linalg.norm(p - po)  # segment length
            if (
                c_mag > np.sin(max_angle * DEG_TO_RAD) and dx > dx_min
            ) or dx + average_step_length > dx_max:
                j = next(k)
                new_points[j] = points[i]  # pivot point
                index[j] = i + 1  # pivot index
                delta_x[j - 1] = dx  # panel length
                delta_turn[j - 1] = np.arcsin(c_mag) / DEG_TO_RAD
                to, po = t, p  # update
        if dx > dx_min:
            j = next(k)
            delta_x[j - 1] = dx  # last segment length
        else:
            delta_x[j - 1] += dx  # last segment length
        new_points[j] = p  # replace/append last point
        index[j] = i + 1  # replace/append last point index
        new_points = new_points[: j + 1]  # trim
        index = index[: j + 1]  # trim
        delta_x = delta_x[:j]  # trim
        return new_points
