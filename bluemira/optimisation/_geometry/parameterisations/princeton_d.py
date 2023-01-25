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
import numpy as np

from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.optimisation._geometry.parameterisations import tools


def f_ineq_constraint(geom: GeometryParameterisation) -> np.ndarray:
    """Inequality constraint for PrincetonD."""
    free_vars = geom.variables.get_normalised_values()
    x1, x2, _ = tools.process_x_norm_fixed(geom.variables, free_vars)
    return np.array([x1 - x2])


def df_ineq_constraint(geom: GeometryParameterisation) -> np.ndarray:
    """Inequality constraint gradient for PrincetonD."""
    opt_vars = geom.variables
    free_vars = opt_vars.get_normalised_values()
    grad = np.zeros((1, len(free_vars)))
    if not geom.variables["x1"].fixed:
        grad[0, tools.get_x_norm_index(opt_vars, "x1")] = 1
    if not geom.variables["x2"].fixed:
        grad[0, tools.get_x_norm_index(opt_vars, "x2")] = -1
    return grad
