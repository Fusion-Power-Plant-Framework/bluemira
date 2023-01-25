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

from bluemira.geometry.parameterisations import TripleArc
from bluemira.optimisation._geometry.parameterisations import tools


def f_ineq_constraint(geom: TripleArc) -> np.ndarray:
    """
    Inequality constraint for TripleArc.

    Constrain such that a1 + a2 is less than or equal to 180 degrees.
    """
    norm_vals = geom.variables.get_normalised_values()
    x_actual = tools.process_x_norm_fixed(geom.variables, norm_vals)
    _, _, _, _, _, a1, a2 = x_actual
    return np.array([a1 + a2 - 180])


def df_ineq_constraint(geom: TripleArc) -> np.ndarray:
    """Inequality constraint gradient for TripleArc."""
    free_vars = geom.variables.get_normalised_values()
    g = np.zeros((len(free_vars), 1))
    if not geom.variables["a1"].fixed:
        idx_a1 = tools.get_x_norm_index(geom.variables, "a1")
        g[idx_a1] = 1
    if not geom.variables["a2"].fixed:
        idx_a2 = tools.get_x_norm_index(geom.variables, "a2")
        g[idx_a2] = 1
    return g[0, :]


def tol() -> np.ndarray:
    """The constraint tolerance for TripleArc."""
    return np.array([np.finfo(float).eps])
