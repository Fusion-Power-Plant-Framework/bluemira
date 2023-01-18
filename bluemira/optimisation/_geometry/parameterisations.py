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

from bluemira.geometry.parameterisations import GeometryParameterisation, PrincetonD


def _get_x_norm_index(variables, name: str):
    """
    Get the index of a variable name in the modified-length x_norm vector

    Parameters
    ----------
    name: str
        Variable name for which to get the index

    Returns
    -------
    idx_x_norm: int
        Index of the variable name in the modified-length x_norm vector
    """
    fixed_idx = variables._fixed_variable_indices
    idx_actual = variables.names.index(name)

    if not fixed_idx:
        return idx_actual

    count = 0
    for idx_fx in fixed_idx:
        if idx_actual > idx_fx:
            count += 1
    return idx_actual - count


def _process_x_norm_fixed(variables, x_norm):
    """
    Utility for processing a set of free, normalised variables, and folding the fixed
    un-normalised variables back into a single list of all actual values.

    Parameters
    ----------
    x_norm: np.ndarray
        Normalised vector of variable values

    Returns
    -------
    x_actual: list
        List of ordered actual (un-normalised) values
    """
    fixed_idx = variables._fixed_variable_indices

    # Note that we are dealing with normalised values when coming from the optimiser
    x_actual = list(variables.get_values_from_norm(x_norm))

    if fixed_idx:
        x_fixed = variables.values
        for i in fixed_idx:
            x_actual.insert(i, x_fixed[i])
    return x_actual


def f_ineq_constraint_princeton_d(geom: GeometryParameterisation) -> np.ndarray:
    """Inequality constraint for PrincetonD."""
    free_vars = geom.variables.get_normalised_values()
    x1, x2, _ = _process_x_norm_fixed(geom.variables, free_vars)
    return np.array([x1 - x2])


def df_ineq_constraint_princeton_d(geom: GeometryParameterisation) -> np.ndarray:
    """Inequality constraint gradient for PrincetonD."""
    opt_vars = geom.variables
    free_vars = opt_vars.get_normalised_values()
    grad = np.zeros((1, len(free_vars)))
    if not geom.variables["x1"].fixed:
        grad[0, _get_x_norm_index(opt_vars, "x1")] = 1
    if not geom.variables["x2"].fixed:
        grad[0, _get_x_norm_index(opt_vars, "x2")] = -1
    return grad


INEQ_CONSTRAINT_REGISTRY = {
    PrincetonD: [
        {
            "f_constraint": f_ineq_constraint_princeton_d,
            "df_constraint": df_ineq_constraint_princeton_d,
            "tolerance": np.array([np.finfo(float).eps]),
        }
    ]
}
