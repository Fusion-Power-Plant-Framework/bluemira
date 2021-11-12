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
Optimisation utilities
"""
import numpy as np
from scipy.optimize._constraints import old_constraint_to_new
from BLUEPRINT.geometry.constants import VERY_BIG
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import distance_between_points, normal, get_intersect


def convert_scipy_constraints(list_of_con_dicts):
    """
    Converts a list of old-style scipy constraint dicts into NonLinearConstraints

    Parameters
    ----------
    list_of_con_dicts

    Returns
    -------
    constraints: List[NonLinearConstraint]
    """
    new_constraints = []
    for i, con in enumerate(list_of_con_dicts):
        new = old_constraint_to_new(i, con)
        new_constraints.append(new)
    return new_constraints


def geometric_constraint(bound, loop, con_type="external"):
    """
    Geometric constraint function in 2-D.

    Parameters
    ----------
    bound: Loop
        The bounding loop constraint
    loop: Loop
        The shape being optimised
    con_type: str
        The type of constraint to apply ["internal", "external"]

    Returns
    -------
    constraint: np.array
        The geometric constraint array
    """

    def get_min_distance(point, vector_line):
        x_inter, z_inter = get_intersect(loop, vector_line)
        distances = []
        for xi, zi in zip(x_inter, z_inter):
            distances.append(distance_between_points(point, [xi, zi]))

        return np.min(distances)

    normals = normal(*bound.d2)
    constraint = np.zeros(len(bound))
    for i, b_point in enumerate(bound.d2.T):
        n_hat = np.array([normals[0][i], normals[1][i]])

        n_hat = VERY_BIG * n_hat
        x_con, z_con = b_point

        line = Loop(
            x=[x_con - n_hat[0], x_con + n_hat[0]],
            z=[z_con - n_hat[1], z_con + n_hat[1]],
        )
        distance = get_min_distance(b_point, line)
        constraint[i] = distance

    return constraint


def dot_difference(bound, loop, side="internal"):
    """
    Utility function for geometric constraints.
    """
    xloop, zloop = loop.d2
    switch = 1 if side == "internal" else -1
    n_xloop, n_zloop = normal(xloop, zloop)
    x_bound, z_bound = bound.d2
    dotp = np.zeros(len(x_bound))
    for j, (x, z) in enumerate(zip(x_bound, z_bound)):
        i = np.argmin((x - xloop) ** 2 + (z - zloop) ** 2)
        dl = [xloop[i] - x, zloop[i] - z]
        dn = [n_xloop[i], n_zloop[i]]
        dotp[j] = switch * np.dot(dl, dn)
    return dotp
