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

"""
Matrix manipulation methods for finite element solver
"""
from copy import deepcopy
from typing import List, Tuple

import numpy as np


def k_condensation(k: np.ndarray, releases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Section 6.5 in J.S. Prz..
    """
    # Initialise matrix partitions
    n = np.count_nonzero(releases)
    kxy = np.zeros((12 - n) * n)
    kyy = np.zeros(n * n)

    count_xy = 0
    count_yy = 0
    for i, free_1 in enumerate(releases):
        for j, free_2 in enumerate(releases):
            if not free_1 and free_2:
                kxy[count_xy] = k[i, j]
                count_xy += 1
            elif free_1 and free_2:
                kyy[count_yy] = k[i, j]
                count_yy += 1
    kxy = kxy.reshape((12 - n, n))
    kyy = kyy.reshape((n, n))
    return kxy, kyy


def cyclic_decomposition(
    k: np.ndarray, p: np.ndarray, l_nodes: List[int], r_nodes: List[int]
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Perform a cyclic symmetry decomposition of the stiffness matrix and load
    vector.

    Parameters
    ----------
    k:
        The stiffness matrix to decompose
    p:
        The load vector to decompose
    l_nodes:
        The left nodes indices where a symmetry condition was specified
    r_nodes:
        The right node indices where a symmetry condition was specified

    Returns
    -------
    k_cyc:
        The partitioned and ordered stiffness matrix
    p_cyc:
        The partitioned and ordered load vector
    selections:
        The list of left, right, and interior node selection arrays
    """

    def selection_vec(node_ids):
        ranges = [list(range(6 * i, 6 * i + 6)) for i in node_ids]
        return np.array(ranges).flatten()

    k = deepcopy(k)
    p = deepcopy(p)

    size = k.shape[0]
    left = selection_vec(l_nodes)
    right = selection_vec(r_nodes)
    interior = np.arange(0, size)
    drop = np.append(left, right)
    drop.sort()
    interior = np.delete(interior, drop)

    k_rr = k[np.ix_(right, right)]
    k_rl = k[np.ix_(right, left)]
    k_ri = k[np.ix_(right, interior)]
    k_lr = k[np.ix_(left, right)]
    k_ll = k[np.ix_(left, left)]
    k_li = k[np.ix_(left, interior)]
    k_ir = k[np.ix_(interior, right)]
    k_il = k[np.ix_(interior, left)]
    k_ii = k[np.ix_(interior, interior)]

    p_r = p[right]
    p_l = p[left]
    p_i = p[interior]

    k_cyc = np.array(
        [[k_rr, k_rl, k_ri], [k_lr, k_ll, k_li], [k_ir, k_il, k_ii]], dtype=object
    )
    p_cyc = np.array([p_r, p_l, p_i], dtype=object)
    selections = [left, right, interior]

    return k_cyc, p_cyc, selections
