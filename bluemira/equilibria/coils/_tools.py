# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tools for Coilgroups
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.magnetostatics.greens import circular_coil_inductance_elliptic, greens_psi

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet


def make_mutual_inductance_matrix(coilset: CoilSet) -> np.ndarray:
    """
    Calculate the mutual inductance matrix of a coilset.

    Parameters
    ----------
    coilset:
        Coilset for which to calculate the mutual inductance matrix

    Returns
    -------
    :
        The symmetric mutual inductance matrix [H]

    Notes
    -----
    Single-filament coil formulation; serves as a useful approximation.

    - **Off-diagonal terms** (:math:`i \\neq j`):

        .. math::
            M_{ij} = n_i n_j G(x_i, z_i, x_j, z_j)

        where :math:`G` is the Green's function for mutual inductance.

    - **Diagonal terms** (:math:`i = j`):

        .. math::
            M_{ii} = n_i^2 L_i

        with :math:`L_i` as the self-inductance using elliptic integrals.

    """
    n_coils = coilset.n_coils()
    M = np.zeros((n_coils, n_coils))  # noqa: N806
    xcoord = coilset.x
    zcoord = coilset.z
    dx = coilset.dx
    dz = coilset.dz
    n_turns = coilset.n_turns

    itri, jtri = np.triu_indices(n_coils, k=1)

    M[itri, jtri] = (
        n_turns[itri]
        * n_turns[jtri]
        * greens_psi(xcoord[itri], zcoord[itri], xcoord[jtri], zcoord[jtri])
    )
    M[jtri, itri] = M[itri, jtri]

    radius = np.hypot(dx, dz)
    for i in range(n_coils):
        M[i, i] = n_turns[i] ** 2 * circular_coil_inductance_elliptic(
            xcoord[i], radius[i]
        )

    return M


def _get_symmetric_coils(coilset: CoilSet) -> list[list]:
    """
    Coilset symmetry utility

    Returns
    -------
    :
        Symmetric coilset
    """
    x, z, dx, dz, currents = coilset.to_group_vecs()
    coil_matrix = np.array([x, np.abs(z), dx, dz, currents]).T

    sym_stack = [[coil_matrix[0], 1]]
    for i in range(1, len(x)):
        coil = coil_matrix[i]

        for j, sym_coil in enumerate(sym_stack):
            if np.allclose(coil, sym_coil[0]):
                sym_stack[j][1] += 1
                break

        else:
            sym_stack.append([coil, 1])

    return sym_stack


def check_coilset_symmetric(coilset: CoilSet) -> bool:
    """
    Check whether or not a CoilSet is purely symmetric about z=0.

    Parameters
    ----------
    coilset:
        CoilSet to check for symmetry

    Returns
    -------
    Whether or not the CoilSet is symmetric about z=0
    """
    sym_stack = _get_symmetric_coils(coilset)
    for coil, count in sym_stack:
        if count != 2 and not np.isclose(coil[1], 0.0):  # noqa: PLR2004
            # therefore z = 0
            return False
    return True


def get_max_current(
    dx: float | np.ndarray, dz: float | np.ndarray, j_max: float | np.ndarray
) -> np.float64 | np.ndarray:
    """
    Get the maximum current in a rectangular coil cross-sectional area

    Parameters
    ----------
    dx:
        Coil half-width [m]
    dz:
        Coil half-height [m]
    j_max:
        Coil current density [A/m^2]

    Returns
    -------
    Maximum current [A]

    Notes
    -----
    .. math::
        I_{\\text{max}} = j_{\\text{max}} \\cdot (4 \\cdot dx \\cdot dz)

    """
    return np.abs(j_max * (4 * dx * dz))


def rename_coilset(coilset: CoilSet):
    """
    Rename the coils.

    Returns
    -------
    Coilset
        The coilset containing the renamed coils
    """
    coil_types = ["PF", "CS", "DUM"]
    coil_numbers = [coilset.get_coiltype(ct) for ct in coil_types]
    for n, ct in zip(coil_numbers, coil_types, strict=False):
        if n is not None:
            for i, coil_name in enumerate(coilset.get_coiltype(ct).name):
                coil_num = i + 1
                coilset[coil_name].name = ct + "_" + str(coil_num)
    coilset.control = coilset.name
    return coilset
