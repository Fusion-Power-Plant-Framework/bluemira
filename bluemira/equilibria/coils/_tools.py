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

from bluemira.magnetostatics.greens import (
    circular_coil_inductance_elliptic,
    greens_psi,
    square_coil_inductance_kirchhoff,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.equilibria.coils import CoilSet


def make_mutual_inductance_matrix(
    coilset: CoilSet,
    *,
    square_coil: bool = False,
) -> np.ndarray:
    """
    Calculate the mutual inductance matrix of a coilset.

    Parameters
    ----------
    coilset:
        Coilset for which to calculate the mutual inductance matrix
    square_coil:
        Whether or not to use a square coil approximation for the
        self-inductance diagonal terms. Defaults to a elliptical
        integral of a circular cross-section coil.

    Returns
    -------
    :
        The symmetric mutual inductance matrix [H]

    Notes
    -----
    Multi-filament coil formulation. The mutual inductance between two coils is
    calculated based on the number of filaments in each coil (numerical
    discretisation, which is then normalised). The number of turns in each coil
    determine the actual multiplier of the mutual inductance.

    - **Off-diagonal terms** (:math:`i \\neq j`):

        .. math::
            M_{ij} = n_i n_j \\sum_{k=0, m=0}^{n_k, n_m} G(x_{i,n}, z_{i,n}, x_{j,m}, z_{j,m})

        where :math:`G` is the Green's function for mutual inductance.

    - **Diagonal terms** (:math:`i = j`):

        .. math::
            M_{ii} = n_i^2 L_i

        with :math:`L_i` as the self-inductance using elliptic integrals.

    """  # noqa: W505, E501
    coils = coilset.all_coils()
    n_coils = len(coils)
    M = np.zeros((n_coils, n_coils))  # noqa: N806
    itri, jtri = np.triu_indices(n_coils, k=1)
    for i, j in zip(itri, jtri, strict=True):
        coil1, coil2 = coils[i], coils[j]
        for xi1, zi1 in zip(coil1._quad_x, coil1._quad_z, strict=True):
            for xi2, zi2 in zip(coil2._quad_x, coil2._quad_z, strict=True):
                M[i, j] += greens_psi(xi1, zi1, xi2, zi2)
        M[i, j] *= (
            coil1.n_turns * coil2.n_turns / (len(coil1._quad_x) * len(coil2._quad_x))
        )

    M[jtri, itri] = M[itri, jtri]

    M[np.diag_indices_from(M)] = coilset.n_turns**2 * np.squeeze(
        square_coil_inductance_kirchhoff(coilset.x, 2 * coilset.dx, 2 * coilset.dz)
        if square_coil
        else (
            circular_coil_inductance_elliptic(
                coilset.x, np.hypot(coilset.dx, coilset.dz)
            )
        )
    )
    return M


def _get_symmetric_coils(
    coilset: CoilSet, rtol: float = 1e-5
) -> tuple[list[npt.NDArray], npt.NDArray, list[list[int]]]:
    """
    Coilset symmetry utility

    Parameters
    ----------
    coilset:
        CoilSet to get symmetric coils from
    rtol:
        Relative tolerance used when comparing coil values,
        rtol = 1.e-5 is the default value for np.allclose.
        The values for the secondary coil in the pair will be
        set to be equal to the primary coil values if they are
        within rtol.

    Returns
    -------
    :
        Symmetric coilset data
    :
        Counts of number of coils in group
    :
        indexes from original coilset
    """
    from bluemira.equilibria.coils._grouping import SymmetricCircuit  # noqa: PLC0415

    x, z, dx, dz, currents = coilset.to_group_vecs()
    coil_matrix = np.array([x, np.abs(z), dx, dz, currents]).T

    sym_stack = [[coil_matrix[0], 1, [0]]]
    for i in range(1, len(x)):
        coil = coil_matrix[i]

        for j, sym_coil in enumerate(sym_stack):
            if np.allclose(coil, sym_coil[0], rtol=rtol):
                coil = sym_coil[0]
                sym_stack[j][1] += 1
                sym_coil[2].append(i)
                break

        else:
            sym_stack.append([coil, 1, [i]])

    coil_data, count, _inds = np.array(sym_stack, dtype=object).T

    indexes = _inds.tolist()
    offset = 0
    coils = coilset._coils
    for no in range(len(indexes)):
        indexes[no] = np.array(indexes[no]) - offset
        if indexes[no].size >= 2 and isinstance(coils[no], SymmetricCircuit):  # noqa: PLR2004
            offset += 1

    return coil_data.tolist(), np.array(count, dtype=int), indexes


def check_coilset_symmetric(coilset: CoilSet, rtol: float | None = None) -> bool:
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
    coils, counts, _ = _get_symmetric_coils(coilset, rtol=rtol or 1e-5)
    for coil, count in zip(coils, counts, strict=True):
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
        if n is not None and (coils := coilset.get_coiltype(ct)) is not None:
            for i, coil_name in enumerate(coils.name):
                coilset[coil_name].name = f"{ct}_{i + 1}"
    coilset.control = coilset.name
    return coilset
