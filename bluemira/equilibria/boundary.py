# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Boundary conditions for equilibria.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.grid import Grid, integrate_dx_dz
from bluemira.magnetostatics.greens import greens_psi
from bluemira.utilities.tools import is_num

__all__ = ["FreeBoundary", "apply_boundary"]


class FreeBoundary:
    """
    Object representing a free boundary condition, accounting for only the
    plasma psi

    Parameters
    ----------
    grid:
        The grid upon which to apply the free Dirichlet boundary condition
    """

    __slots__ = ("dx", "dz", "edges", "f_greens")

    def __init__(self, grid: Grid):
        x, z = grid.x, grid.z
        self.dx, self.dz = grid.dx, grid.dz
        self.edges = grid.edges

        values = np.zeros((len(self.edges), grid.nx, grid.nz))
        for i, (j, k) in enumerate(self.edges):
            g = greens_psi(x, z, x[j, k], z[j, k])
            g[j, k] = 0  # Drop NaNs
            values[i] = g
        self.f_greens = values

    def __call__(self, psi: npt.NDArray, jtor: npt.NDArray):
        """
        Applies a free boundary (Dirichlet) condition using Green's functions

        Parameters
        ----------
        psi:
            The poloidal magnetic flux [V.s/rad]
        jtor:
            The toroidal current density in the plasma [A/m^2]

        Note
        ----
        Modifies psi in-place
        """
        for i, (j, k) in enumerate(self.edges):
            psi[j, k] = integrate_dx_dz(self.f_greens[i] * jtor, self.dx, self.dz)


def apply_boundary(rhs: npt.NDArray, lhs: Union[float, npt.NDArray]):
    """
    Applies a boundary constraint to the boundaries of an array for use on finite
    difference grids.

    Parameters
    ----------
    rhs:
        The right-hand-side of the equality
    lhs:
        The left-hand-side of the equality
        If 0, will apply a fixed boundary condition of 0 to the rhs

    Note
    ----
    Modified rhs in-place; applying lhs boundary condition
    """
    if is_num(lhs):
        # Usually used to apply a 0 boundary condition
        rhs[0, :] = lhs
        rhs[:, 0] = lhs
        rhs[-1, :] = lhs
        rhs[:, -1] = lhs
    else:
        rhs[0, :] = lhs[0, :]
        rhs[:, 0] = lhs[:, 0]
        rhs[-1, :] = lhs[-1, :]
        rhs[:, -1] = lhs[:, -1]
