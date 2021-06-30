# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Boundary conditions for equilibria.
"""

import numpy as np
from bluemira.utilities.tools import is_num
from bluemira.magnetostatics.greens import greens_psi
from bluemira.equilibria.grid import integrate_dx_dz

__all__ = ["FreeBoundary", "apply_boundary"]


class FreeBoundary:
    """
    Object representing a free boundary condition, accounting for only the
    plasma psi

    Parameters
    ----------
    grid: Grid
        The grid upon which to apply the free Dirichlet boundary condition
    """

    __slots__ = ["dx", "dz", "edges", "f_greens"]

    def __init__(self, grid):
        x, z = grid.x, grid.z
        self.dx, self.dz = grid.dx, grid.dz
        self.edges = grid.edges
 
        values = np.zeros((len(self.edges), grid.nx, grid.nz))
        for i, (j, k) in enumerate(self.edges):
            g = greens_psi(x, z, x[j, k], z[j, k])
            g[j, k] = 0  # Drop NaNs
            values[i] = g
        self.f_greens = values

    def __call__(self, psi, jtor):
        """
        Applies a free boundary (Dirichlet) condition using Green's functions

        Parameters
        ----------
        psi: np.array(N, M)
            The poloidal magnetic flux [V.s/rad]
        jtor: np.array(N, M)
            The toroidal current density in the plasma [A/m^2]

        Note
        ----
        Modifies psi in-place
        """
        for i, (j, k) in enumerate(self.edges):
            psi[j, k] = integrate_dx_dz(self.f_greens[i] * jtor, self.dx, self.dz)


def apply_boundary(rhs, lhs):
    """
    Applies a boundary constraint to an array

    Parameters
    ----------
    rhs: np.array(N, M)
        The right-hand-side of the equation
    lhs: np.array(N, M) or 0
        The left-hand-side of the equation
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
