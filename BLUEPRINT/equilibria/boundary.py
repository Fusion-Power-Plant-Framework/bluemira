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
Domain boundary conditions - Dirichlet
"""
import numpy as np
from BLUEPRINT.utilities.tools import is_num
from BLUEPRINT.magnetostatics.greens import greens_psi
from BLUEPRINT.equilibria.gridops import integrate_dx_dz


class FixedBoundary:
    """
    Object representing a fixed boundary condition
    """

    def __init__(self):
        pass

    def __call__(self, psi, *args):
        """
        Applies a fixed boundary condition, setting: psi = 0 on domain boundary

        Modifies psi in-place
        """
        psi[:, 0] = 0
        psi[0, :] = 0
        psi[:, -1] = 0
        psi[-1, :] = 0


class FreeBoundary:
    """
    Object representing a free boundary condition, accounting for only the
    plasma psi

    Parameters
    ----------
    grid: BLUEPRINT::EQUILIBRIA Grid object
        The grid upon which to apply the free Dirichlet boundary condition
    """

    def __init__(self, grid):
        x, z = grid.x, grid.z
        self.dX, self.dZ = grid.dx, grid.dz
        self.edges = grid.edges
        # Small speed optimisation: compute Green's functions on the domain
        # boundary only once
        values = np.zeros((len(self.edges), grid.nx, grid.nz))
        for i, (j, k) in enumerate(self.edges):
            g = greens_psi(x, z, x[j, k], z[j, k])
            g[j, k] = 0  # Drop NaNs
            values[i] = g
        self.values = values

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
            psi[j, k] = integrate_dx_dz(self.values[i] * jtor, self.dX, self.dZ)


class TrueFreeBoundary:
    """
    Object representing a free boundary condition, accounting for the coils and
    plasma psi
    """

    def __init__(self, grid, coilset):
        self.X, self.Z = grid.x, grid.z
        self.dX, self.dZ = grid.dx, grid.dz
        self.edges = grid.edges
        # Small speed optimisation: compute Green's functions on the domain
        # boundary only once
        values = np.zeros((len(self.edges), grid.nx, grid.nz))
        for i, (j, k) in enumerate(self.edges):
            g = greens_psi(self.X, self.Z, self.X[j, k], self.Z[j, k])
            g[j, k] = 0  # Drop NaNs
            values[i] = g
        self.values = values
        self.coilset = coilset

    def __call__(self, psi, jtor):
        """
        Applies a free boundary (Dirichlet) condition using Green's functions

        Parameters
        ----------
        psi: np.array(nx, nz)
            The magnetic flux density map
        jtor: np.array(nx, nz)
            The plasma toroidal current source term
        coilset: CoilSet object
            The coilset with which to calculate the free boundary condition

        Note
        ----
        Modifies psi in-place
        """
        for i, (j, k) in enumerate(self.edges):
            psi[j, k] = integrate_dx_dz(self.values[i] * jtor, self.dX, self.dZ)
            psi[j, k] += self.coilset.psi(self.X[j, k], self.Z[j, k])


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


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
