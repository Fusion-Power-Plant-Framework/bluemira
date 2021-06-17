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
Grad-Shafranov operator classes
"""
from numpy import ones, linspace, reshape
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import lil_matrix
from bluemira.equilibria.error import EquilibriaError


class GSoperator:
    """
    A sparse matrix for the Grad-Shafranov operator.

    Parameters
    ----------
    x_min: float
        Minimum X value on which to calculate the G-S operator
    x_max: float
        Maximum X value on which to calculate the G-S operator
    z_min: float
        Minimum Z value on which to calculate the G-S operator
    z_max: float
        Maximum Z value on which to calculate the G-S operator

    \t:math:`{\\Delta}^{*} = X^{2}{\\nabla}{\\cdot}\\dfrac{1}{X^2}{\\nabla}`

    \t:math:`\\dfrac{1}{(\\Delta Z)^2}\\psi_{i-1, j}+\\bigg(\\dfrac{1}{(\\Delta X)^2}+\\dfrac{1}{2X_j(\\Delta X)}\\bigg)\\psi_{i, j-1}`
    \t:math:`+\\Bigg[2\\Bigg(\\dfrac{1}{(\\Delta X)^2}+\\dfrac{1}{(\\Delta Z)^2}\\Bigg)\\Bigg]\\psi_{i, j}`
    \t:math:`+\\bigg(\\dfrac{1}{(\\Delta X)^2}-\\dfrac{1}{2X_j(\\Delta X)}\\bigg)\\psi_{i, j+1}`
    \t:math:`+\\dfrac{1}{(\\Delta Z)^2}\\psi_{i+1, j}=-\\mu_0 X_j J_{\\phi_{i, j}}`
    """  # noqa (W505)

    def __init__(self, x_min, x_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max

    def __call__(self, nx, nz):
        """
        Create a sparse matrix with given resolution.

        Parameters
        ----------
        nx: int
            The discretisation of the 2-D field in x
        nz: int
            The discretisation of the 2-D field in z
        """
        d_x = (self.x_max - self.x_min) / (nx - 1)
        d_z = (self.z_max - self.z_min) / (nz - 1)
        x = linspace(self.x_min, self.x_max, nx)
        d_x2, d_z2 = d_x ** 2, d_z ** 2
        A = lil_matrix((nx * nz, nx * nz))
        A.setdiag(ones(nx * nz))
        # NOTE: nb.jit doesn't seem to help here :(
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                ind = i * nz + j
                rp = 0.5 * (x[i + 1] + x[i])  # x_{i+1/2}
                rm = 0.5 * (x[i] + x[i - 1])  # x_{i-1/2}
                A[ind, ind] = -(x[i] / d_x2) * (1 / rp + 1 / rm) - 2 / d_z2  # j, l
                A[ind, ind + nz] = (x[i] / d_x2) / rp  # j, l-1
                A[ind, ind - nz] = (x[i] / d_x2) / rm  # j, l+1
                A[ind, ind + 1] = 1 / d_z2  # j-1, l
                A[ind, ind - 1] = 1 / d_z2  # j+1, l
        return A.tocsr()  # Compressed sparse row format


class DirectSolver:
    """
    Direct solve. Lower-upper decomposition
    """

    def __init__(self, A):
        self.solve = factorized(A.tocsc())

    def __call__(self, b):
        """
        Solve the matrix problem by LU decomposition.
        """
        b1d = reshape(b, -1)
        x = self.solve(b1d)
        if np.any(np.isnan(x)):
            raise EquilibriaError("Matrix inversion problem in GS solver.")
        return reshape(x, b.shape)
