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
import numpy as np
import pytest
from BLUEPRINT.equilibria.gradshafranov import GSSolver
from BLUEPRINT.equilibria.gridops import Grid


class TestForcedFBESymmetry:
    def test_symmetrised_GS(self):
        """
        Checks that the solution of the GS equations is symmetric
        when symmetry in the z=0 plane is forced.
        """
        x_min = 0.5
        x_max = 2.5
        z_max = 3.0
        z_min = -z_max
        nx = 5
        nz_half = 8

        # Check for both odd and even nz
        for nz in (2 * nz_half, 2 * nz_half + 1):
            grid = Grid(x_min, x_max, z_min, z_max, nx, nz)

            solver = GSSolver(grid, force_symmetry=True)
            rhs = np.ones((nx, nz))
            solution = solver(rhs)

            # Test if the shapes of the RHS and solution arrays match
            assert np.shape(rhs) == np.shape(solution)

            # Test if the solution vector is symmetric
            assert np.allclose(solution, np.flip(solution, axis=1))


if __name__ == "__main__":
    pytest.main([__file__])
