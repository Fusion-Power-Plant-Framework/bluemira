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

import os
import pytest
import numpy as np
import json
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.find import (
    find_local_minima,
    inv_2x2_matrix,
    find_LCFS_separatrix,
    _in_plasma,
)
from bluemira.equilibria.equilibrium import Equilibrium


DATA = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")


def test_find_local_minima():
    for _ in range(10):
        array = np.ones((100, 100))
        i, j = np.random.randint(0, 99, 2)
        array[i, j] = 0
        ii, jj = find_local_minima(array)
        assert len(ii) == 1
        assert len(jj) == 1
        assert ii[0] == i
        assert jj[0] == j

    array = np.ones((100, 100))
    array[1, 0] = 0
    array[-2, -2] = 0
    array[-2, 1] = 0
    array[1, -2] = 0
    array[0, 50] = 0
    array[50, 0] = 0

    ii, jj = find_local_minima(array)

    assert len(ii) == 6
    assert len(jj) == 6
    assert (np.sort(ii) == np.array([0, 1, 1, 50, 98, 98])).all()
    assert (np.sort(jj) == np.array([0, 0, 1, 50, 98, 98])).all()


def test_inv_2x2_jacobian():
    a, b, c, d = 3.523, 5.0, 6, 0.2
    inv_jac_true = np.linalg.inv(np.array([[a, b], [c, d]]))
    inv_jac = inv_2x2_matrix(a, b, c, d)
    assert np.allclose(inv_jac_true, inv_jac)


class TestFindLCFSSeparatrix:
    def test_other_grid(self):
        fn = os.sep.join([DATA, "eqref_OOB.json"])
        sof = Equilibrium.from_eqdsk(fn)
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
            )
            assert lcfs.closed
            assert not separatrix.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, primary_xp.z])
            assert np.amin(distances) <= grid_tol
            distances = separatrix.distance_to([primary_xp.x, primary_xp.z])
            assert np.amin(distances) <= grid_tol

    def test_double_null(self):
        fn = os.sep.join([DATA, "DN-DEMO_eqref.json"])
        sof = Equilibrium.from_eqdsk(fn)
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
                double_null=True,
            )

            assert lcfs.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, primary_xp.z])
            assert np.amin(distances) <= grid_tol

            assert isinstance(separatrix, list)
            for loop in separatrix:
                assert not loop.closed
                distances = loop.distance_to([primary_xp.x, primary_xp.z])
                assert np.amin(distances) <= grid_tol


class TestInPlasma:
    def test_recursion(self):
        fn = os.sep.join([DATA, "in_plasma_test.json"])
        with open(fn, "rb") as f:
            data = json.load(f)  # noqa (S301)
        x, z = np.array(data["X"]), np.array(data["Z"])
        lcfs = np.array(data["LCFS"])
        result = np.array(data["result"])
        mask = np.zeros_like(x)

        result2 = _in_plasma(x, z, mask, lcfs)
        assert np.allclose(result, result2)


if __name__ == "__main__":
    pytest.main([__file__])
