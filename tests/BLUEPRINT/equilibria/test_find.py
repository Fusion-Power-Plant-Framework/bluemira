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

import pytest
import pickle  # noqa (S403)
import os
import numpy as np
from BLUEPRINT.geometry.loop import MultiLoop
from BLUEPRINT.equilibria.find import (
    _in_plasma,
    find_OX,
    find_LCFS_separatrix,
    get_contours,
)  # noqa
from BLUEPRINT.equilibria.equilibrium import Equilibrium
from BLUEPRINT.base.file import get_BP_path

DATA = get_BP_path("BLUEPRINT/equilibria/test_data", subfolder="tests")


class TestGetContours:
    @staticmethod
    def old(x, z, array, value):
        """
        Not for use just an example method
        """
        from legacycontour._cntr import Cntr

        value_loop = Cntr(x, z, array).trace(value)
        return value_loop[: len(value_loop) // 2]

    @staticmethod
    def _loop_roll(arr):
        """
        Try to Roll arrays so that the are equal.
        """
        loop = all(np.equal(arr[0], arr[-1]))
        new_arr = arr[:-1].copy() if loop else arr.copy()
        new_arr = np.roll(new_arr, -np.argmin(new_arr, axis=1), axis=1)
        if loop:
            np.append(new_arr, new_arr[0])
        return new_arr

    def test_get_contours(self):
        fn = os.sep.join([DATA, "DN-DEMO_eqref.json"])
        sof = Equilibrium.from_eqdsk(fn)
        psi_n = sof.psi_norm()
        x, z = sof.x, sof.z
        value_arr = np.linspace(0.0001, 1.5)
        # generated using the old legacycontour version of get_contour see self.old
        contours_old = np.load(
            os.sep.join([DATA, "legacycontour_output.npy"]), allow_pickle=True
        )

        for v, c in zip(value_arr, contours_old):
            contours_new = get_contours(x, z, psi_n, v)
            for i, j in zip(
                np.argsort([np.sum(i) for i in c]),
                np.argsort([np.sum(i) for i in contours_new]),
            ):
                try:
                    np.testing.assert_allclose(
                        self._loop_roll(c[i]), self._loop_roll(contours_new[j])
                    )
                except AssertionError:
                    try:
                        # 1 axis reversed
                        np.testing.assert_allclose(
                            self._loop_roll(c[i]), self._loop_roll(contours_new[j])[::-1]
                        )
                    except AssertionError:
                        # One is a loop and one isn't, second axis reversed
                        np.testing.assert_allclose(
                            self._loop_roll(c[i]),
                            self._loop_roll(contours_new[j])[:-1, ::-1],
                        )


class TestFind:
    def test_FIESTAfailcase(self):  # noqa (N802)
        name = "fail_array.pkl"
        filename = os.sep.join([DATA, name])
        with open(filename, "rb") as file:
            psi = pickle.load(file)  # noqa (S301)

        nx, nz = psi.shape

        x = np.linspace(1, 10, nx)
        z = np.linspace(-10, 10, nz)
        x, z = np.meshgrid(x, z)
        find_OX(x.T, z.T, psi)


class TestInPlasma:
    def test_recursion(self):
        fn = os.sep.join([DATA, "in_plasma_test.pkl"])
        with open(fn, "rb") as f:
            data = pickle.load(f)  # noqa (S301)
        x, z = data["X"], data["Z"]
        lcfs = data["LCFS"]
        result = data["result"]
        mask = np.zeros_like(x)

        result2 = _in_plasma(x, z, mask, lcfs)
        assert np.allclose(result, result2)


class TestFindLCFSSeparatrix:
    @pytest.mark.longrun
    def test_CREATE_grid(self):
        fn = os.sep.join([DATA, "Equil_AR3d1_2015_04_v2_SOF_CSred_fine_final.eqdsk"])
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
            assert np.round(2 * np.pi * lcfs.area * lcfs.centroid[0], 0) == 2407
            assert lcfs.closed
            assert not separatrix.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, primary_xp.z])
            assert np.amin(distances) <= grid_tol
            distances = separatrix.distance_to([primary_xp.x, primary_xp.z])
            assert np.amin(distances) <= grid_tol

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

            assert isinstance(separatrix, MultiLoop)
            for loop in separatrix.loops:
                assert not loop.closed
                distances = loop.distance_to([primary_xp.x, primary_xp.z])
                assert np.amin(distances) <= grid_tol


if __name__ == "__main__":
    pytest.main([__file__])
