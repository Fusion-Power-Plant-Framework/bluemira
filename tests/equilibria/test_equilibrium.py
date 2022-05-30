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
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt

from bluemira.base.file import get_bluemira_path, try_get_bluemira_private_data_root
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.opt_problems import UnconstrainedTikhonovCurrentGradientCOP
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import PicardIterator
from bluemira.utilities.tools import abs_rel_difference, compare_dicts
from tests.equilibria.setup_methods import _coilset_setup


class TestFields:
    @classmethod
    def setup_class(cls):
        # Let's make a complete **** equilibrium..
        _coilset_setup(cls)
        grid = Grid(4.5, 14, -9, 9, 65, 65)

        profiles = CustomProfile(
            np.linspace(1, 0), -np.linspace(1, 0), R_0=9, B_0=6, I_p=10e6
        )

        eq = Equilibrium(cls.coilset, grid, profiles)

        isoflux = IsofluxConstraint(
            np.array([6, 8, 12, 6]), np.array([0, 7, 0, -8]), 6, 0, 0
        )
        x_point = FieldNullConstraint(8, -8)

        targets = MagneticConstraintSet([isoflux, x_point])

        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, targets, gamma=1e-8
        )

        program = PicardIterator(eq, profiles, opt_problem, relaxation=0.1, plot=False)
        program()
        cls.eq = eq

    def callable_tester(self, f_callable):
        """
        Checks that all different field calls (with different inputs) behave
        as expected
        """
        # This should go without a hitch...
        value = f_callable(8, 0)
        v2 = f_callable(np.array(8), np.array(0))
        v3 = f_callable(np.array([8]), np.array([0]))[0]
        assert np.isclose(v2, value)
        assert np.isclose(v3, value)

        # Now let's check the full field calls
        b1 = f_callable(x=None, z=None)
        b2 = f_callable(self.eq.grid.x, self.eq.grid.z)
        assert np.allclose(b1, b2)

        # Now let's check iterables (X = 4 or 20 is off-grid)
        # (Z = -10 or 10 off-grid)
        x_array = np.array([4, 8, 20, 4, 8, 20, 4, 8, 20])
        z_array = np.array([0, 0, 0, 10, 10, 10, -10, -10, 10])

        b_values = np.zeros(len(z_array))

        for i, (x, z) in enumerate(zip(x_array, z_array)):
            b_values[i] = f_callable(x, z)

        b1 = f_callable(x_array, z_array)

        assert np.allclose(b_values, b1)

    def test_Bx(self):  # noqa :N802
        self.callable_tester(self.eq.Bx)

    def test_Bz(self):  # noqa :N802
        self.callable_tester(self.eq.Bz)

    def test_Bp(self):  # noqa :N802
        self.callable_tester(self.eq.Bp)

    def test_psi(self):
        self.callable_tester(self.eq.psi)

    @pytest.mark.longrun
    def test_out_of_bounds(self):
        plt.close("all")
        eq = self.eq
        f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4)
        psi = eq.psi()
        psi = eq.plasma_psi

        newgrid = Grid(1, 15, -15, 15, 100, 100)  # bigger grid

        newpsi = np.zeros((newgrid.nx, newgrid.nz))

        # NOTE: _plasmacoil should not be accessed directly, this is just to
        # check. also why we have to do the following useless call to init
        # the _plasmacoil
        eq.plasmaBz(0.1, 0)
        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                if not eq.grid.point_inside(x, z):
                    newpsi[i, j] = eq._plasmacoil.psi(x, z)
                else:
                    newpsi[i, j] = eq.psi_func(x, z)

        levels = np.linspace(np.amin(newpsi), np.amax(newpsi), 20)
        ax.plot(*eq.grid.bounds, color="r")
        ax.contour(eq.grid.x, eq.grid.z, psi, levels=levels)
        ax.contour(
            newgrid.x,
            newgrid.z,
            newpsi,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax.set_aspect("equal")
        ax.set_title("plasma_psi")

        Bx = eq.plasma_Bx

        new_bx = np.zeros((newgrid.nx, newgrid.nz))

        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                new_bx[i, j] = eq.plasmaBx(x, z)

        levels = np.linspace(np.amin(new_bx), np.amax(new_bx), 20)

        ax2.plot(*eq.grid.bounds, color="r")
        ax2.contour(eq.grid.x, eq.grid.z, Bx, levels=levels)
        ax2.contour(
            newgrid.x,
            newgrid.z,
            new_bx,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax2.set_aspect("equal")
        ax2.set_title("plasma_Bx")

        Bz = eq.plasma_Bz

        new_bz = np.zeros((newgrid.nx, newgrid.nz))

        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                new_bz[i, j] = eq.plasmaBz(x, z)

        levels = np.linspace(np.amin(new_bz), np.amax(new_bz), 20)

        ax3.plot(*eq.grid.bounds, color="r")
        ax3.contour(eq.grid.x, eq.grid.z, Bz, levels=levels)
        ax3.contour(
            newgrid.x,
            newgrid.z,
            new_bz,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax3.set_aspect("equal")
        ax3.set_title("plasma_Bz")

        Bp = eq.plasma_Bp

        new_bp = np.sqrt(new_bx**2 + new_bz**2)

        levels = np.linspace(np.amin(new_bp), np.amax(new_bp), 20)

        ax4.plot(*eq.grid.bounds, color="r")
        ax4.contour(eq.grid.x, eq.grid.z, Bp, levels=levels)
        ax4.contour(
            newgrid.x,
            newgrid.z,
            new_bp,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax4.set_aspect("equal")
        ax4.set_title("plasma_Bp")


class TestEquilibrium:
    def test_double_null(self):
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        fn = os.sep.join([path, "DN-DEMO_eqref.json"])
        dn = Equilibrium.from_eqdsk(fn)
        assert dn.is_double_null
        fn = os.sep.join([path, "eqref_OOB.json"])
        sn = Equilibrium.from_eqdsk(fn)
        assert not sn.is_double_null

    def test_qpsi_calculation_modes(self):
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        fn = os.sep.join([path, "DN-DEMO_eqref.json"])
        dn = Equilibrium.from_eqdsk(fn)
        with patch.object(dn, "q") as eq_q:
            res = dn.to_dict(qpsi_calcmode=0)
            assert eq_q.call_count == 0
            assert "qpsi" not in res

            res = dn.to_dict(qpsi_calcmode=1)
            assert eq_q.call_count == 1
            assert "qpsi" in res

            res = dn.to_dict(qpsi_calcmode=2)
            assert eq_q.call_count == 1
            assert "qpsi" in res
            assert np.all(res["qpsi"] == 0)  # array is all zeros


class TestEqReadWrite:
    @pytest.mark.parametrize("qpsi_calcmode", [0, 1])
    def test_read_write(self, qpsi_calcmode):
        data_path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        file_name = "eqref_OOB.json"
        file_path = os.sep.join([data_path, file_name])

        new_file_name = "eqref_OOB_temp1.json"
        new_file_path = os.sep.join([data_path, new_file_name])
        eq = Equilibrium.from_eqdsk(file_path)
        eq.to_eqdsk(directory=data_path, filename=new_file_name)
        d1 = eq.to_dict(qpsi_calcmode=qpsi_calcmode)

        eq2 = Equilibrium.from_eqdsk(new_file_path)
        d2 = eq2.to_dict(qpsi_calcmode=qpsi_calcmode)
        os.remove(new_file_path)
        assert compare_dicts(d1, d2, almost_equal=True)


@pytest.mark.private
class TestQBenchmark:
    @classmethod
    def setup_class(cls):
        root = try_get_bluemira_private_data_root()
        path = os.sep.join([root, "equilibria", "STEP_SPR_08"])
        jetto_file = "SPR-008_3_Inputs_jetto.eqdsk_out"
        jetto_file = os.sep.join([path, jetto_file])
        reader = EQDSKInterface()
        jetto = reader.read(jetto_file)
        cls.q_ref = jetto["qpsi"]
        eq_file = "SPR-008_3_Outputs_STEP_eqref.eqdsk"
        eq_file = os.sep.join([path, eq_file])
        cls.eq = Equilibrium.from_eqdsk(eq_file)

    def test_q_benchmark(self):
        n = len(self.q_ref)
        psi_norm = np.linspace(0, 1, n)
        q = self.eq.q(psi_norm)

        assert 100 * np.median(abs_rel_difference(q, self.q_ref)) < 3.5
        assert 100 * np.mean(abs_rel_difference(q, self.q_ref)) < 4.0
