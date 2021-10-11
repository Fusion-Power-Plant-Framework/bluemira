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
import os
import numpy as np
import json
from matplotlib import pyplot as plt
import tests
from unittest.mock import patch
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.constants import MU_0
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.coils import Coil, CoilSet
from tests.BLUEPRINT.equilibria.setup_methods import _coilset_setup, _coil_circuit_setup


class TestCoil:
    @classmethod
    def setup_class(cls):
        # make a default coil
        cls.coil = Coil(4, 4, 10e6)

    def test_field(self):
        c = Coil(1, 0, current=1591550)  # Sollte 5 T am Achse erzeugen
        Bx, Bz = 0, MU_0 * c.current / (2 * c.x)

        assert c.Bx(0.001, 0) == Bx
        assert round(abs(c.Bz(0.001, 0) - Bz), 5) == 0
        z = 4
        Bx, Bz = (
            0,
            MU_0
            * 2
            * np.pi
            * c.x ** 2
            * c.current
            / (4 * np.pi * (z ** 2 + c.x ** 2) ** (3 / 2)),
        )
        assert round(abs(c.Bx(0.001, z) - Bx), 4) == 0
        assert round(abs(c.Bz(0.001, z) - Bz), 5) == 0
        psi_single = c.psi(15, 15)
        c.mesh_coil(0.1)
        assert round(abs(c.Bx(0.001, z) - Bx), 4) == 0
        assert round(abs(c.Bz(0.001, z) - Bz), 3) == 0
        psi_multi = c.psi(15, 15)
        assert round(abs(psi_single - psi_multi), 2) == 0

    def test_mesh(self):
        xmin, xmax = 0.1, 20
        nx, nz = 100, 100

        zmin, zmax = -12, 12
        x_1_d = np.linspace(xmin, xmax, nx)
        z_1_d = np.linspace(zmin, zmax, nz)
        x, z = np.meshgrid(x_1_d, z_1_d, indexing="ij")
        c = Coil(4, 0, current=1591550, dx=0.3, dz=1)

        gbx = c.control_Bx(x, z)
        gbz = c.control_Bz(x, z)
        gbp = np.sqrt(gbx ** 2 + gbz ** 2)
        gp = c.control_psi(x, z)

        f, ax = plt.subplots()
        cc = ax.contourf(x, z, gbx)

        plt.colorbar(cc)
        ax.set_aspect("equal")
        ax.set_xlim([2, 6])
        ax.set_ylim([-3, 3])

        c.mesh_coil(0.1)

        f, ax = plt.subplots()
        gbxn = c.control_Bx(x, z)
        gbzn = c.control_Bz(x, z)
        gbpn = np.sqrt(gbx ** 2 + gbz ** 2)
        gpn = c.control_psi(x, z)

        if tests.PLOTTING:
            c = ax.contourf(x, z, gbxn)
            plt.colorbar(c)
            ax.set_aspect("equal")
            ax.set_xlim([2, 6])
            ax.set_ylim([-3, 3])
            plt.show()

    @staticmethod
    def callable_tester(f_callable):
        """
        Checks that all different field calls (with different inputs) behave
        as expected
        """
        # This should go without a hitch...
        value = f_callable(8, 0)
        v2 = f_callable(np.array(8), np.array(0))
        v3 = f_callable(np.array([8]), np.array([0]))
        assert np.isclose(v2, value)
        assert np.isclose(v3, value)

        # Now let's check iterables (X = 4 or 20 is off-grid)
        # (Z = -10 or 10 off-grid)
        x = np.array([4, 8, 20, 4, 8, 20, 4, 8, 20])
        z = np.array([0, 0, 0, 10, 10, 10, -10, -10, 10])

        b = np.zeros(len(z))

        for i, (xi, zi) in enumerate(zip(x, z)):
            b[i] = f_callable(xi, zi)

        b1 = f_callable(x, z)

        assert np.allclose(b, b1)

    def test_bx(self):
        self.callable_tester(self.coil.Bx)

    def test_bz(self):
        self.callable_tester(self.coil.Bz)

    def test_bp(self):
        self.callable_tester(self.coil.Bp)

    def test_psi(self):
        self.callable_tester(self.coil.psi)

    def test_point_in_coil(self):
        coil = Coil(4, 4, current=10, dx=1, dz=2)
        inside_x = [3, 4, 5, 3, 4, 5, 3, 4, 5]
        inside_z = [2, 2, 2, 3, 3, 3, 6, 6, 6]
        inside = coil._points_inside_coil(inside_x, inside_z)
        assert np.alltrue(inside)
        outside_x = [0, 0, 0, 1, 1, 1, 10, 10, 10, 3, 3, 3]
        outside_z = [0, 4, 6, 0, 4, 6, 0, 4, 6, 1.9, 6.1, 10]
        outside = coil._points_inside_coil(outside_x, outside_z)
        assert np.all(~outside)
        assert np.alltrue(coil._points_inside_coil(coil.x_corner, coil.z_corner))

    def test_array_handling(self):
        data = get_BP_path("bluemira/magnetostatics/test_data", subfolder="tests")
        filename = os.sep.join([data, "new_B_along_z-z.json"])

        coil = Coil(4, 61, current=20e6, dx=0.5, dz=1.0)

        with open(filename, "r") as file:
            data = json.load(file)
            data["x"] = np.array(data["x"], dtype=np.float)
            data["z"] = np.array(data["z"], dtype=np.float)

        x, z = data["x"][1:], data["z"][1:]
        Bp_array = coil.Bp(x, z)

        Bp_list = []
        for xi, zi in zip(x, z):
            Bp_list.append(coil.Bp(xi, zi))

        assert np.allclose(Bp_array, np.array(Bp_list), equal_nan=True)


class TestCoilSet:
    @classmethod
    def setup_class(cls):
        _coilset_setup(cls)
        _coil_circuit_setup(cls)

    def test_get_solenoid(self):
        """
        Test persistence of CS coil ordering
        """
        for name in ["CS_1", "CS_2", "CS_3", "CS_4", "CS_5"]:
            coil = self.coilset.coils[name]
            assert name == coil.name

        _ = self.coilset.get_solenoid()

        for name in ["CS_1", "CS_2", "CS_3", "CS_4", "CS_5"]:
            coil = self.coilset.coils[name]
            assert name == coil.name

    def test_splitter(self):
        cs = CoilSet(self.circuits, R_0=6.5)
        before = cs.n_PF
        before_coils = list(cs.coils.values())
        cs.splitter(True)
        after = cs.n_PF
        after_coils = list(cs.coils.values())
        assert after == 2 * before
        assert len(after_coils) == 2 * len(before_coils)


@pytest.mark.longrun
class TestSemiAnalytic:
    """
    Compare all three control response methods, and that the combination of the
    Greens and semi-analytical methods makes sense (graphically..)
    """

    @classmethod
    def setup_class(cls):
        cls.coil = Coil(4, 4, current=10e6, dx=1, dz=2)
        cls.coil.mesh_coil(0.2)
        cls.grid = Grid(0.1, 8, 0, 8, 100, 100)
        cls.x_corner = np.append(cls.coil.x_corner, cls.coil.x_corner[0])
        cls.z_corner = np.append(cls.coil.z_corner, cls.coil.z_corner[0])

    def test_bx(self):
        gp = self.coil.control_Bx(self.grid.x, self.grid.z)
        gp_greens = self.coil._control_Bx_greens(self.grid.x, self.grid.z)
        gp_analytic = self.coil._control_Bx_analytical(self.grid.x, self.grid.z)

        if tests.PLOTTING:
            f, ax = plt.subplots(1, 3)
            levels = np.linspace(np.amin(gp), np.amax(gp), 20)
            ax[0].contourf(self.grid.x, self.grid.z, gp_greens)
            ax[1].contourf(self.grid.x, self.grid.z, gp, levels=levels)
            ax[2].contourf(self.grid.x, self.grid.z, gp_analytic, levels=levels)
            for axis in ax:
                axis.plot(self.x_corner, self.z_corner, color="r")
                axis.set_aspect("equal")
            ax[0].set_title("Green's functions")
            ax[1].set_title("Combined Green's and semi-analytic")
            ax[2].set_title("Semi-analytic method")

    def test_bz(self):
        gp = self.coil.control_Bz(self.grid.x, self.grid.z)
        gp_greens = self.coil._control_Bz_greens(self.grid.x, self.grid.z)
        gp_analytic = self.coil._control_Bz_analytical(self.grid.x, self.grid.z)

        if tests.PLOTTING:
            f, ax = plt.subplots(1, 3)
            levels = np.linspace(np.amin(gp), np.amax(gp), 20)
            ax[0].contourf(self.grid.x, self.grid.z, gp_greens)
            ax[1].contourf(self.grid.x, self.grid.z, gp, levels=levels)
            ax[2].contourf(self.grid.x, self.grid.z, gp_analytic, levels=levels)
            for axis in ax:
                axis.plot(self.x_corner, self.z_corner, color="r")
                axis.set_aspect("equal")
            ax[0].set_title("Green's functions")
            ax[1].set_title("Combined Green's and semi-analytic")
            ax[2].set_title("Semi-analytic method")


class TestSymmetricCircuit:
    """
    Compare psi, or other fields as a result of Circuits and Coilsets
    """

    @classmethod
    def setup_class(cls):
        _coil_circuit_setup(cls)

    def test_fields(self):
        for i in np.arange(5):  # 2, 1 chosen as generic point
            assert np.isclose(self.circuits[i].psi(2, 1), self.coilsets[i].psi(2, 1))
            assert np.isclose(self.circuits[i].Bx(2, 1), self.coilsets[i].Bx(2, 1))
            assert np.isclose(self.circuits[i].Bz(2, 1), self.coilsets[i].Bz(2, 1))

    def test_points_inside_coil(self):
        for circuit in self.circuits:
            assert circuit._points_inside_coil(circuit.x, -circuit.z)

    def test_splitting(self):
        circuits = self.circuits.copy()
        for n, circ in enumerate(circuits):
            assert circ.splittable
            coils = [c for c in circ.split()]

            assert len(coils) == circ._n_coils
            assert coils[0].name + ".1" == coils[1].name

            for c, cs_c in zip(coils, self.coilsets[n].coils.values()):
                assert not c.splittable
                np.testing.assert_equal(c.psi(2, 1), cs_c.psi(2, 1))
                np.testing.assert_equal(c.Bx(2, 1), cs_c.Bx(2, 1))
                np.testing.assert_equal(c.Bz(2, 1), cs_c.Bz(2, 1))
                assert not c._points_inside_coil(c.x, -c.z)

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot_circuit(self):
        # should show two coils
        with patch("tests.BLUEPRINT.equilibria.setup_methods.Coil.plot") as plt_m:
            self.circuits[0].plot()
            assert plt_m.call_count == 2

        self.circuits[0].plot()
        plt.show()

    def test_mesh_save(self):
        circ = self.circuits[0].copy()
        with patch("tests.BLUEPRINT.equilibria.setup_methods.Coil.mesh_coil") as mc_m:
            assert not circ._meshed
            circ._remesh()
            assert not mc_m.called
            circ.mesh_coil(123)
            assert circ._meshed == 123
            assert mc_m.called
            circ._remesh()
            assert mc_m.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
