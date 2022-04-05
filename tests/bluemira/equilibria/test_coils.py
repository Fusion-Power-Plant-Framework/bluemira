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

import numpy as np
import pytest
from matplotlib import pyplot as plt

import tests
from bluemira.base.constants import MU_0
from bluemira.equilibria.coils import (
    CS_COIL_NAME,
    NO_COIL_NAME,
    PF_COIL_NAME,
    Coil,
    CoilGroup,
    CoilSet,
    SymmetricCircuit,
    check_coilset_symmetric,
    make_mutual_inductance_matrix,
    symmetrise_coilset,
)
from bluemira.equilibria.constants import NBTI_J_MAX
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid


class TestCoil:
    @classmethod
    def setup_class(cls):
        # make a default coil
        cls.coil = Coil(4, 4, 10e6, j_max=NBTI_J_MAX)
        cls.cs_coil = Coil(4, 4, 10e6, ctype="CS", j_max=NBTI_J_MAX)
        cls.no_coil = Coil(4, 4, 10e6, ctype="asrgd", j_max=NBTI_J_MAX)

    def test_name(self):
        def extract_int(coil):
            return int(coil.name.split("_")[-1])

        def extract_prefix(coil):
            return coil.name.split("_")[0]

        assert extract_prefix(self.coil) == PF_COIL_NAME.split("_")[0]
        assert extract_prefix(self.cs_coil) == CS_COIL_NAME.split("_")[0]
        assert extract_prefix(self.no_coil) == NO_COIL_NAME.split("_")[0]
        coil = Coil(4, 4, 10e6, j_max=NBTI_J_MAX)
        cs_coil = Coil(4, 4, 10e6, ctype="CS", j_max=NBTI_J_MAX)
        no_coil = Coil(4, 4, 10e6, ctype="agd", j_max=NBTI_J_MAX)

        assert extract_int(coil) == extract_int(self.coil) + 1
        assert extract_int(cs_coil) == extract_int(self.cs_coil) + 1
        assert extract_int(no_coil) == extract_int(self.no_coil) + 1

    def test_field(self):
        c = Coil(1, 0, current=1591550, dx=0, dz=0)  # Sollte 5 T am Achse erzeugen
        Bx, Bz = 0, MU_0 * c.current / (2 * c.x)

        assert c.Bx(0.001, 0) == Bx
        assert round(abs(c.Bz(0.001, 0) - Bz), 5) == 0
        z = 4
        Bx, Bz = (
            0,
            MU_0
            * 2
            * np.pi
            * c.x**2
            * c.current
            / (4 * np.pi * (z**2 + c.x**2) ** (3 / 2)),
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
        gbp = np.sqrt(gbx**2 + gbz**2)
        gp = c.control_psi(x, z)

        if tests.PLOTTING:
            f, ax = plt.subplots()
            cc = ax.contourf(x, z, gbx)

            plt.colorbar(cc)
            ax.set_aspect("equal")
            ax.set_xlim([2, 6])
            ax.set_ylim([-3, 3])

        c.mesh_coil(0.1)

        gbxn = c.control_Bx(x, z)
        gbzn = c.control_Bz(x, z)
        gbpn = np.sqrt(gbx**2 + gbz**2)
        gpn = c.control_psi(x, z)

        if tests.PLOTTING:
            f, ax = plt.subplots()
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


class TestCoilGroup:
    @staticmethod
    def make_coilgroup():
        coils = [
            Coil(6, 6, ctype="CS", name="CS_8", j_max=NBTI_J_MAX),
            Coil(7, 7, ctype="CS", name="CS_0", j_max=NBTI_J_MAX),
            Coil(8, 8, ctype="plasma", name="plasma_1", j_max=NBTI_J_MAX),
            Coil(4, 4, ctype="PF", name="PF_1", j_max=NBTI_J_MAX),
            Coil(4, 5, ctype="PF", name="PF_0", j_max=NBTI_J_MAX),
        ]

        return CoilGroup(coils)

    def test_init_sort(self):
        group = self.make_coilgroup()
        coil_list = list(group.coils.values())
        assert len(coil_list) == 5
        assert coil_list[0].name == "PF_0"
        assert coil_list[1].name == "PF_1"
        assert coil_list[2].name == "CS_0"
        assert coil_list[3].name == "CS_8"
        assert coil_list[4].name == "plasma_1"

    def test_add(self):
        group = self.make_coilgroup()
        group.add_coil(Coil(3, 3, ctype="PF", name="PF_3", j_max=NBTI_J_MAX))
        group.add_coil(Coil(9, 9, ctype="CS", name="CS_9", j_max=NBTI_J_MAX))
        group.add_coil(Coil(10, 10, ctype="plasma", name="plasma_10", j_max=NBTI_J_MAX))

        coil_list = list(group.coils.values())
        assert len(coil_list) == 8
        assert coil_list[0].name == "PF_0"
        assert coil_list[1].name == "PF_1"
        assert coil_list[2].name == "PF_3"
        assert coil_list[3].name == "CS_0"
        assert coil_list[4].name == "CS_8"
        assert coil_list[5].name == "CS_9"
        assert coil_list[6].name == "plasma_1"
        assert coil_list[7].name == "plasma_10"

    def test_remove(self):
        group = self.make_coilgroup()
        group.remove_coil("PF_0")
        group.remove_coil("PF_1")

        coil_list = list(group.coils.values())
        assert len(coil_list) == 3
        assert coil_list[0].name == "CS_0"

        with pytest.raises(KeyError):
            group.remove_coil("PF_1")


class TestSymmetricCircuit:
    @classmethod
    def setup_class(cls):
        coil = Coil(x=1.5, z=6, current=1e6, dx=0.25, dz=0.5, ctype="PF", name="TEST")
        circuit = SymmetricCircuit(coil)
        mirror_coil = Coil(
            x=1.5, z=-6, current=1e6, dx=0.25, dz=0.5, ctype="PF", name="TEST_MIRROR"
        )

        cls.circuit = circuit
        cls.coils = [coil, mirror_coil]

    def test_fields(self):
        points = [
            [1, 1],
            [2, 2],
            [1.5, 6],
            [1.5, -6],
        ]
        for point in points:
            coil_psi = sum([coil.psi(*point) for coil in self.coils])
            coil_Bx = sum([coil.Bx(*point) for coil in self.coils])
            coil_Bz = sum([coil.Bz(*point) for coil in self.coils])

            circuit_psi = self.circuit.psi(*point)
            circuit_Bx = self.circuit.Bx(*point)
            circuit_Bz = self.circuit.Bz(*point)
            assert np.isclose(coil_psi, circuit_psi)
            assert np.isclose(coil_Bx, circuit_Bx)
            assert np.isclose(coil_Bz, circuit_Bz)

    def test_control(self):
        points = [
            [1, 1],
            [2, 2],
            [1.5, 6],
            [1.5, -6],
        ]
        for point in points:
            coil_psi = sum([coil.control_psi(*point) for coil in self.coils])
            coil_Bx = sum([coil.control_Bx(*point) for coil in self.coils])
            coil_Bz = sum([coil.control_Bz(*point) for coil in self.coils])

            circuit_psi = self.circuit.control_psi(*point)
            circuit_Bx = self.circuit.control_Bx(*point)
            circuit_Bz = self.circuit.control_Bz(*point)
            assert np.isclose(coil_psi, circuit_psi)
            assert np.isclose(coil_Bx, circuit_Bx)
            assert np.isclose(coil_Bz, circuit_Bz)

    def test_current(self):
        self.circuit.set_current(2e6)
        for coil in self.coils:
            coil.set_current(2e6)
        self.test_fields()

    def test_attributes(self):
        self.circuit.x = 4
        assert self.circuit.x == 4
        assert self.circuit["TEST.1"].x == 4
        assert self.circuit["TEST.2"].x == 4

        self.circuit.z = 6
        assert self.circuit.z == 6
        assert self.circuit["TEST.1"].z == 6
        assert self.circuit["TEST.2"].z == -6


class TestCoilSet:
    @classmethod
    def setup_class(cls):
        coil = Coil(
            x=1.5,
            z=6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_2",
        )
        circuit = SymmetricCircuit(coil)

        coil2 = Coil(
            x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5.0, b_max=50, name="PF_1"
        )

        cls.coilset = CoilSet([coil2, circuit])

    def test_group_vecs(self):
        x, z, dx, dz, currents = self.coilset.to_group_vecs()

        assert np.allclose(x, np.array([4, 1.5, 1.5]))
        assert np.allclose(z, np.array([10, 6, -6]))
        assert np.allclose(dx, np.array([1, 0.25, 0.25]))
        assert np.allclose(dz, np.array([0.5, 0.5, 0.5]))
        assert np.allclose(currents, np.array([2e6, 1e6, 1e6]))

    def test_member_attributes(self):
        assert np.isclose(self.coilset["PF_1"].x, 4)
        assert np.isclose(self.coilset["PF_2"].x, 1.5)

        assert np.isclose(self.coilset["PF_1"].z, 10)
        assert np.isclose(self.coilset["PF_2"].z, 6)

        assert np.isclose(self.coilset["PF_1"].dx, 1)
        assert np.isclose(self.coilset["PF_2"].dx, 0.25)

        assert np.isclose(self.coilset["PF_1"].dz, 0.5)
        assert np.isclose(self.coilset["PF_2"].dz, 0.5)

        assert np.isclose(self.coilset["PF_1"].j_max, 5.0)
        assert np.isclose(self.coilset["PF_2"].j_max, 10.0)

        assert np.isclose(self.coilset["PF_1"].b_max, 50)
        assert np.isclose(self.coilset["PF_2"].b_max, 100)

        assert np.isclose(self.coilset["PF_1"].current, 2e6)
        assert np.isclose(self.coilset["PF_2"].current, 1e6)

        coil2 = self.coilset["PF_1"]
        coil2.set_dz(0.77)
        symm_circuit = self.coilset["PF_2"]
        symm_circuit.set_dz(0.6)
        assert np.isclose(coil2.dz, 0.77)
        assert np.isclose(symm_circuit["PF_2.1"].dz, 0.6)
        assert np.isclose(symm_circuit["PF_2.2"].dz, 0.6)

    def test_numbers(self):
        assert self.coilset.n_PF == 2
        assert self.coilset.n_CS == 0
        assert self.coilset.n_coils == 3
        assert len(self.coilset.coils) == 2

    def test_currents(self):
        set_currents = np.array([3e6, 4e6])
        self.coilset.set_control_currents(set_currents)
        currents = self.coilset.get_control_currents()
        assert np.allclose(set_currents, currents)

    def test_material_assignment(self):
        test_j_max = 7.0
        test_b_max = 24.0
        self.coilset.assign_coil_materials("PF", j_max=test_j_max, b_max=test_b_max)
        n_indep_coils = len(self.coilset.coils.values())
        assert len(self.coilset.get_max_currents(0.0)) == n_indep_coils
        assert len(self.coilset.get_max_fields()) == n_indep_coils
        for coil in self.coilset.coils.values():
            assert np.isclose(coil.j_max, test_j_max)
            assert np.isclose(coil.b_max, test_b_max)


class TestCoilSetSymmetry:
    def test_symmetry_check(self):
        coilset = CoilSet(
            [Coil(5, 5, 0, dx=1.0, dz=1.0), Coil(5, -5, 0, dx=1.0, dz=1.0)]
        )

        assert check_coilset_symmetric(coilset)

        coilset = CoilSet(
            [Coil(5, 5, 0, dx=1.0, dz=1.0), Coil(5, -5, dx=1.0, dz=1.0, current=1e6)]
        )

        assert not check_coilset_symmetric(coilset)

        coilset = CoilSet(
            [
                Coil(5, 5, 0, dx=1.0, dz=1.0),
                Coil(5, 0, 0, dx=1.0, dz=1.0),
                Coil(5, -5, 0, dx=1.0, dz=1.0),
            ]
        )
        assert check_coilset_symmetric(coilset)

        coilset = CoilSet(
            [
                Coil(5, 5, 0, dx=1.0, dz=1.0),
                Coil(5, 1, 0, dx=1.0, dz=1.0),
                Coil(5, -5, 0, dx=1.0, dz=1.0),
            ]
        )

        assert not check_coilset_symmetric(coilset)

        coilset = CoilSet(
            [
                Coil(5, 5, 0, dx=1.0, dz=1.0),
                Coil(5, 0, 0, dx=1.0, dz=1.0),
                Coil(5, 1, 0, dx=1.0, dz=1.0),
                Coil(5, -5, 0, dx=1.0, dz=1.0),
            ]
        )

        assert not check_coilset_symmetric(coilset)

    def test_symmetrise(self):
        coilset = CoilSet(
            [
                Coil(5, 5, 1e6, dx=1, dz=1),
                Coil(5, -5, 1e6, dx=1, dz=1),
            ]
        )
        new = symmetrise_coilset(coilset)
        assert len(new.coils) == 1
        assert new.n_coils == 2
        assert isinstance(list(new.coils.values())[0], SymmetricCircuit)

        coilset = CoilSet(
            [
                SymmetricCircuit(Coil(5, 5, 1e6, dx=1, dz=1)),
                SymmetricCircuit(Coil(12, 7, 1e6, dx=1, dz=1)),
                SymmetricCircuit(Coil(4, 9, 1e6, dx=1, dz=1)),
                Coil(5, 0, 1e6, dx=1, dz=1),
            ]
        )
        new = symmetrise_coilset(coilset)
        assert len(new.coils) == len(coilset.coils)
        assert new.n_coils == coilset.n_coils


class TestCoilSizing:
    def test_initialisation(self):
        c = Coil(4, 4, current=0, j_max=None, dx=1, dz=1)
        assert c.dx == 1
        assert c.dz == 1

        c = Coil(4, 4, current=0, j_max=None, dx=1, dz=2)
        assert c.dx == 1
        assert c.dz == 2

        c = Coil(4, 4, current=0, j_max=None, dx=2, dz=1)
        assert c.dx == 2
        assert c.dz == 1

        c = Coil(4, 4, current=0, j_max=None, dx=1, dz=1, flag_sizefix=False)
        assert c.dx == 1
        assert c.dz == 1
        assert c.flag_sizefix

        c = Coil(4, 4, current=0, j_max=10)
        assert c.dx == 0
        assert c.dz == 0
        assert not c.flag_sizefix

    def test_bad_initialisation(self):
        with pytest.raises(EquilibriaError):
            Coil(4, 4, current=1)

        with pytest.raises(EquilibriaError):
            Coil(4, 4, dx=1)
        with pytest.raises(EquilibriaError):
            Coil(4, 4, dz=1)


class TestMutualInductances:
    @classmethod
    def setup_class(cls):
        coil1 = Coil(4, 4, j_max=1)
        coil2 = Coil(5, 5, j_max=1)
        coil3 = Coil(6, 6, j_max=1)
        cls.coilset1 = CoilSet([coil1, coil2, coil3])

    def test_normal(self):
        """
        Just check the symmetry for now
        """
        m = make_mutual_inductance_matrix(self.coilset1)

        assert m.shape == (3, 3)
        idxs = np.triu_indices(3, k=1)
        tri_upper = m[idxs]

        test_m = np.zeros((3, 3))
        test_m[idxs] = tri_upper
        test_m = test_m + test_m.T
        diag = np.diag_indices(3)
        m[diag] = 0.0
        assert np.allclose(m, test_m)


class TestCoilSorting:
    def test_sorting_coilset(self):
        """Ensure sorting of coils properly handles numbers in the coil name."""
        coils = [
            Coil(5, 5, 0, dx=1.0, dz=1.0, name="CS_2"),
            Coil(5, 5, 0, dx=1.0, dz=1.0, name="CS_10"),
        ]
        sorted_coils = CoilGroup.sort_coils(coils)
        sorted_names = list(sorted_coils)
        assert sorted_names[0] == "CS_2"
        assert sorted_names[1] == "CS_10"
