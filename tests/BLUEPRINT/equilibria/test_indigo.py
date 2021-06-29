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

import os
from matplotlib import pyplot as plt
import numpy as np
import pytest
from BLUEPRINT.equilibria.equilibrium import Equilibrium, Breakdown
from BLUEPRINT.equilibria.coils import CoilSet
from BLUEPRINT.equilibria.run import AbInitioEquilibriumProblem
from BLUEPRINT.equilibria.find import in_zone
from BLUEPRINT.equilibria.constraints import XzTesting
from BLUEPRINT.equilibria.shapes import flux_surface_johner
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.utilities.plottools import mathify
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface

import tests
from tests.BLUEPRINT.equilibria.setup_methods import _coilset_setup


class TestEUDEMO1:
    """
    System test on reference CREATE equilibria. Plasma psi is reconstructed
    from psi field - coilset.psi() contribution.
    """

    @classmethod
    def setup_class(cls):
        cls.fp = get_BP_path("eqdsk", subfolder="data")
        cls.eof = os.path.join(cls.fp, "AR3d1_2015_04_v2_EOF_CSred_fine_final.eqdsk")
        cls.sof = os.path.join(cls.fp, "AR3d1_2015_04_v2_SOF_CSred_fine_final.eqdsk")
        cls.coilset = CoilSet.from_eqdsk(cls.eof)

    @pytest.mark.longrun
    def test_boundary_and_psi(self):
        # Takes 285 seconds
        if tests.PLOTTING:
            plot_defaults()

        for e in [self.eof, self.sof]:
            eqdsk = EQDSKInterface()
            r = eqdsk.read(e)
            eq = Equilibrium.from_eqdsk(e)
            if tests.PLOTTING:
                f, ax = plt.subplots()
                ax.plot(r["xbdry"], r["zbdry"], color="b")
            sep = eq.get_separatrix()
            if tests.PLOTTING:
                eq.plot(ax)
                levels = np.linspace(np.amin(r["psi"]), np.amax(r["psi"]), 30)
                x, z = np.meshgrid(r["x"], r["z"], indexing="ij")
                ax.contourf(x, z, r["psi"], cmap="plasma", levels=levels)
                sep.plot(ax=ax, edgecolor="r", fill=False)
            assert np.allclose(r["psi"], 2 * np.pi * eq.psi())

    def test_breakdown(self):
        """
        Fabrizio Franza MIRA benchmark
        """
        if tests.PLOTTING:
            plot_defaults()
        grid = Grid(0.1, 20, -15, 10, 65, 65)
        bd = Breakdown(self.coilset, grid, R_0=9.072)
        currents = 1e6 * np.array(
            [15.6, 2.08, -0.257, 1.49, -0.333, 14.2, 28.1, 27.8, 56.3, 28.1, 28.1]
        )
        bd.coilset.set_control_currents(currents)
        bd.coilset.mesh_coils(0.1)
        bd._remap_greens()
        psi_bd = bd.psi(9.8, 0)
        if tests.PLOTTING:
            f, ax = plt.subplots()
            bd.coilset.plot(ax)
            bd.plot(ax)
        zone = Loop(*circle_seg(2, [9.8, 0]))
        mask = in_zone(bd.x, bd.z, zone.d2.T)
        Bp = bd.Bp()
        assert np.amax(mask * Bp) <= 4e-3
        assert np.isclose(2 * np.pi * psi_bd, 320.3, 1)


@pytest.mark.longrun
class TestFreeBoundary:
    """
    Grid test with FreeBoundary.

    This emulates a cute Jeon et al test
    """

    def test_grids(self):
        """
        Solve the same equilibrium on different-sized grids
        """
        _coilset_setup(self, materials=True)
        self.coilset.reset()
        a = AbInitioEquilibriumProblem(
            9.072,
            5.667,
            3.1,
            19.6e6,
            1.107,
            li=0.8,
            delta=0.3333,
            kappa=1.65,
            n_PF=6,
            n_CS=5,
            r_cs=0,
            tk_cs=0,
            tfbnd=None,
            eqtype="SN",
            rtype="Normal",
            coilset=self.coilset,
        )
        f_s = flux_surface_johner(
            9.072, 0, 9.072 / 3.1, 1.59, 1.9, 0.4, 0.4, -20, 0, 60, 30, n=100
        )
        a.constraints = XzTesting(f_s.x, f_s.z, -0 / (2 * np.pi), n=30)
        a.solve()
        c = AbInitioEquilibriumProblem(
            9.072,
            5.667,
            3.1,
            19.6e6,
            1.107,
            li=0.8,
            delta=0.3333,
            kappa=1.65,
            n_PF=6,
            n_CS=5,
            r_cs=0,
            tk_cs=0,
            tfbnd=None,
            eqtype="SN",
            rtype="Normal",
            coilset=self.coilset,
        )

        # Get approximately the same finite difference grid size
        x_min, x_max = 4, 15
        z_min, z_max = -10, 13
        x_size = x_max - x_min
        z_size = z_max - z_min
        dx_new = a.eq.grid.dx * x_size / a.eq.grid.x_size
        dz_new = a.eq.grid.dz * z_size / a.eq.grid.z_size
        nx_new = int(x_size / dx_new)
        nz_new = int(z_size / dz_new)
        g = Grid(x_min, x_max, z_min, z_max, nx_new, nz_new)
        c.eq.reset_grid(g)
        c.constraints = XzTesting(f_s.x, f_s.z, -0 / (2 * np.pi), n=30)
        c.solve()
        apsi = a.eq.psi()
        cpsi = c.eq.psi()
        levels = np.linspace(np.amin(apsi), np.amax(apsi), 30)
        if tests.PLOTTING:
            f, ax = plt.subplots()
            ax.contour(c.eq.x, c.eq.z, cpsi, cmap="Greys", levels=levels)
            ax.contour(
                a.eq.x, a.eq.z, apsi, cmap="magma", levels=levels, linestyles="dashed"
            )
            ax.set_aspect("equal")
            plt.show()

        # reconstruct interpolated psi on a grid
        cpsiplasma = c.eq.psi_func(a.eq.grid.x_1d, a.eq.grid.z_1d)
        cpsicoilset = c.eq.coilset.psi(a.eq.grid.x, a.eq.grid.z)
        delta = np.amax(np.abs(apsi - (cpsiplasma + cpsicoilset)))
        psi_range = abs(np.amin(apsi)) + np.amax(apsi)
        rel_error = 100 * delta / psi_range
        assert rel_error <= 0.2, rel_error  # relative error in psi [%]


@pytest.mark.longrun
class TestAbInitioBreakdown:
    def test_breakdown(self):
        fp = get_BP_path("eqdsk", subfolder="data")
        fn = os.path.join(fp, "AR3d1_2015_04_v2_EOF_CSred_fine_final.eqdsk")
        eq = Equilibrium.from_eqdsk(fn)
        coilset = eq.coilset.copy()
        coilset.reset()
        # EU-DEMO1 2015 - PROCESS run 2LBJRY
        a = AbInitioEquilibriumProblem(
            9.072,
            5.667,
            3.1,
            19.6e6,
            1.107,
            li=0.8,
            delta=0.3333,
            kappa=1.65,
            n_PF=6,
            n_CS=5,
            r_cs=0,
            tk_cs=0,
            tfbnd=None,
            eqtype="SN",
            rtype="Normal",
            coilset=coilset,
        )
        a.coilset.assign_coil_materials("CS", "NbTi")  # Lo han hecho asi..
        a.breakdown(420e6, 350e6, 250e6)
        # No assert
        a.snapshots["Breakdown"].plot()


@pytest.mark.longrun
@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestFBI:
    def test_breakdown(self):
        fp = get_BP_path("eqdsk", subfolder="data")
        fn = os.path.join(fp, "AR3d1_2015_04_v2_EOF_CSred_fine_final.eqdsk")
        eq = Equilibrium.from_eqdsk(fn)
        coilset = eq.coilset.copy()
        coilset.reset()
        # EU-DEMO1 2015 - PROCESS run 2LBJRY
        currents = (
            np.array(
                [
                    12.38,
                    4.63,
                    -3.41,
                    4.34,
                    -3.2,
                    19.2,
                    28.07,
                    28.07,
                    57.14,
                    28.07,
                    20.18,
                ]
            )
            * 1e6
        )
        coilset.set_control_currents(currents)
        grid = Grid(7, 10, -3, 3, 50, 50)
        bd = Breakdown(coilset, grid, R_0=9)
        bd.psi_bd = 100
        bd.plot()


@pytest.mark.longrun
@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestForces:
    """
    Checks the forces and fields calculated against CREATE breakdown scenario
    """

    def test_create_bd(self):
        """
        EU-DEMO1 2015 - PROCESS run 2LBJRY
        """
        _coilset_setup(self)

        currents = (
            np.array(
                [
                    12.38,
                    4.63,
                    -3.41,
                    4.34,
                    -3.2,
                    19.2,
                    28.07,
                    28.07,
                    57.14,
                    28.07,
                    20.18,
                ]
            )
            * 1e6
        )

        self.coilset.set_control_currents(currents)
        self.coilset.mesh_coils(0.2)
        grid = Grid(0.1, 20, -15, 10, 20, 20)
        bd = Breakdown(self.coilset, grid, R_0=9)
        bd.set_breakdown_point(9.42, 0.16)

        # CREATE results 2015
        field = np.array(
            [5.17, 1.98, 1.24, 1.45, 0.94, 4.6, 11.38, 11.59, 11.9, 11.51, 9.77]
        )
        z_forces = np.array(
            [
                -409.1,
                -21.07,
                19.56,
                -4.58,
                -40.97,
                404.62,
                -820.35,
                -99.99,
                32.45,
                320.87,
                619.04,
            ]
        )

        bp2 = bd.force_field.calc_field(currents)[0]
        fz2 = bd.force_field.calc_force(currents)[0][:, 1] / 1e6

        f, ax = plt.subplots()

        self.coilset.plot(ax)
        bd.plot(ax, Bp=True)

        f, ax = plt.subplots(1, 2)

        def compare_plot(ax_, res_equilibria, res_create, title):

            labels = [mathify(nam) for nam in self.coil_names]

            index = np.arange(len(self.coil_names))
            w = 0.2
            ax_.bar(index, res_equilibria, width=w, label="equilibria", align="center")
            ax_.bar(index - w, res_create, width=w, label="CREATE", align="center")
            ax_.set_xticks(index + w / 2)
            ax_.set_xticklabels(labels)
            ax_.set_title(title)

        compare_plot(ax[0], bp2, field, mathify("B_p"))
        compare_plot(ax[1], fz2, z_forces, mathify("F_z"))
        plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
