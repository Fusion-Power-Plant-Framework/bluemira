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
import numpy as np
from matplotlib import pyplot as plt
import pytest
import tests
from tests.BLUEPRINT.equilibria.setup_methods import _coilset_setup
from BLUEPRINT.utilities.plottools import mathify
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.equilibria.equilibrium import Breakdown, Equilibrium
from BLUEPRINT.equilibria.constraints import XzTesting
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface
from BLUEPRINT.equilibria.optimiser import BreakdownOptimiser, FBIOptimiser
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.profiles import BetaIpProfile
from BLUEPRINT.equilibria.solve import PicardLiAbsIterator


@pytest.mark.longrun
class TestCREATEBenchmark:
    """
    Performs a comparison of the Breakdown optimisation algorithm with a
    known CREATE result for the EU-DEMO1 2015 (April)
    IDM reference: DEMO_D_2AQ5GP
    """

    @classmethod
    def setup_class(cls):
        # Parameter set
        cls.R_0 = 8.938
        cls.B_t = 5.63
        cls.I_p = 19.6
        cls.l_i = 0.8
        cls.beta_p = 1.107
        cls.k_95 = 1.59
        cls.d_95 = 0.33

        # Constraints
        cls.PF_Fz_max = 450e6
        cls.CS_Fz_sum = 300e6
        cls.CS_Fz_sep = 350e6

        _coilset_setup(cls, materials=True)

    def test_breakdown(self):
        self.coilset.mesh_coils(0.3)

        # Here we set up as close to identical an optimisation problem as I can
        # figure out, based on the report. It is entirely possible CREATE use more
        # sophisticated Breakdown constraints (knowing M. Mattei).

        # We're using the same CoilSet as previously optimised by CREATE here, so
        # no coil position or size optimisation is performed.
        optimiser = BreakdownOptimiser(
            9.42,
            0.16,
            r_zone=1.5,  # Not sure about this but hey
            b_zone_max=0.003,
            max_currents=self.coilset.get_max_currents(0),
            max_fields=self.coilset.get_max_fields(),
            PF_Fz_max=self.PF_Fz_max,
            CS_Fz_sum=self.CS_Fz_sum,
            CS_Fz_sep=self.CS_Fz_sep,
        )

        grid = Grid(0.1, self.R_0 * 2, -1.5 * self.R_0, 1.5 * self.R_0, 100, 100)
        bd = Breakdown(self.coilset, grid, psi=None, R_0=self.R_0)
        bd.set_breakdown_point(9.42, 0.16)
        self.coilset.set_control_currents(self.coilset.get_max_currents(0))
        bd._remap_greens()  # Need to reset ForceField...
        currents = optimiser(bd)
        self.coilset.set_control_currents(currents)
        bd._remap_greens()

        # Now we set up a simple Breakdown object with CREATE's current vector
        # (no optimisation performed here)
        create_coilset = self.coilset.copy()
        currents = 1e6 * np.array(
            [12.38, 4.63, -3.41, 4.34, -3.2, 19.2, 28.07, 28.07, 57.14, 28.07, 20.18]
        )
        create_coilset.set_control_currents(currents, update_size=False)
        create_bd = Breakdown(
            create_coilset, grid, psi=create_coilset.psi(grid.x, grid.z), R_0=self.R_0
        )

        if tests.PLOTTING:
            plot_defaults()
            f, ax = plt.subplots(1, 2)
            f.tight_layout(pad=3)
            bd.plot(ax[0])
            self.coilset.plot(ax[0])
            create_bd.plot(ax[1])
            create_coilset.plot(ax[1])
            plt.show()

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

        def compare_plot(ax_, res_equilibria, res_create, title):
            labels = [mathify(nam) for nam in self.coil_names]

            index = np.arange(len(self.coil_names))
            w = 0.2
            ax_.bar(index, res_equilibria, width=w, label="BLUEPRINT", align="center")
            ax_.bar(index - w, res_create, width=w, label="CREATE", align="center")
            ax_.set_xticks(index + w / 2)
            ax_.set_xticklabels(labels)
            ax_.set_title(title)
            ax_.legend()

        if tests.PLOTTING:
            f, ax = plt.subplots(1, 2)
            compare_plot(ax[0], bp2, field, mathify("B_p"))
            compare_plot(ax[1], fz2, z_forces, mathify("F_z"))
            plt.show()

        # Check that the breakdown fluxes are similar +/- 1 V.s
        assert np.round(bd.breakdown_psi, 0) == np.round(create_bd.breakdown_psi, 0)

    def test_start_of_flattop(self):
        path = get_BP_path("eqdsk", subfolder="data")
        filename = "AR3d1_2015_04_v2_SOF_CSred_fine_final.eqdsk"
        filename = os.sep.join([path, filename])
        create_sof = Equilibrium.from_eqdsk(filename, load_large_file=True)
        profiles = BetaIpProfile(self.beta_p, self.I_p * 1e6, self.R_0, self.B_t)

        eq_dict = EQDSKInterface().read(filename)
        eq_dictgrid = eq_dict.copy()
        eq_dictgrid["nx"] = 200
        eq_dictgrid["nz"] = 200
        grid = Grid.from_eqdict(eq_dictgrid)

        sof = Equilibrium(
            self.coilset,
            grid,
            Ip=self.I_p,
            li=self.l_i,
            RB0=[self.R_0, self.B_t],
            profiles=None,
        )
        sof.solve(profiles)
        constraints = XzTesting(
            eq_dict["xbdry"], eq_dict["zbdry"], 143 / 2 / np.pi, n=150
        )

        optimiser = FBIOptimiser(
            self.coilset.get_max_fields(),
            self.PF_Fz_max,
            self.CS_Fz_sum,
            self.CS_Fz_sep,
        )
        optimiser.update_current_constraint(self.coilset.get_max_currents(0))

        iterator = PicardLiAbsIterator(sof, profiles, constraints, optimiser)
        iterator()

        return sof, constraints, create_sof


if __name__ == "__main__":
    pytest.main([__file__])
