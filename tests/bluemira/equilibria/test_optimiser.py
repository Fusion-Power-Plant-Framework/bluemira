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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import PF_COIL_NAME, Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.opt_problems import CoilsetPositionCOP
from bluemira.equilibria.optimiser import PositionOptimiser
from bluemira.geometry._deprecated_loop import Loop
from tests.bluemira.equilibria.setup_methods import _coilset_setup, _make_square


class TestPositionOptimiser:
    @classmethod
    def setup_class(cls):
        cls._coilset_setup = _coilset_setup

    def _track_setup(self):
        # Stripped from
        # BLUEPRINT.examples.equilibria.single_null
        fp = get_bluemira_path("geometry", subfolder="data")
        TF = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        TF = TF.offset(2.4)
        clip = np.where(TF.x >= 3.5)
        self.TF_track = Loop(TF.x[clip], z=TF.z[clip])
        self.TF_track.interpolate(200)

    def _coil_positioning_b4_update(self, pos_opt, positions):
        s, f = 0, 2
        for c_name, coil in self.coilset.coils.items():
            if c_name in ["PF_5", "PF_6"]:
                num = int(c_name.split("_")[1])
                assert self.pos_dict[c_name] == pos_opt.Rmap.L_to_xz(
                    num, positions[-4:][s:f]
                )
                s += 2
                f += 2

    def _coil_positioning(self, pos_opt, positions, coilset):

        for name, pos in zip(self.coil_names, positions[:4][::-1]):
            cpos = pos_opt.XLmap.L_to_xz(pos)
            coil = coilset.coils[name]
            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z)))

        CS_pos = pos_opt.XLmap.L_to_zdz(positions[4:9])
        for name, cpos in zip(self.coil_names[4:9], zip(*CS_pos)):
            coil = coilset.coils[name]

            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z, coil.dz)))

        for name, pos in zip(self.coil_names[9:], positions[9:].reshape(-1, 2)):
            num = int(name.split("_")[1])
            cpos = pos_opt.Rmap.L_to_xz(num, pos)
            coil = coilset.coils[name]
            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z)))

    def PFcoil_region_ordering(self, patch_loc, nlo_retv, pos_opt, eq):
        with patch(
            patch_loc + "nlopt.opt", name="nlopt.opt", return_value=nlo_retv
        ) as nlo:
            with patch(patch_loc + "process_NLOPT_result"):
                with patch.object(pos_opt, "update_positions"):

                    pos_opt.I_star = None
                    pos_opt(eq, constraints=None)
                    np.testing.assert_equal(nlo.call_args[0][1], self.no_coils)

                positions = nlo_retv.optimize.call_args[0][0]

                self._coil_positioning_b4_update(pos_opt, positions)

                with patch.object(pos_opt, "_optimise_currents"):
                    pos_opt.update_positions(positions)

                self._coil_positioning(pos_opt, positions, eq.coilset)

        np.testing.assert_equal(pos_opt.n_L, self.no_coils)

    def get_i_max(self, patch_loc, nlo_retv, pos_opt, eq):
        with patch(patch_loc + "nlopt.opt", name="nlopt.opt", return_value=nlo_retv):
            with patch(patch_loc + "process_NLOPT_result"):
                with patch.object(pos_opt, "update_positions"):
                    pos_opt.I_star = None
                    pos_opt(eq, None)

                    i_max_b4 = pos_opt._get_i_max()
                    pos_opt.flag_PFR = not pos_opt.flag_PFR
                    i_max_after = pos_opt._get_i_max()
        assert not np.array_equal(i_max_after, i_max_b4)

    @pytest.mark.parametrize("function_name", ["get_i_max", "PFcoil_region_ordering"])
    def test_optimiser(self, function_name):

        function = getattr(self, function_name)

        self._track_setup()
        self._coilset_setup(materials=True)

        self.coil_names = (
            self.coil_names[:4] + self.coil_names[6:] + self.coil_names[4:6]
        )
        regions = {}

        for i in range(2):
            coil = self.coilset.coils[f"PF_{i+5}"]
            mn = {"x": coil.x - 2 * coil.dx, "z": coil.z - 2 * coil.dz}
            mx = {"x": coil.x + 2 * coil.dx, "z": coil.z + 2 * coil.dz}
            regions[PF_COIL_NAME.format(i + 5)] = Loop(**_make_square(mn, mx))

        self.no_coils += len(regions)

        self.pos_dict = {}
        for name, coil in self.coilset.coils.items():
            self.pos_dict[name] = (coil.x, coil.z)

        # Stripped from
        # BLUEPRINT.equilibria.run.EquilibriumProblem.optimise_positions
        solenoid = self.coilset.get_solenoid()
        CS_x = solenoid.radius
        CS_zmin = solenoid.z_min
        CS_zmax = solenoid.z_max
        CS_gap = solenoid.gap

        # Stripped from
        # BLUEPRINT.examples.equilibria.single_null
        max_PF_current = 25e6  # [A]
        PF_Fz_max = 400e6  # [N]
        CS_Fz_sum = 300e6  # [N]
        CS_Fz_sep = 250e6  # [N]

        patch_loc = "bluemira.equilibria.optimiser."

        nlo_retv = MagicMock(name="opt.optimize()")

        eq = MagicMock(name="Equilibrium()")
        eq.coilset = self.coilset

        with patch(patch_loc + "FBIOptimiser"):

            pos_opt = PositionOptimiser(
                CS_x=CS_x,
                CS_zmin=CS_zmin,
                CS_zmax=CS_zmax,
                CS_gap=CS_gap,
                max_PF_current=max_PF_current,
                max_fields=self.coilset.get_max_fields(),
                PF_Fz_max=PF_Fz_max,
                CS_Fz_sum=CS_Fz_sum,
                CS_Fz_sep=CS_Fz_sep,
                psi_values=None,
                pfcoiltrack=self.TF_track,
                pf_coilregions=regions,
                pf_exclusions=None,
                CS=True,
                plot=False,
                gif=False,
            )

            function(patch_loc, nlo_retv, pos_opt, eq)


class TestCoilsetOptimiser:
    @classmethod
    def setup_class(cls):
        coil = Coil(
            x=1.5,
            z=6.0,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=1e-5,
            b_max=100,
            ctype="PF",
            name="PF_2",
        )
        circuit = SymmetricCircuit(coil)

        coil2 = Coil(
            x=4.0,
            z=10.0,
            current=2e6,
            dx=0.5,
            dz=0.33,
            j_max=5.0e-6,
            b_max=50.0,
            name="PF_1",
        )

        coil3 = Coil(
            x=4.0,
            z=20.0,
            current=7e6,
            dx=0.5,
            dz=0.33,
            j_max=None,
            b_max=50.0,
            name="PF_3",
        )
        cls.coilset = CoilSet([circuit, coil2, coil3])

        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        cls.pfregions = {}
        for coil in cls.coilset._ccoils:
            xu = coil.x + max_coil_shifts["x_shifts_upper"]
            xl = coil.x + max_coil_shifts["x_shifts_lower"]
            zu = coil.z + max_coil_shifts["z_shifts_upper"]
            zl = coil.z + max_coil_shifts["z_shifts_lower"]

            rect = Loop(x=[xl, xu, xu, xl, xl], z=[zl, zl, zu, zu, zl])

            cls.pfregions[coil.name] = rect

        cls.optimiser = CoilsetPositionCOP(cls.coilset, None, None, cls.pfregions)

    def test_modify_coilset(self):
        # Read
        coilset_state, substates = self.optimiser.read_coilset_state(
            self.coilset, self.optimiser.scale
        )
        # Modify vectors
        x, z, currents = np.array_split(coilset_state, substates)
        x += 1.1
        z += 0.6
        currents += 0.99
        updated_coilset_state = np.concatenate((x, z, currents))
        self.optimiser.set_coilset_state(
            self.optimiser.coilset, updated_coilset_state, self.optimiser.scale
        )

        coilset_state, substates = self.optimiser.read_coilset_state(
            self.coilset, self.optimiser.scale
        )
        state_x, state_z, state_i = np.array_split(coilset_state, substates)
        assert np.allclose(state_x, x)
        assert np.allclose(state_z, z)
        assert np.allclose(state_i, currents)

    def test_current_bounds(self):
        n_control_currents = len(self.coilset.get_control_currents())
        user_max_current = 2.0e9
        user_current_limits = (
            user_max_current * np.ones(n_control_currents) / self.optimiser.scale
        )
        coilset_current_limits = self.optimiser.coilset.get_max_currents(0.0)

        control_current_limits = np.minimum(user_current_limits, coilset_current_limits)
        bounds = (-control_current_limits, control_current_limits)

        assert n_control_currents == len(user_current_limits)
        assert n_control_currents == len(coilset_current_limits)

        optimiser_current_bounds = self.optimiser.get_current_bounds(
            self.optimiser.coilset, user_max_current, self.optimiser.scale
        )
        assert np.allclose(bounds[0], optimiser_current_bounds[0])
        assert np.allclose(bounds[1], optimiser_current_bounds[1])

        # print(self.optimiser.coilset.get_max_currents(0.0))
        # print(self.optimiser.get_current_bounds(10.0) / self.optimiser.scale)

        # self.optimiser.get_current_bounds()

        # optimiser_maxima = 0.9
        # i_max = self.coilset.get_max_currents(max_currents)


if __name__ == "__main__":
    pytest.main([__file__])
