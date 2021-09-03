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
import pytest
import os
import numpy as np
from BLUEPRINT.systems.firstwall import FluxSurface, FirstWallSN, FirstWallDN
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.equilibria.equilibrium import Equilibrium


class TestFluxSurface:
    @classmethod
    def setup_class(cls):
        read_path = get_BP_path("equilibria", subfolder="data/BLUEPRINT")
        eq_name = "EU-DEMO_EOF.json"
        eq_name = os.sep.join([read_path, eq_name])
        eq = Equilibrium.from_eqdsk(eq_name)
        cls.fluxsurface = FluxSurface(eq, 12, 0)

    def test_polar_coordinates(self):
        x = [15, 5]
        z = [1, 1]
        theta = self.fluxsurface.polar_coordinates(x, z)
        assert theta[0] > 7 and theta[0] < 8
        assert theta[1] > 161 and theta[1] < 163

    def test_assign_lfs_hfs_sn(self):
        x = np.array([8, 10, 5, 9])
        z = np.array([-1, -5, 2, 2])
        th = np.array([280, 290, 260, 250])
        p_side = self.fluxsurface.assign_lfs_hfs_sn(x, z, th)
        assert len(p_side[0]) != 0
        assert len(p_side[1][0]) == 2

    def test_assign_top_bottom(self):
        x = np.array([8, 10, 5, 9])
        z = np.array([1, 5, -2, -7])
        p_loc = self.fluxsurface.assign_top_bottom(x, z)
        assert len(p_loc[0]) != 0
        assert len(p_loc[1]) == 2

    def test_find_first_intersection_lfs_sn(self):
        x = np.array([8, 10, 12, 9])
        z = np.array([-1, -5, -2, -5])
        th = np.array([280, 5, 100, 279])
        first = self.fluxsurface.find_first_intersection_lfs_sn(x, z, th)
        assert first[0] == 8
        assert first[1] == -1

    def test_calculate_q_par_omp(self):
        qpar = self.fluxsurface.calculate_q_par_omp(0, 0, 100, 100)
        assert qpar == 0


class TestFirstWallSN:
    @classmethod
    def setup_class(cls):
        read_path = get_BP_path("equilibria", subfolder="data/BLUEPRINT")
        eq_name = "EU-DEMO_EOF.json"
        eq_name = os.sep.join([read_path, eq_name])
        eq = Equilibrium.from_eqdsk(eq_name)
        cls.firstwall = FirstWallSN(FirstWallSN.default_params, {"equilibrium": eq})

    def test_make_preliminary_profile(self):
        prof = self.firstwall.make_preliminary_profile()
        assert hasattr(prof, "x")

    def test_make_divertor(self):
        fw_loop = self.firstwall.make_preliminary_profile()
        div = self.firstwall.make_divertor(fw_loop)
        assert div[0].x[0] == div[0].x[-1]
        assert div[0].z[0] == div[0].z[-1]

    def test_q_parallel_calculation(self):
        qpar = self.firstwall.q_parallel_calculation()
        assert len(self.firstwall.flux_surfaces) == len(qpar)
        for i in range(len(qpar), -1):
            assert qpar[i] > qpar[i + 1]


class TestFirstWallDN:
    @classmethod
    def setup_class(cls):
        read_path = get_BP_path("BLUEPRINT/equilibria/test_data", subfolder="tests")
        eq_name = "DN-DEMO_eqref.json"
        eq_name = os.sep.join([read_path, eq_name])
        eq = Equilibrium.from_eqdsk(eq_name)
        cls.firstwall = FirstWallDN(FirstWallDN.default_params, {"equilibrium": eq})

    def test_make_preliminary_profile(self):
        prof = self.firstwall.make_preliminary_profile()
        assert len(prof.x) == len(prof.z)
        assert prof.x[0] == prof.x[-1]

    def test_make_divertor_outer_target(self):
        tar_out = self.firstwall.make_divertor_outer_target()
        assert tar_out[0][0] > self.firstwall.points["x_point"]["x"]
        assert tar_out[0][0] < tar_out[1][0]

    def test_make_divertor_inner_target(self):
        tar_in = self.firstwall.make_divertor_inner_target()
        assert tar_in[0][0] < self.firstwall.points["x_point"]["x"]
        assert tar_in[0][0] > tar_in[1][0]

    def test_make_flux_surfaces(self):
        assert hasattr(self.firstwall, "flux_surface_hfs")
        assert hasattr(self.firstwall, "flux_surface_hfs")
        assert len(self.firstwall.flux_surface_hfs) < len(
            self.firstwall.flux_surface_lfs
        )

    def test_q_parallel_calculation(self):
        qpar = self.firstwall.q_parallel_calculation()
        assert len(qpar[0]) == len(self.firstwall.flux_surface_lfs)

    def test_modify_fw_profile(self):
        profile = self.firstwall.profile
        prof_up = self.firstwall.modify_fw_profile(profile, 11.5, -2.5, 0.3)
        assert prof_up.x[0] == profile.x[0]
        assert prof_up.x[-1] == profile.x[-1]
        assert len(prof_up.x) == len(profile.x)


if __name__ == "__main__":
    pytest.main([__file__])
