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

from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.radiation_profile import (
    CoreRadiation,
    Radiation,
    STScrapeOffLayerRadiation,
    calculate_line_radiation_loss,
    calculate_z_species,
    electron_density_and_temperature_sol_decay,
    exponential_decay,
    gaussian_decay,
    ion_front_distance,
    radiative_loss_function_values,
    random_point_temperature,
    target_temperature,
    upstream_temperature,
    x_point_temperature,
)

TEST_PATH = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")


class TestRadiation:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        filename = os.sep.join([EQ_PATH, eq_name])
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "DN_fw_shape.json"
        filename = os.sep.join([TEST_PATH, fw_name])
        cls.fw = Loop.from_file(filename)

        p_solver_params = ParameterFrame()
        solver = ChargedParticleSolver(p_solver_params, eq, dx_mp=0.001)
        solver.analyse(cls.fw)
        cls.solver = solver

        # fmt: off
        plasma_params = ParameterFrame([
            ["kappa", "Elongation", 3, "dimensionless", None, "Input"],
        ])
        # fmt: on

        impurity_content = {
            "H": [8.6794e-01],
            "He": [5.1674e-02],
        }

        t_ref_name = "T_ref.npy"
        l_ref_name = "L_ref_He.npy"
        t_filename = os.sep.join([TEST_PATH, t_ref_name])
        l_filename = os.sep.join([TEST_PATH, l_ref_name])
        impurity_data = {
            "He": {"T_ref": np.load(t_filename), "L_ref": np.load(l_filename)},
        }

        cls.rad = Radiation(cls.solver, plasma_params)
        cls.st_core = CoreRadiation(
            cls.solver,
            impurity_content,
            impurity_data,
            plasma_params,
        )
        cls.st_sol = STScrapeOffLayerRadiation(
            cls.solver,
            impurity_content,
            impurity_data,
            plasma_params,
            cls.fw,
        )

    def test_collect_flux_tubes(self):
        psi = np.linspace(1, 1.5, 5)
        ft = self.rad.collect_flux_tubes(psi)
        assert len(ft) == 5

    def test_core_flux_tube_pol_t(self):
        flux_tube = self.solver.eq.get_flux_surface(0.99)
        te = self.rad.flux_tube_pol_t(flux_tube, 100, True)
        assert te[0] == te[-1]
        assert len(te) == len(flux_tube)

    def test_core_flux_tube_pol_n(self):
        flux_tube = self.solver.eq.get_flux_surface(0.99)
        ne_mp = 2e20
        ne = self.rad.flux_tube_pol_n(flux_tube, ne_mp, True)
        assert ne[0] == ne[-1]
        assert len(ne) == len(flux_tube)

    def test_key_temperatures(self):
        self.t_u, q_u = upstream_temperature(
            self.rad.plasma_params.A,
            self.rad.plasma_params.R_0,
            self.rad.plasma_params.kappa,
            self.rad.plasma_params.q_95,
            self.rad.plasma_params.fw_lambda_q_far_omp,
            self.rad.plasma_params.P_rad,
            self.solver.eq,
            self.rad.r_sep_omp,
            self.rad.z_mp,
            self.rad.plasma_params.k_0,
            self.fw,
        )
        t_tar_det = target_temperature(
            q_u,
            self.t_u,
            self.rad.plasma_params.lfs_p_fraction,
            self.rad.plasma_params.div_p_sharing,
            self.rad.plasma_params.n_el_0,
            self.rad.plasma_params.gamma_sheath,
            self.rad.plasma_params.eps_cool,
            self.rad.plasma_params.f_ion_t,
        )
        t_tar_no_det = target_temperature(
            3e10,
            self.t_u,
            self.rad.plasma_params.lfs_p_fraction,
            self.rad.plasma_params.div_p_sharing,
            self.rad.plasma_params.n_el_0,
            self.rad.plasma_params.gamma_sheath,
            self.rad.plasma_params.eps_cool,
            self.rad.plasma_params.f_ion_t,
        )
        t_x = x_point_temperature(
            q_u,
            self.t_u,
            self.solver.eq,
            self.rad.points["x_point"]["x"],
            self.rad.points["x_point"]["z_low"],
            self.rad.plasma_params.k_0,
            self.rad.r_sep_omp,
            self.rad.z_mp,
            self.fw,
        )
        assert self.t_u < 5e-1
        assert t_tar_det < self.t_u * 1e-1
        assert self.t_u > t_x > t_tar_det
        assert t_tar_no_det * 1e-3 > self.rad.plasma_params.f_ion_t

    def test_sol_decay(self):
        t_u = self.rad.plasma_params.T_el_sep
        n_u = self.rad.plasma_params.n_el_sep
        decayed_t, decayed_n = electron_density_and_temperature_sol_decay(
            t_u,
            n_u,
            self.rad.plasma_params.fw_lambda_q_near_omp,
            self.rad.plasma_params.fw_lambda_q_far_omp,
            self.solver.dx_omp,
            self.solver.dx_imp,
        )
        assert decayed_t[0] > decayed_t[-1]
        assert decayed_n[0] > decayed_n[-1]

    def test_gaussian_decay(self):
        decayed_val = gaussian_decay(10, 1, 50)
        gap_1 = decayed_val[0] - decayed_val[1]
        gap_2 = decayed_val[1] - decayed_val[2]
        gap_3 = decayed_val[-2] - decayed_val[-1]
        assert gap_1 < gap_2 < gap_3

    def test_exponential_decay(self):
        decayed_val = exponential_decay(10, 1, 50, True)
        gap_1 = decayed_val[0] - decayed_val[1]
        gap_2 = decayed_val[1] - decayed_val[2]
        gap_3 = decayed_val[-2] - decayed_val[-1]
        assert gap_1 > gap_2 > gap_3

    def test_ion_front_distance(self):
        distance = ion_front_distance(
            6, -9, self.solver.eq, self.rad.points["x_point"]["z_low"], 1e-3, 1, 1, 2e20
        )
        assert distance is not None
        assert np.round(distance, 1) == 2.6

    def test_calculate_z_species(self):
        t_ref = np.array([0, 10])
        z_ref = np.array([10, 20])
        frac = 0.1
        t_test = 5
        z = calculate_z_species(t_ref, z_ref, frac, t_test)
        assert z == 22.5

    def test_rho_core(self):
        rho_core = self.st_core.collect_rho_core_values()
        assert rho_core[0] == 0
        assert rho_core[-1] < 1

    def test_core_electron_density_temperature_profile(self):
        rho = np.linspace(0, 1, 10)
        ne_core, te_core = self.st_core.core_electron_density_temperature_profile(rho)
        assert len(ne_core) == len(te_core)
        assert ne_core[0] > ne_core[-1]
        assert te_core[0] > te_core[-1]

    def test_rho_sol(self):
        rho_sol = self.st_sol.collect_rho_sol_values()
        assert rho_sol[0] > 1

    def test_rad_region_extention(self):
        z_main_low, z_pfr_low = self.st_sol.x_point_radiation_z_ext()
        z_main_up, z_pfr_up = self.st_sol.x_point_radiation_z_ext(low_div=False)
        assert z_main_low > z_pfr_low
        assert z_main_up < z_pfr_up

    def test_core_plot_1d(self):
        with patch.object(self.st_core, "plot_1d_profile") as plot_mock:
            self.st_core.build_mp_rad_profile()
        assert len(plot_mock.call_args[0][0]) == len(self.st_core.rho_core)
        assert plot_mock.call_args[0][0][-1] == self.st_core.rho_core[-1]

    def test_core_plot_2d(self):
        flux_tubes = self.st_core.collect_flux_tubes(self.st_core.rho_core)
        with patch.object(self.st_core, "plot_2d_map") as plot_mock:
            self.st_core.build_core_radiation_map()
        assert len(plot_mock.call_args[0][0]) == len(flux_tubes)
        assert len(plot_mock.call_args[0][0][0]) == len(flux_tubes[0])

    def test_mp_electron_density_temperature_profiles(self):
        te_sol, ne_sol = self.st_sol.mp_electron_density_temperature_profiles()
        assert len(te_sol) == len(ne_sol)
        assert te_sol[0] < self.st_sol.plasma_params.T_el_sep
        assert te_sol[0] > te_sol[-1]

    def test_sol_flux_tube_pol_n(self):
        tube = self.st_sol.flux_tubes_lfs_low[0].loop
        ne_mp = 1.8e20
        n_out = 2e22
        n_tar = 2e20
        rec_i = np.arange(10, 20)
        ne = self.st_sol.flux_tube_pol_n(
            tube, ne_mp, n_rad_out=n_out, rec_i=rec_i, n_tar=n_tar
        )
        assert ne[0] == ne[9]
        assert ne[9] < ne[10]
        assert ne[10] > ne[20]

    def test_build_sector_profiles(self):
        tubes = self.st_sol.flux_tubes_lfs_low
        x_strike = self.st_sol.x_strike_lfs
        z_strike = self.st_sol.z_strike_lfs

        with pytest.raises(
            BuilderError, match="Required recycling region extention: rec_ext"
        ):
            self.st_sol.build_sector_profiles(tubes, x_strike, z_strike, 0.5, self.fw)

        with pytest.raises(
            BuilderError, match="Required extention towards pfr: pfr_ext"
        ):
            self.st_sol.build_sector_profiles(
                tubes, x_strike, z_strike, 0.5, self.fw, x_point_rad=True
            )

    def test_ST_sectors(self):
        # sol flux tubes temperature and density
        t_n_profiles = self.st_sol.build_sol_profiles(self.fw)
        # upstream temperature
        flux_tube_u = t_n_profiles[0][0][0][0]
        # target temperature
        flux_tube_t = t_n_profiles[0][0][0][-1]
        # test sol flux tube poloidal temperature
        assert flux_tube_u != flux_tube_t
        rad_profiles = self.st_sol.build_sol_rad_distribution(*t_n_profiles)
        # number of secotors
        assert len(rad_profiles) == 4
        # number of impurities
        assert len(rad_profiles[0]) == 1
        # number of flux tubes
        assert len(rad_profiles[0][0]) != 0
        # number of radiative points
        assert len(rad_profiles[0][0][0]) != 0
        # sol radiation map builder
        with patch.object(self.st_sol, "plot_2d_map") as plot_mock:
            self.st_sol.build_sol_radiation_map(*rad_profiles, self.fw)
        assert len(plot_mock.call_args[0][0]) == 4
        assert len(plot_mock.call_args[0][1]) == 4
        assert len(plot_mock.call_args[0][0][0]) == len(self.st_sol.flux_tubes_lfs_low)
