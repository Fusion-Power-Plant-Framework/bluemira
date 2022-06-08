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

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.radiation_profile import (
    Radiation,
    ScrapeOffLayer,
    STScrapeOffLayer,
    TwoPointModelTools,
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

        # fmt: off
        rad_params = ParameterFrame([
            ["p_sol", "power entering the SoL", 300e6, "W", None, "Input"],
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
        cls.tpm = TwoPointModelTools(cls.solver, plasma_params, rad_params)
        cls.sol = ScrapeOffLayer(cls.solver, plasma_params)
        cls.st_sol = STScrapeOffLayer(
            cls.solver,
            impurity_content,
            impurity_data,
            plasma_params,
            rad_params,
            cls.fw,
        )

    def test_core_flux_tube_pol_t(self):
        flux_tube = self.solver.eq.get_flux_surface(0.99)
        te = self.rad.flux_tube_pol_t(flux_tube, 100, True)
        assert te[0] == te[-1]
        assert len(te) == len(flux_tube)

    def test_key_temperatures(self):
        self.t_u, q_u = self.tpm.upstream_temperature(self.fw)
        t_tar = self.tpm.target_temperature(q_u, self.t_u)
        t_x = self.tpm.x_point_temperature(q_u, self.t_u, self.fw)
        assert self.t_u < 5e-1
        assert t_tar < self.t_u * 1e-1
        assert self.t_u > t_x > t_tar

    def test_sol_decay(self):
        t_u = self.tpm.plasma_params.T_el_sep
        n_u = self.tpm.plasma_params.n_el_sep
        decayed_t, decayed_n = self.tpm.electron_density_and_temperature_sol_decay(
            t_u, n_u
        )
        assert decayed_t[0] > decayed_t[-1]
        assert decayed_n[0] > decayed_n[-1]

    def test_gaussian_decay(self):
        decayed_val = self.tpm.gaussian_decay(10, 1, 50)
        gap_1 = decayed_val[0] - decayed_val[1]
        gap_2 = decayed_val[1] - decayed_val[2]
        gap_3 = decayed_val[-2] - decayed_val[-1]
        assert gap_1 < gap_2 < gap_3

    def test_exponential_decay(self):
        decayed_val = self.tpm.exponential_decay(10, 1, 50, True)
        gap_1 = decayed_val[0] - decayed_val[1]
        gap_2 = decayed_val[1] - decayed_val[2]
        gap_3 = decayed_val[-2] - decayed_val[-1]
        assert gap_1 > gap_2 > gap_3

    def test_rad_region_extention(self):
        z_main_low, z_pfr_low = self.sol.x_point_radiation_z_ext()
        z_main_up, z_pfr_up = self.sol.x_point_radiation_z_ext(low_div=False)
        assert z_main_low > z_pfr_low
        assert z_main_up < z_pfr_up

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
