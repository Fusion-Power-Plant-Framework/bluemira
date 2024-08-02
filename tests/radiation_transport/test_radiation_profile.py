# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest
from eqdsk.models import Sign

from bluemira.base import constants
from bluemira.base.constants import raw_uc
from bluemira.base.file import get_bluemira_path
from bluemira.codes.process import api
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.midplane_temperature_density import (
    collect_rho_core_values,
    midplane_profiles,
)
from bluemira.radiation_transport.radiation_profile import RadiationSource
from bluemira.radiation_transport.radiation_tools import (
    DetectedRadiation,
    FirstWallRadiationSolver,
    electron_density_and_temperature_sol_decay,
    grid_interpolator,
    interpolated_field_values,
    ion_front_distance,
    linear_interpolator,
    make_wall_detectors,
    pfr_filter,
    radiative_loss_function_values,
    target_temperature,
    upstream_temperature,
)

TEST_PATH = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")


@pytest.mark.skipif(not api.ENABLED, reason="PROCESS is not installed on the system.")
class TestCoreRadiation:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        filename = Path(EQ_PATH, eq_name)
        eq = Equilibrium.from_eqdsk(filename, from_cocos=3, qpsi_sign=Sign.NEGATIVE)
        fw_name = "DN_fw_shape.json"
        filename = Path(TEST_PATH, fw_name)
        fw_shape = Coordinates.from_json(filename)

        midplane_params = {
            "alpha_n": {"value": 1.15, "unit": "dimensionless"},
            "alpha_t": {"value": 1.905, "unit": "dimensionless"},
            "n_e_0": {"value": 21.93e19, "unit": "1/m^3"},
            "n_e_ped": {"value": 8.117e19, "unit": "1/m^3"},
            "n_e_sep": {"value": 1.623e19, "unit": "1/m^3"},
            "rho_ped_n": {"value": 0.94, "unit": "dimensionless"},
            "rho_ped_t": {"value": 0.976, "unit": "dimensionless"},
            "n_points_core_95": {"value": 30, "unit": "dimensionless"},
            "n_points_core_99": {"value": 15, "unit": "dimensionless"},
            "n_points_mantle": {"value": 10, "unit": "dimensionless"},
            "t_beta": {"value": 2.0, "unit": "dimensionless"},
            "T_e_0": {"value": 21.442, "unit": "keV"},
            "T_e_ped": {"value": 5.059, "unit": "keV"},
            "T_e_sep": {"value": 0.16, "unit": "keV"},
        }
        cls.params = {
            "sep_corrector": {"value": 5e-3, "unit": "dimensionless"},
            "det_t": {"value": 0.0015, "unit": "keV"},
            "eps_cool": {"value": 25.0, "unit": "eV"},
            "f_ion_t": {"value": 0.01, "unit": "keV"},
            "fw_lambda_q_near_omp": {"value": 0.003, "unit": "m"},
            "fw_lambda_q_far_omp": {"value": 0.1, "unit": "m"},
            "fw_lambda_q_near_imp": {"value": 0.003, "unit": "m"},
            "fw_lambda_q_far_imp": {"value": 0.1, "unit": "m"},
            "gamma_sheath": {"value": 7.0, "unit": "dimensionless"},
            "k_0": {"value": 2000.0, "unit": "dimensionless"},
            "lfs_p_fraction": {"value": 0.9, "unit": "dimensionless"},
            "P_sep": {"value": 100, "unit": "MW"},
            "theta_inner_target": {"value": 5.0, "unit": "deg"},
            "theta_outer_target": {"value": 5.0, "unit": "deg"},
            **midplane_params,
        }

        cls.config = {
            "f_imp_core": {"H": 1e-1, "He": 1e-2, "Xe": 1e-4, "W": 1e-5},
            "f_imp_sol": {"H": 0, "He": 0, "Ar": 0.003, "Xe": 0, "W": 0},
        }

        profiles = midplane_profiles(params=midplane_params)

        source = RadiationSource(
            eq=eq,
            firstwall_shape=fw_shape,
            params=cls.params,
            midplane_profiles=profiles,
            core_impurities=cls.config["f_imp_core"],
            sol_impurities=cls.config["f_imp_sol"],
        )
        source.analyse(firstwall_geom=fw_shape)
        source.rad_map(fw_shape)

        cls.profiles = profiles
        cls.source = source
        cls.fw_shape = fw_shape

    def test_collect_flux_tubes(self):
        psi = np.linspace(1, 1.5, 5)
        ft = self.source.core_rad.collect_flux_tubes(psi)
        assert len(ft) == 5

    def test_rho_core(self):
        rho_ped = (
            self.params["rho_ped_n"]["value"] + self.params["rho_ped_t"]["value"]
        ) / 2
        rho_core = collect_rho_core_values(rho_ped, 30, 15, 10)
        assert rho_core[0] > 0
        assert rho_core[-1] < 1

    def test_core_electron_density_temperature_profile(self):
        ne_core = self.profiles.ne
        te_core = self.profiles.te
        psi_n = self.profiles.psi_n
        assert len(ne_core) == len(te_core) == len(psi_n)

        # Ensure values are within expected ranges
        assert np.all(ne_core >= self.params["n_e_sep"]["value"])
        assert np.all(te_core >= self.params["T_e_sep"]["value"])
        assert np.all(ne_core <= self.params["n_e_0"]["value"])
        assert np.all(te_core <= self.params["T_e_0"]["value"])

        # Ensure the density and temperature are decreasing towards the edge
        assert ne_core[0] > ne_core[-1]
        assert te_core[0] > te_core[-1]

    def test_calculate_mp_radiation_profile(self):
        self.source.core_rad.calculate_mp_radiation_profile()
        rad_tot = np.sum(np.array(self.source.core_rad.rad_mp, dtype=object), axis=0)
        assert len(self.source.core_rad.rad_mp) == 4
        assert rad_tot[0] > rad_tot[-1]

    def test_core_flux_tube_pol_t(self):
        flux_tube = self.source.eq.get_flux_surface(0.99)
        te = self.source.core_rad.flux_tube_pol_t(flux_tube, 100, core=True)
        assert te[0] == te[-1]
        assert len(te) == len(flux_tube)

    def test_core_flux_tube_pol_n(self):
        flux_tube = self.source.eq.get_flux_surface(0.99)
        ne_mp = 2e20
        ne = self.source.core_rad.flux_tube_pol_n(flux_tube, ne_mp, core=True)
        assert ne[0] == ne[-1]
        assert len(ne) == len(flux_tube)

    def test_mp_electron_density_temperature_profiles(self):
        te_sol_omp, ne_sol_omp = (
            self.source.sol_rad.mp_electron_density_temperature_profiles()
        )
        te_sol_imp, ne_sol_imp = (
            self.source.sol_rad.mp_electron_density_temperature_profiles(omp=False)
        )
        assert te_sol_omp[0] > te_sol_omp[-1]
        assert ne_sol_omp[0] > ne_sol_omp[-1]
        assert te_sol_imp[0] > te_sol_imp[-1]
        assert ne_sol_imp[0] > ne_sol_imp[-1]

    def test_key_temperatures(self):
        t_u = upstream_temperature(
            b_pol=self.source.sol_rad.b_pol_sep_omp,
            b_tot=self.source.sol_rad.b_tot_sep_omp,
            lambda_q_near=self.source.params.fw_lambda_q_near_omp.value,
            p_sol=self.source.params.P_sep.value,
            eq=self.source.eq,
            r_sep_mp=self.source.sol_rad.r_sep_omp,
            z_mp=self.source.sol_rad.z_mp,
            k_0=self.source.params.k_0.value,
            firstwall_geom=self.fw_shape,
        )
        t_u_eV = constants.raw_uc(t_u, "keV", "eV")
        f_ion_t = self.source.params.f_ion_t.value_as("eV")

        t_tar_det = target_temperature(
            self.source.params.P_sep.value,
            t_u_eV,
            self.source.params.n_e_sep.value,
            self.source.params.gamma_sheath.value,
            self.source.params.eps_cool.value_as("eV"),
            f_ion_t,
            self.source.sol_rad.b_pol_out_tar,
            self.source.sol_rad.b_pol_sep_omp,
            self.source.params.theta_outer_target.value,
            self.source.sol_rad.r_sep_omp,
            self.source.sol_rad.x_strike_lfs,
            self.source.params.fw_lambda_q_near_omp.value,
            self.source.sol_rad.b_tot_out_tar,
        )

        t_tar_no_det = target_temperature(
            self.source.params.P_sep.value,
            t_u_eV,
            3e10,
            self.source.params.gamma_sheath.value,
            self.source.params.eps_cool.value_as("eV"),
            f_ion_t,
            self.source.sol_rad.b_pol_out_tar,
            self.source.sol_rad.b_pol_sep_omp,
            self.source.params.theta_outer_target.value,
            self.source.sol_rad.r_sep_omp,
            self.source.sol_rad.x_strike_lfs,
            self.source.params.fw_lambda_q_near_omp.value,
            self.source.sol_rad.b_tot_out_tar,
        )
        assert t_u < 5e-1
        assert t_tar_det <= self.source.params.f_ion_t.value_as("eV")
        assert t_tar_no_det > self.source.params.f_ion_t.value_as("eV")

    def test_sol_decay(self):
        t_u = self.source.params.T_e_sep.value_as("keV")
        n_u = self.source.params.n_e_sep.value
        decayed_t, decayed_n = electron_density_and_temperature_sol_decay(
            t_u,
            n_u,
            self.source.params.fw_lambda_q_near_omp.value,
            self.source.params.fw_lambda_q_far_omp.value,
            self.source.dx_omp,
        )
        assert decayed_t[0] > decayed_t[-1]
        assert decayed_n[0] > decayed_n[-1]

    def test_ion_front_distance(self):
        distance = ion_front_distance(
            6,
            -9,
            self.source.eq,
            self.source.sol_rad.points["x_point"]["z_low"],
            1e-3,
            1,
            1,
            2e20,
        )
        assert distance is not None
        assert distance == pytest.approx(2.619, rel=1e-3)

    def test_radiation_region_boundary(self):
        low_z_main, low_z_pfr = self.source.sol_rad.x_point_radiation_z_ext()
        up_z_main, up_z_pfr = self.source.sol_rad.x_point_radiation_z_ext(low_div=False)
        assert low_z_main > low_z_pfr
        assert up_z_main < up_z_pfr
        in_x_lfs, in_z_low, out_x_lfs, out_z_low = (
            self.source.sol_rad.radiation_region_ends(low_z_main, low_z_pfr)
        )
        _, in_z_up, _, out_z_up = self.source.sol_rad.radiation_region_ends(
            up_z_main, up_z_pfr
        )
        in_x_hfs, _, out_x_hfs, _ = self.source.sol_rad.radiation_region_ends(
            low_z_main, low_z_pfr, lfs=False
        )
        assert in_x_lfs > self.source.sol_rad.points["x_point"]["x"]
        assert out_x_lfs > self.source.sol_rad.points["x_point"]["x"]
        assert in_x_hfs < self.source.sol_rad.points["x_point"]["x"]
        assert out_x_hfs < self.source.sol_rad.points["x_point"]["x"]
        assert in_z_low > out_z_low
        assert in_z_up < out_z_up

    def test_tar_electron_densitiy_temperature_profiles(self):
        ne_array = np.linspace(1e20, 1e19, 5)
        te_array = np.linspace(15, 8, 5)
        te_det, ne_det = self.source.sol_rad.tar_electron_densitiy_temperature_profiles(
            ne_array, te_array, detachment=True
        )
        te_att, ne_att = self.source.sol_rad.tar_electron_densitiy_temperature_profiles(
            ne_array, te_array, detachment=False
        )
        assert all(t_d < t_a for t_d, t_a in zip(te_det, te_att, strict=False))
        assert all(n_d < n_a for n_d, n_a in zip(ne_det, ne_att, strict=False))

    def test_rad_core_by_psi_n(self):
        rad_centre = self.source.rad_core_by_psi_n(0.1)
        rad_edge = self.source.rad_core_by_psi_n(0.9)
        assert rad_centre > rad_edge

    def test_rad_core_by_points(self):
        rad_centre = self.source.rad_core_by_points(10.5, -1)
        rad_edge = self.source.rad_core_by_points(12, -1)
        assert rad_centre > rad_edge

    def test_radiative_loss_function_values(self):
        imp_data_t_ref = [
            data["T_ref"]
            for key, data in self.source.imp_data_core.items()
            if key != "Ar"
        ]
        imp_data_t_ref = imp_data_t_ref[0]
        t_ref = np.array([imp_data_t_ref[0], imp_data_t_ref[2], imp_data_t_ref[4]])
        imp_data_l_ref = [
            data["L_ref"]
            for key, data in self.source.imp_data_core.items()
            if key != "Ar"
        ]
        imp_data_l_ref = imp_data_l_ref[0]
        l_ref = np.array([imp_data_l_ref[0], imp_data_l_ref[2], imp_data_l_ref[4]])
        tvals = np.array([imp_data_t_ref[1], imp_data_t_ref[3]])
        lvals = np.array([imp_data_l_ref[1], imp_data_l_ref[3]])
        l1 = radiative_loss_function_values(tvals, t_ref, l_ref)
        np.testing.assert_allclose(l1, lvals, rtol=2e-1)

    def test_pfr_filter(self):
        x_point_z = self.source.sol_rad.points["x_point"]["z_low"]
        pfr_x_down, pfr_z_down = pfr_filter(self.source.sol_rad.separatrix, x_point_z)
        assert pfr_x_down.shape == (59,)
        assert pfr_z_down.shape == (59,)

        assert np.all(pfr_z_down < x_point_z - 0.01)

    def test_make_wall_detectors(self):
        max_wall_len = 10.0e-2
        X_WIDTH = 0.01
        wall_detectors = make_wall_detectors(
            self.fw_shape.x, self.fw_shape.z, max_wall_len, X_WIDTH
        )
        assert all(detector[2] <= max_wall_len for detector in wall_detectors)
        assert all(np.isclose(detector[1], X_WIDTH) for detector in wall_detectors)
        assert len(wall_detectors) == 532

    def test_FirstWallRadiationSolver(self):
        # Coversion required for CHERAB
        f_sol = linear_interpolator(
            self.source.sol_rad.x_tot,
            self.source.sol_rad.z_tot,
            raw_uc(self.source.sol_rad.rad_tot, "MW", "W"),
        )

        # SOL radiation grid
        x_sol = np.linspace(min(self.fw_shape.x), max(self.fw_shape.x), 4)
        z_sol = np.linspace(min(self.fw_shape.z), max(self.fw_shape.z), 4)

        rad_sol_grid = interpolated_field_values(x_sol, z_sol, f_sol)
        func = grid_interpolator(x_sol, z_sol, rad_sol_grid)
        solver = FirstWallRadiationSolver(
            source_func=func, firstwall_shape=self.fw_shape
        )
        assert solver.fw_shape == self.fw_shape

        wall_loads = solver.solve(50, 1, 10, plot=False)

        # check return types
        assert isinstance(wall_loads, DetectedRadiation)
        assert isinstance(wall_loads.total_power, float)

        # check if the arrays are of same length
        assert len(wall_loads.detector_numbers) == len(wall_loads.detected_power)

        # the solver gives slightly different powers in each run
        # so just asserting the order of total power
        assert np.isclose(wall_loads.total_power, 2.32e8, rtol=0.05)
