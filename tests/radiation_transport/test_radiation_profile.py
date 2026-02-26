# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest

from bluemira.base import constants
from bluemira.base.constants import raw_uc
from bluemira.base.file import get_bluemira_path
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
    get_impurity_data,
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


class ExampleCoreRadiation:
    def __init__(
        self,
        eq_name,
        fw_name,
        sep_corrector_omp,
        sep_corrector_imp,
        lfs_p_fraction,
        tungsten_fraction,
        expected_values,
    ):
        filename = Path(EQ_PATH, eq_name)
        eq = Equilibrium.from_eqdsk(filename, from_cocos=3, qpsi_positive=False)
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
        self.params = {
            "sep_corrector_omp": {"value": sep_corrector_omp, "unit": "dimensionless"},
            "sep_corrector_imp": {"value": sep_corrector_imp, "unit": "dimensionless"},
            "det_t": {"value": 0.0015, "unit": "keV"},
            "eps_cool": {"value": 25.0, "unit": "eV"},
            "f_ion_t": {"value": 0.01, "unit": "keV"},
            "main_ext": {"value": None, "unit": "m"},
            "rec_ext_out_leg": {"value": 2, "unit": "m"},
            "rec_ext_in_leg": {"value": 0.2, "unit": "m"},
            "fw_lambda_q_near_omp": {"value": 0.003, "unit": "m"},
            "fw_lambda_q_far_omp": {"value": 0.1, "unit": "m"},
            "fw_lambda_q_near_imp": {"value": 0.003, "unit": "m"},
            "fw_lambda_q_far_imp": {"value": 0.1, "unit": "m"},
            "lambda_t_factor": {"value": 7, "unit": "dimensionless"},
            "lambda_n_factor": {"value": 1 / 7, "unit": "dimensionless"},
            "gamma_sheath": {"value": 7.0, "unit": "dimensionless"},
            "k_0": {"value": 2000.0, "unit": "dimensionless"},
            "lfs_p_fraction": {"value": lfs_p_fraction, "unit": "dimensionless"},
            "P_sep": {"value": 100, "unit": "MW"},
            "theta_inner_target": {"value": 5.0, "unit": "deg"},
            "theta_outer_target": {"value": 5.0, "unit": "deg"},
            **midplane_params,
        }

        self.config = {
            "f_imp_core": {"H": 1e-2, "He": 1e-2, "Xe": 1e-4, "W": tungsten_fraction},
            "f_imp_sol": {"H": 0, "He": 0, "Ar": 1e-3, "Xe": 0, "W": 0},
            "confinement_core": 0.1,
            "confinement_sol": 10,
        }

        profiles = midplane_profiles(params=midplane_params)

        source = RadiationSource(
            eq=eq,
            firstwall_shape=fw_shape,
            params=self.params,
            midplane_profiles=profiles,
            core_impurities=self.config["f_imp_core"],
            sol_impurities=self.config["f_imp_sol"],
            confinement_time_core=self.config["confinement_core"],
            confinement_time_sol=self.config["confinement_sol"],
        )
        source.analyse(firstwall_geom=fw_shape)
        source.rad_map(fw_shape)

        self.profiles = profiles
        self.source = source
        self.fw_shape = fw_shape
        self.expected_values = expected_values


@pytest.fixture(
    scope="class",
    params=[
        {
            "eq_name": "EU-DEMO_EOF.json",
            "fw_name": "first_wall.json",
            "sep_corrector_omp": 5e-2,
            "sep_corrector_imp": 6e-2,
            "lfs_p_fraction": 1,
            "tungsten_fraction": 1e-4,
            "expected_values": {
                "pfr_down_shape": (46,),
                "wall_det_len": 323,
                "rad_tot": 2695.3397,
                "x_tot": 50965.673,
                "z_tot": -311.79,  # TODO @DarioV86: This seems broken #3328
                "ion_front_dist": 2.440,
                "total_power": 1.574e7,
            },
        },
        {
            "eq_name": "DN-DEMO_eqref.json",
            "fw_name": "DN_fw_shape.json",
            "sep_corrector_omp": 5e-3,
            "sep_corrector_imp": 6e-3,
            "lfs_p_fraction": 0.9,
            "tungsten_fraction": 1e-5,
            "expected_values": {
                "pfr_down_shape": (59,),
                "wall_det_len": 532,
                "rad_tot": 1295.7477,
                "x_tot": 55614.0533,
                "z_tot": 239.84336,
                "ion_front_dist": 2.619,
                "total_power": 2.46e8,
            },
        },
    ],
)
def rad(request):
    return ExampleCoreRadiation(**request.param)


class TestCoreRadiation:
    def test_get_impurity_data(self, rad):
        core_impurities = get_impurity_data(
            rad.config["f_imp_core"], rad.config["confinement_core"]
        )
        core_shape = core_impurities["H"]["T_ref"].shape
        assert len(core_impurities) == len(rad.config["f_imp_core"])
        for values in core_impurities.values():
            assert np.shape(values["T_ref"]) == core_shape
            assert np.shape(values["L_ref"]) == core_shape
            assert np.shape(values["z_ref"]) == core_shape

        sol_impurities = get_impurity_data(
            rad.config["f_imp_sol"], rad.config["confinement_sol"]
        )
        sol_shape = sol_impurities["H"]["T_ref"].shape
        assert len(sol_impurities) == len(rad.config["f_imp_sol"])
        for values in sol_impurities.values():
            assert np.shape(values["T_ref"]) == sol_shape
            assert np.shape(values["L_ref"]) == sol_shape
            assert np.shape(values["z_ref"]) == sol_shape

    def test_collect_flux_tubes(self, rad):
        psi = np.linspace(1, 1.5, 5)
        ft = rad.source.core_rad.collect_flux_tubes(psi)
        assert len(ft) == 5

    def test_rho_core(self, rad):
        rho_ped = (
            rad.params["rho_ped_n"]["value"] + rad.params["rho_ped_t"]["value"]
        ) / 2
        rho_core = collect_rho_core_values(rho_ped, 30, 15, 10)
        assert rho_core[0] > 0
        assert rho_core[-1] < 1

    def test_rad_sol_by_psi_n(self, rad):
        rad_centre = rad.source.rad_sol_by_psi_n(0.1).max()
        rad_edge = rad.source.rad_sol_by_psi_n(0.9).max()
        assert rad_centre > rad_edge

    def test_rad_by_psi_n(self, rad):
        rad_centre = rad.source.rad_by_psi_n(0.1).max()
        rad_edge = rad.source.rad_by_psi_n(0.9).max()
        assert rad_centre > rad_edge

    def test_rad_sol_by_points(self, rad):
        """The rad_sol value of the flux surface intersecting the chosen point of x-z
        should include the rad_sol value of that chosen point of x-z.
        """
        eq = rad.source.eq
        x, z = eq.x.flatten(), eq.z.flatten()
        psi_n = eq.psi_norm().flatten()
        interp_grid = linear_interpolator(x, z, psi_n)

        i = len(rad.source.rad_tot) // 2  # pick a random index within range
        x_tot, z_tot = rad.source.x_tot.flatten()[i], rad.source.z_tot.flatten()[i]
        psi_norm_tot = interpolated_field_values(x_tot, z_tot, interp_grid)

        rad_sol_pt = rad.source.rad_sol_by_points([x_tot], [z_tot]).flatten()[0]
        rad_sol_psi = rad.source.rad_sol_by_psi_n(psi_norm_tot[0][0]).flatten()
        assert rad_sol_pt in rad_sol_psi

    def test_rad_by_points(self, rad):
        """The 'rad' value of the flux surface intersecting the chosen point of x-z
        should include the 'rad' value of that chosen point of x-z.
        """
        eq = rad.source.eq
        x, z = eq.x.flatten(), eq.z.flatten()
        psi_n = eq.psi_norm().flatten()
        interp_grid = linear_interpolator(x, z, psi_n)

        i = len(rad.source.rad_tot) // 2  # pick a point in the SOL.
        x_tot, z_tot = rad.source.x_tot.flatten()[i], rad.source.z_tot.flatten()[i]
        psi_norm_tot = interpolated_field_values(x_tot, z_tot, interp_grid)

        rad_tot_pt = rad.source.rad_by_points(x_tot, z_tot).flatten()[0]
        rad_tot_psi = rad.source.rad_by_psi_n(psi_norm_tot[0][0]).flatten()
        assert rad_tot_pt in rad_tot_psi

    def test_core_electron_density_temperature_profile(self, rad):
        ne_core = rad.profiles.ne
        te_core = rad.profiles.te
        psi_n = rad.profiles.psi_n
        assert len(ne_core) == len(te_core) == len(psi_n)

        # Ensure values are within expected ranges
        assert np.all(ne_core >= rad.params["n_e_sep"]["value"])
        assert np.all(te_core >= rad.params["T_e_sep"]["value"])
        assert np.all(ne_core <= rad.params["n_e_0"]["value"])
        assert np.all(te_core <= rad.params["T_e_0"]["value"])

        # Ensure the density and temperature are decreasing towards the edge
        assert ne_core[0] > ne_core[-1]
        assert te_core[0] > te_core[-1]

    def test_calculate_mp_radiation_profile(self, rad):
        rad.source.core_rad.calculate_mp_radiation_profile()
        rad_tot = np.sum(np.array(rad.source.core_rad.rad_mp, dtype=object), axis=0)
        assert len(rad.source.core_rad.rad_mp) == 4
        assert rad_tot[0] > rad_tot[-1]

    def test_calculate_core_distribution(self, rad):
        # calls calculate_core_distribution() internally
        rad.source.core_rad.calculate_core_radiation_map()
        rad_tot = rad.expected_values["rad_tot"]
        x_tot = rad.expected_values["x_tot"]
        z_tot = rad.expected_values["z_tot"]
        assert np.sum(rad.source.core_rad.rad_tot) == pytest.approx(rad_tot)
        assert np.sum(rad.source.core_rad.x_tot) == pytest.approx(x_tot)
        assert np.sum(rad.source.core_rad.z_tot) == pytest.approx(z_tot)

    def test_core_flux_tube_pol_t(self, rad):
        flux_tube = rad.source.eq.get_flux_surface(0.99)
        te = rad.source.core_rad.flux_tube_pol_t(flux_tube, 100, core=True)
        assert te[0] == te[-1]
        assert len(te) == len(flux_tube)

    def test_core_flux_tube_pol_n(self, rad):
        flux_tube = rad.source.eq.get_flux_surface(0.99)
        ne_mp = 2e20
        ne = rad.source.core_rad.flux_tube_pol_n(flux_tube, ne_mp, core=True)
        assert ne[0] == ne[-1]
        assert len(ne) == len(flux_tube)

    def test_mp_electron_density_temperature_profiles(self, rad):
        te_sol_omp, ne_sol_omp = (
            rad.source.sol_rad.mp_electron_density_temperature_profiles()
        )
        te_sol_imp, ne_sol_imp = (
            rad.source.sol_rad.mp_electron_density_temperature_profiles(omp=False)
        )
        assert te_sol_omp[0] > te_sol_omp[-1]
        assert ne_sol_omp[0] > ne_sol_omp[-1]
        assert te_sol_imp[0] > te_sol_imp[-1]
        assert ne_sol_imp[0] > ne_sol_imp[-1]

    def test_key_temperatures(self, rad):
        t_u = upstream_temperature(
            b_pol=rad.source.sol_rad.b_pol_sep_omp,
            b_tot=rad.source.sol_rad.b_tot_sep_omp,
            lambda_q_near=rad.source.params.fw_lambda_q_near_omp.value,
            p_sol=rad.source.params.P_sep.value,
            eq=rad.source.eq,
            r_sep_mp=rad.source.sol_rad.r_sep_omp,
            z_mp=rad.source.sol_rad.z_mp,
            k_0=rad.source.params.k_0.value,
            firstwall_geom=rad.fw_shape,
        )
        t_u_eV = constants.raw_uc(t_u, "keV", "eV")
        f_ion_t = rad.source.params.f_ion_t.value_as("eV")

        t_tar_det = target_temperature(
            rad.source.params.P_sep.value,
            t_u_eV,
            rad.source.params.n_e_sep.value,
            rad.source.params.gamma_sheath.value,
            rad.source.params.eps_cool.value_as("eV"),
            f_ion_t,
            rad.source.sol_rad.b_pol_out_tar,
            rad.source.sol_rad.b_pol_sep_omp,
            rad.source.params.theta_outer_target.value,
            rad.source.sol_rad.r_sep_omp,
            rad.source.sol_rad.x_strike_lfs,
            rad.source.params.fw_lambda_q_near_omp.value,
            rad.source.sol_rad.b_tot_out_tar,
        )

        t_tar_no_det = target_temperature(
            rad.source.params.P_sep.value,
            t_u_eV,
            3e10,
            rad.source.params.gamma_sheath.value,
            rad.source.params.eps_cool.value_as("eV"),
            f_ion_t,
            rad.source.sol_rad.b_pol_out_tar,
            rad.source.sol_rad.b_pol_sep_omp,
            rad.source.params.theta_outer_target.value,
            rad.source.sol_rad.r_sep_omp,
            rad.source.sol_rad.x_strike_lfs,
            rad.source.params.fw_lambda_q_near_omp.value,
            rad.source.sol_rad.b_tot_out_tar,
        )
        assert t_u < 5e-1
        assert t_tar_no_det > rad.source.params.f_ion_t.value_as("eV")
        assert t_tar_det <= rad.source.params.f_ion_t.value_as("eV")  # needs review

    def test_sol_decay(self, rad):
        t_u = rad.source.params.T_e_sep.value_as("keV")
        n_u = rad.source.params.n_e_sep.value
        decayed_t, decayed_n = electron_density_and_temperature_sol_decay(
            t_u,
            n_u,
            rad.source.params.fw_lambda_q_near_omp.value,
            rad.source.params.fw_lambda_q_far_omp.value,
            rad.source.dx_omp,
        )
        assert decayed_t[0] > decayed_t[-1]
        assert decayed_n[0] > decayed_n[-1]

    def test_ion_front_distance(self, rad):
        distance = ion_front_distance(
            6,
            -9,
            rad.source.eq,
            rad.source.sol_rad.points["x_point"]["z_low"],
            1e-3,
            1,
            1,
            2e20,
        )
        assert distance is not None
        assert distance == pytest.approx(rad.expected_values["ion_front_dist"], rel=1e-3)

    def test_radiation_region_boundary(self, rad):
        low_z_main, low_z_pfr = rad.source.sol_rad.x_point_radiation_z_ext()
        up_z_main, up_z_pfr = rad.source.sol_rad.x_point_radiation_z_ext(low_div=False)
        assert low_z_main > low_z_pfr
        assert up_z_main < up_z_pfr
        in_x_lfs, in_z_low, out_x_lfs, out_z_low = (
            rad.source.sol_rad.radiation_region_ends(low_z_main, low_z_pfr)
        )
        _, in_z_up, _, out_z_up = rad.source.sol_rad.radiation_region_ends(
            up_z_main, up_z_pfr
        )
        in_x_hfs, _, out_x_hfs, _ = rad.source.sol_rad.radiation_region_ends(
            low_z_main, low_z_pfr, lfs=False
        )
        assert in_x_lfs > rad.source.sol_rad.points["x_point"]["x"]
        assert out_x_lfs > rad.source.sol_rad.points["x_point"]["x"]
        assert in_x_hfs < rad.source.sol_rad.points["x_point"]["x"]
        assert out_x_hfs < rad.source.sol_rad.points["x_point"]["x"]
        assert in_z_low > out_z_low
        assert in_z_up < out_z_up

    def test_tar_electron_densitiy_temperature_profiles(self, rad):
        ne_array = np.linspace(1e20, 1e19, 5)
        te_array = np.linspace(15, 8, 5)
        te_det, ne_det = rad.source.sol_rad.tar_electron_densitiy_temperature_profiles(
            ne_array, te_array, detachment=True
        )
        te_att, ne_att = rad.source.sol_rad.tar_electron_densitiy_temperature_profiles(
            ne_array, te_array, detachment=False
        )
        assert all(t_d < t_a for t_d, t_a in zip(te_det, te_att, strict=False))
        assert all(n_d < n_a for n_d, n_a in zip(ne_det, ne_att, strict=False))

    def test_rad_core_by_psi_n(self, rad):
        rad_centre = rad.source.rad_core_by_psi_n(0.1)
        rad_edge = rad.source.rad_core_by_psi_n(0.9)
        assert rad_centre > rad_edge

    def test_rad_core_by_points(self, rad):
        rad_centre = rad.source.rad_core_by_points(10.5, -1)
        rad_edge = rad.source.rad_core_by_points(12, -1)
        assert rad_centre > rad_edge

    def test_radiative_loss_function_values(self, rad):
        imp_data_t_ref = [
            data["T_ref"]
            for key, data in rad.source.imp_data_core.items()
            if key != "Ar"
        ]
        imp_data_t_ref = imp_data_t_ref[0]
        t_ref = np.array([imp_data_t_ref[0], imp_data_t_ref[2], imp_data_t_ref[4]])
        imp_data_l_ref = [
            data["L_ref"]
            for key, data in rad.source.imp_data_core.items()
            if key != "Ar"
        ]
        imp_data_l_ref = imp_data_l_ref[0]
        l_ref = np.array([imp_data_l_ref[0], imp_data_l_ref[2], imp_data_l_ref[4]])
        tvals = np.array([imp_data_t_ref[1], imp_data_t_ref[3]])
        lvals = np.array([imp_data_l_ref[1], imp_data_l_ref[3]])
        l1 = radiative_loss_function_values(tvals, t_ref, l_ref)
        np.testing.assert_allclose(l1, lvals, rtol=2e-1)

    def test_pfr_filter(self, rad):
        x_point_z = rad.source.sol_rad.points["x_point"]["z_low"]
        pfr_x_down, pfr_z_down = pfr_filter(rad.source.sol_rad.separatrix, x_point_z)
        assert np.all(pfr_z_down < x_point_z - 0.01)

        assert pfr_x_down.shape == rad.expected_values["pfr_down_shape"]
        assert pfr_z_down.shape == rad.expected_values["pfr_down_shape"]

    def test_make_wall_detectors(self, rad):
        max_wall_len = 10.0e-2
        X_WIDTH = 0.01
        wall_detectors = make_wall_detectors(
            rad.fw_shape.x, rad.fw_shape.z, max_wall_len, X_WIDTH
        )
        assert all(detector.y_width <= max_wall_len for detector in wall_detectors)
        assert all(np.isclose(detector.x_width, X_WIDTH) for detector in wall_detectors)
        assert len(wall_detectors) == rad.expected_values["wall_det_len"]

    def test_FirstWallRadiationSolver(self, rad):
        # Coversion required for CHERAB
        f_sol = linear_interpolator(
            rad.source.sol_rad.x_tot,
            rad.source.sol_rad.z_tot,
            raw_uc(rad.source.sol_rad.rad_tot, "MW", "W"),
        )

        # SOL radiation grid
        x_sol = np.linspace(min(rad.fw_shape.x), max(rad.fw_shape.x), 4)
        z_sol = np.linspace(min(rad.fw_shape.z), max(rad.fw_shape.z), 4)

        rad_sol_grid = interpolated_field_values(x_sol, z_sol, f_sol)
        func = grid_interpolator(x_sol, z_sol, rad_sol_grid)
        solver = FirstWallRadiationSolver(source_func=func, firstwall_shape=rad.fw_shape)
        assert solver.fw_shape == rad.fw_shape

        wall_loads = solver.solve(50, 1, 10, plot=False)

        # check return types
        assert isinstance(wall_loads, DetectedRadiation)
        assert isinstance(wall_loads.total_power, float)

        # check if the arrays are of same length
        assert len(wall_loads.detector_numbers) == len(wall_loads.detected_power)

        # the solver gives slightly different powers in each run
        # so just asserting the order of total power
        exp_total_power = rad.expected_values["total_power"]
        assert np.isclose(wall_loads.total_power, exp_total_power, rtol=0.05)
