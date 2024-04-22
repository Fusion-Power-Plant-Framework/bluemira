# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest

from bluemira.base import constants
from bluemira.base.file import get_bluemira_path
from bluemira.codes.process import api
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.midplane_temperature_density import MidplaneProfiles
from bluemira.radiation_transport.radiation_profile import RadiationSource
from bluemira.radiation_transport.radiation_tools import (
    calculate_line_radiation_loss,
    calculate_z_species,
    electron_density_and_temperature_sol_decay,
    exponential_decay,
    gaussian_decay,
    ion_front_distance,
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
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "DN_fw_shape.json"
        filename = Path(TEST_PATH, fw_name)
        fw_shape = Coordinates.from_json(filename)

        cls.params = {
            "sep_corrector": {"value": 5e-3, "unit": "dimensionless"},
            "alpha_n": {"value": 1.15, "unit": "dimensionless"},
            "alpha_t": {"value": 1.905, "unit": "dimensionless"},
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
            "n_e_0": {"value": 21.93e19, "unit": "1/m^3"},
            "n_e_ped": {"value": 8.117e19, "unit": "1/m^3"},
            "n_e_sep": {"value": 1.623e19, "unit": "1/m^3"},
            "P_sep": {"value": 100, "unit": "MW"},
            "rho_ped_n": {"value": 0.94, "unit": "dimensionless"},
            "rho_ped_t": {"value": 0.976, "unit": "dimensionless"},
            "n_points_core_95": {"value": 30, "unit": "dimensionless"},
            "n_points_core_99": {"value": 15, "unit": "dimensionless"},
            "n_points_mantle": {"value": 10, "unit": "dimensionless"},
            "t_beta": {"value": 2.0, "unit": "dimensionless"},
            "T_e_0": {"value": 21.442, "unit": "keV"},
            "T_e_ped": {"value": 5.059, "unit": "keV"},
            "T_e_sep": {"value": 0.16, "unit": "keV"},
            "theta_inner_target": {"value": 5.0, "unit": "deg"},
            "theta_outer_target": {"value": 5.0, "unit": "deg"},
        }

        cls.config = {
            "f_imp_core": {"H": 1e-1, "He": 1e-2, "Xe": 1e-4, "W": 1e-5},
            "f_imp_sol": {"H": 0, "He": 0, "Ar": 0.003, "Xe": 0, "W": 0},
        }

        profiles = MidplaneProfiles(params=cls.params)
        psi_n = profiles.psi_n
        ne_mp = profiles.ne_mp
        te_mp = profiles.te_mp

        source = RadiationSource(
            eq=eq,
            firstwall_shape=fw_shape,
            params=cls.params,
            psi_n=psi_n,
            ne_mp=ne_mp,
            te_mp=te_mp,
            core_impurities=cls.config["f_imp_core"],
            sol_impurities=cls.config["f_imp_sol"],
        )
        source.analyse(firstwall_geom=fw_shape)

        cls.profiles = profiles
        cls.source = source
        cls.fw_shape = fw_shape

    def test_collect_flux_tubes(self):
        psi = np.linspace(1, 1.5, 5)
        ft = self.source.core_rad.collect_flux_tubes(psi)
        assert len(ft) == 5

    def test_rho_core(self):
        rho_core = self.profiles.collect_rho_core_values()
        assert rho_core[0] > 0
        assert rho_core[-1] < 1

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

    def test_mp_electon_density_temperature_profiles(self):
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


def test_gaussian_decay():
    decayed_val = gaussian_decay(10, 1, 50)
    gap_1 = decayed_val[0] - decayed_val[1]
    gap_2 = decayed_val[1] - decayed_val[2]
    gap_3 = decayed_val[-2] - decayed_val[-1]
    assert gap_1 < gap_2 < gap_3


def test_exponential_decay():
    decayed_val = exponential_decay(10, 1, 50, decay=True)
    gap_1 = decayed_val[0] - decayed_val[1]
    gap_2 = decayed_val[1] - decayed_val[2]
    gap_3 = decayed_val[-2] - decayed_val[-1]
    assert gap_1 > gap_2 > gap_3


def test_calculate_z_species():
    t_ref = np.array([0, 10])
    z_ref = np.array([10, 20])
    frac = 0.1
    t_test = 5
    z = calculate_z_species(t_ref, z_ref, frac, t_test)
    assert z == pytest.approx(22.5)


def test_calculate_line_radiation_loss():
    ne = 1e20
    p_loss = 1e-31
    frac = 0.01
    rad = calculate_line_radiation_loss(ne, p_loss, frac)
    assert rad == pytest.approx(0.796, abs=1e-3)
