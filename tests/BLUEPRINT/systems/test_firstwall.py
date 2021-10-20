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
import pytest
import os
import numpy as np
from BLUEPRINT.systems.firstwall import FluxSurface, FirstWallSN, FirstWallDN
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.geomtools import get_intersect


DATA_PATH = get_bluemira_path("equilibria", subfolder="data")


#  Helper function to load an equilibrium
def load_equilibrium(eq_name):
    return Equilibrium.from_eqdsk(os.sep.join([DATA_PATH, eq_name]))


# Helper function to make a dummy keep-out-zone from equilibrium
def make_koz(x_x_point, x_low, x_high, z_inner, z_outer):
    koz_x = [x_low, x_low, x_x_point, x_x_point, x_high, x_high, x_x_point, x_x_point]
    koz_z = [z_inner, -z_inner, -z_inner, -z_outer, -z_outer, z_outer, z_outer, z_inner]
    koz = Loop(x=koz_x, y=None, z=koz_z)
    koz.close()
    return koz


def make_input_loops_sn(eq):
    o_point, x_point = eq.get_OX_points()
    z_x_point = x_point[0][1]
    x_x_point = x_point[0][0]
    z_inner = z_x_point - 1.25
    z_outer = z_x_point - 2.25
    x_low = x_x_point - 3
    x_high = x_x_point + 4.5
    koz = make_koz(x_x_point, x_low, x_high, z_inner, z_outer)
    vessel = make_koz(x_x_point, x_low, x_high, z_inner - 1.3, z_outer - 1.3)
    return koz, vessel


def make_input_loops_dn(eq):
    o_point, x_point = eq.get_OX_points()
    z_x_point = x_point[1][1]
    x_x_point = x_point[1][0]
    z_inner = z_x_point - 1
    z_outer = z_x_point - 3
    x_low = x_x_point - 3
    x_high = x_x_point + 5.5
    koz = make_koz(x_x_point, x_low, x_high, z_inner, z_outer)
    vessel = make_koz(x_x_point, x_low, x_high, z_inner - 1.5, z_outer - 1.5)
    return koz, vessel


def firstwall_sn_inputs():
    eq = load_equilibrium("EU-DEMO_EOF.json")
    koz, vessel = make_input_loops_sn(eq)
    inputs = {
        "equilibrium": eq,
        "koz": koz,
        "vv_inner": vessel,
        "strike_pts_from_koz": False,
        "pick_flux_from_psinorm": False,
        "SN": True,
        "div_vertical_outer_target": False,
        "div_vertical_inner_target": False,
    }
    return inputs


def firstwall_dn_inputs():
    eq = load_equilibrium(
        "DN-DEMO_eqref.json",
    )
    koz, vessel = make_input_loops_dn(eq)
    inputs = {
        "equilibrium": eq,
        "koz": koz,
        "vv_inner": vessel,
        "strike_pts_from_koz": False,
        "pick_flux_from_psinorm": False,
        "DEMO_DN": True,
        "div_vertical_outer_target": True,
        "div_vertical_inner_target": False,
    }
    return inputs


# Helper function to load a SN first wall
def load_firstwall_sn():
    inputs = firstwall_sn_inputs()
    fw = FirstWallSN(FirstWallSN.default_params, inputs)
    return fw


# Helper function to load a DN first wall
def load_firstwall_dn():
    inputs = firstwall_dn_inputs()
    fw = FirstWallDN(FirstWallDN.default_params, inputs)
    return fw


# Driver test class to call the methods in FirstWallSN without fully initialising
class FirstWallSNDriver(FirstWallSN):
    def __init__(self):
        self.config = FirstWallSN.default_params
        self.inputs = firstwall_sn_inputs()
        self.init_params()
        self.init_equilibrium()


# Driver test class to call the methods in FirstWallDN without fully initialising
class FirstWallDNDriver(FirstWallDN):
    def __init__(self):
        self.config = FirstWallDN.default_params
        self.inputs = firstwall_dn_inputs()
        self.init_params()
        self.init_equilibrium()


class TestFluxSurface:
    @classmethod
    def setup_class(cls):
        eq = load_equilibrium("EU-DEMO_EOF.json")
        cls.fluxsurface = FluxSurface(eq, 12, 0)

    def test_assign_lfs_hfs_sn(self):
        x = np.array([8, 10, 5, 9])
        z = np.array([-1, -5, 2, 2])
        p_side = self.fluxsurface.assign_lfs_hfs_sn(x, z)
        assert len(p_side[0]) != 0
        assert len(p_side[1][0]) == 2

    def test_assign_top_bottom(self):
        x = np.array([8, 10, 5, 9])
        z = np.array([1, 5, -2, -7])
        p_loc = self.fluxsurface.assign_top_bottom(x, z)
        assert len(p_loc[0]) != 0
        assert len(p_loc[1]) == 2

    def test_calculate_q_par_omp(self):
        qpar = self.fluxsurface.calculate_q_par_omp(0, 0, 100, 100)
        assert qpar == 0


# Method to check attributes of a first wall
def check_firstwall(firstwall):
    # Check all the attributes are sensible
    assert isinstance(firstwall.profile, Loop)
    assert isinstance(firstwall.inner_profile, Loop)
    assert isinstance(firstwall.geom, dict)
    assert isinstance(firstwall.geom["2D profile"], Shell)
    assert isinstance(firstwall.geom["Inboard wall"], Loop)
    assert isinstance(firstwall.geom["Outboard wall"], Loop)
    assert isinstance(firstwall.divertor_loops, list)
    assert isinstance(firstwall.divertor_cassettes, list)

    inboard = firstwall.geom["Inboard wall"]
    outboard = firstwall.geom["Outboard wall"]

    # Inboard / outboard intersect in SN case
    if len(firstwall.divertor_loops) == 1:
        int_x, int_z = get_intersect(inboard, outboard)
        n_ints = len(int_x)
        assert n_ints >= 1 and len(int_z) == n_ints

    # Check intersections with the divertor
    for div in firstwall.divertor_loops:
        assert isinstance(div, Loop)
        for sec_compare in [inboard, outboard]:
            int_x, int_z = get_intersect(div, sec_compare)
            n_ints = len(int_x)
            assert n_ints >= 1 and len(int_z) == n_ints

    return True


class TestFirstWallSN:
    # Setup for *every* test in class
    def setup_method(self):
        self.firstwall = FirstWallSNDriver()

    def test_make_preliminary_profile(self):
        prof = self.firstwall.make_preliminary_profile()
        assert hasattr(prof, "x")

    # Test build for different combinations of thicknesses
    @pytest.mark.parametrize("tk_in", [0.1])
    @pytest.mark.parametrize("tk_out_diff", [0.0, 0.05, -0.05])
    @pytest.mark.parametrize("tk_div_diff", [0.0, 0.05, -0.05])
    def test_build_firstwall(self, tk_in, tk_out_diff, tk_div_diff):
        self.firstwall.params.tk_fw_in = tk_in
        self.firstwall.params.tk_fw_out = tk_in + tk_out_diff
        self.firstwall.params.tk_fw_div = tk_in + tk_div_diff
        self.firstwall.build()
        assert check_firstwall(self.firstwall)

    def test_q_parallel_calculation(self):
        self.firstwall.build()
        qpar = self.firstwall.q_parallel_calculation()
        assert len(self.firstwall.flux_surfaces) == len(qpar)
        for i in range(len(qpar), -1):
            assert qpar[i] > qpar[i + 1]


class TestFirstWallDN:
    # Setup for *every* test in class
    def setup_method(self):
        self.firstwall = FirstWallDNDriver()

    def test_make_preliminary_profile(self):
        prof = self.firstwall.make_preliminary_profile()
        assert len(prof.x) == len(prof.z)
        assert prof.x[0] == prof.x[-1]

    def test_make_divertor_outer_target(self):
        flux_loops = self.firstwall.pick_flux_loops()
        inner_strike, outer_strike = self.firstwall.find_strike_points(flux_loops)
        tangent = self.firstwall.get_tangent_vector(outer_strike, flux_loops[0])
        tar_out = self.firstwall.make_divertor_target(
            outer_strike,
            tangent,
            vertical_target=True,
            outer_target=True,
        )
        assert tar_out[0][0] > self.firstwall.points["x_point"]["x"]
        assert tar_out[0][0] < tar_out[1][0]

    def test_make_divertor_inner_target(self):
        flux_loops = self.firstwall.pick_flux_loops()
        inner_strike, outer_strike = self.firstwall.find_strike_points(flux_loops)
        tangent = self.firstwall.get_tangent_vector(inner_strike, flux_loops[1])
        tar_in = self.firstwall.make_divertor_target(
            inner_strike,
            tangent,
            vertical_target=False,
            outer_target=False,
        )
        assert tar_in[0][0] < self.firstwall.points["x_point"]["x"]
        assert tar_in[0][0] > tar_in[1][0]

    @pytest.mark.parametrize("ints_from_psi", [True, False])
    def test_make_divertor_from_koz(self, ints_from_psi):
        self.firstwall.inputs["strike_pts_from_koz"] = True
        self.firstwall.inputs["pick_flux_from_psinorm"] = ints_from_psi

        # Make a fake firstwall loop
        # (just needs to intersect yz plane containing x point)
        x_x_point = self.firstwall.points["x_point"]["x"]
        z_x_point = self.firstwall.points["x_point"]["z_low"]
        fw_x_right = x_x_point + self.firstwall.params.xpt_outer_gap + 0.5
        fw_x_left = x_x_point - self.firstwall.params.xpt_outer_gap - 0.5
        fw_z_top = z_x_point + 1.0
        fw_z_bot = z_x_point - 1.0
        fw_x = [fw_x_right, fw_x_right, fw_x_left, fw_x_left]
        fw_z = [fw_z_top, fw_z_bot, fw_z_bot, fw_z_top]
        fw_loop = Loop(x=fw_x, y=None, z=fw_z)
        fw_loop.close()

        # Make the divertor
        div = self.firstwall.make_divertor(fw_loop)[0]

        # Check the loop is closed
        assert div.closed

        # Get the inner/outer target end points
        flux_loops = self.firstwall.pick_flux_loops()
        inner_strike, outer_strike = self.firstwall.find_strike_points(flux_loops)
        tangent_out = self.firstwall.get_tangent_vector(outer_strike, flux_loops[0])
        tar_out = self.firstwall.make_divertor_target(
            outer_strike, tangent_out, vertical_target=True, outer_target=True
        )
        tangent_in = self.firstwall.get_tangent_vector(inner_strike, flux_loops[1])
        tar_in = self.firstwall.make_divertor_target(
            inner_strike, tangent_in, vertical_target=False, outer_target=False
        )
        # Get the minimum and maximum x,z
        max_x = round(tar_out[1][0], 5)
        min_x = round(tar_in[1][0], 5)
        max_z = round(self.firstwall.points["x_point"]["z_low"], 5)
        min_z = round(tar_out[0][1], 5)
        # min_z = round(max_z - 2.25,5)

        # Check the bounds
        div_x_max = np.max(div.x)
        div_x_min = np.min(div.x)
        div_z_max = np.max(div.z)
        div_z_min = np.min(div.z)
        assert div_x_max == max_x
        assert div_x_min == min_x
        assert div_z_max == max_z
        assert div_z_min == min_z

    # Test build for different combinations of thicknesses
    @pytest.mark.parametrize("tk_in", [0.1])
    @pytest.mark.parametrize("tk_out_diff", [0.0, 0.05, -0.05])
    @pytest.mark.parametrize("tk_div_diff", [0.0, 0.05, -0.05])
    def test_build_firstwall(self, tk_in, tk_out_diff, tk_div_diff):
        self.firstwall.params.tk_fw_in = tk_in
        self.firstwall.params.tk_fw_out = tk_in + tk_out_diff
        self.firstwall.params.tk_fw_div = tk_in + tk_div_diff
        self.firstwall.build()
        assert check_firstwall(self.firstwall)

    def test_make_flux_surfaces(self):
        self.firstwall.build()
        assert hasattr(self.firstwall, "flux_surface_hfs")
        assert hasattr(self.firstwall, "flux_surface_hfs")
        assert len(self.firstwall.flux_surface_hfs) < len(
            self.firstwall.flux_surface_lfs
        )

    def test_q_parallel_calculation(self):
        self.firstwall.build()
        qpar = self.firstwall.q_parallel_calculation()
        assert len(qpar[0]) == len(self.firstwall.flux_surface_lfs)

    def test_modify_fw_profile(self):
        profile = self.firstwall.make_preliminary_profile()
        prof_up = self.firstwall.modify_fw_profile(profile, 11.5, -2.5)
        assert prof_up.x[0] == profile.x[0]
        assert prof_up.x[-1] == profile.x[-1]
        assert len(prof_up.x) == len(profile.x)


if __name__ == "__main__":
    pytest.main([__file__])
