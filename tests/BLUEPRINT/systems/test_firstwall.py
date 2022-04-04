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

import pytest

import BLUEPRINT.geometry.loop as old_loop  # noqa :N813
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.firstwall import FirstWallDN, FirstWallSN, get_tangent_vector
from BLUEPRINT.systems.optimisation_callbacks import FW_optimiser

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
        "DEMO_like_divertor": True,
        "div_vertical_outer_target": False,
        "div_vertical_inner_target": False,
        "dx_mp": 0.05,
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
        "SN": False,
        "DEMO_like_divertor": True,
        "div_vertical_outer_target": True,
        "div_vertical_inner_target": False,
        "dx_mp": 0.05,
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


# Method to check attributes of a first wall
def check_firstwall(firstwall):
    # Check all the attributes are sensible
    assert isinstance(firstwall.profile, (Loop, old_loop.Loop))
    assert isinstance(firstwall.inner_profile, (Loop, old_loop.Loop))
    assert isinstance(firstwall.geom, dict)
    assert isinstance(firstwall.geom["2D profile"], Shell)
    assert isinstance(firstwall.geom["Inboard wall"], (Loop, old_loop.Loop))
    assert isinstance(firstwall.geom["Outboard wall"], (Loop, old_loop.Loop))
    assert isinstance(firstwall.geom["Preliminary profile"], (Loop, old_loop.Loop))
    assert isinstance(firstwall.geom["Inner profile"], (Loop, old_loop.Loop))
    assert isinstance(firstwall.divertor_loops, list)
    assert isinstance(firstwall.divertor_cassettes, list)

    inboard = firstwall.geom["Inboard wall"]
    outboard = firstwall.geom["Outboard wall"]

    return True


class TestFirstWallSN:
    # Setup for *every* test in class
    def setup_method(self):
        self.firstwall = load_firstwall_sn()
        self.firstwall.build(FW_optimiser)

    def test_build_callback(self):
        wall2 = load_firstwall_sn()
        wall2.build()
        assert wall2.__getstate__() != self.firstwall.__getstate__()
        wall3 = load_firstwall_sn()

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
        self.firstwall.build(FW_optimiser)
        assert check_firstwall(self.firstwall)


class TestFirstWallDN:
    # Setup for *every* test in class
    def setup_method(self):
        self.firstwall = load_firstwall_dn()
        self.firstwall.build(FW_optimiser)

    def test_make_preliminary_profile(self):
        prof = self.firstwall.make_preliminary_profile()
        assert len(prof.x) == len(prof.z)
        assert prof.x[0] == prof.x[-1]

    def test_make_divertor_outer_target(self):
        div_builder = self.firstwall.divertor_builder
        flux_loops = div_builder.pick_flux_loops()
        inner_strike, outer_strike = div_builder.find_strike_points(flux_loops)
        tangent = get_tangent_vector(outer_strike, flux_loops[0])
        tar_out = div_builder.make_divertor_target(
            outer_strike,
            tangent,
            vertical_target=True,
            outer_target=True,
        )
        tar_pfr_end = tar_out[0]
        tar_sol_end = tar_out[1]
        assert tangent[0] < 0
        assert tar_sol_end[0] > self.firstwall.points["x_point"]["x"]
        assert tar_sol_end[0] > tar_pfr_end[0]

    def test_make_divertor_inner_target(self):
        div_builder = self.firstwall.divertor_builder
        flux_loops = div_builder.pick_flux_loops()
        inner_strike, outer_strike = div_builder.find_strike_points(flux_loops)
        tangent = get_tangent_vector(inner_strike, flux_loops[1])
        tar_in = div_builder.make_divertor_target(
            inner_strike,
            tangent,
            vertical_target=False,
            outer_target=False,
        )
        tar_pfr_end = tar_in[0]
        tar_sol_end = tar_in[1]
        assert tangent[0] < 0
        assert tar_pfr_end[0] < self.firstwall.points["x_point"]["x"]
        assert tar_pfr_end[0] > tar_sol_end[0]

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
        builder = self.firstwall.divertor_builder
        div = builder.make_divertor(fw_loop)[0]

        # Check the loop is closed
        assert div.closed

        # Get the inner/outer target end points
        flux_loops = builder.pick_flux_loops()
        inner_strike, outer_strike = builder.find_strike_points(flux_loops)
        tangent_out = get_tangent_vector(outer_strike, flux_loops[0])
        tar_out = builder.make_divertor_target(
            outer_strike, tangent_out, vertical_target=True, outer_target=True
        )
        tangent_in = get_tangent_vector(inner_strike, flux_loops[1])
        tar_in = builder.make_divertor_target(
            inner_strike, tangent_in, vertical_target=False, outer_target=False
        )
        # Get the minimum and maximum x,z
        max_x = round(tar_out[1][0], 5)
        min_x = round(tar_in[1][0], 5)
        max_z = round(self.firstwall.points["x_point"]["z_low"], 5)
        min_z = round(tar_out[0][1], 5)
        # min_z = round(max_z - 2.25,5)

    # Test build for different combinations of thicknesses
    @pytest.mark.parametrize("tk_in", [0.1])
    @pytest.mark.parametrize("tk_out_diff", [0.0, 0.05, -0.05])
    @pytest.mark.parametrize("tk_div_diff", [0.0, 0.05, -0.05])
    def test_build_firstwall(self, tk_in, tk_out_diff, tk_div_diff):
        self.firstwall.params.tk_fw_in = tk_in
        self.firstwall.params.tk_fw_out = tk_in + tk_out_diff
        self.firstwall.params.tk_fw_div = tk_in + tk_div_diff
        self.firstwall.build(FW_optimiser)
        assert check_firstwall(self.firstwall)

    def test_modify_fw_profile(self):
        profile = self.firstwall.make_preliminary_profile()
        prof_up = self.firstwall.modify_fw_profile(profile, 11.5, -2.5)
        assert prof_up.x[0] == profile.x[0]
        assert prof_up.x[-1] == profile.x[-1]
        assert len(prof_up.x) == len(profile.x)
