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
"""
Testing routines for the central column shield system
"""
import numpy as np
import pytest

from bluemira.base.parameter import ParameterFrame
from bluemira.geometry._deprecated_tools import get_intersect
from bluemira.geometry.error import GeometryError
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.geomtools import circle_seg, make_box_xz
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.blanket import STBreedingBlanket


def setup_fw_loop(x_low, x_mid, x_high, z_mid, z_high):
    fw_x = [x_low, x_mid, x_high, x_high, x_mid, x_low]
    fw_z = [-z_high, -z_high, -z_mid, z_mid, z_high, z_high]
    fw_dummy = Loop(x=fw_x, z=fw_z)
    fw_dummy.close()
    return fw_dummy


# Create analytic profiles for inputs
def setup_bb_inputs(x_low, x_mid, x_high, z_mid, z_high, x_diff=2.0, z_diff=2.0):
    # Dummy first wall loop (outer edge is trapezium shape)
    fw_dummy = setup_fw_loop(x_low, x_mid, x_high, z_mid, z_high)

    # Dummy vv profile is a rectangle
    vv_x_low = x_low - x_diff
    vv_x_high = x_high + x_diff
    vv_z_high = z_high + z_diff
    vv_z_low = -vv_z_high
    vv_dummy = make_box_xz(vv_x_low, vv_x_high, vv_z_low, vv_z_high)

    inputs = {}
    inputs["fw_outboard"] = fw_dummy
    inputs["vv_inner"] = vv_dummy
    return inputs


def calc_area(h1, h2, h3, thickness):
    return (h1 + h2 + h3) * thickness


def test_build_banana_blanket():

    # Make a blanket with dummy first wall and vacuum vessel
    x_low = 7
    x_mid = 8
    x_high = 10
    z_mid = 3
    z_high = 5
    inputs = setup_bb_inputs(x_low, x_mid, x_high, z_mid, z_high)
    params = ParameterFrame(STBreedingBlanket.default_params.to_records())
    blanket = STBreedingBlanket(params, inputs)

    # Define heights of sections in our first wall
    h1 = z_high - z_mid
    h2 = 2 * z_mid
    h3 = h1

    # Fetch thicknesses
    bz_thickness = blanket.params.tk_bb_bz
    man_thickness = blanket.params.tk_bb_man

    # Check existence of attributes
    assert isinstance(blanket.geom, dict)
    assert "OB 2D profile bz" in blanket.geom
    assert "OB 2D profile manifold" in blanket.geom
    bz = blanket.geom["OB 2D profile bz"]
    man = blanket.geom["OB 2D profile manifold"]
    assert isinstance(bz, Loop)
    assert isinstance(man, Loop)

    # Check both loops are inside the vv by making a shell
    assert Shell(inputs["vv_inner"], bz)
    assert Shell(inputs["vv_inner"], man)

    # Check neither intersect with the first wall
    x_ints, z_ints = get_intersect(inputs["fw_outboard"], bz)
    assert len(x_ints) == 0 and len(z_ints) == 0
    x_ints, z_ints = get_intersect(inputs["fw_outboard"], man)
    assert len(x_ints) == 0 and len(z_ints) == 0

    # Check they intersect each other at four points
    x_ints, z_ints = get_intersect(bz, man)
    assert len(x_ints) == 4 and len(z_ints) == 4

    # Check areas
    area_bz = bz.area
    area_bz_check = calc_area(h1, h2, h3, bz_thickness)
    assert pytest.approx(area_bz, 1e-5) == area_bz_check

    area_man = man.area
    area_man_check = calc_area(h1, h2, h3, man_thickness)
    assert pytest.approx(area_man, 1e-5) == area_man_check


def test_build_banana_blanket_fail_params():

    # Make a blanket with dummy first wall and vacuum vessel
    inputs = setup_bb_inputs(7, 8, 10, 3, 5)
    params = ParameterFrame(STBreedingBlanket.default_params.to_records())

    # Change thicknesses to be too big
    params.tk_bb_bz = 2.0
    params.tk_bb_man = 1.0

    with pytest.raises(GeometryError):
        blanket = STBreedingBlanket(params, inputs)


def test_build_banana_blanket_fail_subtract():

    vv_dummy = make_box_xz(3, 7, -3, 3)

    # With a circular first wall we will fail to extract the
    # concave banana shape
    circle_x, circle_z = circle_seg(1, (4, 0))
    fw_dummy = Loop(x=circle_x, y=None, z=circle_z)
    fw_dummy.close()

    inputs = {}
    inputs["fw_outboard"] = fw_dummy
    inputs["vv_inner"] = vv_dummy

    params = ParameterFrame(STBreedingBlanket.default_params.to_records())
    with pytest.raises(SystemsError):
        blanket = STBreedingBlanket(params, inputs)


def test_build_immersion_blanket():

    # Make a blanket with dummy first wall and vacuum vessel
    x_low = 7
    x_mid = 8
    x_high = 10
    z_mid = 3
    z_high = 5
    x_diff = 2
    z_diff = 2
    inputs = setup_bb_inputs(x_low, x_mid, x_high, z_mid, z_high, x_diff, z_diff)
    params = ParameterFrame(STBreedingBlanket.default_params.to_records())
    params.blanket_type = "immersion"
    blanket = STBreedingBlanket(params, inputs)

    # Check existence of attributes
    assert isinstance(blanket.geom, dict)
    assert "OB 2D profile bz" in blanket.geom
    assert "OB 2D profile manifold upper" in blanket.geom
    assert "OB 2D profile manifold lower" in blanket.geom
    bz = blanket.geom["OB 2D profile bz"]
    man_up = blanket.geom["OB 2D profile manifold upper"]
    man_low = blanket.geom["OB 2D profile manifold lower"]
    assert isinstance(bz, Loop)
    assert isinstance(man_up, Loop)
    assert isinstance(man_low, Loop)

    # Check area of  breeding zone: rectangle minus trapezium
    # 1) trapezium
    h_1 = 2 * z_high  # long edge
    h_2 = 2 * z_mid  # short edge
    w_trap = x_high - x_mid
    area_trap = w_trap * (h_1 + h_2) / 2.0
    # 2) rectangle
    fw_sep = blanket.params.g_bb_fw
    vv_x_high = x_high + x_diff
    w_bz = vv_x_high - (x_mid + fw_sep)
    h_bz = h_1
    area_rect = h_bz * w_bz
    # 3 ) bz
    area_bz_check = area_rect - area_trap
    area_bz = bz.area
    assert pytest.approx(area_bz, 1e-5) == area_bz_check

    # Check area of manifold parts (two identical rectangles)
    w_man = blanket.params.tk_bb_man
    vv_z_high = z_high + z_diff
    h_vv = 2 * vv_z_high
    h_man = (h_vv - h_1) / 2
    area_man_check = w_man * h_man
    area_man_up = man_up.area
    area_man_low = man_low.area
    assert pytest.approx(area_man_up, 1e-5) == area_man_check
    assert pytest.approx(area_man_low, 1e-5) == area_man_check

    # Check vertical orientation of manifold parts
    z_max_up = np.max(man_up.z)
    z_max_low = np.max(man_low.z)
    assert z_max_up > z_max_low


def test_unknown_blanket():
    # Make a blanket with dummy first wall and vacuum vessel
    inputs = setup_bb_inputs(7, 8, 10, 3, 5)
    params = ParameterFrame(STBreedingBlanket.default_params.to_records())

    # Change blanket type
    params.blanket_type = "Dummy"

    with pytest.raises(SystemsError):
        blanket = STBreedingBlanket(params, inputs)
