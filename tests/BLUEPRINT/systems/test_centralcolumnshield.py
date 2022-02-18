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
from math import acos, pi, sqrt

import pytest

import bluemira.geometry._deprecated_loop as new_loop
from bluemira.base.parameter import ParameterFrame
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.centralcolumnshield import CentralColumnShield
from tests.BLUEPRINT.systems.test_firstwall import load_firstwall_dn, load_firstwall_sn


# Pretend that the fw profile is a circle
def setup_fw_loop(radius, circle_centre):
    circle_x, circle_z = circle_seg(radius, circle_centre, angle=360, npoints=1000)
    firstwall = Loop(x=circle_x, y=None, z=circle_z)
    firstwall.close()
    return firstwall


# Pretend that the vv profile is a square
def setup_vv_loop(length, centre):
    xmid = centre[0]
    zmid = centre[1]
    xmin = xmid - length / 2.0
    xmax = xmid + length / 2.0
    zmin = zmid - length / 2.0
    zmax = zmid + length / 2.0
    square_x = [xmin, xmin, xmax, xmax]
    square_z = [zmin, zmax, zmax, zmin]
    vessel = Loop(x=square_x, y=None, z=square_z)
    vessel.close()
    return vessel


# Retrieve default CCS params
def setup_cc_params():
    params = ParameterFrame(CentralColumnShield.default_params.to_records())
    params.g_ccs_vv_add = 0.1
    return params


# Create analytic profiles for inputs
def setup_cc_inputs(centre_fw, centre_vv, radius, length):
    firstwall = setup_fw_loop(radius, centre_fw)
    vessel = setup_vv_loop(length, centre_vv)
    inputs = {}
    inputs["FW_outer"] = firstwall
    inputs["VV_inner"] = vessel
    return inputs


def calc_analytic_area(xmid, zmid, radius, length, params):

    r_cut = params.r_ccs
    vv_offset = params.g_ccs_vv_add
    fw_offset = params.g_ccs_fw

    # Cut square:
    height = length - 2 * vv_offset
    xmin = xmid - (height / 2.0)
    width = r_cut - xmin
    area_rect = height * width

    # Cut circle (segment)
    r_seg = radius + fw_offset
    arc_x = abs(xmid - r_cut)
    ratio = arc_x / r_seg

    # The full circle area
    area_cut = pi * r_seg**2

    # Calculate segment area if we intersect circle
    area_seg = 0.0
    if ratio < 1.0:
        area_seg = (acos(ratio) - ratio * sqrt(1 - ratio**2)) * r_seg**2

    if r_cut > xmid:
        # Past the midpoint, subtract the segment from circle
        area_cut = area_cut - area_seg
    else:
        # Before midpoint, we just have a segment
        area_cut = area_seg

    # Area remaining after subtracting cut circle
    return area_rect - area_cut


# Helper function to test we raise correct errors
def fail_construct_ccs(inputs, params, error, errortype=GeometryError):

    with pytest.raises(errortype) as err:
        ccs = CentralColumnShield(params, inputs)

    return str(err.value) == error


# Test construction of ccs for fixed fw and vv but changing radius
ccs_test_radii = [1.2, 1.725, 2.25, 2.775]


@pytest.mark.parametrize("test_radii", ccs_test_radii)
def test_construct_ccs(test_radii):
    xmid = 2.25
    zmid = 0.0
    centre = (xmid, zmid)
    radius = 1.0
    length = 4.0
    params = setup_cc_params()
    inputs = setup_cc_inputs(centre, centre, radius, length)

    params.r_ccs = test_radii
    ccs = CentralColumnShield(params, inputs)

    # Get the 2D profile of the central column shield
    profile = ccs.geom["2D profile"]

    # Return the area of the profile
    area_test = profile.area

    # Analytically calculate area of segment cut from rectangle
    area_expect = calc_analytic_area(xmid, zmid, radius, length, params)

    # Compare results are within 1e-4
    assert area_test == pytest.approx(area_expect, rel=1e-4)


# Test failure when FW not inside VV
fw_params = [
    {"xmid": 2.25, "zmid": 0.0, "radius": 5.0},  # Radius too big
    {"xmid": 2.25, "zmid": 0.8501, "radius": 1.0},  # Touches top
    {"xmid": 2.25, "zmid": 0.8501, "radius": 1.0},  # Touches top
    {"xmid": 1.399, "zmid": 0.0, "radius": 1.0},  # Touches left
    {"xmid": 3.101, "zmid": 0.0, "radius": 1.0},  # Touches right
]


@pytest.mark.parametrize("fw_test_params", fw_params)
def test_construct_fw_not_in_vv(fw_test_params):
    xmid = fw_test_params["xmid"]
    zmid = fw_test_params["zmid"]
    radius = fw_test_params["radius"]
    length = 4.0
    centre_fw = (xmid, zmid)
    centre_vv = (2.25, 0.0)
    inputs = setup_cc_inputs(centre_fw, centre_vv, radius, length)
    params = setup_cc_params()
    error = CentralColumnShield.FIRSTWALL_ERR
    assert fail_construct_ccs(inputs, params, error)


# Test failure when midplane thickness too small
def test_construct_fail_thickness():
    xmid = 1.45
    zmid = 0.0
    radius = 1.0
    length = 4.0
    centre_fw = (xmid, zmid)
    centre_vv = (2.25, 0.0)
    inputs = setup_cc_inputs(centre_fw, centre_vv, radius, length)
    params = setup_cc_params()
    params.r_ccs = 1.5
    error = CentralColumnShield.THICKNESS_ERR
    assert fail_construct_ccs(inputs, params, error)


def test_construct_fail_input_types():
    inputs = {}
    inputs["FW_outer"] = None
    inputs["VV_inner"] = None
    params = setup_cc_params()
    error = CentralColumnShield.INPUT_TYPE_ERR
    assert fail_construct_ccs(inputs, params, error, TypeError)


offset_params = [
    {"fw_offset": 0.05, "vv_offset": -0.01},
    {"fw_offset": -0.01, "vv_offset": 0.05},
]


# Test failure when offsets are non-positive
@pytest.mark.parametrize("offset_test_params", offset_params)
def test_construct_fail_offset_value(offset_test_params):
    radius = 1.0
    length = 4.0
    centre = (2.25, 0.0)
    inputs = setup_cc_inputs(centre, centre, radius, length)
    params = setup_cc_params()
    params.g_ccs_vv_add = offset_test_params["vv_offset"]
    params.g_ccs_fw = offset_test_params["fw_offset"]
    error = CentralColumnShield.OFFSET_VAL_ERR
    assert fail_construct_ccs(inputs, params, error, ValueError)


ccs_test_fail_radii = [
    {"value": 3.3, "msg": CentralColumnShield.LARGE_RADIUS_ERR},
    {"value": 0.2, "msg": CentralColumnShield.SMALL_RADIUS_ERR},
]


# Test failure when central column shield radius is too large/small
@pytest.mark.parametrize("fail_radii", ccs_test_fail_radii)
def test_construct_fail_ccs_radius(fail_radii):
    radius = 1.0
    length = 4.0
    centre = (2.25, 0.0)
    inputs = setup_cc_inputs(centre, centre, radius, length)
    params = setup_cc_params()
    params.r_ccs = fail_radii["value"]
    error = fail_radii["msg"]
    assert fail_construct_ccs(inputs, params, error, ValueError)


# Integration test
@pytest.mark.parametrize("is_single_null", [True, False])
def test_build_from_fw(is_single_null):
    if is_single_null:
        firstwall = load_firstwall_sn()
    else:
        firstwall = load_firstwall_dn()
    firstwall.build()

    vv_inner = firstwall.inputs["vv_inner"]
    fw_outer = firstwall.geom["2D profile"].outer
    div_cassettes = firstwall.divertor_cassettes

    inputs = {"FW_outer": fw_outer, "VV_inner": vv_inner, "Div_cassettes": div_cassettes}
    params = ParameterFrame(CentralColumnShield.default_params.to_records())
    params.r_ccs = 9
    ccs = CentralColumnShield(params, inputs)
    assert isinstance(ccs.geom["2D profile"], (Loop, new_loop.Loop))
