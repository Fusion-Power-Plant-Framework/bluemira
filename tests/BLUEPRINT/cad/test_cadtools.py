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
import pickle  # noqa :S403
import warnings
from unittest.mock import patch

import numpy as np
import OCC

from bluemira.base.file import get_bluemira_path
from bluemira.utilities.tools import compare_dicts
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    check_good_STL,
    extrude,
    get_properties,
    loft,
    make_bezier_curve,
    make_box,
    make_circle,
    make_face,
    make_mixed_face,
    make_mixed_shell,
    make_shell,
    make_spline_face,
    make_wire,
    revolve,
    save_as_STEP,
    save_as_STEP_assembly,
    save_as_STL,
    sweep,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.offset import offset_clipper
from BLUEPRINT.geometry.shell import Shell

TESTS = get_bluemira_path("BLUEPRINT/cad/test_data", subfolder="tests")


class TestOrientedBox:
    def test_box(self):
        # basically checks for gp_Pnt errors (Standard_Real)
        make_box((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))
        make_box((0.0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))
        make_box(
            (0.0, 0, -15),  # This was failing at some point...
            (29.423558412096913, 5.852709660483847, 0),
            (-5.96737102138974, 29.423558412096913, 0),
            (0, 0, 30),
        )


class TestCADOperations:
    path = TESTS

    def test_loft(self):
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = Loop(x=[0.2, 2.8, 2.8, 0.2, 0.2], y=[0.2, 0.2, 2.8, 2.8, 0.2], z=3)
        so.translate([1, 3, 0])
        si, so = make_wire(si), make_wire(so)
        l_1 = loft(si, so)
        e_1 = Loop(x=[-1, 1, 1, -1, -1], z=[-1, -1, 1, 1, -1])
        e_1 = extrude(make_face(e_1), vec=[1, 3, 0])
        l_1 = boolean_cut(l_1, e_1)
        save_as_STL(l_1, os.sep.join([self.path, "LOFT_SOLID"]))

        su = Loop(x=[0.2, 2.8, 0.2, 0.2], y=[0.2, 0.2, 2.8, 0.2], z=5)
        su = make_wire(su)

        l_1 = loft(si, so, su)
        e_1 = Loop(x=[-1, 1, 1, -1, -1], z=[-1, -1, 1, 1, -1])
        e_1 = extrude(make_face(e_1), vec=[1, 3, 0])
        l_1 = boolean_cut(l_1, e_1)
        save_as_STL(l_1, os.sep.join([self.path, "LOFT_WHACK"]))

    def test_sweep(self):
        a = np.array([[0, 0, 2, 3, 3], [0, 2, 3, 3, 5], [0, 0, 0, 0, 0]])
        bezier = make_bezier_curve(a)
        circle = make_circle([0, 0, 0], [0, 1, 0], 1)
        s_1 = sweep(circle, bezier)
        save_as_STL(s_1, os.sep.join([self.path, "SWEEP"]))

    def test_operations(self):
        si = Loop(x=[1, 2, 2, 1, 1], z=[1, 1, 2, 2, 1])
        so = si.offset(0.01)
        face2 = make_face(so)
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = si.offset(0.01)

        shell_2 = Shell(si, so)

        face = make_face(si)

        e_1 = extrude(face, length=2, axis="z")
        save_as_STL(e_1, os.sep.join([self.path, "MC_PAD"]))

        r_1 = revolve(face2, axis=None, angle=90)
        save_as_STL(r_1, os.sep.join([self.path, "FACE_REVOLVE"]))

        f_r = revolve(face2, axis=None, angle=359)
        save_as_STL(f_r, os.sep.join([self.path, "FULL_FACE_REVOLVE"]))

        f_r = revolve(face2, axis=None, angle=400)
        save_as_STL(f_r, os.sep.join([self.path, "FULL_FACE_OVERREVOLVE"]))

        shell_old = make_shell(shell_2)

        ext_old = extrude(shell_old, length=2, axis="z")
        save_as_STL(ext_old, os.sep.join([self.path, "SHELL_EXTRUSION_OLD"]))
        # EXT = extrude(SHELL, length=2, axis='z')
        # save_as_STL(EXT, os.sep.join([self.path, 'SHELL_EXTRUSION']))

        si = Loop(x=[1, 2, 2, 1, 1], z=[1, 1, 2, 2, 1])
        so = si.offset(0.3)
        shell_1 = Shell(si, so)
        face1 = make_face(so)
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = si.offset(0.3)
        shell_2 = Shell(si, so)

        face = make_face(si)

        shell_one = make_shell(shell_1)
        shell_two = make_shell(shell_2)

        extrude_one = extrude(shell_one, length=3, axis="y")
        extrude_two = extrude(shell_two, length=1.5, axis="z")
        cut1 = extrude(face1, length=3, axis="y")
        cut2 = extrude(face, length=1.5, axis="z")
        extrude_two = boolean_cut(extrude_two, cut1)

        extrude_one = boolean_cut(extrude_one, cut2)

        fuse_1 = boolean_fuse(extrude_one, extrude_two)
        # save_as_STEP(S, os.sep.join([self.path, 'COMPLEX_ONE']))
        save_as_STL(fuse_1, os.sep.join([self.path, "COMPLEX_ONE"]))

        loop = Loop.from_file(os.sep.join([self.path, "plasmaloop.json"]))
        spline = make_spline_face(loop)
        loop.translate([0, 1, 0], update=True)

        face = make_face(loop)

        loop.translate([0, -2, 0], update=True)

        r_1 = revolve(make_face(loop, spline=True), axis=None, angle=360)
        w_1 = Loop(
            x=[8, 10, 10, 8, 8], y=[-10, -10, -10, -10, -10], z=[-2, -2, 2, 2, -2]
        )
        cut_1 = extrude(make_face(w_1), length=20, axis="y")
        # =============================================================================
        #
        #         CUTPLASMA = boolean_cut(R, CUT)
        #         S = Shell(W, W.offset(0.5))
        #         WW = extrude(make_shell(S), length=20, axis='y')
        #
        #         CUTPLASMA = boolean_fuse(CUTPLASMA, WW)
        #         show_CAD_OCC(CUTPLASMA)
        #         save_as_STL(CUTPLASMA, os.sep.join([self.path, 'COMPLEX_TWO']))
        #         save_as_STEP(CUTPLASMA, os.sep.join([self.path, 'COMPLEX_TWO']))
        # =============================================================================

        ploop = Loop(x=[9.0, 9.0, 7.0], y=[-10.0, -5.0, -2.0], z=[0.0, 0.0, 1.0])
        path = make_wire(ploop)
        sweep_1 = sweep(make_face(w_1), path)

        save_as_STL(sweep_1, os.sep.join([self.path, "SWEEP"]))
        sweep_1 = sweep(face, path)

        loop = Loop.from_file(os.sep.join([self.path, "princetonD.json"]))
        loop_2 = loop.offset(0.5)
        shell = make_shell(Shell(loop, loop_2))
        wp = extrude(shell, length=1, axis="y")
        loop_3 = loop_2.offset(1)
        loop_3.translate([0, -0.5, 0])
        loop_4 = loop_3.offset(-2.5)
        shell = Shell(loop_4, loop_3)
        case = extrude(make_shell(shell), length=2, axis="y")
        case = boolean_cut(case, wp)

        save_as_STL(case, os.sep.join([self.path, "CASE"]))
        save_as_STL(wp, os.sep.join([self.path, "WINDINGPACK"]))
        filename = os.sep.join([self.path, "NEST_TEST"])

        # aocxchange does not support logging features in latest pytest versions. It
        # creates log entries with multiple arguments, that pytest then tries to format.
        # We patch the newer pytest function here with a dummy function
        with patch(
            "_pytest.logging.ColoredLevelFormatter.format",
            lambda self, record: str(record),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                partname = "Test_NAME_OF_PART_BP_REACTOR"
                save_as_STEP_assembly(case, wp, filename=filename, partname=partname)

        if hasattr(OCC, "PYTHONOCC_VERSION_MAJOR") and OCC.PYTHONOCC_VERSION_MAJOR >= 7:
            with open(filename + ".STP", "r") as file:
                assert self._line_checker(file, partname)

    # This fails on my laptop...
    # =============================================================================
    #     def test_mixed(self):
    #         IB = Loop.from_file(os.sep.join([self.path, 'IB_test.json']))
    #         OB = Loop.from_file(os.sep.join([self.path, 'OB_test.json']))
    #         f = make_mixed_face(IB)
    #         part = revolve(f, angle=100)
    #         f2 = make_mixed_face(OB)
    #         part2 = revolve(f2, angle=15)
    #         save_as_STL(part, os.sep.join([self.path, 'IBmixed']))
    #         save_as_STL(part2, os.sep.join([self.path, 'OBmixed']))
    # =============================================================================

    @staticmethod
    def _line_checker(file, string):
        for line in file:
            if string in line:
                return True
        return False

    def test_STL_meshes(self):  # noqa :N802
        results = {}
        for file in [f for f in os.listdir(self.path) if f.endswith(".stl")]:
            filename = os.sep.join([self.path, file])
            watertight = check_good_STL(filename)
            results[file] = watertight

        message = "\nWasserdicht oder nicht: \n"
        dic = str(results)
        dic = message + dic.replace(",", "\n")
        assert all(list(results.values())), dic


class TestSTEPExport:
    path = TESTS

    def test_214protocol(self):
        si = Loop(x=[1, 2, 2, 1, 1], z=[1, 1, 2, 2, 1])
        so = si.offset(0.01)
        face2 = make_face(so)
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = si.offset(0.01)
        shell_2 = Shell(si, so)
        wire = make_wire(si)
        wire2 = make_wire(so)
        face = make_face(si)

        e_1 = extrude(face, length=2, axis="z")
        file = os.sep.join([self.path, "mc_pad_step_214"])
        save_as_STEP(e_1, file, standard="AP214", partname="testname214")
        # FILE SCHEMA string alternates and does not work for testing!
        with open(file + ".stp", "r", encoding="cp1252") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("#1 = APPLICATION_PROTOCOL_DEFINITION"):
                    idx = i
                if line.startswith("#7 = PRODUCT"):
                    assert "testname214" in line
            apd = lines[idx + 1]
            assert "2000" in apd

    def test_203protocol(self):
        si = Loop(x=[1, 2, 2, 1, 1], z=[1, 1, 2, 2, 1])
        so = si.offset(0.01)
        face2 = make_face(so)
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = si.offset(0.01)
        shell_2 = Shell(si, so)
        wire = make_wire(si)
        wire2 = make_wire(so)
        face = make_face(si)

        e_1 = extrude(face, length=2, axis="z")
        file = os.sep.join([self.path, "mc_pad_step_203"])
        save_as_STEP(e_1, file, standard="AP203", partname="testname203")
        with open(file + ".stp", "r", encoding="cp1252") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("#1 = APPLICATION_PROTOCOL_DEFINITION"):
                    idx = i
                if line.startswith("#7 = PRODUCT"):
                    assert "testname203" in line
            apd = lines[idx + 1]
            assert "1994" in apd

    def test_defaultname(self):
        si = Loop(x=[1, 2, 2, 1, 1], z=[1, 1, 2, 2, 1])
        so = si.offset(0.01)
        face2 = make_face(so)
        si = Loop(x=[1.2, 1.8, 1.8, 1.2, 1.2], y=[1.2, 1.2, 1.8, 1.8, 1.2])
        so = si.offset(0.01)
        shell_2 = Shell(si, so)
        wire = make_wire(si)
        wire2 = make_wire(so)
        face = make_face(si)

        e_1 = extrude(face, length=2, axis="z")
        file = os.sep.join([self.path, "name_test"])
        save_as_STEP(e_1, file, standard="AP203", partname=None)
        with open(file + ".stp", "r", encoding="cp1252") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("#7 = PRODUCT"):
                    assert "BLUEPRINT_name_test" in line


# Regression tests may fail if tweaking MFM algorithm selection parameters
class TestMixedFaces:
    path = TESTS

    def test_face(self):
        """
        Tests some blanket faces which combine splines and polygons. Checks the
        3-D geometric properties of the results with some regression results
        done when everything was working correctly
        """
        i_b = Loop.from_file(os.sep.join([self.path, "IB_test.json"]))
        o_b = Loop.from_file(os.sep.join([self.path, "OB_test.json"]))
        f = make_mixed_face(i_b)
        part = revolve(f, angle=100)
        f2 = make_mixed_face(o_b)
        part2 = revolve(f2, angle=15)

        props = get_properties(part)

        true_props = {
            "CoG": {
                "x": 3.5051207124203336,
                "y": 4.17724020150331,
                "z": 1.1683908129112384,
            },
            "Volume": 106.04366428842452,
            "Area": 348.29727090293534,
        }
        assert compare_dicts(props, true_props, almost_equal=True, verbose=False)

        props = get_properties(part2)

        true_props = {
            "CoG": {
                "x": 11.583613492691775,
                "y": 1.52501164739993,
                "z": -0.18976515896142937,
            },
            "Volume": 42.949740303457624,
            "Area": 121.54648513674483,
        }

        assert compare_dicts(props, true_props, almost_equal=True, verbose=False)

    def test_shell(self):
        """
        Tests some shell mixed faces
        """
        fn = os.sep.join([self.path, "shell_mixed_test.pkl"])
        with open(fn, "rb") as file:
            d = pickle.load(file)  # noqa :S301
        # This was back in the day when I used to pickle Loop dicts only..
        inner = Loop(**d)
        outer = offset_clipper(inner, 0.2, method="miter")
        shell = Shell(inner, outer)
        true_area = shell.area
        face = make_mixed_shell(shell)
        area = get_properties(face)["Area"]
        assert np.isclose(true_area, area, rtol=5e-3)
        # show_CAD_OCC(F)

    def test_shell2(self):
        """
        Tests another shell, trickier this time
        """
        fn = os.sep.join([self.path, "failing_mixed_shell.json"])
        shell = Shell.from_file(fn)
        true_area = shell.area
        face = make_mixed_shell(shell)
        area = get_properties(face)["Area"]
        assert np.isclose(true_area, area, rtol=5e-3)

    def test_shell3(self):
        """
        Opposite tricky shell
        """
        fn = os.sep.join([self.path, "tf_wp_tricky.json"])
        shell = Shell.from_file(fn)
        true_area = shell.area
        face = make_mixed_shell(shell)
        area = get_properties(face)["Area"]
        assert np.isclose(true_area, area, rtol=5e-3)

    def test_face2(self):
        """
        Tests a TF face which combine splines and polygons. Checks the
        3-D geometric properties of the results with some regression results
        done when everything was working correctly
        """
        fn = os.sep.join([self.path, "TF_case_in_test.json"])
        inner = Loop.from_file(fn)
        face = make_mixed_face(inner)

        part = extrude(face, vec=[0, 1, 0])

        props = get_properties(part)

        if OCC.VERSION == "0.18.1":
            true_props = {
                "CoG": {"x": 9.458790543925387, "y": 0.5, "z": -2.130186477920172e-05},
                "Volume": 185.18032364061102,
                "Area": 423.9937851471826,
            }
        else:
            true_props = {
                "CoG": {"x": 9.458901793964065, "y": 0.5, "z": -2.1267084509526304e-05},
                "Volume": 184.931161612167,
                "Area": 423.42739905831604,
            }

        assert compare_dicts(props, true_props, almost_equal=True, verbose=False)

    def test_face3(self):
        """
        Tests a divertor face which combine splines and polygons. Checks the
        3-D geometric properties of the results with some regression results
        done when everything was working correctly
        """
        fn1 = os.sep.join([self.path, "div_test_mfm.json"])
        fn2 = os.sep.join([self.path, "div_test_mfm2.json"])

        div1 = Loop.from_file(fn1)
        div2 = Loop.from_file(fn2)

        face1 = make_mixed_face(div1)
        face2 = make_mixed_face(div2)

        part1 = extrude(face1, vec=[0, 2, 0])
        part2 = extrude(face2, vec=[0, 2, 0])

        props1 = get_properties(part1)
        props2 = get_properties(part2)

        true_props = {
            "CoG": {
                "x": 8.032021916381938,
                "y": 0.9900000000000005,
                "z": -6.44420412332881,
            },
            "Volume": 4.587743503495517,
            "Area": 29.217329193714182,
        }
        assert compare_dicts(props1, true_props, almost_equal=True, verbose=False)

        true_props = {
            "CoG": {
                "x": 8.035475284774666,
                "y": 0.9900000000000002,
                "z": -6.447539796697995,
            },
            "Volume": 4.6210952635486455,
            "Area": 29.178274456894467,
        }

        assert compare_dicts(props2, true_props, almost_equal=True, verbose=False)

    def test_face4(self):
        """
        Tests a particularly tricky face which can result in a SIGSEGV...
        """
        fn1 = os.sep.join([self.path, "divertor_seg_fault_LDS.json"])
        div1 = Loop.from_file(fn1)
        face = make_mixed_face(div1)
        area = get_properties(face)["Area"]
        assert np.isclose(div1.area, area, rtol=5e-3)


class TestGetProperties:
    def test_cube(self):
        square = Loop(x=[0, 1, 1, 0, 0], z=[0, 0, 1, 1, 0])
        face = make_face(square)
        cube = extrude(face, vec=[0, 1, 0])
        properties = get_properties(cube)
        assert np.isclose(properties["Volume"], 1.0)
        assert np.isclose(properties["CoG"]["x"], 0.5)
        assert np.isclose(properties["CoG"]["y"], 0.5)
        assert np.isclose(properties["CoG"]["z"], 0.5)

    def test_cylinder(self):
        circle = make_circle([0, 0, 0], [0, 0, 1], 1)
        cylinder = extrude(circle, vec=[0, 0, 1])
        properties = get_properties(cylinder)
        assert np.isclose(properties["Volume"], np.pi)
        assert np.isclose(properties["CoG"]["x"], 0.0)
        assert np.isclose(properties["CoG"]["y"], 0.0)
        assert np.isclose(properties["CoG"]["z"], 0.5)

    def test_hollow_cylinder(self):
        circle = make_circle([0, 0, 0], [0, 0, 1], 1)
        circle2 = make_circle([0, 0, 0], [0, 0, 1], 0.5)
        hollow_circle = boolean_cut(circle, circle2)
        hollow_cylinder = extrude(hollow_circle, vec=[0, 0, 1])
        properties = get_properties(hollow_cylinder)
        true_volume = np.pi * (1**2 - 0.5**2)
        assert np.isclose(properties["Volume"], true_volume)
        assert np.isclose(properties["CoG"]["x"], 0.0)
        assert np.isclose(properties["CoG"]["y"], 0.0)
        assert np.isclose(properties["CoG"]["z"], 0.5)
