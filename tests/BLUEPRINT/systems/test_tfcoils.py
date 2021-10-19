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
Testing routines for different TF coil optimisations
"""
import os
import time
from BLUEPRINT.geometry.geomtools import make_box_xz
import pytest
import matplotlib.pyplot as plt
import tests
import numpy as np
import shutil
import tempfile
import OCC


from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from bluemira.equilibria.shapes import flux_surface_manickam
from BLUEPRINT.cad.cadtools import get_properties


# Make temporary sub-directory for tests.
@pytest.fixture
def tempdir():
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)


class TestTFCoil:
    """
    Test the TF coil ripple optimisation for different TF coil shapes and test CAD
    generation.
    """

    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ["R_0", "Major radius", 9, "m", None, "Input"],
            ["B_0", "Toroidal field at R_0", 6, "T", None, "Input"],
            ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
            ["rho_j", "TF coil WP current density", 18.25, "MA/m^2", None, "Input"],
            ["tk_tf_nose", "TF coil inboard nose thickness", 0.6, "m", None, "Input"],
            ["tk_tf_wp", "TF coil winding pack thickness", 0.5, "m", None, "PROCESS"],
            ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.04, "m", None, "Input"],
            ["tk_tf_ins", "TF coil ground insulation thickness", 0.08, "m", None, "Input"],
            ["tk_tf_insgap", "TF coil WP insertion gap", 0.1, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
            ["r_tf_in", "Inboard radius of the TF coil inboard leg", 3.2, "m", None, "PROCESS"],
            ["ripple_limit", "Ripple limit constraint", 0.6, "%", None, "Input"],
            ['tk_tf_inboard', 'TF coil inboard thickness', 1.14, 'm', None, 'Input', 'PROCESS'],
        ]
        # fmt: on

        cls.parameters = ParameterFrame(params)

        read_path = get_BP_path("Geometry", subfolder="data/BLUEPRINT")
        # Load a target plasma separatrix
        lcfs = flux_surface_manickam(3.42, 0, 2.137, 2.9, 0.55, n=40)
        lcfs.close()

        # Load a keep-out zone for the TF coil shape
        name = os.sep.join([read_path, "KOZ.json"])
        ko_zone = Loop.from_file(name)

        cls.to_tf = {
            "name": "Example_PolySpline_TF",
            "plasma": lcfs,
            "koz_loop": ko_zone,
            "shape_type": "S",  # This is the shape parameterisation to use
            "conductivity": "SC",  # Resistive (R) or Superconducting (SC)
            "obj": "L",  # This is the optimisation objective: minimise length
            "wp_shape": "N",  # This is the winding pack shape choice for the inboard leg
            "ny": 1,  # This is the number of current filaments to use in y
            "nr": 1,  # This is the number of current filaments to use in x
            "nrip": 30,  # This is the number of points on the separatrix to calculate ripple for
            "read_folder": read_path,  # This is the path that the shape will be read from
            "write_folder": None,  # This is the path that the shape will be written to (replace in tests)
        }

    @pytest.mark.longrun
    def test_shape_optimisation(self):
        for shape in ["S", "D", "P"]:
            self.optimisation_assistant(shape)

    def optimisation_assistant(self, tf_type):
        self.to_tf["shape_type"] = tf_type

        tf = ToroidalFieldCoils(self.parameters, self.to_tf)
        tic = time.time()
        tf.optimise()
        tock = time.time() - tic

        if tests.PLOTTING:
            f, ax = plt.subplots()
            tf.plot_xz(ax)
            tf.plot_ripple(ax)
            plt.show()

        # Check quality of result
        assert tf.max_ripple <= 0.6021
        # Check for optimisation run time
        assert tock < 200

    def test_cad_components(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)
        # Ensure we've got all the geometry that we need to generate CAD
        tf1._generate_xz_plot_loops()

        CAD = tf1.build_CAD()
        expected_names = ["Toroidal field coils_case", "Toroidal field coils_wp"]
        assert CAD.component["names"] == expected_names


class TestTaperedPictureFrameTF:
    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ["R_0", "Major radius", 3.42, "m", None, "Input"],
            ["B_0", "Toroidal field at R_0", 2.2, "T", None, "Input"],
            ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
            ["tk_tf_nose", "Bucking Cylinder Thickness", 0.17, "m", None, "Input"],
            ["tk_tf_wp", "TF coil winding pack thickness", 0.4505, "m", None, "PROCESS"],
            ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02252, "m", None, "Input"],  # casthi
            ["tk_tf_ins", "TF coil ground insulation thickness", 0.0, "m", None, "Input"],
            ["tk_tf_insgap", "TF coil WP insertion gap", 0.0, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
            ["r_tf_in", "Inner Radius of the TF coil inboard leg", 0.176, "m", None, "PROCESS"],
            ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.6265, "m", None, "PROCESS"],
            ['TF_ripple_limit', 'TF coil ripple limit', 1.0, '%', None, 'Input'],
            ["npts", "Number of points", 200, "N/A", "Used for vessel and plasma", "Input"],
            ["h_cp_top", "Height of the Tapered Section", 6.199, "m", None, "PROCESS"],
            ["r_cp_top", "Radial Position of Top of taper", 0.8934, "m", None, "PROCESS"],
            ['tk_tf_outboard', 'TF coil outboard thickness', 1, 'm', None, 'Input', 'PROCESS'],
            ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'N/A', None, 'Input'],
            ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
            ["tk_tf_ob_casing", "TF leg conductor casing general thickness", 0.1, "m", None, "PROCESS"],
            ['tk_tf_inboard', 'TF coil inboard thickness', 0.4505, 'm', None, 'Input', 'PROCESS'],
        ]
        # fmt: on
        cls.parameters = ParameterFrame(params)
        read_path = get_BP_path("Geometry", subfolder="data/BLUEPRINT")
        lcfs = flux_surface_manickam(3.42, 0, 2.137, 2.9, 0.55, n=40)
        lcfs.close()
        name = os.sep.join([read_path, "KOZ_PF_test1.json"])
        ko_zone = Loop.from_file(name)
        cls.to_tf = {
            "name": "Example_PolySpline_TF",
            "plasma": lcfs,
            "koz_loop": ko_zone,
            "shape_type": "TP",  # This is the shape parameterisation to use
            "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
            "conductivity": "R",  # Resistive (R) or Superconducting (SC)
            "npoints": 400,
            "obj": "L",  # This is the optimisation objective: minimise length
            "ny": 1,  # This is the number of current filaments to use in y
            "nr": 1,  # This is the number of current filaments to use in x
            "nrip": 10,  # This is the number of points on the separatrix to calculate ripple for
            "read_folder": read_path,  # This is the path that the shape will be read from
            "write_folder": None,  # This is the path that the shape will be written to (replace in tests)
        }
        cls.lcfs = lcfs
        cls.ko_zone = ko_zone

    @pytest.mark.skipif(
        not (
            hasattr(OCC, "PYTHONOCC_VERSION_MAJOR") and OCC.PYTHONOCC_VERSION_MAJOR >= 7
        ),
        reason="OCC volume bug",
    )
    def test_tapered_TF(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)
        tf1.optimise()

        # Test CAD Model

        CAD = tf1.build_CAD()
        vol_b_cyl = get_properties(CAD.component["shapes"][0])["Volume"]
        vol_leg_conductor = get_properties(CAD.component["shapes"][1])["Volume"]
        vol_cp_conductor = get_properties(CAD.component["shapes"][2])["Volume"]
        vol_leg_casing = get_properties(CAD.component["shapes"][3])["Volume"]

        true_vol_b_cyl = 0.46925
        true_vol_leg_conductor = 12.3735
        true_vol_cp_conductor = 2.5788
        true_vol_leg_casing = 10.0850

        assert np.isclose(vol_b_cyl, true_vol_b_cyl, rtol=1e-3)
        assert np.isclose(vol_leg_conductor, true_vol_leg_conductor, rtol=1e-2)
        assert np.isclose(vol_cp_conductor, true_vol_cp_conductor, rtol=1e-2)
        assert np.isclose(vol_leg_casing, true_vol_leg_casing, rtol=1e-1)

        if tests.PLOTTING:
            f1, ax = plt.subplots()
            self.lcfs.plot(ax, edgecolor="r", fill=False)
            self.ko_zone.plot(ax, edgecolor="b", fill=False)
            tf1.plot_ripple(ax)
            plt.show()
        assert tf1.cage.get_max_ripple() <= 1.002 * 1.1

    def test_cad_components(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)

        # Ensure we've got all the geometry that we need to generate CAD
        tf1._generate_xz_plot_loops()

        CAD = tf1.build_CAD()
        expected_names = [
            "Toroidal field coils_b_cyl",
            "Toroidal field coils_leg_conductor",
            "Toroidal field coils_cp_conductor",
            "Toroidal field coils_case",
        ]
        assert CAD.component["names"] == expected_names


class TestSCPictureFrameTF:
    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ["R_0", "Major radius", 3.639, "m", None, "Input"],
            ["B_0", "Toroidal field at R_0", 2.0, "T", None, "Input"],
            ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
            ["tk_tf_nose", "TF coil inboard nose thickness", 0.0377, "m", None, "Input"],
            ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.02, 'm', None, 'Input'],
            ["tk_tf_wp", "TF coil winding pack thickness", 0.569, "m", None, "PROCESS"],
            ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02, "m", None, "Input"],
            ["tk_tf_ins", "TF coil ground insulation thickness", 0.008, "m", None, "Input"],
            ["tk_tf_insgap", "TF coil WP insertion gap", 1.0E-7, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
            ["r_tf_in", "Inboard radius of the TF coil inboard leg", 0.148, "m", None, "PROCESS"],
            ["tf_wp_depth", 'TF coil winding pack depth (in y)', 0.3644, 'm', 'Including insulation', 'PROCESS'],
            ["ripple_limit", "Ripple limit constraint", 0.6, "%", None, "Input"],
            ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
            ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
            ['tk_tf_inboard', 'TF coil inboard thickness', 0.6267, 'm', None, 'Input', 'PROCESS'],

        ]
        # fmt: on
        cls.parameters = ParameterFrame(params)
        read_path = get_BP_path("Geometry", subfolder="data/BLUEPRINT")
        lcfs = flux_surface_manickam(3.639, 0, 2.183, 2.8, 0.543, n=40)
        lcfs.close()
        name = os.sep.join([read_path, "KOZ_PF_test1.json"])
        ko_zone = Loop.from_file(name)
        cls.to_tf = {
            "name": "Example_PolySpline_TF",
            "plasma": lcfs,
            "koz_loop": ko_zone,
            "shape_type": "P",  # This is the shape parameterisation to use
            "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
            "conductivity": "SC",  # Resistive (R) or Superconducting (SC)
            "npoints": 200,
            "obj": "L",  # This is the optimisation objective: minimise length
            "ny": 3,  # This is the number of current filaments to use in y
            "nr": 2,  # This is the number of current filaments to use in x
            "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
            "read_folder": read_path,  # This is the path that the shape will be read from
            "write_folder": None,  # This is the path that the shape will be written to (replace in tests)
        }
        cls.lcfs = lcfs
        cls.ko_zone = ko_zone

    def test_pictureframe_SC_TF(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)
        tf1.optimise()

        # Test CAD Model

        CAD = tf1.build_CAD()
        vol_casing = get_properties(CAD.component["shapes"][0])["Volume"]
        vol_wp = get_properties(CAD.component["shapes"][1])["Volume"]
        true_vol_casing = 2.4991
        true_vol_wp = 9.5117

        assert np.isclose(vol_casing, true_vol_casing, rtol=1e-2)
        assert np.isclose(vol_wp, true_vol_wp, rtol=1e-2)

        if tests.PLOTTING:
            f1, ax = plt.subplots()
            self.lcfs.plot(ax, edgecolor="r", fill=False)
            self.ko_zone.plot(ax, edgecolor="b", fill=False)
            tf1.plot_ripple(ax)
            plt.show()
        assert tf1.cage.get_max_ripple() <= 1.002 * 1.1

    def test_cad_components(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)

        # Ensure we've got all the geometry that we need to generate CAD
        tf1._generate_xz_plot_loops()

        CAD = tf1.build_CAD()
        expected_names = [
            "Toroidal field coils_case",
            "Toroidal field coils_wp",
        ]
        assert CAD.component["names"] == expected_names


class TestCurvedPictureframeTF:
    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ["R_0", "Major radius", 3.639, "m", None, "Input"],
            ["B_0", "Toroidal field at R_0", 2.0, "T", None, "Input"],
            ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
            ["tk_tf_nose", "TF coil inboard nose thickness", 0.0377, "m", None, "Input"],
            ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.02, 'm', None, 'Input'],
            ["tk_tf_wp", "TF coil winding pack thickness", 0.569, "m", None, "PROCESS"],
            ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02, "m", None, "Input"],
            ["tk_tf_ins", "TF coil ground insulation thickness", 0.008, "m", None, "Input"],
            ["tk_tf_insgap", "TF coil WP insertion gap", 1.0E-7, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
            ["r_tf_in", "Inboard radius of the TF coil inboard leg", 0.148, "m", None, "PROCESS"],
            ["TF_ripple_limit", "Ripple limit constraint", 0.65, "%", None, "Input"],
            ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
            ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
            ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.75, "m", None, "PROCESS"],
            ["h_cp_top", "Height of the Tapered Section", 6.199, "m", None, "PROCESS"],
            ["r_cp_top", "Radial Position of Top of taper", 0.8934, "m", None, "PROCESS"],
            ["tf_wp_depth", "TF coil winding pack depth (in y)", 0.4625, "m", "Including insulation", "PROCESS"],
            ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
            ['h_tf_max_in', 'Plasma side TF coil maximum height', 12.0, 'm', None, 'PROCESS'],
            ["r_tf_curve", "Radial position of the CP-leg conductor joint", 2.5, "m", None, "PROCESS"],
            ['tk_tf_inboard', 'TF coil inboard thickness', 0.6267, 'm', None, 'Input', 'PROCESS'],

        ]
        # fmt: on
        cls.parameters = ParameterFrame(params)
        read_path = get_BP_path("Geometry", subfolder="data/BLUEPRINT")
        lcfs = flux_surface_manickam(3.42, 0, 2.137, 2.9, 0.55, n=40)
        lcfs.close()
        name = os.sep.join([read_path, "KOZ_PF_test1.json"])
        ko_zone = make_box_xz(1, 9, -9, 9)
        cls.to_tf = {
            "name": "Example_PolySpline_TF",
            "plasma": lcfs,
            "koz_loop": ko_zone,
            "shape_type": "CP",  # This is the shape parameterisation to use
            "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
            "conductivity": "SC",  # Resistive (R) or Superconducting (SC)
            "npoints": 800,
            "obj": "L",  # This is the optimisation objective: minimise length
            "ny": 3,  # This is the number of current filaments to use in y
            "nr": 2,  # This is the number of current filaments to use in x
            "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
            "read_folder": read_path,  # This is the path that the shape will be read from
            "write_folder": None,  # This is the path that the shape will be written to (replace in tests)
        }
        cls.lcfs = lcfs
        cls.ko_zone = ko_zone

    def test_curved_pictureframe_SC_TF(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)
        tf1.optimise()

        # Test CAD Model

        CAD = tf1.build_CAD()
        vol_casing = get_properties(CAD.component["shapes"][2])["Volume"]
        vol_tapered_cp = get_properties(CAD.component["shapes"][1])["Volume"]
        vol_leg_conductor = get_properties(CAD.component["shapes"][0])["Volume"]

        true_vol_tapered_cp = 2.2678
        true_vol_leg_conductor = 10.8514
        true_vol_casing = 3.0716
        assert np.isclose(vol_casing, true_vol_casing, rtol=1e-2)
        assert np.isclose(vol_tapered_cp, true_vol_tapered_cp, rtol=1e-2)
        assert np.isclose(vol_leg_conductor, true_vol_leg_conductor, rtol=1e-2)

        if tests.PLOTTING:
            f1, ax = plt.subplots()
            self.lcfs.plot(ax, edgecolor="r", fill=False)
            self.ko_zone.plot(ax, edgecolor="b", fill=False)
            tf1.plot_ripple(ax)
            plt.show()
        assert tf1.cage.get_max_ripple() <= 1.002 * 1.1

    def test_cad_components(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)

        # Ensure we've got all the geometry that we need to generate CAD
        tf1._generate_xz_plot_loops()

        CAD = tf1.build_CAD()
        expected_names = [
            "Toroidal field coils_leg_conductor",
            "Toroidal field coils_cp_conductor",
            "Toroidal field coils_case",
        ]
        assert CAD.component["names"] == expected_names


if __name__ == "__main__":
    pytest.main([__file__])
