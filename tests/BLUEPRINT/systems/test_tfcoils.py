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
import pytest
import matplotlib.pyplot as plt
import tests
import numpy as np
import shutil
import tempfile

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.parameter import ParameterFrame
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from BLUEPRINT.equilibria.shapes import flux_surface_manickam
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
            "obj": "L",  # This is the optimisation objective: minimise length
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
            ["R_0", "Major radius", 2.5, "m", None, "Input"],
            ["B_0", "Toroidal field at R_0", 3, "T", None, "Input"],
            ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
            ["rho_j", "TF coil WP current density", 30, "MA/m^2", None, "Input"],
            ["tk_tf_nose", "Bucking Cylinder Thickness", 0.17, "m", None, "Input"],
            ["tk_tf_wp", "TF coil winding pack thickness", 0.4505, "m", None, "PROCESS"],
            ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.04, "m", None, "Input"],
            ["tk_tf_ins", "TF coil ground insulation thickness", 0.0, "m", None, "Input"],
            ["tk_tf_insgap", "TF coil WP insertion gap", 0.0, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
            ["r_tf_in", "Inner Radius of the TF coil inboard leg", 0.176, "m", None, "PROCESS"],
            ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.6265, "m", None, "PROCESS"],
            ["ripple_limit", "Ripple limit constraint", 1.1, "%", None, "Input"],
            ["npts", "Number of points", 200, "N/A", "Used for vessel and plasma", "Input"],
            ["tf_wp_width", "TF coil winding pack radial width", 0.76, "m", "Including insulation", "PROCESS"],
            ["h_cp_top", "Height of the Tapered Section", 6.199, "m", None, "PROCESS"],
            ["r_cp_top", "Radial Position of Top of taper", 0.8934, "m", None, "PROCESS"],
            ["tf_wp_depth", "TF coil winding pack depth (in y)", 0.4625, "m", "Including insulation", "PROCESS"],

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
            "obj": "L",  # This is the optimisation objective: minimise length
            "ny": 1,  # This is the number of current filaments to use in y
            "nr": 1,  # This is the number of current filaments to use in x
            "nrip": 10,  # This is the number of points on the separatrix to calculate ripple for
            "read_folder": read_path,  # This is the path that the shape will be read from
            "write_folder": None,  # This is the path that the shape will be written to (replace in tests)
        }
        cls.lcfs = lcfs
        cls.ko_zone = ko_zone

    def test_tapered_TF(self, tempdir):
        self.to_tf["write_folder"] = tempdir
        tf1 = ToroidalFieldCoils(self.parameters, self.to_tf)
        tf1.optimise()

        # Test CAD Model

        CAD = tf1.build_CAD()
        vol_b_cyl = get_properties(CAD.component["shapes"][0])["Volume"]
        vol_leg_conductor = get_properties(CAD.component["shapes"][1])["Volume"]
        vol_cp_conductor = get_properties(CAD.component["shapes"][2])["Volume"]
        print(vol_leg_conductor)
        true_vol1 = 5.6872
        true_vol_leg_conductor = 15.6989
        true_vol_cp_conductor = 2.6215

        assert np.isclose(vol_b_cyl, true_vol1, rtol=1e-3)
        assert np.isclose(vol_leg_conductor, true_vol_leg_conductor, rtol=1e-3)
        assert np.isclose(vol_cp_conductor, true_vol_cp_conductor, rtol=1e-3)
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
        ]
        assert CAD.component["names"] == expected_names


if __name__ == "__main__":
    pytest.main([__file__])
