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
Tests for EU-DEMO build.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import tests
from bluemira.base.components import Component
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import get_log_level, set_log_level
from bluemira.builders.EUDEMO.pf_coils import PFCoilsBuilder
from bluemira.builders.EUDEMO.plasma import PlasmaBuilder, PlasmaComponent
from bluemira.builders.EUDEMO.reactor import EUDEMOReactor
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder, TFCoilsComponent
from bluemira.geometry.coordinates import Coordinates

PARAMS_DIR = os.path.join(get_bluemira_root(), "tests", "bluemira", "builders", "EUDEMO")


@pytest.mark.reactor
class TestEUDEMO:
    """
    Test the EU-DEMO design procedure.
    """

    def setup_class(self):
        params = {}
        with open(os.path.join(PARAMS_DIR, "template.json")) as fh:
            params = json.load(fh)

        with open(os.path.join(PARAMS_DIR, "params.json")) as fh:
            config = json.load(fh)
            for key, val in config.items():
                params[key]["value"] = val

        build_config = {}
        with open(os.path.join(PARAMS_DIR, "build_config.json")) as fh:
            build_config = json.load(fh)

        orig_log_level = get_log_level()
        set_log_level("DEBUG")
        try:
            self.reactor = EUDEMOReactor(params, build_config)
            # print(self.reactor._file_manager.reference_data_dirs)
            # raise ValueError
            self.component = self.reactor.run()
        finally:
            set_log_level(orig_log_level)

    def test_plasma_build(self):
        """
        Test the results of the plasma build.
        """
        plasma_builder = self.reactor.get_builder(EUDEMOReactor.PLASMA)
        assert plasma_builder is not None
        assert plasma_builder.design_problem is not None

        plasma_component: PlasmaComponent = self.component.get_component(
            EUDEMOReactor.PLASMA
        )
        assert plasma_component is not None
        assert plasma_component.equilibrium is not None

        reference_eq_dir = self.reactor.file_manager.reference_data_dirs["equilibria"]
        reference_eq_name = f"{self.reactor.params.Name.value}_eqref.json"
        reference_eq_path = os.path.join(reference_eq_dir, reference_eq_name)
        reference_eq_vals = {}
        with open(reference_eq_path, "r") as fh:
            reference_eq_vals: dict = json.load(fh)
        ref_lcfs = Coordinates(
            {"x": reference_eq_vals["xbdry"], "z": reference_eq_vals["zbdry"]}
        )

        lcfs = Coordinates(plasma_component.equilibrium.get_LCFS().xyz)
        np.testing.assert_allclose(ref_lcfs.length, lcfs.length, rtol=1e-3)
        np.testing.assert_allclose(
            ref_lcfs.center_of_mass[0], lcfs.center_of_mass[0], rtol=1e-3
        )

    def test_tf_build(self):
        """
        Test the results of the TF build.
        """
        tf_builder: TFCoilsBuilder = self.reactor.get_builder(EUDEMOReactor.TF_COILS)
        assert tf_builder is not None
        assert tf_builder.design_problem is not None

        tf_component: TFCoilsComponent = self.component.get_component(
            EUDEMOReactor.TF_COILS
        )
        assert tf_component is not None

        # Check field at origin
        field = tf_component.field(
            self.reactor.params.R_0.value, 0.0, self.reactor.params.z_0.value
        )
        assert field is not None
        assert field == pytest.approx([0, -5.0031, 0])

    def test_tf_save(self):
        """
        Test the TF coil geometry parameterisation can be saved.
        """
        tf_builder: TFCoilsBuilder = self.reactor.get_builder("TF Coils")
        tempdir = tempfile.mkdtemp()
        try:
            the_path = os.sep.join([tempdir, "tf_coils_param.json"])
            tf_builder.save_shape(the_path)
            assert os.path.isfile(the_path)
            with open(the_path, "r") as fh:
                assert len(fh.readlines()) > 0
        finally:
            shutil.rmtree(tempdir)

    def test_pf_coils_built(self):
        """
        Test the results of the PF build.
        """
        tf_builder = self.reactor.get_builder(EUDEMOReactor.PF_COILS)
        assert tf_builder is not None

        tf_component = self.component.get_component(EUDEMOReactor.PF_COILS)
        assert tf_component is not None

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot_xz(self):
        """
        Display the results.
        """
        Component(
            "xz view",
            children=[
                self.component.get_component(EUDEMOReactor.PLASMA).get_component("xz"),
                self.component.get_component(EUDEMOReactor.TF_COILS).get_component("xz"),
            ],
        ).plot_2d()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot_xy(self):
        """
        Display the results.
        """
        Component(
            "xy view",
            children=[
                self.component.get_component(EUDEMOReactor.PLASMA).get_component("xy"),
                self.component.get_component(EUDEMOReactor.TF_COILS).get_component("xy"),
            ],
        ).plot_2d()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_show_cad(self):
        """
        Display the results.
        """
        Component(
            "xyz view",
            children=[
                self.component.get_component(EUDEMOReactor.PLASMA).get_component("xyz"),
                self.component.get_component(EUDEMOReactor.TF_COILS).get_component(
                    "xyz"
                ),
            ],
        ).show_cad()

    def test_show_segment_cad(self):
        component = Component("Segment View")
        plasma_builder: PlasmaBuilder = self.reactor.get_builder("Plasma")
        tf_coils_builder: TFCoilsBuilder = self.reactor.get_builder("TF Coils")
        pf_coils_builder: PFCoilsBuilder = self.reactor.get_builder("PF Coils")
        component.add_child(plasma_builder.build_xyz(degree=270))
        component.add_child(tf_coils_builder.build_xyz(degree=270))
        component.add_child(pf_coils_builder.build_xyz(degree=270))
        if tests.PLOTTING:
            component.show_cad()
