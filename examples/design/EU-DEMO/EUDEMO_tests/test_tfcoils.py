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
Test first wall silhouette designer.
"""

import copy
import os

import numpy as np
import pytest

from bluemira.base.error import DesignError
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import make_circle, make_polygon
from EUDEMO_builders.tf_coils import TFCoilBuilder, TFCoilDesigner

EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")
DATA = get_bluemira_path("design/EU-DEMO/EUDEMO_tests/test_data", subfolder="examples")

OPTIMISER_MODULE_REF = "bluemira.builders.tf_coils"

CONFIG = {
    "param_class": f"TripleArc",
    "variables_map": {},
    "run_mode": "mock",
    "name": "First Wall",
    "problem_class": f"{OPTIMISER_MODULE_REF}::RippleConstrainedLengthGOP",
}
PARAMS = {
    "R_0": {"name": "R_0", "value": 10.5},
    "r_tf_current_ib": {"name": "r_tf_current_ib", "value": 1},
    "r_tf_in": {"name": "r_tf_in", "value": 3.2},
    "tk_tf_wp": {"name": "tk_tf_wp", "value": 0.5},
    "tk_tf_wp_y": {"name": "tk_tf_wp_y", "value": 0.5},
    "tf_wp_width": {"name": "tf_wp_width", "value": 0.76},
    "tf_wp_depth": {"name": "tf_wp_depth", "value": 1.05},
    "tk_tf_front_ib": {"name": "tk_tf_front_ib", "value": 0.04},
    "g_ts_tf": {"name": "g_ts_tf", "value": 0.05},
    "TF_ripple_limit": {"name": "TF_ripple_limit", "value": 0.6},
    "z_0": {"name": "z_0", "value": 0.0},
    "B_0": {"name": "B_0", "value": 6},
    "n_TF": {"name": "n_TF", "value": 12},
    "tk_tf_ins": {"name": "tk_tf_ins", "value": 0.08},
    "tk_tf_insgap": {"name": "tk_tf_insgap", "value": 0.1},
    "tk_tf_nose": {"name": "tk_tf_nose", "value": 0.6},
}


class TestTFCoilDesigner:
    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(EQDATA, "eqref_OOB.json"))
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())
        cls.lcfs = make_polygon(cls.eq.get_LCFS().xyz, closed=True)
        cls.vvts_koz = make_circle(10, center=(15, 0, 0), axis=(0.0, 1.0, 0.0))

    # def test_parameterisation_read(self):
    #     config = copy.deepcopy(CONFIG)
    #     config.update(
    #         {"run_mode": "read", "file_path": os.path.join(DATA, "wall_polyspline.json")}
    #     )

    #     designer = TFCoilDesigner(
    #         PARAMS, build_config=config, separatrix=self.lcfs, keep_out_zone=self.vvts_koz
    #     )
    #     param = designer.execute()

    #     assert param.variables["flat"].value == 0
    #     assert param.variables["x1"].value == PARAMS["r_fw_ib_in"]["value"]

    def test_read_no_file(self):
        config = copy.deepcopy(CONFIG)
        config.update({"run_mode": "read"})

        designer = TFCoilDesigner(
            PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        with pytest.raises(ValueError):
            designer.execute()

    def test_run_no_problem_class(self):
        config = copy.deepcopy(CONFIG)
        config.update({"run_mode": "run"})
        del config["problem_class"]

        designer = TFCoilDesigner(
            PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        with pytest.raises(ValueError):
            designer.execute()

    def test_run_check_parameters(self):
        config = copy.deepcopy(CONFIG)
        config["run_mode"] = "run"
        config.update(
            {
                "problem_settings": {"n_koz_points": 101},
                "optimisation_settings": {
                    "algorithm_name": "COBYLA",
                    "parameters": {"initial_step": 1e-4},
                    "conditions": {"max_eval": 25},
                },
            }
        )
        designer = TFCoilDesigner(
            PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )
        d_run = designer.execute()

        designer_mock = TFCoilDesigner(
            PARAMS,
            build_config=CONFIG,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )
        d_mock = designer_mock.execute()
        assert d_run.create_shape().length != d_mock.create_shape().length
        assert designer.problem_settings == config["problem_settings"]
        assert designer.opt_config == config["optimisation_settings"]
        assert (
            designer.algorithm_name == config["optimisation_settings"]["algorithm_name"]
        )
        assert designer.opt_parameters == config["optimisation_settings"]["parameters"]
        assert designer.opt_conditions == config["optimisation_settings"]["conditions"]

    def test_shape_is_closed(self):
        designer = TFCoilDesigner(
            PARAMS,
            build_config=CONFIG,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        assert designer.execute().create_shape().is_closed()

    # def test_height_derived_from_params_given_PolySpline_mock_mode(self):
    #     params = copy.deepcopy(PARAMS)
    #     params.update(
    #         {
    #             "R_0": {"name": "R_0", "value": 10},
    #             "kappa_95": {"name": "kappa_95", "value": 2},
    #             "A": {"name": "A", "value": 2},
    #         }
    #     )
    #     config = copy.deepcopy(CONFIG)
    #     config.update({"param_class": f"{WALL_MODULE_REF}::WallPolySpline"})

    #     designer = TFCoilDesigner(
    #         params, build_config=config, separatrix=self.lcfs, keep_out_zone=self.vvts_koz
    #     )
    #     bounding_box = designer.execute().create_shape().bounding_box

    #     # expected_height = 2*(R_0/A)*kappa_95 = 20
    #     assert np.isclose(bounding_box.z_max - bounding_box.z_min, 20.0)

    # def test_width_matches_params_given_PrincetonD_mock_mode(self):

    #     vm = {
    #         "x1": {  # ib radius
    #             "value": "r_fw_ib_in",
    #         },
    #         "x2": {  # ob radius
    #             "value": "r_fw_ob_in",
    #         },
    #         "dz": -2,
    #     }
    #     config = copy.deepcopy(CONFIG)

    #     config.update(
    #         {"param_class": f"{WALL_MODULE_REF}::WallPrincetonD", "variables_map": vm}
    #     )

    #     designer = TFCoilDesigner(
    #         PARAMS, build_config=config, separatrix=self.lcfs, keep_out_zone=self.vvts_koz
    #     )

    #     bounding_box = designer.execute().create_shape().bounding_box

    #     width = bounding_box.x_max - bounding_box.x_min
    #     assert width == pytest.approx(
    #         PARAMS["r_fw_ob_in"]["value"] - PARAMS["r_fw_ib_in"]["value"]
    #     )

    # def test_DesignError_for_small_silhouette(self):

    #     config = copy.deepcopy(CONFIG)
    #     config.update({"param_class": f"{WALL_MODULE_REF}::WallPrincetonD"})

    #     designer = TFCoilDesigner(
    #         PARAMS, build_config=config, separatrix=self.lcfs, keep_out_zone=self.vvts_koz
    #     )

    #     with pytest.raises(DesignError):
    #         designer.execute()
