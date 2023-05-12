# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Tests for EU-DEMO Lower Port Duct Designer
"""

# import pytest

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_fuse, make_circle, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.designer import LowerPortDuctDesignerParams


class TestLowerPortDuctDesigner:
    """Test Lower Port Duct Designer"""

    @classmethod
    def setup_class(cls):
        cls.divertor_xz_silhouette = BluemiraWire(
            [
                make_polygon(
                    [
                        [-1, 0, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                    ]
                ),
                make_circle(1, start_angle=-180, end_angle=0, axis=(0, -1, 0)),
            ]
        )

    def setup_method(self):
        self.params = make_parameter_frame(
            {
                "n_TF": {"value": 18, "unit": "dimensionless"},
                "n_div_cassettes": {"value": 3, "unit": "dimensionless"},
                "tf_coil_thickness": {"value": 0.65, "unit": "m"},
                "lp_duct_tf_offset": {"value": 0.5, "unit": "m"},
                "lp_duct_div_pad_outer": {"value": 0.3, "unit": "m"},
                "lp_duct_div_pad_inner": {"value": 0.1, "unit": "m"},
                "lp_height": {"value": 3, "unit": "m"},
                "lp_width": {"value": 3, "unit": "m"},
                "lp_duct_angle": {"value": 0, "unit": "degrees"},
                "lp_duct_wall_tk": {"value": 0.02, "unit": "m"},
            },
            LowerPortDuctDesignerParams,
        )

    def test_duct_div_hole(self):
        ...
