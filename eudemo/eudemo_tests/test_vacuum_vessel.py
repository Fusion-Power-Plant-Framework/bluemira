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
from bluemira.geometry.parameterisations import PictureFrame
from eudemo.vacuum_vessel import VacuumVesselBuilder


class TestVacuumVesselBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "r_vv_ib_in": {"value": 5.1, "unit": "m"},
            "r_vv_ob_in": {"value": 14.5, "unit": "m"},
            "tk_vv_in": {"value": 0.6, "unit": "m"},
            "tk_vv_out": {"value": 1.1, "unit": "m"},
            "g_vv_bb": {"value": 0.02, "unit": "m"},
            "n_TF": {"value": 16, "unit": ""},
            "vv_in_off_deg": {"value": 80, "unit": "deg"},
            "vv_out_off_deg": {"value": 160, "unit": "deg"},
        }

        cls.picture_frame = PictureFrame(
            {
                "x1": {"value": 2},
                "ro": {"value": 3.5},
                "ri": {"value": 3},
                "x2": {"value": 10},
            },
        ).create_shape()

    def test_components_and_segments(self):
        builder = VacuumVesselBuilder(self.params, {}, self.picture_frame)
        vacuum_vessel = builder.build()

        assert vacuum_vessel.get_component("xz")
        assert vacuum_vessel.get_component("xy")

        xyz = vacuum_vessel.get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == 2
