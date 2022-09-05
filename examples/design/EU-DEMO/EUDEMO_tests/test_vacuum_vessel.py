# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
from bluemira.geometry.parameterisations import PictureFrame
from EUDEMO_builders.vacuum_vessel import VacuumVesselBuilder


class TestVacuumVesselBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "r_vv_ib_in": {"name": "r_vv_ib_in", "value": 5.1},
            "r_vv_ob_in": {"name": "r_vv_ob_in", "value": 14.5},
            "tk_vv_in": {"name": "tk_vv_in", "value": 0.6},
            "tk_vv_out": {"name": "tk_vv_out", "value": 1.1},
            "g_vv_bb": {"name": "g_vv_bb", "value": 0.02},
            "n_TF": {"name": "n_TF", "value": 16},
        }
        cls.picture_frame = PictureFrame(
            {"x1": {"value": 2}, "ro": {"value": 6}, "ri": {"value": 3}},
        ).create_shape()

    def test_components_and_segments(self):
        builder = VacuumVesselBuilder(self.params, {}, self.picture_frame)
        cryostat_ts = builder.build()

        assert cryostat_ts.component().get_component("xz")
        assert cryostat_ts.component().get_component("xy")

        xyz = cryostat_ts.component().get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == self.params["n_TF"]["value"]
