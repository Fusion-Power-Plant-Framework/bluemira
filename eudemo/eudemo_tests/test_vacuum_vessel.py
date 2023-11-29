# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
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
