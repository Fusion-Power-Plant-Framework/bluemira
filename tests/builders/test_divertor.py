# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from bluemira.builders.divertor import DivertorBuilder
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon


class TestDivertorBuilder:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "n_div_cassettes": {"value": 3, "unit": "m"},
            "c_rm": {"value": 0.02, "unit": "m"},
            "n_TF": {"value": 12, "unit": ""},
        }
        cls.div_koz = BluemiraFace(
            make_polygon([[1, 0, -5], [1, 0, -10], [5, 0, -10], [5, 0, -5]], closed=True)
        )

    def test_components_and_segments(self):
        builder = DivertorBuilder(self.params, {}, self.div_koz)
        cryostat_ts = builder.build()

        assert cryostat_ts.get_component("xz")
        assert cryostat_ts.get_component("xy")

        xyz = cryostat_ts.get_component("xyz")
        assert xyz
        assert len(xyz.leaves) == self.params["n_div_cassettes"]["value"]
