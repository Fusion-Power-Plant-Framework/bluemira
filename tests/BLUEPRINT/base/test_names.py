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

from BLUEPRINT.base.names import name_long_short, name_short_long


class TestNamemapper:
    def test_map_short_long(self):
        d = {"VV": 0, "Plasma": 1, "RS": 2, "Cryostat vacuum vessel": 3, "DUMP": 4}
        dd = name_long_short(d)
        ddtrue = {"VV": 0, "PL": 1, "RS": 2, "CR": 3, "DUMP": 4}
        assert dd == ddtrue

    def test_map_long_short(self):
        d = {"VV": 0, "Plasma": 1, "RS": 2, "Cryostat vacuum vessel": 3, "DUMP": 4}
        dd = name_short_long(d)
        ddtrue = {
            "Reactor vacuum vessel": 0,
            "Plasma": 1,
            "Radiation shield": 2,
            "Cryostat vacuum vessel": 3,
            "DUMP": 4,
        }
        assert dd == ddtrue
