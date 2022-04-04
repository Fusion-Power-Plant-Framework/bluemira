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

import pytest

from bluemira.codes.process import teardown
from bluemira.codes.process.api import ENABLED as PROCESS_ENABLED
from bluemira.codes.process.mapping import mappings
from tests.bluemira.codes.process import INDIR


@pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
class TestMFileReader:
    @classmethod
    def setup_class(cls):
        cls.mapping = {p_map.name: bm_key for bm_key, p_map in mappings.items()}
        units = {p_map.name: p_map.unit for val, p_map in mappings.items()}
        cls.bmfile = teardown.BMFile(INDIR, cls.mapping, units)
        return cls

    def test_extraction(self):
        inp = list(self.mapping.values())
        out = self.bmfile.extract_outputs(inp)
        assert len(inp) == len(out)
