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

import pytest

from bluemira.structural.constants import LOAD_TYPES
from bluemira.structural.error import StructuralError
from bluemira.structural.loads import distributed_load, point_load


class TestPointLoad:
    def test_bad(self):
        with pytest.raises(StructuralError):
            point_load(100, 0.15, 5, "Fxy")

    def test_good(self):
        for load_type in LOAD_TYPES:
            r = point_load(100, 0.14, 10, load_type)
            assert len(r) == 12


class TestDistributedLoad:
    def test_bad(self):
        for string in ["Mx", "My", "Mz"]:
            with pytest.raises(StructuralError):
                distributed_load(0.3, 1, string)

    def test_good(self):
        for string in ["Fx", "Fy", "Fz"]:
            r = distributed_load(0.3, 1, string)
            assert len(r) == 12
