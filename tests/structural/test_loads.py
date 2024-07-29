# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.structural.constants import LoadType
from bluemira.structural.error import StructuralError
from bluemira.structural.loads import distributed_load, point_load


class TestPointLoad:
    def test_bad(self):
        with pytest.raises(StructuralError):
            point_load(100, 0.15, 5, "Fxy")

    def test_good(self):
        for load_type in LoadType:
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
