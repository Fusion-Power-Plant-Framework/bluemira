# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from BLUEPRINT.beams.node import Node
import pytest


class TestNode:
    def test_distances(self):
        n1 = Node(0, 0, 0, 0)
        for _ in range(100):
            v = 1000 * np.random.rand(3) - 1000
            dx, dy, dz = v
            n2 = Node(dx, dy, dz, 1)
            assert np.isclose(n1.distance_to_other(n2), np.sqrt(np.sum(v ** 2)))

    def test_assignment(self):
        with pytest.raises(AttributeError):
            node = Node(0, 0, 0, 0)
            node.dummy = 4

    def test_defaultsupports(self):
        node = Node(0, 0, 0, 0)
        assert not node.supports.all()
        node.add_support(np.array([True, True, True, True, True, True]))
        assert node.supports.all()
        node.add_support(np.array([True, True, True, True, True, False]))
        assert not node.supports[5]
        assert node.supports[:5].all()


if __name__ == "__main__":
    pytest.main([__file__])
