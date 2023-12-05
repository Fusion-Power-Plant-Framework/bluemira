# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.structural.node import Node


class TestNode:
    def test_distances(self):
        n1 = Node(0, 0, 0, 0)
        rng = np.random.default_rng()
        for _ in range(100):
            v = 1000 * rng.random(3) - 1000
            dx, dy, dz = v
            n2 = Node(dx, dy, dz, 1)
            assert np.isclose(n1.distance_to_other(n2), np.sqrt(np.sum(v**2)))

    def test_assignment(self):
        with pytest.raises(AttributeError):
            Node(0, 0, 0, 0).dummy = 4

    def test_defaultsupports(self):
        node = Node(0, 0, 0, 0)
        assert not node.supports.all()
        node.add_support(np.array([True, True, True, True, True, True]))
        assert node.supports.all()
        node.add_support(np.array([True, True, True, True, True, False]))
        assert not node.supports[5]
        assert node.supports[:5].all()
        node.clear_supports()
        assert not node.supports.all()
