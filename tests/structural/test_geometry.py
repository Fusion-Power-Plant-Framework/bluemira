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

import numpy as np
from matplotlib import pyplot as plt

from bluemira.geometry.coordinates import Coordinates
from bluemira.structural.crosssection import IBeam
from bluemira.structural.geometry import Geometry
from bluemira.structural.material import SS316


def add_node(geometry, *node):
    for n in node:
        geometry.add_node(*n)


class TestKMatrix:
    def test_k(self):
        geometry = Geometry()
        i_300_200 = IBeam(0.2, 0.3, 0.05, 0.04)
        add_node(geometry, (4, 5, 6), (7, 8, 9), (8, 8, 9))
        geometry.add_element(0, 1, i_300_200, SS316)
        geometry.add_element(0, 1, i_300_200, SS316)
        geometry.add_element(1, 2, i_300_200, SS316)

        k_matrix = geometry.k_matrix()

        fig, ax = plt.subplots()
        ax.matshow(k_matrix)
        plt.show()
        plt.close(fig)

        assert np.allclose(k_matrix, k_matrix.T)


class TestMembership:
    @classmethod
    def setup_class(cls):
        geometry = Geometry()
        add_node(geometry, (-1, 0, -1), (0, 0, 0), (1, 1, 1), (2, 2, 0))
        i_300_200 = IBeam(0.2, 0.3, 0.05, 0.04)
        geometry.add_element(0, 1, i_300_200, SS316)
        geometry.add_element(1, 2, i_300_200, SS316)
        geometry.add_element(2, 3, i_300_200, SS316)
        cls.geometry = geometry

    def test_node_membership(self):
        self.geometry.add_node(0, 0, 0)
        assert len(self.geometry.nodes) == 4

        self.geometry.add_node(-1, 0, -1)
        assert len(self.geometry.nodes) == 4

        self.geometry.add_node(2, 2, 0)
        assert len(self.geometry.nodes) == 4

        self.geometry.add_node(2, 2, 2)
        assert len(self.geometry.nodes) == 5

    def test_element_membership(self):
        i_300_300 = IBeam(0.3, 0.3, 0.05, 0.04)

        elem_id = self.geometry.add_element(0, 1, i_300_300, SS316)
        assert len(self.geometry.elements) == 3
        assert elem_id == 0
        # Check the properties were modified
        eiyy = i_300_300.i_yy * SS316.E
        assert self.geometry.elements[0]._properties["EIyy"] == eiyy

        elem_id = self.geometry.add_element(2, 3, i_300_300, SS316)
        assert len(self.geometry.elements) == 3
        assert elem_id == 2
        # Check the properties were modified
        eiyy = i_300_300.i_yy * SS316.E
        assert self.geometry.elements[0]._properties["EIyy"] == eiyy

        elem_id = self.geometry.add_element(3, 4, i_300_300, SS316)
        assert len(self.geometry.elements) == 4
        assert elem_id == 3


class TestRemove:
    def test_remove_node(self):
        x_section = IBeam(1, 1, 0.25, 0.5)
        g = Geometry()
        add_node(g, (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
        g.add_element(0, 1, x_section, SS316)
        g.add_element(1, 2, x_section, SS316)
        g.add_element(2, 3, x_section, SS316)

        assert g.n_nodes == 4
        assert g.n_elements == 3
        assert g.nodes[2].connections == {1, 2}
        k = g.k_matrix()
        assert k.shape == (6 * 4, 6 * 4)
        g.remove_node(3)
        assert g.n_nodes == 3
        assert g.n_elements == 2
        assert g.nodes[2].connections == {1}
        k = g.k_matrix()
        assert k.shape == (6 * 3, 6 * 3)

    def test_remove_element(self):
        x_section = IBeam(1, 1, 0.25, 0.5)
        g = Geometry()
        add_node(
            g,
            (0, 0, 0),  # 0
            (1, 0, 0),  # 1
            (1, 1, 0),  # 2
            (1, 2, 0),  # 3
            (0, 2, 0),  # 4
            (0, 1, 0),  # 5
        )
        g.add_element(0, 1, x_section, SS316)  # 0
        g.add_element(1, 2, x_section, SS316)  # 1
        g.add_element(2, 3, x_section, SS316)  # 2
        g.add_element(3, 4, x_section, SS316)  # 3
        g.add_element(4, 5, x_section, SS316)  # 4
        g.add_element(5, 0, x_section, SS316)  # 5
        g.add_element(2, 5, x_section, SS316)  # 6

        assert g.n_elements == 7
        assert g.nodes[2].connections == {1, 2, 6}
        k_1 = g.k_matrix()
        g.remove_element(6)
        assert g.n_elements == 6
        assert g.nodes[2].connections == {1, 2}
        k_2 = g.k_matrix()
        assert k_1.shape == k_2.shape
        assert not np.allclose(k_1, k_2)

    def test_node_element_complicated(self):
        x_section = IBeam(1, 1, 0.25, 0.5)
        g = Geometry()
        add_node(
            g,
            (0, 0, 0),  # 0
            (1, 0, 0),  # 1
            (1, 1, 0),  # 2
            (1, 2, 0),  # 3
            (0, 2, 0),  # 4
            (0, 1, 0),  # 5
            (-1, 0, 0),  # 6
            (-1, -1, 0),  # 7
            (-1, -2, 0),  # 8
            (0, -2, 0),  # 9
            (0, -1, 0),  # 10
        )
        g.add_element(0, 1, x_section, SS316)  # 0
        g.add_element(1, 2, x_section, SS316)  # 1
        g.add_element(2, 3, x_section, SS316)  # 2
        g.add_element(3, 4, x_section, SS316)  # 3
        g.add_element(4, 5, x_section, SS316)  # 4
        g.add_element(5, 0, x_section, SS316)  # 5
        g.add_element(2, 5, x_section, SS316)  # 6
        g.add_element(0, 6, x_section, SS316)  # 7
        g.add_element(6, 7, x_section, SS316)  # 8
        g.add_element(7, 8, x_section, SS316)  # 9
        g.add_element(8, 9, x_section, SS316)  # 10
        g.add_element(9, 10, x_section, SS316)  # 11

        # Prior to removal
        assert g.nodes[0].connections == {0, 5, 7}

        for i in [1, 2, 3, 4, 5][::-1]:
            g.remove_node(i)
        assert g.n_nodes == 6
        assert g.n_elements == 5
        # We've removed 5 Nodes, and implicitly removed 7 Elements
        # The Elements and Node connections will have been re-numbered
        assert g.nodes[0].connections == {0}


class TestMove:
    @staticmethod
    def _add_element(g):
        x_section = IBeam(0.1, 0.1, 0.025, 0.05)

        g.add_element(0, 1, x_section, SS316)  # 0
        g.add_element(1, 2, x_section, SS316)  # 1
        g.add_element(2, 3, x_section, SS316)  # 2
        g.add_element(3, 4, x_section, SS316)  # 3
        g.add_element(4, 5, x_section, SS316)  # 4
        g.add_element(5, 0, x_section, SS316)  # 5
        g.add_element(2, 5, x_section, SS316)  # 6

    def test_move_node(self):
        g = Geometry()
        add_node(
            g,
            (0, 0, 0),  # 0
            (1, 0, 0),  # 1
            (1, 2, 0),  # 2
            (1, 4, 0),  # 3
            (0, 4, 0),  # 4
            (0, 2, 0),  # 5
        )
        self._add_element(g)

        g.move_node(2, dx=-0.5)
        assert g.n_nodes == 6
        assert g.n_elements == 7

        g.move_node(2, dx=-0.5)
        assert g.n_nodes == 5
        assert g.n_elements == 6

    def test_move_node2(self):
        g = Geometry()
        add_node(
            g,
            (0, 0, 0),  # 0
            (1, 0, 0),  # 1
            (1, 1, 0),  # 2
            (1, 2, 0),  # 3
            (0, 2, 0),  # 4
            (0, 1, 0),  # 5
        )
        self._add_element(g)

        g.move_node(5, dx=0.5)
        assert g.n_nodes == 6
        assert g.n_elements == 7

        g.move_node(5, dx=0.5)
        assert g.n_nodes == 5
        assert g.n_elements == 6

    def test_add_coordinates(self):
        x_section = IBeam(0.1, 0.1, 0.025, 0.05)
        g = Geometry()
        c = Coordinates({"x": [0, 1, 2, 3], "z": [0, -1, 1, -1]})
        g.add_coordinates(c, x_section, SS316)
        assert g.n_nodes == 4
        assert g.n_elements == 3
