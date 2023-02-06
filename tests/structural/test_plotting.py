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

import matplotlib.pyplot as plt

from bluemira.geometry.coordinates import Coordinates
from bluemira.structural.crosssection import RectangularBeam
from bluemira.structural.geometry import Geometry
from bluemira.structural.loads import LoadCase
from bluemira.structural.material import SS316
from bluemira.structural.model import FiniteElementModel
from bluemira.structural.plotting import GeometryPlotter


class TestPlotting:
    def test_no_errors(self):
        fem = FiniteElementModel()

        geometry = Geometry()
        rect_beam = RectangularBeam(0.5, 0.5)
        nodes = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 2], [1, 0, 2]]
        for node in nodes:
            geometry.add_node(*node)
        for i in range(5):
            geometry.add_element(i, i + 1, rect_beam, SS316)
        geometry.add_element(0, 3, rect_beam, SS316)
        geometry.add_element(2, 5, rect_beam, SS316)

        l_loop = Coordinates([[2.5, 1.5, 1.5], [0, 0, 0], [0, 0, 2]])
        u_loop = Coordinates([[0, 0, 1, 1], [0, 0, 0, 0], [2, 0, 0, 2]])
        u_loop.translate((3, 0, 0))
        e_loop = Coordinates(
            {"x": [1, 0, 0, 1, 0, 0, 1], "z": [2, 2, 1.005, 1.005, 1, 0, 0]}
        )
        e_loop.translate((4.5, 0, 0))

        p_loop = Coordinates({"x": [0, 0, 0, 1, 1, 0.01], "z": [0, 1, 2, 2, 1, 1]})
        p_loop.translate((6, 0, 0))
        r_loop = Coordinates({"x": [0, 0, 0, 1, 1, 0.01, 1], "z": [0, 1, 2, 2, 1, 1, 0]})
        r_loop.translate((7.5, 0, 0))

        i_loop = Coordinates(
            {"x": [0, 0.49, 0.49, 0, 1, 0.51, 0.51, 1], "z": [0, 0, 2, 2, 2, 2, 0, 0]}
        )
        i_loop.translate((9, 0, 0))
        n_loop = Coordinates({"x": [0, 0, 1, 1], "z": [0, 2, 0, 2]})
        n_loop.translate((10.5, 0, 0))
        t_loop = Coordinates({"x": [0, 0.49, 0.5, 0.51, 1], "z": [2, 2, 0, 2, 2]})
        t_loop.translate((12, 0, 0))

        for letter in [l_loop, u_loop, e_loop, p_loop, r_loop, i_loop, n_loop, t_loop]:
            letter.rotate(base=(0, 0, 1), direction=(0, 0, 1), degree=30)
            geometry.add_coordinates(letter, rect_beam, SS316)

        fem.set_geometry(geometry)
        fem.add_support(0, True, True, True, True, True, True)

        fem.add_support(44, False, False, False, True, True, True)
        load_case = LoadCase()
        load_case.add_node_load(4, 1000, "Fx")

        load_case.add_distributed_load(20, -2000, "Fz")
        load_case.add_node_load(9, -2000, "My")
        load_case.add_node_load(9, 2000, "My")

        fem.apply_load_case(load_case)

        GeometryPlotter(geometry)
        plt.show()
