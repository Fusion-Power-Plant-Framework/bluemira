# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from matproplib.conditions import STPConditions

from bluemira.geometry.coordinates import Coordinates
from bluemira.materials.basic import SS316
from bluemira.structural.crosssection import RectangularBeam
from bluemira.structural.geometry import Geometry
from bluemira.structural.loads import LoadCase
from bluemira.structural.model import FiniteElementModel
from bluemira.structural.plotting import GeometryPlotter


class TestPlotting:
    def test_no_errors(self):
        op_cond = STPConditions()
        ss316 = SS316()
        fem = FiniteElementModel()

        geometry = Geometry()
        rect_beam = RectangularBeam(0.5, 0.5)
        nodes = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 2], [1, 0, 2]]
        for node in nodes:
            geometry.add_node(*node)
        for i in range(5):
            geometry.add_element(i, i + 1, rect_beam, ss316, op_cond)
        geometry.add_element(0, 3, rect_beam, ss316, op_cond)
        geometry.add_element(2, 5, rect_beam, ss316, op_cond)

        l_loop = Coordinates([[2.5, 1.5, 1.5], [0, 0, 0], [0, 0, 2]])
        u_loop = Coordinates([[0, 0, 1, 1], [0, 0, 0, 0], [2, 0, 0, 2]])
        u_loop.translate((3, 0, 0))
        e_loop = Coordinates({
            "x": [1, 0, 0, 1, 0, 0, 1],
            "z": [2, 2, 1.005, 1.005, 1, 0, 0],
        })
        e_loop.translate((4.5, 0, 0))

        p_loop = Coordinates({"x": [0, 0, 0, 1, 1, 0.01], "z": [0, 1, 2, 2, 1, 1]})
        p_loop.translate((6, 0, 0))
        r_loop = Coordinates({"x": [0, 0, 0, 1, 1, 0.01, 1], "z": [0, 1, 2, 2, 1, 1, 0]})
        r_loop.translate((7.5, 0, 0))

        i_loop = Coordinates({
            "x": [0, 0.49, 0.49, 0, 1, 0.51, 0.51, 1],
            "z": [0, 0, 2, 2, 2, 2, 0, 0],
        })
        i_loop.translate((9, 0, 0))
        n_loop = Coordinates({"x": [0, 0, 1, 1], "z": [0, 2, 0, 2]})
        n_loop.translate((10.5, 0, 0))
        t_loop = Coordinates({"x": [0, 0.49, 0.5, 0.51, 1], "z": [2, 2, 0, 2, 2]})
        t_loop.translate((12, 0, 0))

        for letter in [l_loop, u_loop, e_loop, p_loop, r_loop, i_loop, n_loop, t_loop]:
            letter.rotate(base=(0, 0, 1), direction=(0, 0, 1), degree=30)
            geometry.add_coordinates(letter, rect_beam, ss316, op_cond)

        fem.set_geometry(geometry)
        fem.add_support(0, dx=True, dy=True, dz=True, rx=True, ry=True, rz=True)

        fem.add_support(44, dx=False, dy=False, dz=False, rx=True, ry=True, rz=True)
        load_case = LoadCase()
        load_case.add_node_load(4, 1000, "Fx")

        load_case.add_distributed_load(20, -2000, "Fz")
        load_case.add_node_load(9, -2000, "My")
        load_case.add_node_load(9, 2000, "My")

        fem.apply_load_case(load_case)

        GeometryPlotter(geometry)
