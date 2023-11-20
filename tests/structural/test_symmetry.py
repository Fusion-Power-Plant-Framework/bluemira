# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy

import numpy as np

from bluemira.geometry.tools import make_circle
from bluemira.structural.crosssection import IBeam
from bluemira.structural.material import SS316
from bluemira.structural.model import FiniteElementModel
from bluemira.structural.transformation import cyclic_pattern


class TestCyclicSymmetry:
    def test_symmetry(self):
        model = FiniteElementModel()

        xsection = IBeam(0.4, 0.6, 0.2, 0.1)

        i1 = model.add_node(5, 0, 0)
        i2 = model.add_node(9, 0, 3)
        model.add_element(i1, i2, xsection, SS316)
        model.add_support(0, dx=True, dy=True, dz=True, rx=True, ry=True, rz=True)

        circle = make_circle(radius=9, center=(0, 0, 3), start_angle=0, end_angle=30)
        coordinates = circle.discretize(ndiscr=15)

        sym_nodes = []
        n1 = model.add_node(*coordinates.points[0])
        sym_nodes.append(n1)
        for point in coordinates.points[1:]:
            n2 = model.add_node(*point)
            model.add_element(n1, n2, xsection, SS316)
            n1 = n2

        sym_nodes.append(n1)
        model.apply_cyclic_symmetry(*sym_nodes, [0, 0, 0], [0, 0, 1])

        model.add_distributed_load(10, 200e5, "Fy")

        # apply loads so they get patterned too
        fullmodel = deepcopy(model)
        fullmodel._apply_load_case(fullmodel.load_case)
        fullmodel.clear_load_case()  # To avoid duplicate loads on the first sector
        fullmodel.geometry = cyclic_pattern(
            fullmodel.geometry,
            np.array([0, 0, 1]),
            30,
            int(360 / 30),
        )
        # Deconstruct cyclic_symmetry
        fullmodel.cycle_sym = None
        fullmodel.cycle_sym_ids = []

        result = model.solve(sparse=False)
        fullresult = fullmodel.solve(sparse=True)

        result.plot(100, stress=True, pattern=True)
        fullresult.plot(100, stress=True)

        left = model.cycle_sym.left_nodes
        right = model.cycle_sym.right_nodes

        # Checks that the single sector model symmetry node displacements are identical
        for i, j in zip(left, right, strict=False):
            n_left = model.geometry.nodes[i]
            n_right = model.geometry.nodes[j]
            assert np.isclose(n_left.displacements[2], n_right.displacements[2])

        # Check that the full model symmetry node displacements are identical
        for i, j in zip(left, right, strict=False):
            n_left = fullmodel.geometry.nodes[i]
            n_right = fullmodel.geometry.nodes[j]
            assert np.isclose(n_left.displacements[2], n_right.displacements[2])

        # Now check that the results are equal
        for i in left:
            n = model.geometry.nodes[i]
            n_full = fullmodel.geometry.nodes[i]
            assert np.allclose(n.displacements, n_full.displacements), (
                n.displacements / n_full.displacements
            )

        for i in right:
            n = model.geometry.nodes[i]
            n_full = fullmodel.geometry.nodes[i]
            assert np.allclose(n.displacements, n_full.displacements), (
                n.displacements / n_full.displacements
            )
