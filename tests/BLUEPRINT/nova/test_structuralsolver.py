# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import numpy as np
from matplotlib import pyplot as plt
import pytest

from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.nova.structuralsolver import StructuralSolver

import tests


@pytest.mark.longrun
@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestStructuralSymmetry:
    def test_full_cage(self, reactor):
        s_sfull = StructuralSolver(
            reactor.ATEC, reactor.TF.cage, reactor.EQ.snapshots["EOF"].eq
        )
        s_sfull.pattern()
        s_sfull.model.add_gravity_loads()
        s_sfull.solve(sparse=True)

        solver = StructuralSolver(
            reactor.ATEC, reactor.TF.cage, reactor.EQ.snapshots["EOF"].eq
        )
        solver.model.add_gravity_loads()
        solver.solve(sparse=False)

        left = solver.model.cycle_sym.left_nodes
        right = solver.model.cycle_sym.right_nodes

        # Checks that the single s
        for i, j in zip(left, right):
            n_left = solver.model.geometry.nodes[i]
            n_right = solver.model.geometry.nodes[j]
            assert np.isclose(n_left.displacements[2], n_right.displacements[2])

        for i, j in zip(left, right):
            n_left = s_sfull.model.geometry.nodes[i]
            n_right = s_sfull.model.geometry.nodes[j]
            assert np.isclose(n_left.displacements[2], n_right.displacements[2])

        for i in left:
            n = solver.model.geometry.nodes[i]
            n_full = s_sfull.model.geometry.nodes[i]
            assert np.allclose(n.displacements, n_full.displacements)

        for i in right:
            n = solver.model.geometry.nodes[i]
            n_full = s_sfull.model.geometry.nodes[i]
            assert np.allclose(n.displacements, n_full.displacements)

        node_sym = []
        for n in s_sfull.model.geometry.nodes:
            if n.symmetry:
                node_sym.append(n)

        x, y, z = (
            np.zeros(len(node_sym)),
            np.zeros(len(node_sym)),
            np.zeros(len(node_sym)),
        )
        dx, dy, dz = (
            np.zeros(len(node_sym)),
            np.zeros(len(node_sym)),
            np.zeros(len(node_sym)),
        )

        for i, n in enumerate(node_sym):
            x[i] = n.x
            y[i] = n.y
            z[i] = n.z
            dx[i] = n.displacements[0]
            dy[i] = n.displacements[1]
            dz[i] = n.displacements[2]

        # Sort into rings
        z = np.around(z, 6)
        z_vals = np.unique(z)

        indices = [np.where(z == i)[0] for i in z_vals]

        loops = []
        for index in indices:
            loop = Loop(x[index], y[index], z[index])
            loops.append(loop)

        scale = 10

        dx *= scale
        dy *= scale
        dz *= scale

        dloops = []
        for index in indices:
            loop = Loop(x[index] + dx[index], y[index] + dy[index], z[index] + dz[index])
            dloops.append(loop)

        ax = Plot3D()

        for loop in loops:
            loop.plot(ax, fill=False)
        for loop in dloops:
            loop.plot(ax, fill=False)

        _, ax = plt.subplots()
        for index in indices:
            delta = np.sqrt(dx[index] ** 2 + dy[index] ** 2 + dz[index] ** 2)
            ax.plot(range(len(index)), delta / scale)

        ax.set_title("Symmetry ring displacements per sector")
        ax.set_xlabel("sector")
        ax.set_ylabel("total displacement [m]")

        # Look at reaction forces
        gs_nodes = []
        for n in s_sfull.model.geometry.nodes:
            if n.supports.all():
                gs_nodes.append(n)

        f_z = []
        for n in gs_nodes:
            f_z.append(n.reactions[2])

        _, ax = plt.subplots()
        ax.plot(np.arange(0, len(f_z) / 2, 0.5), f_z)
        ax.set_title("Fz reaction forces at gravity supports (2 per sector)")
        ax.set_ylabel("Fz [N]")
        ax.set_xlabel("sector")


if __name__ == "__main__":
    pytest.main([__file__, "--plotting-on"])
