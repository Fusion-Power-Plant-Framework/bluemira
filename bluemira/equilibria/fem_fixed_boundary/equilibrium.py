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

"""Fixed boundary equilibrium class"""
import numpy as np

from bluemira.builders.plasma import MakeParameterisedPlasma
from bluemira.base.config import Configuration
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfin
import matplotlib.pyplot as plt

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
import time

from bluemira.base.logs import set_log_level
from bluemira.equilibria.fem_fixed_boundary.transport_solver import (
    PlasmodTransportSolver,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    plot_scalar_field,
    plot_profile,
)


def solve_plasmod_fixed_boundary(plasma, plasmod_solver, delta95_t, kappa_95_t):

    m = meshing.Mesh()
    buffer = m(plasma)

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )

    # initialize the Grad-Shafranov solver
    p = 1
    gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=p)

    print("\nSolving...")

    # solve the Grad-Shafranov equation
    solve_start = time.time()  # compute the time it takes to solve
    psi = gs_solver.solve(
        plasmod_solver.pprime,
        plasmod_solver.ffprime,
        plasmod_solver.I_p,
        tol=1e-3,
        max_iter=50,
    )
    solve_end = time.time()

    plasma.shape.boundary[0].mesh_options = {"lcar": 0.05, "physical_group": "lcfs"}
    plasma.shape.mesh_options = {"lcar": 0.05, "physical_group": "plasma_face"}

    m = meshing.Mesh()
    buffer = m(plasma)

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )

    dolfin.plot(mesh)
    plt.show()

    points = mesh.coordinates()
    psi_data = np.array([gs_solver.psi(x) for x in points])

    levels = np.linspace(0.0, gs_solver.psi_ax, 25)

    axis, cntr, _ = plot_scalar_field(
        points[:, 0], points[:, 1], psi_data, levels=levels, axis=None, tofill=True
    )
    plt.show()

    axis, cntr, _ = plot_scalar_field(
        points[:, 0],
        points[:, 1],
        psi_data,
        levels=[gs_solver.psi_ax * 0.05],
        axis=None,
        tofill=False,
    )
    plt.show()
