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

"""
Comparison of GS solution with Solovev analytic solution
"""

import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import Solovev, plot_scalar_field
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_bspline
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

start_time = time.time()

R0 = 9.07
A = 3.1
delta = 0.5
kappa = 1.7

a = R0 / A

A1 = -6.84256806e-02
A2 = -6.52918977e-02

solovev = Solovev(R0, a, kappa, delta, A1, A2)
levels = 50
axis, cntr, cntrf, points, psi_exact = solovev.plot_psi(
    6.0, -5, 6.0, 10.0, 100, 100, levels=levels
)
plt.show()

levels_p = np.linspace(0.0, max(psi_exact) * 1.05, 10)
levels_m = np.linspace(min(psi_exact) * 1.05, 0.0, 10)
levels = sorted(np.unique(np.concatenate([levels_m, levels_p])))

axis, cntr, cntrf, points, psi_exact = solovev.plot_psi(
    6.0, -5, 6.0, 10.0, 100, 100, levels=levels
)
plt.show()

# ------------------------------------------------------------------------------
ind0 = np.where(np.array(levels) == 0.0)[0][0]
Dp = cntr.collections[ind0].get_paths()[0].vertices
Dp = np.hstack((Dp, np.zeros((Dp.shape[0], 1), dtype=Dp.dtype)))

curve1 = make_bspline(Dp[0 : int(len(Dp) / 2)], label="curve1")
curve2 = make_bspline(Dp[int(len(Dp) / 2 - 1) : len(Dp)], label="curve2")
lcfs = BluemiraWire([curve1, curve2], "LCFS")
lcfs.mesh_options = {"lcar": 0.2, "physical_group": "lcfs"}

plasma_face = BluemiraFace(lcfs, "plasma_face")
plasma_face.mesh_options = {"lcar": 0.2, "physical_group": "plasma_face"}

plasma = PhysicalComponent("Plasma", shape=plasma_face)
plasma.plot_options.view = "xy"
plasma.plot_2d()
plt.show()
# ------------------------------------------------------------------------------
m = meshing.Mesh()
buffer = m(plasma)

msh_to_xdmf("Mesh.msh", dimensions=2, directory=".", verbose=True)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

# ------------------------------------------------------------------------------
# initialize the Grad-Shafranov solver
p = 3
gs_solver = FemMagnetostatic2d(mesh, p_order=p)

# Set the right hand side of the Grad-Shafranov equation, as a function of psi
g = dolfin.Expression(
    "1/mu0*(-x[0]*A1 + A2/x[0])", A1=solovev.A1, A2=solovev.A2, mu0=MU_0, degree=2
)
# ------------------------------------------------------------------------------
# Solve the equation

print("\nSolving...")

# solve the Grad-Shafranov equation
solve_start = time.time()  # compute the time it takes to solve
psi1 = gs_solver.solve(g)
solve_end = time.time()

points1 = mesh.coordinates()
psi1_data = np.array([psi1(x) for x in points1])
# solovev.psi.set_allow_extrapolation(True)
psi1_exact = solovev.psi(points1)

levels = np.linspace(0.0, 110, 25)

axis, cntr, _ = plot_scalar_field(
    points1[:, 0], points1[:, 1], psi1_exact, levels=levels, axis=None, tofill=False
)

plt.show()

axis = None
axis, cntr, _ = plot_scalar_field(
    points1[:, 0], points1[:, 1], psi1_data, levels=20, axis=axis, tofill=True
)
plt.show()

error = abs(psi1_data - psi1_exact)

levels = np.linspace(0.0, max(error) * 1.1, 50)
axis, cntr, _ = plot_scalar_field(
    points1[:, 0], points1[:, 1], error, levels=levels, axis=None, tofill=True
)
plt.show()

# L2_error_h = dolfin.errornorm(solovev.psi, gs_solver.psi)
