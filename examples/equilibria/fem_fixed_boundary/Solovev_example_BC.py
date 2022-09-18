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

import matplotlib.pyplot as plt
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import Solovev, plot_scalar_field
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import interpolate_bspline, make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfin


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
xmin = 5
zmin = -6
dx = 8
dz = 12
nx = 100
nz = 100
axis, cntr, cntrf, points, psi_exact = solovev.plot_psi(
    xmin, zmin, dx, dz, nx, nz, levels=levels
)
plt.show()
#-------------------------------------------------------------------------------
# create a corresponding geometrical domain
d_points = Coordinates({'x':[xmin, xmin + dx, xmin + dx, xmin],
                        'y':[0, 0, 0, 0],
                        'z':[zmin, zmin, zmin + dz, zmin + dz]})

rect = make_polygon(d_points, "boundary", True)
rect.mesh_options = {'lcar': 0.1, 'physical_group': 'boundary'}
face = BluemiraFace(rect)
face.mesh_options = {'lcar': 0.1, 'physical_group': 'domain'}

m = meshing.Mesh()
buffer = m(face)

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

# initialize the Grad-Shafranov solver
c = solovev.coeff
A1 = solovev.A1
A2 = solovev.A2

p = 2
gs_solver = FemMagnetostatic2d(mesh, boundaries, p_order=p)

# Set the right hand side of the Grad-Shafranov equation, as a function of psi
g = dolfin.Expression(
    "1/mu0*(-x[0]*A1 + A2/x[0])", pi=np.pi, A1=A1, A2=A2, mu0=MU_0, degree=p
)

# boundary conditions definition
# the Dirichlet boundary condition (in this case it is the exact solution)
dirichletBCFunction = dolfin.Expression(
    'c1 + c2*pow(x[0],2) + c3*(pow(x[0],4) - 4*pow(x[0],2)*pow(x[1],2)) + c4*(pow(x['
    '0],2)*std::log(x[0]) -pow(x[1],2)) + A1*pow(x[0],4)/8 -A2*pow(x[1],2)/2',
    c1 = c[0], c2 = c[1], c3 = c[2], c4 = c[3], A1 = A1, A2 = A2, degree=p)

#check BC with psi_exact
num = 0
if psi_exact[num] - dirichletBCFunction(points[num][0], points[num][1])*2*np.pi > 1e-8:
    raise ValueError('dirichletBCFunction are not equal to psi_exact')

# ------------------------------------------------------------------------------
# Solve the equation

print("\nSolving...")

# solve the Grad-Shafranov equation
solve_start = time.time()
psi1 = gs_solver.solve(g, dirichletBCFunction, labels['boundary'])
solve_end = time.time()

print(f"\nSolved in {solve_end - solve_start} seconds")

psi1_data = np.array([psi1(x)*2*np.pi for x in points])
diff = psi1_data - psi_exact
eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(psi_exact, ord=2)
print(eps)

psi1_data = np.array([psi1(x)*2*np.pi for x in points])
levels = np.linspace(0.0, 110, 25)

axis, cntr, _ = plot_scalar_field(
    points[:, 0], points[:, 1], psi_exact, levels=50, axis=None, tofill=True
)
plt.show()

axis = None
axis, cntr, _ = plot_scalar_field(
    points[:, 0], points[:, 1], psi1_data, levels=50, axis=axis, tofill=True
)
plt.show()

bc_data = np.array([dirichletBCFunction(x)*2*np.pi for x in points])
axis = None
axis, cntr, _ = plot_scalar_field(
    points[:, 0], points[:, 1], bc_data, levels=50, axis=axis,
    tofill=True
)
plt.show()

#
# error = abs(psi1_data - psi1_exact)
#
# levels = np.linspace(0.0, max(error) * 1.1, 50)
# axis, cntr, _ = plot_scalar_field(
#     points1[:, 0], points1[:, 1], error, levels=levels, axis=None, tofill=True
# )
# plt.show()
#
# # L2_error_h = dolfin.errornorm(solovev.psi, gs_solver.psi)
#
# diff = psi1_data - psi1_exact
# eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(psi1_exact, ord=2)
# print(eps)