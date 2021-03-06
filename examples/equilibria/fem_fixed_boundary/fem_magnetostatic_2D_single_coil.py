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
Application of the dolfin fem 2D magnetostatic to a single coil problem
"""

# %%[markdown]

# # Introduction

# In this example, we will show how to use the fem_magnetostatic_2D solver to find the
# magnetic field generated by a simple coil. The coil axis is the z-axis. Solution is
# calculated on the xz plane.

# # Imports

# Import necessary module definitions.

# %%

import dolfin
import matplotlib.pyplot as plt
import numpy as np

import bluemira.geometry.tools as tools
import bluemira.magnetostatics.greens as greens
from bluemira.base.components import Component, PhysicalComponent
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import ScalarSubFunc, b_coil_axis
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

# %%[markdown]

# # Creation of the geometry

# Definition of coil and enclosure parameters

# %%
r_enclo = 100
lcar_enclo = 0.5

rc = 5
drc = 0.01
lcar_coil = 0.01

# %%[markdown]

# create the coil (rectangular cross section) and set the mesh options

# %%

poly_coil = tools.make_polygon(
    [[rc - drc, rc + drc, rc + drc, rc - drc], [0, 0, 0, 0], [-drc, -drc, +drc, +drc]],
    closed=True,
    label="poly_enclo",
)

poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
coil = BluemiraFace(poly_coil)
coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

# %%[markdown]

# create the enclosure (rectangular cross section) and set the mesh options

# %%
poly_enclo = tools.make_polygon(
    [[0, r_enclo, r_enclo, 0], [0, 0, 0, 0], [-r_enclo, -r_enclo, r_enclo, r_enclo]],
    closed=True,
    label="poly_enclo",
)

poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}
enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

# %%[markdown]

# create the different components

# %%
c_universe = Component(name="universe")
c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

# %%[markdown]

# # Mesh

# Create the mesh (by default, mesh is stored in the file Mesh.msh")

# %%

m = meshing.Mesh()
m(c_universe, dim=2)

# %%[markdown]

# Convert the mesh in xdmf for reading in fenics.

# %%

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".")

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

# %%[markdown]

# # Setup EM problem

# Finally, instantiate the em solver

# %%

em_solver = FemMagnetostatic2d(mesh, boundaries, 3)

# %%[markdown]

# Define source term (coil current distribution) for the fem problem

# %%

Ic = 1e6
jc = Ic / coil.area
markers = [labels["coil"]]
functions = [jc]
jtot = ScalarSubFunc(functions, markers, subdomains)

# %%[markdown]

# plot the source term
# Note: depending on the geometric dimension of the coil, enclosure, and mesh
# characteristic length, the plot could be not so "explanatory".

# %%

f_space = dolfin.FunctionSpace(mesh, "DG", 0)
f = dolfin.Function(f_space)
f.interpolate(jtot)
dolfin.plot(f, title="Source term")
plt.show()

# %%[markdown]

# solve the em problem and calculate the magnetic field B

# %%

em_solver.solve(jtot)
em_solver.calculate_b()

# %%[markdown]

# Compare the obtained B with both the theoretical value

# 1) Along the z axis (analytical solution)

# %%
z_points_axis = np.linspace(0, r_enclo, 200)
r_points_axis = np.zeros(z_points_axis.shape)
Bz_axis = np.array(
    [em_solver.B(x) for x in np.array([r_points_axis, z_points_axis]).T]
).T[1]
B_teo = np.array([b_coil_axis(rc, 0, z, Ic) for z in z_points_axis])

fig, ax = plt.subplots()
ax.plot(z_points_axis, Bz_axis, label="B_calc")
ax.plot(z_points_axis, B_teo, label="B_teo")
plt.legend()
plt.show()

diff = Bz_axis - B_teo

fig, ax = plt.subplots()
ax.plot(z_points_axis, diff, label="B_calc - B_teo")
plt.legend()
plt.show()

# %%[markdown]

# 1) Along a radial path at z_offset (solution from green function)

# %%

z_offset = 40 * drc

points_x = np.linspace(0, r_enclo, 200)
points_z = np.zeros(z_points_axis.shape) + z_offset

g_psi, g_bx, g_bz = greens.greens_all(rc, 0, points_x, points_z)
g_psi *= Ic
g_bx *= Ic
g_bz *= Ic
B_fem = np.array([em_solver.B(x) for x in np.array([points_x, points_z]).T])
Bx_fem = B_fem.T[0]
Bz_fem = B_fem.T[1]

fig, ax = plt.subplots()
ax.plot(z_points_axis, Bx_fem, label="Bx_fem")
ax.plot(z_points_axis, g_bx, label="Green Bx")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(z_points_axis, Bz_fem, label="Bz_fem")
ax.plot(z_points_axis, g_bz, label="Green Bz")
plt.legend()
plt.show()

diff1 = Bx_fem - g_bx
diff2 = Bz_fem - g_bz

fig, ax = plt.subplots()
ax.plot(z_points_axis, diff1, label="B_calc - GreenBx")
ax.plot(z_points_axis, diff2, label="B_calc - GreenBz")
plt.legend()
plt.show()
