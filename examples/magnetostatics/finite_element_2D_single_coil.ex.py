# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Application of the dolfin fem 2D magnetostatic to a single coil problem
"""

# %% [markdown]
# # 2-D FEM magnetostatic single coil
# ## Introduction
#
# In this example, we will show how to use the fem_magnetostatic_2D solver to find the
# magnetic field generated by a simple coil. The coil axis is the z-axis. Solution is
# calculated on the xz plane.
#
# ## Imports
#
# Import necessary module definitions.

# %%
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx.io import XDMFFile
from dolfinx.plot import vtk_mesh
from matplotlib.axes import Axes
from mpi4py import MPI

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics import greens
from bluemira.magnetostatics.fem_utils import (
    Association,
    create_j_function,
    model_to_mesh,
)
from bluemira.magnetostatics.finite_element_2d import Bz_coil_axis, FemMagnetostatic2d
from bluemira.mesh import meshing

<<<<<<< HEAD
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

=======
# %%
>>>>>>> 99f124d2 (🎨 Various updates and cleanup for dolfinx 0.7.1)
rank = MPI.COMM_WORLD.rank

ri = 0.01  # Inner radius of copper wire
rc = 3  # Outer radius of copper wire
R = 100  # Radius of domain
I_wire = 10e6  # wire's current
gdim = 2  # Geometric dimension of the mesh
model_rank = 0
mesh_comm = MPI.COMM_WORLD

# Define geometry for wire cylinder
nwire = 20  # number of wire divisions
lwire = 0.1  # mesh characteristic length for each segment

nenclo = 20  # number of external enclosure divisions
lenclo = 1  # mesh characteristic length for each segment

# enclosure
theta_encl = np.linspace(np.pi / 2, -np.pi / 2, nenclo)
r_encl = R * np.cos(theta_encl)
z_encl = R * np.sin(theta_encl)

# adding (0,0) to improve mesh quality
enclosure_points = [
    [0, 0, 0],
    *[[r_encl[ii], z_encl[ii], 0] for ii in range(r_encl.size)],
]


poly_enclo1 = make_polygon(enclosure_points[0:2])
poly_enclo1.mesh_options = {"lcar": 0.05, "physical_group": "poly_enclo1"}
poly_enclo2 = make_polygon(enclosure_points[1:])
poly_enclo2.mesh_options = {"lcar": 1, "physical_group": "poly_enclo2"}
poly_enclo = BluemiraWire([poly_enclo1, poly_enclo2])
poly_enclo.close("poly_enclo")
poly_enclo.mesh_options = {"lcar": 1, "physical_group": "poly_enclo"}

# coil
theta_coil = np.linspace(0, 2 * np.pi, nwire)
r_coil = rc + ri * np.cos(theta_coil[:-1])
z_coil = ri * np.sin(theta_coil)

coil_points = [[r_coil[ii], z_coil[ii], 0] for ii in range(r_coil.size)]

poly_coil = make_polygon(coil_points, closed=True)
lcar_coil = np.ones([poly_coil.vertexes.shape[1], 1]) * lwire
poly_coil.mesh_options = {"lcar": 0.01, "physical_group": "poly_coil"}

coil = BluemiraFace([poly_coil])
coil.mesh_options.physical_group = "coil"

enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options.physical_group = "enclo"

c_universe = Component(name="universe")
c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

# %% [markdown]
#
# ## Mesh
#
# Create the mesh (by default, mesh is stored in the file Mesh.msh")

# %%
directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [Path(directory, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]

meshing.Mesh(meshfile=meshfiles)(c_universe, dim=2)

(mesh, ct, ft), labels = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
gmsh.write("Mesh.msh")
gmsh.finalize()

with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)

# pyvista.start_xvfb()

pyvista.OFF_SCREEN = False
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb()

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
grid.set_active_scalars("Marker")
actor = plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    cell_tag_fig = plotter.screenshot("cell_tags.png")


<<<<<<< HEAD
from bluemira.magnetostatics.fem_utils import (
    calculate_area,
    create_j_function,
    integrate_f,
)
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d
from bluemira.magnetostatics.fem_utils import create_j_function

=======
>>>>>>> 99f124d2 (🎨 Various updates and cleanup for dolfinx 0.7.1)
# %%
em_solver = FemMagnetostatic2d(2)
em_solver.set_mesh(mesh, ct)

# %% [markdown]
#
# Define source term (coil current distribution) for the fem problem

# %%
coil_tag = labels["coil"][1]
functions = [(1, coil_tag, I_wire)]
jtot = create_j_function(mesh, ct, [Association(1, coil_tag, I_wire)])

# %% [markdown]
#
# solve the em problem and calculate the magnetic field B

# %%
em_solver.define_g(jtot)
em_solver.solve()
em_solver.calculate_b()

# %% [markdown]
#
# Compare the obtained B with both the theoretical value
#
# 1) Along the z axis (analytical solution)

# %%
z_points_axis = np.linspace(0, R, 200)
r_points_axis = np.zeros(z_points_axis.shape)
b_points = np.array([r_points_axis, z_points_axis, 0 * z_points_axis]).T

Bz_axis, b_points = em_solver.B._eval_new(b_points)
Bz_axis = Bz_axis[:, 1]
bz_points = b_points[:, 1]
B_z_teo = np.array([Bz_coil_axis(rc, 0, z, I_wire) for z in bz_points])

ax: Axes
_, ax = plt.subplots()
ax.plot(bz_points, Bz_axis, label="B_calc")
ax.plot(bz_points, B_z_teo, label="B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("B (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(bz_points, Bz_axis - B_z_teo, label="B_calc - B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
ax.legend()
plt.show()

# %% [markdown]
#
# 1) Along a radial path at z_offset (solution from green function)

# %%
z_offset = 40 * ri

points_x = np.linspace(0, R, 200)
points_z = np.zeros(z_points_axis.shape) + z_offset

new_points = np.array([points_x, points_z, 0 * points_z]).T
B_fem, new_points = em_solver.B._eval_new(new_points)
Bx_fem = B_fem.T[0]
Bz_fem = B_fem.T[1]

g_psi, g_bx, g_bz = greens.greens_all(rc, 0, new_points[:, 0], new_points[:, 1])
g_psi *= I_wire
g_bx *= I_wire
g_bz *= I_wire

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem, label="Bx_fem")
ax.plot(new_points[:, 0], g_bx, label="Green Bx")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bx (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bz_fem, label="Bz_fem")
ax.plot(new_points[:, 0], g_bz, label="Green Bz")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bz (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem - g_bx, label="B_calc - GreenBx")
ax.plot(new_points[:, 0], Bz_fem - g_bz, label="B_calc - GreenBz")
ax.legend()
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
plt.show()

# from bluemira.magnetostatics.fem_utils import plot_meshtags
# pyvista.OFF_SCREEN = False
# plot_meshtags(mesh)
