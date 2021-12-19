# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                    J. Morris, D. Short
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
Testing the fixed-boundary equilibrium solver.
"""
from bluemira.equilibria.shapes import JohnerLCFS

from bluemira.base.config import Configuration

from bluemira.equilibria.fem_fixed_boundary.plasma import Plasma
from bluemira.equilibria.fem_fixed_boundary.dolfinSolver import GradShafranovLagrange
import bluemira.equilibria.fem_fixed_boundary.tools as tools
import bluemira.equilibria.fem_fixed_boundary.transport_solver as transport_solver

from bluemira.mesh import meshing
import bluemira.mesh.msh2xdmf as msh2xdmf

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.plane import BluemiraPlane

import matplotlib.pyplot as plt

import dolfin
import numpy as np

# %% Geometry (EU-DEMO 2017)
## Plasma shape creation
R0 = 8.938
A = 3.1
p = JohnerLCFS(
    {
        "r_0": {"value": R0},
        "a": {"value": R0 / A},
        "kappa_u": {"value": 1.70},
        "kappa_l": {"value": 1.85},
        "delta_u": {"value": 0.50},
        "delta_l": {"value": 0.50},
    }
)

# Plasma LCFS
lcfs = p.create_shape(label="LCFS")
lcfs.mesh_options = {"lcar": 0.3, "physical_group": "LCFS"}

# Plasma cross-section
plasma_face = BluemiraFace(lcfs, label="plasma_surface")
plasma_face.mesh_options = {"lcar": 0.5, "physical_group": "plasma"}

# Plasma component
plasma_comp = Plasma(name="plasma", shape=plasma_face)
plasma_shell = BluemiraShell(plasma_face)

# plot plasma_comp (default)
ax = plasma_comp.plot_2d(show=False)
ax.grid(True)
ax.set_title("Plasma shape (default plot options)")
plt.show()

# Now it is necessary to set the plasma solvers:
# 1) mhd_solver
# Note: with the new version of plasmod interfaces into codes, I cannot
# run plasmod in mock mode. Since I don't have plasmod installed, I am
# going to set an empty plasma transport solver (just a backup solution
# for the moment).
plasmod = "bluemira.codes.plasmod"

PLASMOD_PATH = "../plasmod_bluemira"

new_params = {
    "A": 3.1,
    "B_0": 5.3,
    "R_0": 8.93,
    "q_95": 3.23,
}
build_config = {
    "problem settings": {
        "Pfus_req": 2000,
        "i_modeltype": "GYROBOHM_2",
    },
    "mode": "mock",
    "binary": f"{PLASMOD_PATH}/plasmod.o",
}

mhd_solver = transport_solver.TransportSolver(
    plasmod, params=Configuration(new_params), build_config=build_config
)

plasma_comp.set_mhd_solver(mhd_solver)

pprime = plasma_comp._pprime

# Te = plasmod_solver.get_profile("Te")
x = mhd_solver.solver.get_profile("x")
fig, ax = plt.subplots()
ax.plot(x, pprime(x))
ax.set(xlabel="x (-)", ylabel="pprime")
ax.grid()
plt.show()


ffprime = plasma_comp._ffprime

# Te = plasmod_solver.get_profile("Te")
x = mhd_solver.solver.get_profile("x")
fig, ax = plt.subplots()
ax.plot(x, ffprime(x))
ax.set(xlabel="x (-)", ylabel="ffprime")
ax.grid()
plt.show()


# 2) gs_solver
# Note: to create a gs_solver, the plasma mesh is needed. In the following
# the mesh is created with direct code. In the future, probably, it is better
# to integrate this part in a "ad-hoc" plasma method (or gs_solver method).

# %%[markdown] generate mesh
# Note: due to problem with msh2xdmf when doing mesh2d, i.e. only first
# 2 spatial compononents are saved into mesh coordinates, it is necessary
# to change the plasma plane.
xz_plane = BluemiraPlane(axis=[1, 0, 0], angle=-90)
plasma_face.change_plane(xz_plane)

# plot plasma_comp (only LCFS) - jsut to check that the plane has been changed
plasma_comp._plotter.set_plane("xy")
plasma_comp._plotter.options.show_faces = False
plasma_comp._plotter.options.wire_options = {"color": "red", "linestyle": "dashed"}
ax = plasma_comp.plot_2d(show=False)
ax.grid(True)
ax.set_title("Plasma contour")
plt.show()

# mesh is initialized and created
m = meshing.Mesh()
buffer = m(plasma_shell)

# %%[markdown]
# Convert the me in xdmf for reading in fenics
msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")
mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
    prefix="Mesh",
    dim=2,
    directory=".",
    subdomains=True,
)

# %%[markdown] Plot mesh
# Note: this plot function works only for 2D meshes.
dolfin.plot(mesh)

#%%[markdown] Define the GSE solver
gs_solver = GradShafranovLagrange(mesh, p=2)

# set plasma Grad-Shafranov solver
plasma_comp.set_gs_solver(gs_solver)

# This value should be declared into the set of plasma or reactor parameters
# For this example, it is declared here
Ip = 1.9e7
Ap = plasma_face.area
print("Average current density [A/m²] = " + str(Ip / Ap))

# initialize the plasma current density to a constant value
# (this is due to the fact that the msh_solver is None
plasma_curr_density = plasma_comp.curr_density(Ip / Ap)

g = tools.func_to_dolfinFunction(plasma_curr_density, gs_solver.V)
# The next code would have produced the same result
# g = dolfin.Expression(str(Ip / Ap), degree=2)

gs_solver.solve(g)

plasma_curr_density = plasma_comp.curr_density()
g = tools.func_to_dolfinFunction(plasma_curr_density, gs_solver.V)
gs_solver.solve(g)

psi = gs_solver.psi

#%%[markdown] plot poloidal flux
fig, ax = plt.subplots()
c = dolfin.plot(psi, title="Fancy plot", mode="color")
dolfin.plot(mesh)
ax.set_xlabel("$r$ [m]")
ax.set_ylabel("$z$ [m]")
c.set_cmap("viridis")
fig.colorbar(c)
fig.show()
fig.savefig("polo_flux.png")

#%% calculate solution on a vertical line across r = R0
psi = gs_solver.psi
z = np.linspace(-1, 1, 101)
points = [(R0, z_) for z_ in z]  # 2D points
psi_line = np.array([psi(point) for point in points])

fig, ax = plt.subplots()
plt.plot(z, psi_line, "ko-", linewidth=2)
ax.grid(True)
ax.set_xlabel("$z$ [m]")
ax.set_ylabel("$\Psi$ [Wb]")
plt.title("$\Psi$ vs. $z$")
plt.show()

#%%[markdown] calculate max value --> axis
psi_ax = psi.vector().max()
print("Max polo flux at axis:", psi_ax, "Wb")

#%%[markdown] get 95 % flux surface
v = mesh.coordinates()
x = v[:, 0]
z = v[:, 1]
psi_v = psi.compute_vertex_values()

psi_norm = (psi_ax - psi_v) / (psi_ax)

levels = np.linspace(0, 1, 21)
axis, cntr, cntrf = tools.plot2d_scalar_field(
    x, z, psi_norm, levels=levels, axis=None, to_fill=False, show=True
)
path_coordinates = []
for index in range(len(levels)):
    path = cntr.collections[index].get_paths()
    if path:
        path_coordinates.append(path[0].vertices)
    else:
        path_coordinates.append([])
plt.close("all")
