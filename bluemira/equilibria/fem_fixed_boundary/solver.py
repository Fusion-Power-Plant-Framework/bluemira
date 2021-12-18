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

from plasma import Plasma
import bluemira.equilibria.fem_fixed_boundary.tools as tools
import bluemira.codes.plasmod as plasmod
from dolfinSolver import GradShafranovLagrange
from bluemira.mesh import meshing
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.plane import BluemiraPlane
from bluemira.equilibria.shapes import JohnerLCFS
import bluemira.display as display
from bluemira.display.plotter import FacePlotter
import matplotlib.pyplot as plt
import bluemira.mesh.msh2xdmf as msh2xdmf
import dolfin
import numpy as np

# creation of a plasma
## create a geometry
## create a face

#%% close all figures and clear console


# %% Geometry (EU-DEMO 2017)
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
lcfs = p.create_shape(label="LCFS")
lcfs.mesh_options = {"lcar": 0.3, "physical_group": "LCFS"}

# plot shape
my_options = display.plotter.get_default_options()
print(my_options)
f, ax = plt.subplots()
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.grid(True)
ax.set_title("Plasma shape")
display.plot_2d(
    lcfs, show=False, ax=ax, wire_options={"color": "red", "linestyle": "dashed"}
)

# Face
plasma_face = BluemiraFace(lcfs, label="plasma_surface")
plasma_face.mesh_options = {"lcar": 0.5, "physical_group": "plasma"}

# plot face
fig, ax = plt.subplots()
fplotter = FacePlotter(plane="xz", face_options={"color": "red"})
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.grid(True)
fplotter.plot_2d(plasma_face, ax=ax, show=False)
ax.set_title("Face plot without points (default)")


# %%[markdown] generate mesh
xz_plane = BluemiraPlane(axis=[1, 0, 0], angle=-90)
plasma_face.change_plane(xz_plane)

plasma_comp = Plasma(name="plasma", shape=plasma_face)
plasma_shell = BluemiraShell(plasma_face)

m = meshing.Mesh()
buffer = m(plasma_shell)
# print(m.get_gmsh_dict(buffer))

# %%[markdown]
# Convert the me in xdmf for reading in fenics

msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")
mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
    prefix="Mesh",
    dim=2,
    directory=".",
    subdomains=True,
)

#%%[markdown] Plot mesh
dolfin.plot(mesh)

#%%[markdown] solve GSE for a constant current density distribution
gs_solver = GradShafranovLagrange(mesh, p=2)
Ip = 1.9e7
Ap = plasma_face.area
print("Average current density [A/m²] = " + str(Ip / Ap))
g = dolfin.Expression(str(Ip / Ap), degree=2)
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

# todo get flux surface contours for 2D FEM functions, e.g. trincontour plots
levels = [psi_ax*0.05]
axis, cntr, cntrf = tools.plot2d_scalar_field(x, z, psi_v, levels=levels, axis=None, to_fill=False, show=True)
path = []
for index in range(len(levels)):
    path.append(cntr.collections[index].get_paths()[0].vertices)


# plt.close("all")

"""
mhd_solver = plasmod.Solver(...)
plasmod_parameters = {}
"""
# plasma.set_mhd_solver = mhd_solver
# plasma.set_gs_solver = gs_solver

# Remember that this
#     g = plasma.J_to_dolfinFunction(solver.V)
# will become this
# g = tools.func_to_dolfinFunction(plasma.curr_density, gs_solver.V)

# implement the dolfinUpdate that adjust the current density
# check the method in core.py for PlasmaFreeGS


#
# plasma = self.getPlasma()
#
# if plasma is None:
#     raise ValueError("No plasma has been found")
#
# if solver is None:
#     if (not hasattr(plasma.shape,
#                     'physicalGroups')) or plasma.shape.physicalGroups is None:
#         plasma.shape.physicalGroups = {1: "external", 2: "plasma"}
#     else:
#         if not 1 in plasma.shape.physicalGroups:
#             plasma.shape.physicalGroups[1] = "external"
#
#         if not 2 in plasma.shape.physicalGroups:
#             plasma.shape.physicalGroups[2] = "plasma"
#
#     mesh_dim = 2
#
#     if plasma.J is None:
#         raise ValueError('Plamsa Jp must to be defined')
#
#     fullmeshfile = os.path.join(meshdir, meshfile)
#
#     print(fullmeshfile)
#
#     if createmesh:
#         #### Mesh Generation ####
#         mesh = mirapy.core.Mesh("plasma")
#         mesh.meshfile = fullmeshfile
#         if not Pax is None:
#             # P0lcar = plasma.shape.lcar/2.
#             mesh.embed = [(Pax, Pax_lcar)]
#         mesh(plasma)
#
#     # Run the conversion
#     mirapy.msh2xdmf.msh2xdmf(meshfile, dim=mesh_dim)
#
#     # Run the import
#     prefix, _ = os.path.splitext(fullmeshfile)
#
#     mesh, boundaries, subdomains, labels = mirapy.msh2xdmf.import_mesh_from_xdmf(
#         prefix=prefix,
#         dim=mesh_dim,
#         directory=meshdir,
#         subdomains=True,
#     )
#
#     solver = mirapy.dolfinSolver.GradShafranovLagrange(mesh, p=p)
#
# # Calculate plasma geometrical parameters
# plasma.calculatePlasmaParameters(solver.mesh)
#
# eps = 1.0  # error measure ||u-u_k||
# i = 0  # iteration counter
# while eps > tol and i < maxiter:
#     prev = solver.psi.compute_vertex_values()
#     i += 1
#     plasma.psi = solver.psi
#     g = plasma.J_to_dolfinFunction(solver.V)
#     solver.solve(g)
#     diff = solver.psi.compute_vertex_values() - prev
#     eps = numpy.linalg.norm(diff, ord=numpy.Inf)
#     print('iter = {} eps = {}'.format(i, eps))
#     plasma.dolfinUpdate(solver.V)
#
# self.__solvers['fixed_boundary'] = solver
# plasma.updateFilaments(solver.V)
