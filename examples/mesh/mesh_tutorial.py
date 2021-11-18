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
Some examples of using bluemira mesh module.
"""

# %%

from bluemira.equilibria.shapes import JohnerLCFS

# %%[markdown]
# Creation of a simple geometry

# %%
p = JohnerLCFS()

# %%

from bluemira.equilibria.shapes import JohnerLCFS
import bluemira.display as display
from bluemira.mesh import meshing
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.plane import BluemiraPlane
import bluemira.mesh.msh2xdmf as msh2xdmf
import dolfin
import matplotlib.pyplot as plt
# %%[markdown]

# Creation of a simple geometry

# %%

p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")
lcfs.mesh_options = {'lcar': 0.3, 'physical_group': 'LCFS'}
plasma_face = BluemiraFace(lcfs, label="plasma_surface")
plasma_face.mesh_options = {"lcar": 0.5, "physical_group": "plasma"}

# create an external boundary. Just for semplicity I just scale the JohnerLCFS curve
p_ext = JohnerLCFS()
sol_ext_boundary = p_ext.create_shape()
bari = sol_ext_boundary.center_of_mass
sol_ext_boundary.scale(1.2)
new_bari = sol_ext_boundary.center_of_mass
diff = bari - new_bari
v = (diff[0], diff[1], diff[2])
sol_ext_boundary.translate(v)
display.plot_2d(sol_ext_boundary)

sol = BluemiraFace([sol_ext_boundary, lcfs.deepcopy()])
sol.mesh_options = {"lcar": 0.5, "physical_group": "sol"}

f, ax = plt.subplots()
fplotter = display.plotter.FacePlotter(plane="xz")
fplotter.options.show_points = False
ax = fplotter.plot_2d(plasma_face, ax=ax, show=False)
fplotter.options.face_options= {'c':'red'}
ax = fplotter.plot_2d(sol, ax=ax, show=False)
plt.show()


plane = BluemiraPlane(axis=[1,0,0], angle=90)
plasma_face.change_plane(plane)
sol.change_plane(plane)

compound = BluemiraShell([plasma_face, sol])

# %%[markdown]

# Mesh creation

# %%
m = meshing.Mesh()
buffer = m(compound)
print(m.get_gmsh_dict(buffer))

# %%[markdown]

# Convert the me in xdmf for reading in fenics

# %%

msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")

mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
    prefix="Mesh",
    dim=2,
    directory=".",
    subdomains=True,
)
# # %%[markdown]

# # Plot the mesh

# # %%
# # If the mesh is made by 3D points, the plot with dolfin doesn't work
# dolfin.plot(mesh)

print(mesh.coordinates())