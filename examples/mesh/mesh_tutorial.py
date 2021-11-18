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
#import bluemira.display as display
from bluemira.mesh import meshing
from bluemira.geometry.face import BluemiraFace
import bluemira.mesh.msh2xdmf as msh2xdmf
import dolfin
# %%[markdown]

# Creation of a simple geometry

# %%

p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")
lcfs.mesh_options = {'lcar': 0.3, 'physical_group': 'LCFS'}
face = BluemiraFace(lcfs, label="plasma_surface")
face.mesh_options = {"lcar": 0.1, "physical_group": "surface"}
#display.plot_2d(face)

# %%[markdown]

# Mesh creation

# %%
m = meshing.Mesh()
buffer = m(face)
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