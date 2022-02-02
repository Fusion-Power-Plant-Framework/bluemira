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

import os

import bluemira.geometry.tools as tools
from bluemira.base.file import get_bluemira_root
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.shell import BluemiraShell
from bluemira.mesh import meshing

HAS_MSH2XDMF = False
try:
    from bluemira.utilities.tools import get_module

    msh2xdmf = get_module(
        os.path.join(get_bluemira_root(), "..", "msh2xdmf", "msh2xdmf.py")
    )

    HAS_MSH2XDMF = True
except ImportError as err:
    print(f"Unable to import msh2xdmf, dolfin examples will not run: {err}")

# %%[markdown]

# Creation of a simple geometry

# %%

p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")
lcfs.change_plane(BluemiraPlane(axis=[1.0, 0.0, 0.0], angle=-90))
lcfs.mesh_options = {"lcar": 0.75, "physical_group": "LCFS"}
face = BluemiraFace(lcfs, label="plasma_surface")
face.mesh_options = {"lcar": 1, "physical_group": "surface"}

poly = tools.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
poly.mesh_options = {"lcar": 0.75, "physical_group": "poly"}
coil = BluemiraFace(poly)
# coil.mesh_options = {"lcar": 1, "physical_group": "coil"}

shell = BluemiraShell([face, coil])
# %%[markdown]

# Mesh creation

# %%
m = meshing.Mesh()
buffer = m(shell)
print(m.get_gmsh_dict(buffer))

# %%[markdown]

# Convert the mesh in xdmf for reading in fenics. Note that this requires the msh2xdmf
# module to be available.

# %%

if HAS_MSH2XDMF:
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
import dolfin
import matplotlib.pyplot as plt

dolfin.plot(mesh)
plt.show()

if HAS_MSH2XDMF:
    print(mesh.coordinates())
