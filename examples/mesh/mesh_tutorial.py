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

# %%[markdown]

# # Introduction

# In this example, we will show how to use the mesh module to create a 2D mesh for fem
# application

# # Imports

# Import necessary module definitions.

# %%

import os

import dolfin
import matplotlib.pyplot as plt

import bluemira.geometry.tools as tools
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_root
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.wire import BluemiraWire
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

# # Geometry

# Creation of a simple 2D geometry, i.e. a Johner shape + a coil with casing

# %%

p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")

poly1 = tools.make_polygon(
    [[0, 1, 1], [0, 0, 1], [0, 0, 0]], closed=False, label="poly1"
)
poly2 = tools.make_polygon(
    [[1, 0, 0], [1, 1, 0], [0, 0, 0]], closed=False, label="poly2"
)
poly3 = tools.make_polygon(
    [[0.25, 0.75, 0.75], [0.25, 0.25, 0.75], [0, 0, 0]], closed=False, label="poly3"
)
poly4 = tools.make_polygon(
    [[0.75, 0.25, 0.25], [0.75, 0.75, 0.25], [0, 0, 0]], closed=False, label="poly4"
)
poly_out = BluemiraWire([poly1, poly2], label="poly_out")
poly_in = BluemiraWire([poly3, poly4], label="poly_in")
coil_out = BluemiraFace([poly_out, poly_in], label="coil_out")
coil_in = BluemiraFace([poly_in], label="coil_in")

# %%[markdown]

# Note: due to a limitation of the mesh converter for importing of the mesh into a fem
# solver, the 2D geometry must to lie on the xy plane. For this reason, a rotation of
# the geometry is applied.

# %%
lcfs.change_placement((BluemiraPlacement(axis=[1.0, 0.0, 0.0], angle=-90)))

# %%[markdown]

# # Mesh setup

# setup characteristic mesh length

# %%

lcfs.mesh_options = {"lcar": 0.75, "physical_group": "LCFS"}
face = BluemiraFace(lcfs, label="plasma_surface")
face.mesh_options = {"lcar": 0.5, "physical_group": "surface"}

poly1.mesh_options = {"lcar": 0.25, "physical_group": "poly1"}
poly2.mesh_options = {"lcar": 0.25, "physical_group": "poly2"}
poly3.mesh_options = {"lcar": 0.75, "physical_group": "poly3"}
poly4.mesh_options = {"lcar": 0.75, "physical_group": "poly4"}
coil_out.mesh_options = {"lcar": 1, "physical_group": "coil"}
coil_in.mesh_options = {"lcar": 0.3, "physical_group": "coil"}

# %%[markdown]

# In order to mesh all the geometry in one, the best solution is to create a component
# tree as in the following

# %%
c_all = Component(name="all")
c_plasma = PhysicalComponent(name="plasma", shape=face, parent=c_all)
c_coil = Component(name="coil", parent=c_all)
c_coil_in = PhysicalComponent(name="coil_in", shape=coil_in, parent=c_coil)
c_coil_out = PhysicalComponent(name="coil_out", shape=coil_out, parent=c_coil)

# %%[markdown]

# Initialize and create the mesh

# %%

m = meshing.Mesh()
buffer = m(c_all)
print(m.get_gmsh_dict(buffer))

# %%[markdown]

# # Convert to xdmf

# Convert the mesh in xdmf for reading in fenics (fem tool). Note that this requires the
# msh2xdmf module to be available.

# %%

if HAS_MSH2XDMF:
    msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")

    mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
        prefix="Mesh",
        dim=2,
        directory=".",
        subdomains=True,
    )
    dolfin.plot(mesh)
    plt.show()

    print(mesh.coordinates())
