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

import dolfin
import matplotlib.pyplot as plt

import bluemira.geometry.tools as tools
from bluemira.base.components import Component, PhysicalComponent
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

# Creation of a simple geometry

p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")

lcfs.mesh_options = {"lcar": 0.75, "physical_group": "LCFS"}
face = BluemiraFace(lcfs, label="plasma_surface")
face.mesh_options = {"lcar": 0.5, "physical_group": "surface"}

poly1 = tools.make_polygon(
    [[0, 1, 1], [0, 0, 0], [0, 0, 1]], closed=False, label="poly1"
)
poly2 = tools.make_polygon(
    [[1, 0, 0], [0, 0, 0], [1, 1, 0]], closed=False, label="poly2"
)
poly1.mesh_options = {"lcar": 0.25, "physical_group": "poly1"}
poly2.mesh_options = {"lcar": 0.25, "physical_group": "poly2"}

poly3 = tools.make_polygon(
    [[0.25, 0.75, 0.75], [0, 0, 0], [0.25, 0.25, 0.75]], closed=False, label="poly3"
)
poly4 = tools.make_polygon(
    [[0.75, 0.25, 0.25], [0, 0, 0], [0.75, 0.75, 0.25]], closed=False, label="poly4"
)
poly3.mesh_options = {"lcar": 0.75, "physical_group": "poly3"}
poly4.mesh_options = {"lcar": 0.75, "physical_group": "poly4"}

poly_out = BluemiraWire([poly1, poly2], label="poly_out")
poly_in = BluemiraWire([poly3, poly4], label="poly_in")
coil = BluemiraFace([poly_out, poly_in], label="coil")
coil.mesh_options = {"lcar": 1, "physical_group": "coil"}

shell = BluemiraShell([face, coil])

comp = Component(name="comp")
pcomp1 = PhysicalComponent(name="pcomp1", shape=coil, parent=comp)
pcomp2 = PhysicalComponent(name="pcomp2", shape=face, parent=comp)

# Mesh creation

m = meshing.Mesh()
buffer = m(comp)
print(m.get_gmsh_dict(buffer))

# Convert the mesh in xdmf for reading in fenics.

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    dimensions=(0, 1),
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

print(mesh.coordinates())
