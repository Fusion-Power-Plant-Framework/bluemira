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
Application test for the caching geoemtry (ivanmaione/develop_simple_caching_2)
"""

# %%[markdown]

# # Introduction

# This is a simple example to test geometry caching module capabilities

# # Imports

# Import necessary module definitions.

# %%

import dolfin  # noqa
import matplotlib.pyplot as plt

import bluemira.display as display
import bluemira.geometry.tools as geotools
import bluemira.mesh.tools as meshtools
from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh.meshing import Mesh

# %%[markdown]

# # Geometry caching test

# Creation of a simple 2-D geometry
# 1. a square

# %%
points = Coordinates({"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]})
wire_in = geotools.make_polygon(points, "wire_in", True)

wire_plotter = display.plotter.WirePlotter()
wire_plotter.options.view = "xy"
wire_plotter.plot_2d(wire_in)

# %%[markdown]

# 2. a D-shape

# %%
wires = []
wires.append(geotools.make_polygon([[0, 3], [0, 0], [0, 0]], label="wire_out1"))
wires.append(geotools.make_circle(1, (3, 1, 0), 270, 360, label="circle_out2"))
wires.append(geotools.make_polygon([[4, 4], [1, 3], [0, 0]], label="wire_out3"))
wires.append(geotools.make_circle(1, (3, 3, 0), 0, 90, label="circle_out4"))
wires.append(geotools.make_polygon([[3, 0], [4, 4], [0, 0]], label="wire_out5"))
wires.append(geotools.make_polygon([[0, 0], [4, 0], [0, 0]], label="wire_out6"))
wire_out = BluemiraWire(wires, label="wire_out")
print(wire_out)
print(f"wire_out is closed: {wire_out.is_closed()}")
wire_plotter.plot_2d(wire_out)


# %%[markdown]

# Create a component (just for plotting purpose in this case)

# %%
root = Component("test_comp")
root.plot_options.view = "xy"
root.plot_options.wire_options["linewidth"] = 2
comp_in = PhysicalComponent("comp_in", wire_in, parent=root)
comp_in.plot_options.view = "xy"
comp_in.plot_options.wire_options["linewidth"] = 2
comp_out = PhysicalComponent("comp_out", wire_out, parent=root)
comp_out.plot_options.view = "xy"
comp_out.plot_options.wire_options["linewidth"] = 2
root.plot_2d(show=True)

# %%[markdown]

# Apply a translation to wire_in

# %%
wire_in.translate((1, 1, 0))
root.plot_2d(show=True)

# %%[markdown]

# Change one of the sub-wires in wire_out.
# Warning: This is a non-conventional operation

# %%
new_wire = geotools.make_polygon([[3, 4], [0, 1], [0, 0]], "wire_out2")
wire_out.boundary[1] = new_wire

# # or
# old_wire = wire_out.search("circle_out2")[0]
# old_wire.boundary = new_wire.boundary
# old_wire.label = new_wire.label
root.plot_2d(show=True)

# %%[markdown]

# As can be seen, root shape is not changed. This is because the wire_out shape has
# not been updated (there is no collback that "informs" wire_out on a change in the
# internal structure of a wire.
#
# wire_out must be updated.

# %%
wire_out._set_boundary(wire_out.boundary)

print(wire_out)
root.plot_2d(show=True)

# %%[markdown]

# # Meshing
# Set mesh options for comp_out and comp_in

# %%
comp_out.shape.mesh_options.lcar = 0.1
comp_out.shape.mesh_options.physical_group = "out"

comp_in.shape.mesh_options.lcar = 0.1
comp_in.shape.mesh_options.physical_group = "in"

# %%[markdown]

# Create a face
# Note: wire mesh options are still stored in face through its "self.boundary".
# However, it is necessary to define (at least) the mesh_options.physical_group for
# face, otherwise it will not be exported during the mesh process.

# %%
face = BluemiraFace([wire_out, wire_in], label="face")
# face.mesh_options.lcar = 0.1
face.mesh_options.physical_group = "face"

# %%[markdown]

# Create the respective component (for plotting and meshing purpose)

# %%
face_comp = PhysicalComponent("face_comp", face)
face_comp.plot_options.view = "xy"
face_comp.plot_options.wire_options["linewidth"] = 2
face_comp.plot_2d(show=True)

# %%[markdown]

# Mesh face_comp and import in dolfin

# %%
m = Mesh()
buffer = m(face_comp)
meshtools.msh_to_xdmf("Mesh.msh", dimensions=2, directory=".")

mesh, boundaries, _, _ = meshtools.import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)

dolfin.plot(mesh)
plt.show()

# %%[markdown]

# Since no geometry operations have been made that called a "recreation" of the
# self.boundary property, it is still possible to act on any wire property to have an
# effect on face_comp. For example, it is possible to change the mesh size of a
# sub-wire in wire_out (in this case, "circle_out4"):

# %%
obj = wire_out.search("circle_out4")[0]
obj.mesh_options.lcar = 0.02
obj.mesh_options.physical_group = "circle"

buffer = m(face_comp)
meshtools.msh_to_xdmf("Mesh.msh", dimensions=2, directory=".")

mesh, boundaries, _, _ = meshtools.import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)

dolfin.plot(mesh)
plt.show()
