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

import freecad  # noqa: F401
import FreeCAD

import bluemira.geometry as geo

from bluemira.mesh.meshing import Mesh

# Defining my parameters
import bluemira.geometry.tools

r1 = 0
r2 = r1 + 1
z1 = 0
z2 = z1 + 1

r3 = r2
r4 = r3 + 1
z3 = 0
z4 = z3 + 0.5
r5 = r4 + z4 - z3

p1 = FreeCAD.Vector(r1, 0, z1)
p2 = FreeCAD.Vector(r2, 0, z1)
p3 = FreeCAD.Vector(r2, 0, z2 + 0.5)
p4 = FreeCAD.Vector(r1, 0, z2)
ell_c = FreeCAD.Vector(r2, 0, z2)

Points = [p4, p1, p2, p3]

p5 = FreeCAD.Vector(r3, 0, z3)
p6 = FreeCAD.Vector(r5, 0, z3)
p7 = FreeCAD.Vector(r4, 0, z4)
p8 = FreeCAD.Vector(r3, 0, z4)

radius = r5 - r4
c = FreeCAD.Vector(r4, 0, z3)

Points2 = [p7, p8, p5, p6]


poly1 = geo.tools.make_polygon(Points, "poly1")
ellip = geo.tools.make_ellipse([1,0,1], 1, 0.5, [1,0,0], [0,0,0.5],
                               90, 180)
print(ellip.discretize(2))
ser_poly1 = bluemira.geometry.tools.serialize_shape(poly1)
print(ser_poly1)
des_poly1 = bluemira.geometry.tools.deserialize_shape(ser_poly1)
print(des_poly1)
ser_poly1_2 = bluemira.geometry.tools.serialize_shape(des_poly1)
print(ser_poly1_2)

wire1 = geo.wire.BluemiraWire([poly1, ellip])
face1 = geo.face.BluemiraFace(wire1, "face1")

ser_face1 = bluemira.geometry.tools.serialize_shape(face1)
print(ser_face1)
des_face1 = bluemira.geometry.tools.deserialize_shape(ser_face1)
print(des_face1)

poly2 = geo.tools.make_polygon(Points2[::-1], "poly2")
circle = geo.tools.make_circle(
    radius=radius,
    center=c,
    start_angle=270,
    end_angle=360,
    axis=[0.0, 1.0, 0.0],
    label="circle",
)

boundary = geo.wire.BluemiraWire([poly2, circle])
face2 = geo.face.BluemiraFace(boundary, "face2")
shell1 = geo.shell.BluemiraShell([face1, face2], "shell1")
print(f"shell1: {shell1}")

ser_shell1 = geo.tools.serialize_shape(shell1)
print(f"ser_shell1: {ser_shell1}")

des_shell1 = geo.tools.deserialize_shape(ser_shell1)
print(f"des_shell1: {des_shell1}")

# shell1.mesh_options = {"lcar": 0.5, "physical_group": "domain"}
face1.mesh_options = {"lcar": 0.1, "physical_group": "top_domain"}
face2.mesh_options = {"lcar": 0.3, "physical_group": "bot_domain"}
poly2.mesh_options = {"physical_group": "poly2_fg"}
# circle.mesh_options = {"lcar": 0.01}

m = Mesh(meshfile=["Mesh.geo_unrolled", "Mesh.msh"])
buffer = m(shell1)
print(m.get_gmsh_dict(buffer))

try:
    import msh2xdmf
    msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")

    mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
        prefix="Mesh",
        dim=2,
        directory=".",
        subdomains=True,
    )
except ImportError as e:
    print(f"Unable to import msh2xdmf: {e}")


# import json
#
# with open("mesh_dict.json", "w") as outfile:
#     json.dump(buffer, outfile)

# ser1 = geo.tools.serialize_shape(shell1)
# print(ser1)
#
# des1 = geo.tools.deserialize_shape(ser1)
# print(des1)
# print(des1.print_mesh_options(True))
