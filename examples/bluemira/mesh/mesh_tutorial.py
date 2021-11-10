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

r3 = r2 + 0.2
r4 = r3 + 1
z3 = 0
z4 = z3 + 0.5

p1 = FreeCAD.Vector(r1, 0, z1)
p2 = FreeCAD.Vector(r2, 0, z1)
p3 = FreeCAD.Vector(r2, 0, z2)
p4 = FreeCAD.Vector(r1, 0, z2)

Points = [p1, p2, p3, p4]

p5 = FreeCAD.Vector(r3, 0, z3)
p6 = FreeCAD.Vector(r4, 0, z3)
p7 = FreeCAD.Vector(r4, 0, z4)
p8 = FreeCAD.Vector(r3, 0, z4)

Points2 = [p5, p6, p7, p8]

poly1 = geo.tools.make_polygon(Points, "poly1", True)
ser_poly1 = bluemira.geometry.tools.serialize_shape(poly1)
print(ser_poly1)
des_poly1 = bluemira.geometry.tools.deserialize_shape(ser_poly1)
print(des_poly1)
ser_poly1_2 = bluemira.geometry.tools.serialize_shape(des_poly1)
print(ser_poly1_2)

face1 = geo.face.BluemiraFace(poly1, "face1")

ser_face1 = bluemira.geometry.tools.serialize_shape(face1)
print(ser_face1)
des_face1 = bluemira.geometry.tools.deserialize_shape(ser_face1)
print(des_face1)

poly2 = geo.tools.make_polygon(Points2, "poly2", True)
face2 = geo.face.BluemiraFace(poly2, "face2")
shell1 = geo.shell.BluemiraShell([face1, face2], "shell1")
print(f"shell1: {shell1}")

ser_shell1 = geo.tools.serialize_shape(shell1)
print(f"ser_shell1: {ser_shell1}")

des_shell1 = geo.tools.deserialize_shape(ser_shell1)
print(f"des_shell1: {des_shell1}")

face1.mesh_options = {'lcar': 0.1}
poly2.mesh_options = {'physical_group': 'test'}

m = Mesh()
buffer = m(shell1)
print(m.get_gmsh_dict(buffer))

# import json
#
# with open("mesh_dict.json", "w") as outfile:
#     json.dump(buffer, outfile)
