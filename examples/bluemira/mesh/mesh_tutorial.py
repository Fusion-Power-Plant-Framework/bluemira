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
import Part

import bluemira.codes._freecadapi as cadapi
import bluemira.geometry as geo

from bluemira.mesh.meshing import Mesh

# Defining my parameters
import bluemira.geometry.tools

r1 = 49.5
r2 = 87
rn = 13.5
z1 = 23
b2 = 18
z2 = z1 + b2

# Points of the Bezier curve
p1 = FreeCAD.Vector(r1, 0, 0)
p2 = FreeCAD.Vector(r1, 0, (-z1 - 1) / 2)
p3 = FreeCAD.Vector(r1, 0, -z1)
p4 = FreeCAD.Vector((r1 + r2) / 2, 0, -z1)
p5 = FreeCAD.Vector(r2, 0, -z1)

Points = [p1, p2, p3, p4, p5]

# Creating the Bezier curve
bez = Part.BezierCurve()
bez.setPoles(Points)
bez_curve = Part.Edge(bez)
Part.show(bez_curve)

# Obtaining the length of the Bezier curve
L = bez_curve.Length

# find the parameter corresponding to the given distance
distance = 0.3 * L
parameter = bez.parameterAtDistance(distance, bez.FirstParameter)

# creating the 2 parts
part_1 = bez.copy()
part_1.segment(part_1.FirstParameter, parameter)
wire1 = Part.Wire(part_1.toShape())

part_2 = bez.copy()
part_2.segment(parameter, part_2.LastParameter)
wire2 = Part.Wire(part_2.toShape())

# display the 2 parts
# Part.show(part_1.toShape())
# Part.show(part_2.toShape())

ser_bz = cadapi.serialize_shape(bez_curve)
print(ser_bz)
wire = Part.Wire(Part.Shape(bez_curve))
ser_wire = cadapi.serialize_shape(wire)
print(ser_wire)
des_wire = cadapi.deserialize_shape(ser_wire)
print(des_wire)

bmwire = geo.wire.BluemiraWire(wire)
ser_bmwire = bluemira.geometry.tools.serialize_shape(bmwire)
des_bmwire = bluemira.geometry.tools.deserialize_shape(ser_bmwire)
print(des_bmwire)

bmwire2 = geo.wire.BluemiraWire(
    [
        geo.wire.BluemiraWire(wire1, label="wire1"),
        geo.wire.BluemiraWire(wire2, label="wire2"),
    ],
    label="full_wire",
)
bmwire2.close()

m = Mesh()
buffer = m(bmwire2)

print(m.get_gmsh_dict(buffer))

bmface = geo.face.BluemiraFace(bmwire2)
ser_bmface = geo.tools.serialize_shape(bmface)
print(ser_bmface)

des_bmface = geo.tools.deserialize_shape(ser_bmface)

print(des_bmface)
