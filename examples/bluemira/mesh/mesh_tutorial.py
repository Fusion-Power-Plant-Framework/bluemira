import freecad
import FreeCAD
import Part

# Defining my parameters
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
bezCurve = Part.Edge(bez)
Part.show(bezCurve)

# Obtaining the length of the Bezier curve
L = bezCurve.Length

# find the parameter corresponding to the given distance
distance = 0.3 * L
parameter = bez.parameterAtDistance(distance, bez.FirstParameter)

# creating the 2 parts
part_1 = bez.copy()
part_1.segment(part_1.FirstParameter, parameter)
part_2 = bez.copy()
part_2.segment(parameter, part_2.LastParameter)

# display the 2 parts
# Part.show(part_1.toShape())
# Part.show(part_2.toShape())

import bluemira.geometry._freecadapi as fcapi
ser = fcapi.serialize_shape(bezCurve)
print(ser)
