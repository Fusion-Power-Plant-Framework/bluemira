#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""
# import from freecad
import freecad
import Part
from FreeCAD import Base

import mirapy.geo as geo
import mirapy.plotting as plotting

import numpy

def print_freecad_properties(obj):
    print("------------------------------------------------")
    print("|- {} properties:".format(obj))
    print("|- Length: {}".format(obj.Length))
    # Area is 0 if no faces are created
    print("|- Area: {}".format(obj.Area))
    # volume is 0 if not solids are created
    print("|- Volume: {}".format(obj.Volume))
    print("|- BoundBox: {}".format(obj.BoundBox))
    print("|- CenterOfMass: {}".format(obj.CenterOfMass))
    print("|- MatrixOfInertia: {}".format(obj.MatrixOfInertia))
    print("------------------------------------------------")

# create a point list as vertexes of a polygon
pntslist = [Base.Vector(), Base.Vector(1., 0., 0.),
            Base.Vector(1., 1., 0.), Base.Vector(0., 1., 0.),
            Base.Vector()]

# define a wire
wire = Part.makePolygon(pntslist)

w1 = Part.makePolygon(pntslist[0:4])
print("wire w1 is closed? {}".format(w1.isClosed()))
plotting.plotWire2D(w1)
print("Forcing wire w1 to be closed...")
w1 = geo.Utils.close_wire(w1)
print("Check: wire w1 is closed? {}".format(w1.isClosed()))
plotting.plotWire2D(w1)

# define a shape from wire with label "polygon"
shape = geo.Shape(wire, label="polygon")

# plot shape
axis, _ = plotting.plot2D(shape)
axis.set_title("Plot shape2D")

# create a shape2D object
# note: condition to create a Shape2D object is that the boundary is closed.
print("Shape polygon is close? {}".format(shape.isClosed()))  # True
shape2d = geo.Shape2D(boundary=shape, label="face")

# plot shape2D
axis, _ = plotting.plot2D(shape2d)
axis.set_title("Plot shape2D")

# it is possible to create an hole in a Shape2D
# create a new Shape that represents the boundary of the hole
pntslisthole = [Base.Vector(0.25, 0.25, 0.), Base.Vector(0.75, 0.25, 0.),
                Base.Vector(0.75, 0.75, 0.), Base.Vector(0.25, 0.75, 0.),
                Base.Vector(0.25, 0.25, 0.), ]

whole = Part.makePolygon(pntslisthole)
shole = geo.Shape(whole, "hole_boundary")
s2hole = geo.Shape2D(shape, holes=[shole])

axis, _ = plotting.plot2D(s2hole)
axis.set_title("Plot shape2D with hole")


####### modify the geometry #######
# rotation/translation of geometry is possible by means of
# Base.Placement objects in FreeCAD.
placement = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 90.)
shape2d.Placement = placement

# default 2D plotting of shapes is made on the xy-plane
axis, _ = plotting.plot2D(shape2d)
axis.set_title("Plotting an xz shape on xy-plane")
# however, plotting plane can be changed specifying the appropriate
# placement operation (the placement operation affects only the
# shape visualization, not the shape's geometry).
axis, _ = plotting.plot2D(shape2d, plane=placement.inverse())
axis.set_title("Plot on an arbitrary reference plane")

# to move a shape 
translation =  Base.Placement()
translation.move(Base.Vector(2,0,0))

shape2d.Placement = translation
axis, _ = plotting.plot2D(shape2d)
axis.set_title("Translated shape")

# printing some wire and face properties
print_freecad_properties(shape2d.getSingleWires()[0])
print_freecad_properties(shape2d.face)
