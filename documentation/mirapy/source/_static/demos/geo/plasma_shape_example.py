#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""
import freecad
import Part
from FreeCAD import Base

import mirapy.geo as geo
import mirapy.plotting as plotting
import mirapy.algebra as algebra

import numpy
import math

# Initialize the input dictionary
data = {}
#Reactor parameters
data['R0']          = 8.938;            # (m)  Major radius
data['A']           = 3.1;              # (-)   Aspect ratio

#Plasma parameters
data['kappaXU']     = 1.68;             # (-)   Upper plasma elongation at the X point
data['kappaXL']     = 1.88;             # (-)   Lower plasma elongation at the X point
data['deltaXU']     = 0.50;             # (-)   Upper plasma triangularity at the X point
data['deltaXL']     = 0.50;             # (-)   Lower plasma triangularity at the X point
data['psiPU']       = 0.0/180*math.pi;       # (-)   Plasma shape angle outboard upper
data['psiMU']       = math.pi;       # (-)   Plasma shape angle inboard upper
data['psiPL']       = 30.0/180*math.pi;      # (-)   Plasma shape angle outboard lower
data['psiML']       = 120.0/180*math.pi;      # (-)   Plasma shape angle inboard lower

# a function for the creation of the plasma shape
def create_plasma(R0, A, kappaXU, kappaXL, deltaXU, deltaXL,
                  psiPU, psiMU, psiPL, psiML):

    a = R0/A

    # reference points
    XU = Base.Vector(R0-a*deltaXU,a*kappaXU,0)
    XL = Base.Vector(R0-a*deltaXL,-a*kappaXL,0)
    Po = Base.Vector(R0+a,0,0)
    Pi = Base.Vector(R0-a,0,0)

    # creation of connection curves. Auxiliary functions in mirapy.algebra
    # are used to easily define cubic beziers and arc of ellipse with
    # points and tangent constraints
    C1 = algebra.bezier2pointstangent(XL, psiPL, Po, math.pi/2)
    C2 = algebra.arcofellipse2pointstangent(Po,math.pi/2,XU,psiPU) 
    C3 = algebra.arcofellipse2pointstangent(XU,psiMU,Pi,math.pi/2)
    C4 = algebra.bezier2pointstangent(Pi,3*math.pi/2.,XL,psiML + math.pi)

    # creation of the plasma shape
    wire = Part.Wire(Part.Shape([C1,C2,C3,C4]).Edges)
    shape = geo.Shape(wire, "Separatrix")
    plasma = geo.Shape2D(shape, "plasma")

    # definition of geo constraints for later visualization
    tuples = {"XU": (XU, None),
              "XL": (XL, None),
              "Po": (Po, None),
              "Pi": (Pi, None),
              "XUp": (XU, numpy.rad2deg(psiPU)),
              "XUm": (XU, numpy.rad2deg(psiMU)),
              "XLp": (XL, numpy.rad2deg(psiPL)),
              "XLm": (XL, numpy.rad2deg(psiML)),
              "Pop": (Po, 90.),
              "Pom": (Po, 270.),
              "Pip": (Pi, 90.),
              "Pim": (Pi, 270.),              
              }

    geoConstrDict = {}
    
    for k,v in tuples.items():
        geoConstrDict[k] = geo.geoConstraint(v[0], angle=v[1],
                                                 lscale=1,label=k)
    
    return plasma, geoConstrDict


plasma, geoConstrDict = create_plasma(**data)
# plot of plasma shape with constraints
# default plot on xy plane
axis, _ = plotting.plot2D(plasma)
for k, v in geoConstrDict.items():
    axis, _ = plotting.plot2D(v, axis=axis, poptions={})
axis.set_title("Example of plasma shape creation \n"
               "plot on default xy-plane")
axis.set_xlabel('x')
axis.set_ylabel('y')

# test rotating shape and constraints
placement = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 90.)
plasma.Placement = placement
for k, v in geoConstrDict.items():
    v.Placement = placement

axis, _ = plotting.plot2D(plasma, plane='xz')
for k, v in geoConstrDict.items():
    axis, _ = plotting.plot2D(v, axis=axis, plane='xz', poptions={})
axis.set_title("Rotated plasma shape - xz-plane")
axis.set_xlabel('x')
axis.set_ylabel('z')
