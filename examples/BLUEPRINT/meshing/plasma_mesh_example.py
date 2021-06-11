#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""
# recycling code from geo/simple_shape_creation.py
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
# Reactor parameters
data["R0"] = 8.938
# (m)  Major radius
data["A"] = 3.1
# (-)   Aspect ratio

# Plasma parameters
data["kappaXU"] = 1.68
# (-)   Upper plasma elongation
data["kappaXL"] = 1.88
# (-)   Lower plasma elongation
data["deltaXU"] = 0.50
# (-)   Upper plasma triangularity
data["deltaXL"] = 0.50
# (-)   Lower plasma triangularity
data["psiPU"] = 0.0 / 180 * math.pi
# (-)   Plasma shape angle outboard upper
data["psiMU"] = math.pi
# (-)   Plasma shape angle inboard upper
data["psiPL"] = 30.0 / 180 * math.pi
# (-)   Plasma shape angle outboard lower
data["psiML"] = 120.0 / 180 * math.pi
# (-)   Plasma shape angle inboard lower

# a function for the creation of the plasma shape
def create_plasma(R0, A, kappaXU, kappaXL, deltaXU, deltaXL, psiPU, psiMU, psiPL, psiML):

    a = R0 / A

    # reference points
    XU = Base.Vector(R0 - a * deltaXU, a * kappaXU, 0)
    XL = Base.Vector(R0 - a * deltaXL, -a * kappaXL, 0)
    Po = Base.Vector(R0 + a, 0, 0)
    Pi = Base.Vector(R0 - a, 0, 0)

    # creation of connection curves. Auxiliary functions in mirapy.algebra
    # are used to easily define cubic beziers and arc of ellipse with
    # points and tangent constraints
    C1 = algebra.bezier2pointstangent(XL, psiPL, Po, math.pi / 2)
    C2 = algebra.arcofellipse2pointstangent(Po, math.pi / 2, XU, psiPU)
    C3 = algebra.arcofellipse2pointstangent(XU, psiMU, Pi, math.pi / 2)
    C4 = algebra.bezier2pointstangent(Pi, 3 * math.pi / 2.0, XL, psiML + math.pi)

    # creation of the plasma shape
    wire = Part.Wire(Part.Shape([C1, C2, C3, C4]).Edges)
    shape = geo.Shape(wire, "Separatrix")
    plasma = geo.Shape2D(shape, "plasma")

    # definition of geo constraints for later visualization
    tuples = {
        "XU": (XU, None),
        "XL": (XL, None),
        "Po": (Po, None),
        "Pi": (Pi, None),
        "XUp": (XU, numpy.rad2deg(psiPU)),
        "XUm": (XU, numpy.rad2deg(psiMU)),
        "XLp": (XL, numpy.rad2deg(psiPL)),
        "XLm": (XL, numpy.rad2deg(psiML)),
        "Pop": (Po, 90.0),
        "Pom": (Po, 270.0),
        "Pip": (Pi, 90.0),
        "Pim": (Pi, 270.0),
    }

    geoConstrDict = {}

    for k, v in tuples.items():
        geoConstrDict[k] = geo.geoConstraint(v[0], angle=v[1], lscale=1, label=k)

    return plasma, geoConstrDict


plasma, geoConstrDict = create_plasma(**data)

# code for meshing
import mirapy.meshing as meshing
import mirapy.msh2xdmf as msh2xdmf
import os
import dolfin

# change the mesh size
plasma.boundary[0].lcar = 0.25

# define physical groups
plasma.physicalGroups = {1: "LCFS", 2: "plasma"}

# inizialize the mesh
# just as an example, the mesh_dim is set to 1, i.e. only lines and
# points will be meshed. Resulting mesh is saved into
# Mesh1D.geo_unrolled and Mesh1D.msh. Import them in gmsh for an easy
# visualization.
m = meshing.Mesh(meshfile=["Mesh1D.geo_unrolled", "Mesh1D.msh"], mesh_dim=1)

# switch to a 2D mesh
m.mesh_dim = 2

# test embedding two point into the plasma surface with different meshsize
m.embed = [(Base.Vector(9.0, 0.0, 0.0), 0.01), (Base.Vector(8.0, -4, 0), 0.9)]

# exporting test with different format
m.meshfile = ["Mesh2D.geo_unrolled", "Mesh2D.msh", "Mesh2D.step", "Mesh2D.brep"]

# mesh
m(plasma)

# Run the conversion to fenics mesh format
meshdir = "."
meshfile = "Mesh2D.msh"
msh2xdmf.msh2xdmf(meshfile, dim=m.mesh_dim, directory=meshdir)

# Run the import
prefix, _ = os.path.splitext(meshfile)

# extract the mesh, boundaries, subdomains, labels
mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh_from_xdmf(
    prefix=os.path.splitext(meshfile)[0],
    dim=m.mesh_dim,
    directory=meshdir,
    subdomains=True,
)


# even if the extracted mesh is 2D, mesh coordinates are stored as 3D vector,
# so the standard plot visualization of fenics will not work. To test, try:
# >>> dolfin.plot(mesh)
# Indeed, fenics documentation advise to save the mesh in a format that can
# be opened in Paraview.

# To overcome this problem,
# following commands are just a simple trick to remove one component in the
# coordinate vector allowing the plot here.
def create_dolfin_mesh(points, cells):
    # https://bitbucket.org/fenics-project/dolfin/issues/845/initialize-mesh-from-vertices
    editor = dolfin.MeshEditor()
    mesh = dolfin.Mesh()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(points.shape[0])
    editor.init_cells(cells.shape[0])
    for k, point in enumerate(points):
        editor.add_vertex(k, point)
    for k, cell in enumerate(cells):
        editor.add_cell(k, cell)
    editor.close()
    return mesh


mesh2d = create_dolfin_mesh(mesh.coordinates()[:, 0:2], mesh.cells())
boundaries_mvc = dolfin.MeshValueCollection(
    "size_t", mesh2d, dim=mesh2d.topology().dim() - 1
)
boundaries_mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh2d, boundaries_mvc)
for cell_index in range(mesh2d.num_cells()):
    boundaries_mf.set_value(cell_index, boundaries.array()[cell_index])

submains_mvc = dolfin.MeshValueCollection("size_t", mesh2d, dim=mesh2d.topology().dim())
submains_mf = dolfin.cpp.mesh.MeshFunctionSizet(mesh2d, submains_mvc)
for cell_index in range(mesh2d.num_cells()):
    submains_mf.set_value(cell_index, subdomains.array()[cell_index])

dolfin.plot(mesh2d)
dolfin.plot(submains_mf)
