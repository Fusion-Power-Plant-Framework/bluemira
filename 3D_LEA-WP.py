import gmsh
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx import mesh, fem, io, common, default_scalar_type, geometry, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter, distribute_entity_data, gmshio
from dolfinx.mesh import create_mesh, meshtags_from_entities
from dolfinx.plot import vtk_mesh
from dolfinx.io import XDMFFile
from ufl import (FacetNormal, sqrt, FiniteElement, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, VectorElement, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, nabla_div, rhs, sym)
import pyvista

# Mesh
mesh_path = "3d_solid.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

pyvista.set_jupyter_backend("static")

topology, cell_types, geometry = plot.vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

p = pyvista.Plotter()
p.add_mesh(grid,show_edges=True)
p.view_xy()
p.show_axes()
p.show()

from dolfinx.plot import vtk_mesh
from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "3d_solid.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)



# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))


def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

ds = Measure("ds", domain=domain)

def epsilon(u):
    return sym(grad(u))


def sigma(u):
    return lambda_ * nabla_div(u) * Identity(len(u)) + 2 * mu * epsilon(u)


u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


with io.XDMFFile(domain.comm, "trry.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)


s = sigma(uh) - 1. / 3 * tr(sigma(uh)) * Identity(len(uh))
von_Mises = sqrt(3. / 2 * inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

with io.XDMFFile(domain.comm, "trry.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)
    xdmf.write_function(stresses)
