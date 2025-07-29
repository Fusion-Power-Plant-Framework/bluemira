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
from ufl import (FacetNormal, sqrt, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, nabla_div, rhs, sym)
import pyvista

# Mesh
mesh_path = "3d-barr.msh"
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

with XDMFFile(MPI.COMM_WORLD, "solid_bar.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)




E = fem.Constant(domain, 200000000000.0)
nu = fem.Constant(domain,0.3)

lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)

rho = 8000.0
g = 9.81


# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))


def clamped_boundary(x):
    return np.isclose(x[2], 0)


u_zero = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
bc = fem.dirichletbc(u_zero, fem.locate_dofs_geometrical(V, clamped_boundary), V)
bcs = [bc]


dx = Measure("dx", domain=domain, subdomain_data=cell_markers)
dS = Measure("dS", domain=domain, subdomain_data=facet_markers)
ds = Measure("ds", domain=domain, subdomain_data=facet_markers)


T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

def epsilon(u):
    return sym(grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lmbda * nabla_div(u) * Identity(len(u)) + 2 * mu * epsilon(u)


u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(domain, default_scalar_type((10000.0, 0, 0)))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx


problem = LinearProblem(a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
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


