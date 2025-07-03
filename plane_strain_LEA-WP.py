# Plane Strain Linear Elastic Analysis 

import gmsh
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import ufl
from mpi4py import MPI
from dolfinx.io import XDMFFile, VTKFile
from petsc4py import PETSc
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx import mesh, fem, io, common, default_scalar_type, geometry
import dolfinx.fem.petsc
from dolfinx.io import VTXWriter, distribute_entity_data, gmshio
from dolfinx.mesh import create_mesh, meshtags_from_entities
from dolfinx.plot import vtk_mesh
from dolfinx.io import XDMFFile
from ufl import (FacetNormal, FiniteElement, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, VectorElement, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
from dolfinx_mpc.utils import (create_point_to_point_constraint, determine_closest_block, rigid_motions_nullspace, facet_normal_approximation)
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import create_normal_approximation
from dolfinx_mpc import MultiPointConstraint



# Mesh
mesh_path = "neufrustum4.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

# Check for Measures

import pyvista
from dolfinx import plot

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

with XDMFFile(MPI.COMM_WORLD, "neufrustum4.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)
    
###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###

# Checks the Measure for the Complete Mesh Domain and Not Just Each Tags
import ufl

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
areaj = fem.assemble_scalar(fem.form(1.0 * dx(2)))
print(f"Computed area_areaj = {areaj:.7f}")

areav = fem.assemble_scalar(fem.form(1.0 * dx(1)))
print(f"Computed area_areav = {areav:.7f}")

Ten = 100000.0

sigma_z = (0.5*Ten)/(areaj + areav)
print(f"The longitudinal stress (sigma_Z)=", sigma_z)


length_1 = fem.assemble_scalar(fem.form(1.0 * dS(2)))
print(f"Computed length_l1 = {length_1:.7f}")

length_12 = fem.assemble_scalar(fem.form(1.0 * ds(1)))
print(f"Computed length_l2 = {length_12:.7f}")



####--------------------------------------------------------------------------------------------------------------------------------------------------------------
#Defining Material Properties through subdomains

Q = fem.functionspace(domain, ("DG", 0))
E = fem.Function(Q)
nu = fem.Function(Q)

my_surface_cells = cell_markers.find(1)
E.x.array[my_surface_cells] = np.full_like(my_surface_cells, 2e11, dtype=default_scalar_type)
nu.x.array[my_surface_cells] = np.full_like(my_surface_cells, 0.3, dtype=default_scalar_type)

EFGH_cells = cell_markers.find(2)
E.x.array[EFGH_cells] = np.full_like(EFGH_cells, 2e11, dtype=default_scalar_type)
nu.x.array[EFGH_cells] = np.full_like(EFGH_cells, 0.3, dtype=default_scalar_type)

IJKL_cells = cell_markers.find(3)
E.x.array[IJKL_cells] = np.full_like(IJKL_cells, 2e11, dtype=default_scalar_type)
nu.x.array[IJKL_cells] = np.full_like(IJKL_cells, 0.3, dtype=default_scalar_type)

MNOP_cells = cell_markers.find(4)
E.x.array[MNOP_cells] = np.full_like(MNOP_cells, 2e11, dtype=default_scalar_type)
nu.x.array[MNOP_cells] = np.full_like(MNOP_cells, 0.3, dtype=default_scalar_type)

QRST_cells = cell_markers.find(5)
E.x.array[QRST_cells] = np.full_like(QRST_cells, 2e11, dtype=default_scalar_type)
nu.x.array[QRST_cells] = np.full_like(QRST_cells, 0.3, dtype=default_scalar_type)


####------------------------------------------------------------------------------------------------------------------------------------------------------------
gdim = 2

def strain(u, repr ="vectorial"):
    eps_t = sym(grad(u))
    if repr =="vectorial":
        return as_vector([eps_t[0,0], eps_t[1,1], 2*eps_t[0,1]])
    elif repr =="tensorial":
        return eps_t

E = fem.Constant(domain, 200000000000.0)
nu = fem.Constant(domain,0.3)

lmbda = E*nu/(1+nu)/(1-2*nu)
mu = E/2/(1+nu)
#z = E/(1-nu**2)
zz = E/((1+nu)*(1-2*nu))

C = as_matrix([[zz*(1-nu), zz*nu, 0.0],[zz*nu, zz*(1-nu), 0.0],[0.0, 0.0, 0.5*zz*(1-2*nu)]])

def stress(u, repr ="vectorial"):
    sigv = dot(C, strain(u))
    if repr =="vectorial":
        return sigv
    elif repr =="tensorial":
        return as_matrix([[sigv[0], sigv[2]], [sigv[2], sigv[1]]])


# Define Function Space
degree = 2
V = fem.functionspace(domain, ("P",degree, (gdim,)))



#Define Variational Problem
du = TrialFunction(V)
u_ = TestFunction(V)
#u = fem.Function(V, name = "Displacement")
a_form = inner(stress(du),strain(u_))*dx


T = fem.Constant(domain, 1000000.0)


#Self-weight on the surface
n = FacetNormal(domain)

L_form = dot(T*n,u_) * ds(8)


# Get the facet markers for boundary edges (ps1 and ps2)

mpc = MultiPointConstraint(V)

for tag in [5, 6]:  # ps1 and ps2
    normal_vec = create_normal_approximation(V, facet_markers, tag)
    mpc.create_slip_constraint(V, (facet_markers, tag), normal_vec)

mpc.finalize()

uh = fem.Function(mpc.function_space, name="Displacement")

# --- Solve ---
problem = LinearProblem(a_form, L_form, mpc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh_mpc = problem.solve()
uh.name = "u"




# Compute stress and strain components
# Use DG(0) space for storing element-wise quantities
DG0 = fem.functionspace(domain, ("DG", 0, (3,)))  # for vectorial stress/strain

# STRAIN
strain_expr = strain(uh_mpc, "vectorial")
strain_fn = fem.Function(DG0, name="Strain")
for i in range(3):
    scalar_space = fem.functionspace(domain, ("DG", 0))
    strain_i = fem.Function(scalar_space)
    strain_i.interpolate(fem.Expression(strain_expr[i], scalar_space.element.interpolation_points()))
    strain_fn.x.array[i::3] = strain_i.x.array

# STRESS
stress_expr = stress(uh_mpc, "vectorial")
stress_fn = fem.Function(DG0, name="Stress")
for i in range(3):
    scalar_space = fem.functionspace(domain, ("DG", 0))
    stress_i = fem.Function(scalar_space)
    stress_i.interpolate(fem.Expression(stress_expr[i], scalar_space.element.interpolation_points()))
    stress_fn.x.array[i::3] = stress_i.x.array


with VTKFile(domain.comm, "neufrus41_vs.pvd", "w") as vtk:
    vtk.write_function(strain_fn)
    vtk.write_function(uh_mpc)
    vtk.write_function(stress_fn)
