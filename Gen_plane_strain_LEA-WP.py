from __future__ import annotations

# Standard libraries
import os
import math
from pathlib import Path

# Third-party packages
import numpy as np
import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import bmat
import matplotlib.pyplot as plt

# MPI and PETSc
from mpi4py import MPI
from petsc4py import PETSc

# GMSH and PyVista
import gmsh
import pyvista

# DOLFINx core
from dolfinx import mesh,fem, io, plot, common, default_real_type, default_scalar_type, geometry

# DOLFINx FEM
from dolfinx.fem import (Function, Constant, form, dirichletbc,locate_dofs_topological)
from dolfinx.fem.petsc import (assemble_matrix, assemble_vector, apply_lifting, apply_lifting_nest, set_bc, set_bc_nest)

# DOLFINx I/O
from dolfinx.io import (XDMFFile, VTKFile, VTXWriter, gmshio, distribute_entity_data)

# DOLFINx mesh utils
from dolfinx.mesh import (create_mesh, meshtags_from_entities)

# DOLFINx C++ utilities
from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.cpp.la.petsc import (get_local_vectors, scatter_local_vectors)

# UFL
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction, as_tensor, as_vector, dot, dx, ds, inner, sym, tr, grad, nabla_grad, extract_blocks, MixedFunctionSpace)

# Basix
import basix
import basix.ufl as bul
from basix.ufl import element, mixed_element

# dolfinx_mpc
import dolfinx_mpc
from dolfinx_mpc import (MultiPointConstraint, LinearProblem)
from dolfinx_mpc.utils import (create_point_to_point_constraint, determine_closest_block, gather_PETScMatrix, gather_PETScVector, gather_transformation_matrix, facet_normal_approximation,create_normal_approximation)

# scifem utilities
from scifem import (create_real_functionspace,assemble_scalar)

## Importing mesh from GMSH
# Mesh
mesh_path = "neufrustum4.msh"
domain, cell_tags, facet_tags = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

# Check for Measures

pyvista.set_jupyter_backend("static")

topology, cell_types, geometry = plot.vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

p = pyvista.Plotter()
p.add_mesh(grid,show_edges=True)
p.view_xy()
p.show_axes()
p.show()

with XDMFFile(MPI.COMM_WORLD, "neufrustum4.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_tags, domain.geometry)


# Function spaces
degree = 2
gdim = domain.topology.dim
fdim = gdim -1 
# Create the function space
cellname = domain.ufl_cell().cellname()
Ve = basix.ufl.element(basix.ElementFamily.P, cellname, 2, shape=(domain.geometry.dim,), dtype=default_real_type)
Qe = basix.ufl.element(basix.ElementFamily.P, cellname, 1, dtype=default_real_type)

V = fem.functionspace(domain, Ve)
Q = fem.functionspace(domain, Qe)

# Dirichlet BC on u_x = 0 at left edge (tag 2)
left_facets = np.flatnonzero(facet_tags.values == 2)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), domain.topology.dim - 1, left_facets)
bc_x = dirichletbc(PETSc.ScalarType(0.0), left_dofs_x, V.sub(0))
#bcs = [bc_x]
bcs = []

# Slip condition MPC on displacement space V only
mpc = MultiPointConstraint(V)
for tag in [5, 6]:
    n_vec = create_normal_approximation(V, facet_tags, tag)
    mpc.create_slip_constraint(V, (facet_tags, tag), n_vec, bcs=[])
mpc.finalize()

mpc_q = dolfinx_mpc.MultiPointConstraint(Q)
mpc_q.finalize()

# Material properties
E = Constant(domain, 2e11)
nu = Constant(domain, 0.3)
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

f = fem.Constant(domain, default_scalar_type((0, 0)))
(u, ezz) = TrialFunction(V), TrialFunction(Q)
(v, ezz_test) = TestFunction(V), TestFunction(Q)


# Helper functions for strains and stresses
def eps_3d(u, ezz):
    return sym(as_tensor([[u[0].dx(0), u[0].dx(1), 0],
                          [u[1].dx(0), u[1].dx(1), 0],
                          [0, 0, ezz]]))

def sigma_3d(u, ezz):
    return lmbda * tr(eps_3d(u, ezz)) * Identity(3) + 2 * mu * eps_3d(u, ezz)


# Bilinear form blocks
a00 = fem.form(inner(sigma_3d(u, 0), eps_3d(v, 0)) * dx)
a01 = fem.form(inner(sigma_3d(Constant(domain, (0.0, 0.0)), ezz), eps_3d(v, 0)) * dx)
a10 = fem.form(inner(sigma_3d(u, 0), eps_3d(Constant(domain, (0.0, 0.0)), ezz_test)) * dx)
a11 = fem.form(inner(sigma_3d(Constant(domain, (0.0, 0.0)), ezz), eps_3d(Constant(domain, (0.0, 0.0)), ezz_test)) * dx)


a = [[a00, a01],[a10, a11]]


# RHS forms
T = Constant(domain, 1e6)
ds = Measure("ds", domain=domain, subdomain_data=facet_tags)
L0 = dot(T * FacetNormal(domain), v) * ds(8)
L1 = Constant(domain, PETSc.ScalarType(0.0)) * ezz_test * dx
L = [fem.form(L0), fem.form(L1)]

# Constraints applied only on displacement block (V)
constraints = [mpc, mpc_q]

# Assemble system
with common.Timer("~Assemble LHS and RHS"):
    A = dolfinx_mpc.create_matrix_nest(a, constraints)
    dolfinx_mpc.assemble_matrix_nest(A, a, constraints, bcs)
    A.assemble()

    b = dolfinx_mpc.create_vector_nest(L, constraints)
    dolfinx_mpc.assemble_vector_nest(b, L, constraints)

    apply_lifting_nest(b, a, bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc_nest(b, fem.bcs_by_block(fem.extract_function_spaces(L), bcs))


# Preconditioner
P11 = fem.petsc.assemble_matrix(fem.form(ezz * ezz_test * dx))
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])  # type: ignore
P.assemble()

# ---------------------- Solve variational problem -----------------------
ksp = PETSc.KSP().create(domain.comm)  # type: ignore
ksp.setOperators(A, P)
ksp.setMonitor(
    lambda ctx, it, r: PETSc.Sys.Print(  # type: ignore
        f"Iteration: {it:>4d}, |r| = {r:.3e}"
    )
)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-8)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)  # type: ignore

nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("ezz", nested_IS[0][1]))

ksp_u, ksp_ezz = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_ezz.setType("preonly")
ksp_ezz.getPC().setType("jacobi")

ksp.setFromOptions()

Uh = b.copy()
ksp.solve(b, Uh)

for Uh_sub in Uh.getNestSubVecs():
    Uh_sub.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,  # type: ignore
        mode=PETSc.ScatterMode.FORWARD,  # type: ignore
    )  # type: ignore

# ----------------------------- Put NestVec into DOLFINx Function - ---------
uh = fem.Function(mpc.function_space)
uh.x.petsc_vec.setArray(Uh.getNestSubVecs()[0].array)

ezzh = fem.Function(mpc_q.function_space)
ezzh.x.petsc_vec.setArray(Uh.getNestSubVecs()[1].array)

uh.x.scatter_forward()
ezzh.x.scatter_forward()

# Backsubstitute to update slave dofs in solution vector
mpc.backsubstitution(uh)
mpc_q.backsubstitution(ezzh)

from basix.ufl import element

cellname = domain.ufl_cell().cellname()  # Get the string like "triangle"
V0_elem = element("P", cellname, 2, shape=(2,))
V0 = fem.functionspace(domain, V0_elem)

uh_interp = fem.Function(V0, name="displacement")
uh_interp.interpolate(uh)


# Output strain/stress
TensorSpace = fem.functionspace(domain, ("DG", 0, (3, 3)))
strain_func = fem.Function(TensorSpace, name="strain")
stress_func = fem.Function(TensorSpace, name="stress")

# Define expressions
strain_expr = eps_3d(uh, ezzh)
stress_expr = sigma_3d(uh, ezzh)

# Interpolate
strain_func.interpolate(fem.Expression(strain_expr, TensorSpace.element.interpolation_points()))
stress_func.interpolate(fem.Expression(stress_expr, TensorSpace.element.interpolation_points()))

# Write results
if MPI.COMM_WORLD.rank == 0 and not os.path.exists("results"):
    os.makedirs("results")
MPI.COMM_WORLD.barrier()

with VTKFile(domain.comm, "jason3", "w") as vtk:
    vtk.write_function(uh_interp)
    vtk.write_function(ezzh)
    vtk.write_function(strain_func)
    vtk.write_function(stress_func)

