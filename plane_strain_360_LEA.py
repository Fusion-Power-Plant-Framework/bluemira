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
from dolfinx.mesh import create_mesh, meshtags_from_entities, locate_entities
from dolfinx.plot import vtk_mesh
from ufl import (FacetNormal, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)

# Mesh
mesh_path = "poly16b.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

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
zz = E/((1+nu)*(1-2*nu))

C = as_matrix([[zz*(1-nu), zz*nu, 0.0],[zz*nu, zz*(1-nu), 0.0],[0.0, 0.0, 0.5*zz*(1-2*nu)]])

def stress(u, repr ="vectorial"):
    sigv = dot(C, strain(u))
    if repr =="vectorial":
        return sigv
    elif repr =="tensorial":
        return as_matrix([[sigv[0], sigv[2]], [sigv[2], sigv[1]]])

# Function Space
degree = 2
V = fem.functionspace(domain, ("P",degree, (gdim,)))


#Variational Problem
du = TrialFunction(V)
u_ = TestFunction(V)
a_form = inner(stress(du),strain(u_))*dx

T = fem.Constant(domain, 1000000.0)

n = FacetNormal(domain)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_markers)

L_form = dot(T*n,u_) * (ds(38) + ds(39) + ds(40) + ds(41) + ds(42) + ds(43)+ ds(44) + ds(45) + ds(46) + ds(47) + ds(48) + ds(49) + ds(50) + ds(51) + ds(52) + ds(53))

tdim = domain.topology.dim
domain.topology.create_connectivity(1, 0)     
domain.topology.create_connectivity(0, tdim)    

# vertices from a facet tag
e2v = domain.topology.connectivity(1, 0)
def verts_from_facet_tag(tag: int):
    facets = facet_markers.find(tag)            
    vs = []
    for e in facets:
        vs.extend(e2v.links(e))                
    if len(vs) == 0:
        return np.array([], dtype=np.int32)
    return np.unique(np.array(vs, dtype=np.int32))

# get vertex sets for each tagged boundary
verts_bottom = verts_from_facet_tag(34)
verts_top    = verts_from_facet_tag(33)
verts_left   = verts_from_facet_tag(35)
verts_right  = verts_from_facet_tag(36)

dofs_ux_bottom = fem.locate_dofs_topological(V.sub(1), 0, verts_bottom)
dofs_ux_top    = fem.locate_dofs_topological(V.sub(1), 0, verts_top)
dofs_uy_left   = fem.locate_dofs_topological(V.sub(0), 0, verts_left)
dofs_uy_right  = fem.locate_dofs_topological(V.sub(0), 0, verts_right)

zero = PETSc.ScalarType(0.0)
bc_x_bottom = fem.dirichletbc(zero, dofs_ux_bottom, V.sub(1))  # ux=0 on bottom vertices
bc_x_top    = fem.dirichletbc(zero, dofs_ux_top,    V.sub(1))  # ux=0 on top vertices
bc_y_left   = fem.dirichletbc(zero, dofs_uy_left,   V.sub(0))  # uy=0 on left vertices
bc_y_right  = fem.dirichletbc(zero, dofs_uy_right,  V.sub(0))  # uy=0 on right vertices

bcs = [bc_x_bottom, bc_x_top, bc_y_left, bc_y_right]


uh = fem.Function(V, name="Displacement")
problem = fem.petsc.LinearProblem(a_form, L_form, bcs=bcs, u=uh)
uh = problem.solve() 

V0 = fem.functionspace(domain, ("DG", 0, (3,)))
sig_exp = fem.Expression(stress(uh), V0.element.interpolation_points())
sig = fem.Function(V0, name="Stress")
sig.interpolate(sig_exp)

str_exp = fem.Expression(strain(uh), V0.element.interpolation_points())
strn = fem.Function(V0, name="Strain")
strn.interpolate(str_exp)

vtk = io.VTKFile(domain.comm, "poly16b.pvd", "u")
vtk.write_function(uh, 0)
vtk.write_function(sig, 0)
vtk.write_function(strn, 0)
vtk.close()