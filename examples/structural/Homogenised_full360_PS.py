# Plane Strain Linear Elastic Analysis for homogenous full 360 deg model

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
from ufl import (FacetNormal, as_vector, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import create_normal_approximation

if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()
gmsh.model.add("WP_full2D_360_pln_strn")

# Geometry parameters (all in SI units)

gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0

Rout  = 2.15   # outer vault radius [m]
Rin   = 1.11   # inner vault radius [m]
deg   = np.pi / 180.0
theta = 22.5 * deg         # sector angle [rad]
alpha = theta / 2.0        # half-angle [rad]

# Sector centred about +Y axis
a1 = np.pi/2 - alpha       # ~78.75° (left radial edge)
a2 = np.pi/2 + alpha       # ~101.25° (right radial edge)

# Mesh sizes [m]
lc_outer = 0.40     # typical element size in vault
lc_rect  = 0.030   # typical element size in WP + Insulator

# WP rectangle (in radial direction) [m]
Rin_hole  = 1.85    # inner radius of WP block
Rout_hole = 2.05    # outer radius of WP block
half_w    = 0.30    # half-width of WP block in x-direction (total width 0.60 m)

# Insulator rectangle (inside WP) [m]
Rin_ins  = 1.87          # inner radius of Insulator region
Rout_ins = 2.03          # outer radius of Insulator region
half_w2  = 0.560 / 2.0   # half-width of Insulator region (total width 0.560 m = 0.28 m half-width)


# Sector geometry (vault)

p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_outer, 1)
p1 = gmsh.model.geo.addPoint(Rout*np.cos(a1), Rout*np.sin(a1), 0.0, lc_outer, 2)
p2 = gmsh.model.geo.addPoint(Rout*np.cos(a2), Rout*np.sin(a2), 0.0, lc_outer, 3)
p3 = gmsh.model.geo.addPoint(Rin*np.cos(a1),  Rin*np.sin(a1),  0.0, lc_outer, 4)
p4 = gmsh.model.geo.addPoint(Rin*np.cos(a2),  Rin*np.sin(a2),  0.0, lc_outer, 5)

Louter = gmsh.model.geo.addLine(2, 3, 1)   # outer radius curve ("top")
Linner = gmsh.model.geo.addLine(5, 4, 2)   # inner radius curve ("bottom")
Lleft  = gmsh.model.geo.addLine(4, 2, 3)   # left radial edge
Lright = gmsh.model.geo.addLine(3, 5, 4)   # right radial edge

gmsh.model.geo.addCurveLoop([1, 3, 2, 4], 1)  # vault sector loop


# WP rectangle (centre at x=0, between Rin_hole and Rout_hole)

p6 = gmsh.model.geo.addPoint(-half_w, Rin_hole,  0.0, lc_rect)
p7 = gmsh.model.geo.addPoint( half_w, Rin_hole,  0.0, lc_rect)
p8 = gmsh.model.geo.addPoint( half_w, Rout_hole, 0.0, lc_rect)
p9 = gmsh.model.geo.addPoint(-half_w, Rout_hole, 0.0, lc_rect)

L5 = gmsh.model.geo.addLine(p6, p7)
L6 = gmsh.model.geo.addLine(p7, p8)
L7 = gmsh.model.geo.addLine(p8, p9)
L8 = gmsh.model.geo.addLine(p9, p6)

gmsh.model.geo.addCurveLoop([L5, L6, L7, L8], 2)  # WP loop


# Insulator rectangle (inside WP)

p10 = gmsh.model.geo.addPoint(-half_w2, Rin_ins,  0.0, lc_rect)
p11 = gmsh.model.geo.addPoint( half_w2, Rin_ins,  0.0, lc_rect)
p12 = gmsh.model.geo.addPoint( half_w2, Rout_ins, 0.0, lc_rect)
p13 = gmsh.model.geo.addPoint(-half_w2, Rout_ins, 0.0, lc_rect)

L9  = gmsh.model.geo.addLine(p10, p11)
L10 = gmsh.model.geo.addLine(p11, p12)
L11 = gmsh.model.geo.addLine(p12, p13)
L12 = gmsh.model.geo.addLine(p13, p10)

gmsh.model.geo.addCurveLoop([L9, L10, L11, L12], 3)  # Insulator loop


# Main surfaces (NO jackets/cables)
#   1: vault (sector minus WP rectangle)
#   2: WP (WP minus Insulator rectangle)
#   3: Insulator (solid)

gmsh.model.geo.addPlaneSurface([1, 2], 1)   # vault
gmsh.model.geo.addPlaneSurface([2, 3], 2)   # WP
gmsh.model.geo.addPlaneSurface([3], 3)      # Insulator (no holes)

gmsh.model.geo.synchronize()


# Rotate whole geometry about origin by 11.25 deg

rot = -11.25 * deg  # radians

ents = gmsh.model.getEntities()  # (dim, tag) for points/curves/surfaces now exists
gmsh.model.geo.rotate(
    ents,
    0.0, 0.0, 0.0,   # rotation center
    0.0, 0.0, 1.0,   # axis (z)
    rot
)

gmsh.model.geo.synchronize()


# Replicate to 360° and create individual 2D physical groups
# (Vault/WP/Insulator per sector)
# No 1D boundary physical groups are created.
# IMPORTANT: Do NOT call removeAllDuplicates() after this.

n_sectors = 16
dphi = theta

# template surfaces from the original sector
vault_s0 = 1
wp_s0    = 2
ins_s0   = 3

template_surfs = [(2, vault_s0), (2, wp_s0), (2, ins_s0)]

# surface tags sector-by-sector (k=0 is original)
vault_by_sector = {0: vault_s0}
wp_by_sector    = {0: wp_s0}
ins_by_sector   = {0: ins_s0}

# replication 15 more times (total 16 sectors)

for k in range(1, n_sectors):
    new = gmsh.model.geo.copy(template_surfs)  # same ordering as template_surfs
    gmsh.model.geo.rotate(new, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0,  k * dphi)

    vault_by_sector[k] = new[0][1]
    wp_by_sector[k]    = new[1][1]
    ins_by_sector[k]   = new[2][1]

gmsh.model.geo.synchronize()

# INDIVIDUAL 2D physical groups
# Tag scheme:
#   vault_k : 100 + k
#   WP_k    : 200 + k
#   Ins_k   : 300 + k

for k in range(n_sectors):
    tv = 100 + k
    tw = 200 + k
    ti = 300 + k

    gmsh.model.addPhysicalGroup(2, [vault_by_sector[k]], tag=tv)
    gmsh.model.setPhysicalName(2, tv, f"vault_{k+1}")

    gmsh.model.addPhysicalGroup(2, [wp_by_sector[k]], tag=tw)
    gmsh.model.setPhysicalName(2, tw, f"WP_{k+1}")

    gmsh.model.addPhysicalGroup(2, [ins_by_sector[k]], tag=ti)
    gmsh.model.setPhysicalName(2, ti, f"Insulator_{k+1}")


# Mesh settings (all in m)

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.03)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.40)


# Mesh and export

gmsh.model.mesh.generate(gdim)
gmsh.write("WP_full2D_360_pln_strn.msh")

domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)

gmsh.fltk.run()
gmsh.finalize()


# Mesh
mesh_path = "WP_full2D_360_pln_strn.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)


dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)


Q = fem.functionspace(domain, ("DG", 0))
E  = fem.Function(Q, name="E")
nu = fem.Function(Q, name="nu")

# --- tag lists from your Gmsh physical tag scheme ---
vault_tags = list(range(100, 116))   # 16 vault sectors
wp_tags    = list(range(200, 216))   # 16 WP sectors
ins_tags   = list(range(300, 316))   # 16 Insulator sectors

# vault: concrete-ish, WP+jacket: steel-ish, cable: copper-ish, insulator: polymer-ish
materials = {
    "vault":     (2e11, 0.30),
    "WP":        (2e11, 0.30),
    "Insulator": (2e11, 0.30),
}

def set_material_for_tags(tags, E_val, nu_val):
    for tag in tags:
        cells = cell_markers.find(tag)
        if cells.size == 0:
            continue
        E.x.array[cells]  = np.full_like(cells, E_val,  dtype=default_scalar_type)
        nu.x.array[cells] = np.full_like(cells, nu_val, dtype=default_scalar_type)

set_material_for_tags(vault_tags,  *materials["vault"])
set_material_for_tags(wp_tags,     *materials["WP"])
set_material_for_tags(ins_tags,    *materials["Insulator"])


# Constitutive Relation
gdim = 2

def strain(u, repr ="vectorial"):
    eps_t = sym(grad(u))
    if repr =="vectorial":
        return as_vector([eps_t[0,0], eps_t[1,1], 2*eps_t[0,1]])
    elif repr =="tensorial":
        return eps_t


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
a_form = inner(stress(du),strain(u_))*dx

tdim = domain.topology.dim
domain.topology.create_connectivity(1, 0)       # facets -> vertices
domain.topology.create_connectivity(0, tdim)    # vertices -> cells  (required by locate_dofs_topological)

tol = 1e-6

# Dirichlet bc at points
def point1(x):
    return np.isclose(x[0], 0.0, atol=tol) & np.isclose(x[1], 1.11, atol=tol)

def point2(x):
    return np.isclose(x[0], -1.11, atol=tol) & np.isclose(x[1], 0.0, atol=tol)

def point3(x):
    return np.isclose(x[0], 0.0, atol=tol) & np.isclose(x[1], -1.11, atol=tol)

def point4(x):
    return np.isclose(x[0], 1.11, atol=tol) & np.isclose(x[1], 0.0, atol=tol)

v1 = locate_entities(domain, 0, point1)
v2 = locate_entities(domain, 0, point2)
v3 = locate_entities(domain, 0, point3)
v4 = locate_entities(domain, 0, point4)

dofs_x1 = fem.locate_dofs_topological(V.sub(0), 0, v1)  # x-component dofs at v1
dofs_y1 = fem.locate_dofs_topological(V.sub(1), 0, v2)  # y-component dofs at v2
dofs_x2 = fem.locate_dofs_topological(V.sub(0), 0, v3)  # x-component dofs at v3
dofs_y2 = fem.locate_dofs_topological(V.sub(1), 0, v4)  # y-component dofs at v4

zero = PETSc.ScalarType(0.0)
bc_x1 = fem.dirichletbc(zero, dofs_x1, V.sub(0))  # ux=0 at (0,1.11)
bc_y1 = fem.dirichletbc(zero, dofs_y1, V.sub(1))  # uy=0 at (-1.11,0)
bc_x2 = fem.dirichletbc(zero, dofs_x2, V.sub(0))  # ux=0 at (0,-1.11)
bc_y2 = fem.dirichletbc(zero, dofs_y2, V.sub(1))  # uy=0 at (1.11,0)

bcs = [bc_x1, bc_y1, bc_x2, bc_y2]


x = ufl.SpatialCoordinate(domain)
r = ufl.sqrt(x[0]**2 + x[1]**2 + 1e-16)     # avoid division by 0 at origin
e_r = ufl.as_vector((x[0]/r, x[1]/r))       # radial unit vector

f0 = fem.Constant(domain, default_scalar_type(2008.9286))  # N/m^3 (2D plane strain: per thickness)
f_radial = f0 * e_r

L_form = 0
for tag in ins_tags:
    L_form += ufl.dot(f_radial, u_) * dx(tag)

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

vtk = io.VTKFile(domain.comm, "WP-full360-simp_pln_strn.pvd", "u")
vtk.write_function(uh, 0)
vtk.write_function(sig, 0)
vtk.write_function(strn, 0)
vtk.close()

with XDMFFile(MPI.COMM_WORLD, "WP-full360-simp_pln_strn.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(cell_markers, domain.geometry)
    #xdmf.write_function(uh,0)
    xdmf.write_function(strn,0)
    xdmf.write_function(sig,0)

