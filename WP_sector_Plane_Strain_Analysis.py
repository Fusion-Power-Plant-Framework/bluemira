# Plane Strain Linear Elastic Analysis of a TF Coil WP

#-----------------------------------------------------------------
# Imported Libraries
# ----------------------------------------------------------------- 

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
from ufl import (FacetNormal, as_vector, as_matrix, Identity, Measure, TestFunction, tr, TrialFunction, as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx_mpc.utils import create_normal_approximation

#----------------------------------------------------------------
# Meshing
#-----------------------------------------------------------------
if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()
gmsh.model.add("WP_full2d")

# ------------------------------------------------------------
# Geometry parameters
# ------------------------------------------------------------
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0

Rout  = 2150.0
Rin   = 1110.0
deg   = np.pi / 180.0
theta = 22.5 * deg
alpha = theta / 2.0

a1 = np.pi/2 - alpha          # ~78.75°
a2 = np.pi/2 + alpha          # ~101.25°

# Mesh sizes
lc_outer = 80.0    # vault
lc_rect  = 30.0    # WP + Insulator
lc_jack  = 10.0    # jacket rectangles
lc_cable = 5.0     # cable inner rectangles (finer)

# Jacket hole grid (columns × rows)
n_cols = 5
n_rows = 3
dx = 35.0
dy = 17.5

# WP rectangle (in radius)
Rin_hole  = 1850.0
Rout_hole = 2050.0
half_w    = 300.0      # half-width of WP block (total width 600)

# Insulator rectangle (inside WP)
Rin_ins  = 1870.0
Rout_ins = 2030.0
half_w2  = 560.0 / 2.0  # half-width of Insulator region (total 560)

# Cable offset inside each jacket rectangle
offset = 5.0

# ------------------------------------------------------------
# Sector geometry (vault)
# ------------------------------------------------------------
p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_outer, 1)
p1 = gmsh.model.geo.addPoint(Rout*np.cos(a1), Rout*np.sin(a1), 0.0, lc_outer, 2)
p2 = gmsh.model.geo.addPoint(Rout*np.cos(a2), Rout*np.sin(a2), 0.0, lc_outer, 3)
p3 = gmsh.model.geo.addPoint(Rin*np.cos(a1),  Rin*np.sin(a1),  0.0, lc_outer, 4)
p4 = gmsh.model.geo.addPoint(Rin*np.cos(a2),  Rin*np.sin(a2),  0.0, lc_outer, 5)

Louter = gmsh.model.geo.addLine(2, 3, 1)   # top
Linner = gmsh.model.geo.addLine(5, 4, 2)   # bottom
Lleft  = gmsh.model.geo.addLine(4, 2, 3)   # left
Lright = gmsh.model.geo.addLine(3, 5, 4)   # right

gmsh.model.geo.addCurveLoop([1, 3, 2, 4], 1)  # vault sector loop

# ------------------------------------------------------------
# WP rectangle
# ------------------------------------------------------------
p6 = gmsh.model.geo.addPoint(-half_w, Rin_hole,  0.0, lc_rect)
p7 = gmsh.model.geo.addPoint( half_w, Rin_hole,  0.0, lc_rect)
p8 = gmsh.model.geo.addPoint( half_w, Rout_hole, 0.0, lc_rect)
p9 = gmsh.model.geo.addPoint(-half_w, Rout_hole, 0.0, lc_rect)

L5 = gmsh.model.geo.addLine(p6, p7)
L6 = gmsh.model.geo.addLine(p7, p8)
L7 = gmsh.model.geo.addLine(p8, p9)
L8 = gmsh.model.geo.addLine(p9, p6)

gmsh.model.geo.addCurveLoop([L5, L6, L7, L8], 2)  # WP loop

# ------------------------------------------------------------
# Insulator rectangle
# ------------------------------------------------------------
p10 = gmsh.model.geo.addPoint(-half_w2, Rin_ins,  0.0, lc_rect)
p11 = gmsh.model.geo.addPoint( half_w2, Rin_ins,  0.0, lc_rect)
p12 = gmsh.model.geo.addPoint( half_w2, Rout_ins, 0.0, lc_rect)
p13 = gmsh.model.geo.addPoint(-half_w2, Rout_ins, 0.0, lc_rect)

L9  = gmsh.model.geo.addLine(p10, p11)
L10 = gmsh.model.geo.addLine(p11, p12)
L11 = gmsh.model.geo.addLine(p12, p13)
L12 = gmsh.model.geo.addLine(p13, p10)

gmsh.model.geo.addCurveLoop([L9, L10, L11, L12], 3)  # Insulator outer loop

# ------------------------------------------------------------
# Jacket rectangles (5 × 3) inside Insulator
# ------------------------------------------------------------
width_ins   = 2.0 * half_w2             # 560
height_ins  = Rout_ins - Rin_ins        # 160
usable_w    = width_ins  - (n_cols + 1) * dx
usable_h    = height_ins - (n_rows + 1) * dy

hole_w = usable_w / n_cols
hole_h = usable_h / n_rows

jacket_loops = []
cable_loops = []

loop_j = 1000  # jacket loop tag start
loop_c = 2000  # cable  loop tag start

for j in range(n_rows):
    for i in range(n_cols):

        # Outer jacket rectangle
        x_left  = -half_w2 + dx + i * (hole_w + dx)
        x_right = x_left + hole_w

        y_bot   = Rin_ins + dy + j * (hole_h + dy)
        y_top   = y_bot + hole_h

        pa = gmsh.model.geo.addPoint(x_left,  y_bot, 0.0, lc_jack)
        pb = gmsh.model.geo.addPoint(x_right, y_bot, 0.0, lc_jack)
        pc = gmsh.model.geo.addPoint(x_right, y_top, 0.0, lc_jack)
        pd = gmsh.model.geo.addPoint(x_left,  y_top, 0.0, lc_jack)

        la = gmsh.model.geo.addLine(pa, pb)
        lb = gmsh.model.geo.addLine(pb, pc)
        lc_ = gmsh.model.geo.addLine(pc, pd)
        ld = gmsh.model.geo.addLine(pd, pa)

        gmsh.model.geo.addCurveLoop([la, lb, lc_, ld], loop_j)
        jacket_loops.append(loop_j)

        # Inner cable rectangle (offset from jacket edges)
        cx_left  = x_left  + offset
        cx_right = x_right - offset
        cy_bot   = y_bot   + offset
        cy_top   = y_top   - offset

        pa2 = gmsh.model.geo.addPoint(cx_left,  cy_bot, 0.0, lc_cable)
        pb2 = gmsh.model.geo.addPoint(cx_right, cy_bot, 0.0, lc_cable)
        pc2 = gmsh.model.geo.addPoint(cx_right, cy_top, 0.0, lc_cable)
        pd2 = gmsh.model.geo.addPoint(cx_left,  cy_top, 0.0, lc_cable)

        la2 = gmsh.model.geo.addLine(pa2, pb2)
        lb2 = gmsh.model.geo.addLine(pb2, pc2)
        lc2 = gmsh.model.geo.addLine(pc2, pd2)
        ld2 = gmsh.model.geo.addLine(pd2, pa2)

        gmsh.model.geo.addCurveLoop([la2, lb2, lc2, ld2], loop_c)
        cable_loops.append(loop_c)

        loop_j += 1
        loop_c += 1

# ------------------------------------------------------------
# Main surfaces:
#   1: vault (sector minus WP rectangle)
#   2: WP (WP minus Insulator rectangle)
#   3: Insulator (Insulator minus jackets)
#   jacket_i: each jacket loop with cable loop as hole
#   cable_i: each cable loop as its own surface
# ------------------------------------------------------------
gmsh.model.geo.addPlaneSurface([1, 2], 1)                 # vault
gmsh.model.geo.addPlaneSurface([2, 3], 2)                 # WP
gmsh.model.geo.addPlaneSurface([3] + jacket_loops, 3)     # Insulator minus jackets

jacket_surfaces = []
cable_surfaces = []

surf_j_base = 3000
surf_c_base = 4000

for idx, (jl, cl) in enumerate(zip(jacket_loops, cable_loops)):
    jacket_surf_id = surf_j_base + idx
    cable_surf_id  = surf_c_base + idx

    # Jacket = outer rect minus inner cable rect
    gmsh.model.geo.addPlaneSurface([jl, cl], jacket_surf_id)
    # Cable = inner rect
    gmsh.model.geo.addPlaneSurface([cl], cable_surf_id)

    jacket_surfaces.append(jacket_surf_id)
    cable_surfaces.append(cable_surf_id)

gmsh.model.geo.synchronize()

# ------------------------------------------------------------
# Physical groups: boundaries (vault)
# ------------------------------------------------------------
left_pg = gmsh.model.addPhysicalGroup(1, [3], tag=1)
gmsh.model.setPhysicalName(1, left_pg, "left")

right_pg = gmsh.model.addPhysicalGroup(1, [4], tag=2)
gmsh.model.setPhysicalName(1, right_pg, "right")

top_pg = gmsh.model.addPhysicalGroup(1, [1], tag=3)
gmsh.model.setPhysicalName(1, top_pg, "top")

bottom_pg = gmsh.model.addPhysicalGroup(1, [2], tag=4)
gmsh.model.setPhysicalName(1, bottom_pg, "bottom")

# ------------------------------------------------------------
# Physical groups: vault, WP, Insulator
# ------------------------------------------------------------
vault_pg = gmsh.model.addPhysicalGroup(2, [1], tag=10)
gmsh.model.setPhysicalName(2, vault_pg, "vault")

wp_pg = gmsh.model.addPhysicalGroup(2, [2], tag=11)
gmsh.model.setPhysicalName(2, wp_pg, "WP")

ins_pg = gmsh.model.addPhysicalGroup(2, [3], tag=12)
gmsh.model.setPhysicalName(2, ins_pg, "Insulator")

# ------------------------------------------------------------
# Physical groups: individual jackets & cables + aggregate groups
# ------------------------------------------------------------

# Aggregate physical groups for all jackets and all cables
jackets_all_pg = gmsh.model.addPhysicalGroup(2, jacket_surfaces, tag=20)
gmsh.model.setPhysicalName(2, jackets_all_pg, "jackets_all")

cables_all_pg = gmsh.model.addPhysicalGroup(2, cable_surfaces, tag=21)
gmsh.model.setPhysicalName(2, cables_all_pg, "cables_all")

# ------------------------------------------------------------
# Mesh settings
# ------------------------------------------------------------
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 4.0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 80.0)

# ------------------------------------------------------------
# Mesh
# ------------------------------------------------------------
gmsh.model.mesh.generate(gdim)
gmsh.write("WP_full2d.msh")

domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)

gmsh.fltk.run()
gmsh.finalize()

#-------------------------------------------------------------------------
# FE Analysis
#-------------------------------------------------------------------------

mesh_path = "WP_full2d.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)


dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_markers)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_markers)


#-------------------------------------------------------------------------
# Material Properties
#-------------------------------------------------------------------------

Q = fem.functionspace(domain, ("DG", 0))
E = fem.Function(Q)
nu = fem.Function(Q)

vault_cells = cell_markers.find(10)
E.x.array[vault_cells] = np.full_like(vault_cells, 2e11, dtype=default_scalar_type)
nu.x.array[vault_cells] = np.full_like(vault_cells, 0.3, dtype=default_scalar_type)

wp_cells = cell_markers.find(11)
E.x.array[wp_cells] = np.full_like(wp_cells, 2e11, dtype=default_scalar_type)
nu.x.array[wp_cells] = np.full_like(wp_cells, 0.3, dtype=default_scalar_type)

cable_cells = cell_markers.find(21)
E.x.array[cable_cells] = np.full_like(cable_cells, 2e11, dtype=default_scalar_type)
nu.x.array[cable_cells] = np.full_like(cable_cells, 0.3, dtype=default_scalar_type)

insulator_cells = cell_markers.find(12)
E.x.array[insulator_cells] = np.full_like(insulator_cells, 2e11, dtype=default_scalar_type)
nu.x.array[insulator_cells] = np.full_like(insulator_cells, 0.3, dtype=default_scalar_type)

jacket_cells = cell_markers.find(20)
E.x.array[jacket_cells] = np.full_like(jacket_cells, 2e11, dtype=default_scalar_type)
nu.x.array[jacket_cells] = np.full_like(jacket_cells, 0.3, dtype=default_scalar_type)

#-------------------------------------------------------------------------
# Constitutive Relations
#-------------------------------------------------------------------------

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


fy = fem.Constant(domain, 10000.0)
f  = as_vector((0.0, fy))  


# Body load on surface
L_form = dot(f,u_) * dx(21)

# Get the facet markers for boundary edges (ps1 and ps2)

mpc = MultiPointConstraint(V)

for tag in [3, 4]:  # ps1 and ps2
    normal_vec = create_normal_approximation(V, facet_markers, tag)
    mpc.create_slip_constraint(V, (facet_markers, tag), normal_vec)

mpc.finalize()

uh = fem.Function(mpc.function_space, name="Displacement")

# --- Solve ---
problem = LinearProblem(a_form, L_form, mpc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh_mpc = problem.solve()
uh.name = "u"

#-------------------------------------------------------------------------
# Postprocessing-Visualization
#-------------------------------------------------------------------------

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


with VTKFile(domain.comm, "WP_full2d.pvd", "w") as vtk:
    vtk.write_function(strain_fn)
    vtk.write_function(uh_mpc)
    vtk.write_function(stress_fn)