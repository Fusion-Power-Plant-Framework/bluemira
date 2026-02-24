import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio


# Init

if gmsh.isInitialized():
    gmsh.finalize()
gmsh.initialize()
gmsh.model.add("WP_full2d")


# Geometry parameters (all in SI units: metres)

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
lc_outer = 0.080    # typical element size in vault
lc_rect  = 0.030    # typical element size in WP + Insulator
lc_jack  = 0.010    # typical element size in jacket rectangles
lc_cable = 0.005    # typical element size in cable rectangles (finer)

# Jacket hole grid (columns × rows)
n_cols = 5
n_rows = 3
dx = 0.035          # horizontal gap between jackets (and to Insulator edge) [m]
dy = 0.0175         # vertical gap between jackets (and to Insulator edge) [m]

# WP rectangle (in radial direction) [m]
Rin_hole  = 1.85    # inner radius of WP block
Rout_hole = 2.05    # outer radius of WP block
half_w    = 0.30    # half-width of WP block in x-direction (total width 0.60 m)

# Insulator rectangle (inside WP) [m]
Rin_ins  = 1.87     # inner radius of Insulator region
Rout_ins = 2.03     # outer radius of Insulator region
half_w2  = 0.560 / 2.0  # half-width of Insulator region (total width 0.560 m = 0.28 m half-width)

# Cable offset inside each jacket rectangle [m]
offset = 0.005      # offset from jacket edges to cable edges


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

gmsh.model.geo.addCurveLoop([L9, L10, L11, L12], 3)  # Insulator outer loop


# Jacket rectangles (5 × 3) inside Insulator

width_ins   = 2.0 * half_w2             # total width of Insulator in x [m]
height_ins  = Rout_ins - Rin_ins        # total height of Insulator in y [m]
usable_w    = width_ins  - (n_cols + 1) * dx
usable_h    = height_ins - (n_rows + 1) * dy

hole_w = usable_w / n_cols              # jacket width [m]
hole_h = usable_h / n_rows              # jacket height [m]

jacket_loops = []
cable_loops  = []

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


# Main surfaces:
#   1: vault (sector minus WP rectangle)
#   2: WP (WP minus Insulator rectangle)
#   3: Insulator (Insulator minus jackets)
#   jacket_i: each jacket loop with cable loop as hole
#   cable_i: each cable loop as its own surface

gmsh.model.geo.addPlaneSurface([1, 2], 1)                 # vault
gmsh.model.geo.addPlaneSurface([2, 3], 2)                 # WP
gmsh.model.geo.addPlaneSurface([3] + jacket_loops, 3)     # Insulator minus jackets

jacket_surfaces = []
cable_surfaces  = []

surf_j_base = 3000
surf_c_base = 4000

for idx, (jl, cl) in enumerate(zip(jacket_loops, cable_loops)):
    jacket_surf_id = surf_j_base + idx
    cable_surf_id  = surf_c_base + idx

    # Jacket = outer rectangle minus inner cable rectangle
    gmsh.model.geo.addPlaneSurface([jl, cl], jacket_surf_id)
    # Cable = inner rectangle
    gmsh.model.geo.addPlaneSurface([cl], cable_surf_id)

    jacket_surfaces.append(jacket_surf_id)
    cable_surfaces.append(cable_surf_id)

gmsh.model.geo.synchronize()


# Physical groups: boundaries (vault)

left_pg = gmsh.model.addPhysicalGroup(1, [3], tag=1)
gmsh.model.setPhysicalName(1, left_pg, "left")

right_pg = gmsh.model.addPhysicalGroup(1, [4], tag=2)
gmsh.model.setPhysicalName(1, right_pg, "right")

top_pg = gmsh.model.addPhysicalGroup(1, [1], tag=3)
gmsh.model.setPhysicalName(1, top_pg, "top")

bottom_pg = gmsh.model.addPhysicalGroup(1, [2], tag=4)
gmsh.model.setPhysicalName(1, bottom_pg, "bottom")


# Physical groups: vault, WP, Insulator

vault_pg = gmsh.model.addPhysicalGroup(2, [1], tag=10)
gmsh.model.setPhysicalName(2, vault_pg, "vault")

wp_pg = gmsh.model.addPhysicalGroup(2, [2], tag=11)
gmsh.model.setPhysicalName(2, wp_pg, "WP")

ins_pg = gmsh.model.addPhysicalGroup(2, [3], tag=12)
gmsh.model.setPhysicalName(2, ins_pg, "Insulator")


# Physical groups: individual jackets & cables

phys_j_base = 30   # jacket_* tags
phys_c_base = 50   # cable_* tags

for idx, (js, cs) in enumerate(zip(jacket_surfaces, cable_surfaces)):
    # individual jacket_i
    jp = phys_j_base + idx
    gmsh.model.addPhysicalGroup(2, [js], tag=jp)
    gmsh.model.setPhysicalName(2, jp, f"jacket_{idx+1}")

    # individual cable_i
    cp = phys_c_base + idx
    gmsh.model.addPhysicalGroup(2, [cs], tag=cp)
    gmsh.model.setPhysicalName(2, cp, f"cable_{idx+1}")


# Mesh settings (all in metres)

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.004)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.080)

# Mesh and export

gmsh.model.mesh.generate(gdim)
gmsh.write("WP_full2d.msh")

domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)

gmsh.fltk.run()
gmsh.finalize()

