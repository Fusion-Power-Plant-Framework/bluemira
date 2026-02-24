import pyvista
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import plot

# Import the GPS class (adjust import path to wherever you put it)
from bluemira.structural.finite_element import FEMGeneralizedPlainStrainLEA2D

# --- Load mesh
mesh_path = "WP_full2d.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

material_tags = [10, 11, 12] + list(range(30, 45)) + list(range(50, 65))
body_tags = list(range(50, 65))
slip_tags = [1, 2]

# Optional: visualize mesh
topology, cell_types, geometry = plot.vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
pyvista.set_jupyter_backend("static")
p = pyvista.Plotter()
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show_axes()
p.show()

# --- Instantiate GPS solver
solver = FEMGeneralizedPlainStrainLEA2D(domain, cell_markers, facet_markers, degree=2, ezz_degree=1)

# --- Materials (cell tag -> (E, nu))
solver.set_materials({t: (2e11, 0.30) for t in material_tags})

# --- Slip constraints (same idea, but applied to V only inside class)
solver.apply_slip_conditions(slip_tags)

# --- Loads
# Option A: normal pressure on a boundary (outward normal traction = p*n)
#gps.set_normal_pressure(1e6, tag=tag_for_pressure)

# (Option B: if you want vector traction instead)
# gps.add_boundary_traction(facet_tag=tag_for_pressure, tx=0.0, ty=1e6)

# --- Apply body load on cable tags (e.g., self-weight or traction)
solver.set_body_load(fx=0.0, fy=10000.0, tags=body_tags)

# --- Solve (returns both)
uh, ezzh = solver.solve(ksp_setup="fieldsplit_gamg", rtol=1e-8)
if MPI.COMM_WORLD.rank == 0: print("solved", flush=True)


# --- Export
# If you want a nice P2 displacement for viz, interpolate to an unconstrained space:
solver.export_solution_vtk("results/WP_detailSec-GPS.pvd", interpolate_u=True)