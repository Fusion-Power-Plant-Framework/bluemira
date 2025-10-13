import pyvista
from dolfinx import plot
from dolfinx.io import gmshio
from mpi4py import MPI

from bluemira.structural.finite_element import FEMPlainStrainLEA2D  # Adjust if needed

# --- Load mesh
mesh_path = "poly16b.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(
    mesh_path, MPI.COMM_WORLD, gdim=2
)

# --- (Optional) Visualize mesh
topology, cell_types, geometry = plot.vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
pyvista.set_jupyter_backend("static")
p = pyvista.Plotter()
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show_axes()
p.show()

# --- Instantiate solver
solver = FEMPlainStrainLEA2D(
    mesh=domain,
    cell_markers=cell_markers,
    facet_markers=facet_markers,
    degree=2,
    repr="vectorial",
)

# --- Set materials
solver.set_materials({
    37: (200e9, 0.3)  # Tag 1 has E=200 GPa, nu=0.3
})

# --- Apply normal pressure to a set of boundary tags
for tag in range(38, 54):  # Tags 38 to 53 inclusive
    solver.set_normal_pressure(1e6, tag)

# --- Set Dirichlet BCs on edges
solver.set_dirichlet_bcs([
    (1, 34, 0.0),  # ux = 0 on bottom
    (1, 33, 0.0),  # ux = 0 on top
    (0, 35, 0.0),  # uy = 0 on left
    (0, 36, 0.0),  # uy = 0 on right
])

# --- Solve
uh = solver.solve()

# --- Postprocess
strain, stress = solver.postprocess_stress_strain()

# --- Export results
solver.export_vtk("plane_strain_360_LEA_2D_example.pvd", uh, strain, stress)
