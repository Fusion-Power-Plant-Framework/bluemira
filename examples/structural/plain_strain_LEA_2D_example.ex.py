import pyvista
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import plot

from bluemira.structural.finite_element import FEMPlainStrainLEA2D  # Adjust path as needed

# --- Load mesh
mesh_path = "neufrustum4.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)

# Optional: visualize the mesh with PyVista
topology, cell_types, geometry = plot.vtk_mesh(domain, 2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
pyvista.set_jupyter_backend("static")
p = pyvista.Plotter()
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show_axes()
p.show()

# --- Instantiate the solver class
solver = FEMPlainStrainLEA2D(
    mesh=domain,
    cell_markers=cell_markers,
    facet_markers=facet_markers,
    degree=2,
    repr="vectorial"
)

# --- Assign material properties
solver.set_materials({
    9: (200e9, 0.3)
})

# --- Apply pressure on boundary tag 8 (e.g., self-weight or traction)
solver.set_normal_pressure(1e6, tag=8)

# --- Apply slip conditions on boundary tags 5 and 6 (ps1 and ps2)
solver.apply_slip_conditions(slip_tags=[5, 6])

# --- Solve the system
uh = solver.solve()

# --- Postprocess stress and strain
strain, stress = solver.postprocess_stress_strain()

# --- Export results to VTK
solver.export_vtk("jason2.pvd", uh, strain, stress)
