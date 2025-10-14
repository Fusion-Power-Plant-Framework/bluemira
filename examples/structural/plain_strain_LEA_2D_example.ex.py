import pyvista
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import plot



from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

from bluemira.structural.finite_element import FEMPlainStrainLEA2D  # Adjust path as needed




# --- Load mesh
mesh_path = "neufrustum4.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)
tag_for_material = 9
tag_for_pressure = 8
slip_tags = [5, 6]

# import numpy as np
# # Parameters
# R1 = 0.5098
# R2 = R1 * 2
# theta_deg = -22.5  # angle in degrees
#
# # Convert degrees to radians
# theta_rad = np.radians(theta_deg)
#
# # Define original points
# P0 = np.array([0, R1, 0])
# P1 = np.array([0, R2, 0])
#
# # Rotation matrix around the z-axis (counterclockwise)
# rotation_matrix_z = np.array([
#     [np.cos(theta_rad), -np.sin(theta_rad), 0],
#     [np.sin(theta_rad),  np.cos(theta_rad), 0],
#     [0,                 0,                 1]
# ])
#
# # Apply rotation
# P3 = rotation_matrix_z @ P0
# P2 = rotation_matrix_z @ P1
#
# poly = tools.make_polygon(
#     [P0, P1, P2, P3], closed=True, label="poly"
# )
#
# lcar = R1/50
# poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}
#
# surf = BluemiraFace(poly, label="surf")
# surf.mesh_options = {"physical_group": "coil"}
#
# from pathlib import Path
# meshfiles = [
#     Path(".", p).as_posix() for p in ["Mesh.geo_unrolled", "neufrustum4_2.msh"]
# ]
# m = meshing.Mesh(meshfile=meshfiles)
# m(surf)
#
# msh_to_xdmf("neufrustum4_2.msh", dimensions=(0, 1), directory=".")
#
# mesh_path = "neufrustum4_2.msh"
# domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, gdim=2)


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
    tag_for_material: (200e9, 0.3)
})

# --- Apply pressure on boundary tag 8 (e.g., self-weight or traction)
solver.set_normal_pressure(1e6, tag=tag_for_pressure)

# --- Apply slip conditions on boundary tags 5 and 6 (ps1 and ps2)
solver.apply_slip_conditions(slip_tags=slip_tags)

# --- Solve the system
uh = solver.solve()

# --- Postprocess stress and strain
strain, stress = solver.postprocess_stress_strain()

# --- Export results to VTK
solver.export_vtk("jason2.pvd", uh, strain, stress)
