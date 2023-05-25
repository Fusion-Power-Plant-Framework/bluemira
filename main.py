# import sys

# try:
#     import gmsh
# except ImportError:
#     print("This demo requires gmsh to be installed")
#     sys.exit(0)


# from dolfinx.io import XDMFFile, gmshio

# from mpi4py import MPI

# gmsh.initialize()

# # Choose if Gmsh output is verbose
# gmsh.option.setNumber("General.Terminal", 0)
# model = gmsh.model()
# model.add("Sphere")
# model.setCurrent("Sphere")
# sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)

# # Synchronize OpenCascade representation with gmsh model
# model.occ.synchronize()

# # Add physical marker for cells. It is important to call this function
# # after OpenCascade synchronization
# model.add_physical_group(dim=3, tags=[sphere])

# # Generate the mesh
# model.mesh.generate(dim=3)

# # Create a DOLFINx mesh (same mesh on each rank)
# msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
# msh.name = "Sphere"
# cell_markers.name = f"{msh.name}_cells"
# facet_markers.name = f"{msh.name}_facets"

# with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_meshtags(cell_markers)
#     msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
#     file.write_meshtags(facet_markers)

import dolfinx.plot as plot
import numpy as np
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import (
    CellType,
    compute_midpoints,
    create_unit_cube,
    create_unit_square,
    meshtags,
)
from mpi4py import MPI

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# If environment variable PYVISTA_OFF_SCREEN is set to true save a png
# otherwise create interactive plot
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

# Set some global options for all plots
transparent = False
figsize = 800
pyvista.rcParams["background"] = [0.5, 0.5, 0.5]


def plot_scalar():
    # We start by creating a unit square mesh and interpolating a
    # function into a degree 1 Lagrange space
    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
    V = FunctionSpace(msh, ("Lagrange", 1))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

    # To visualize the function u, we create a VTK-compatible grid to
    # values of u to
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array

    # The function "u" is set as the active scalar for the mesh, and
    # warp in z-direction is set
    grid.set_active_scalars("u")
    warped = grid.warp_by_scalar()

    # A plotting window is created with two sub-plots, one of the scalar
    # values and the other of the mesh is warped by the scalar values in
    # z-direction
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text(
        "Scalar contour field", font_size=14, color="black", position="upper_edge"
    )
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text(
        "Warped function", position="upper_edge", font_size=14, color="black"
    )
    sargs = dict(
        height=0.8,
        width=0.1,
        vertical=True,
        position_x=0.05,
        position_y=0.05,
        fmt="%1.2e",
        title_font_size=40,
        color="black",
        label_font_size=25,
    )
    subplotter.set_position([-3, 2.6, 0.3])
    subplotter.set_focus([3, -1, -0.15])
    subplotter.set_viewup([0, 0, 1])
    subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
    if pyvista.OFF_SCREEN:
        subplotter.screenshot(
            "2D_function_warp.png",
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
    else:
        subplotter.show()


def plot_meshtags():
    # Create a mesh
    msh = create_unit_square(MPI.COMM_WORLD, 25, 25)

    # Create a geometric indicator function
    def in_circle(x):
        return np.array(
            (x.T[0] - 0.5) ** 2 + (x.T[1] - 0.5) ** 2 < 0.2**2, dtype=np.int32
        )

    # Create cell tags - if midpoint is inside circle, it gets value 1,
    # otherwise 0
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    midpoints = compute_midpoints(
        msh, msh.topology.dim, list(np.arange(num_cells, dtype=np.int32))
    )
    cell_tags = meshtags(
        msh, msh.topology.dim, np.arange(num_cells), in_circle(midpoints)
    )

    # Create VTK mesh
    cells, types, x = plot.create_vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvita grid
    grid.cell_data["Marker"] = cell_tags.values
    grid.set_active_scalars("Marker")

    # Create a plotter with two subplots, and add mesh tag plot to the
    # first sub-window
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text(
        "Mesh with markers", font_size=14, color="black", position="upper_edge"
    )
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.view_xy()

    # We can visualize subsets of data, by creating a smaller topology
    # (set of cells). Here we create VTK mesh data for only cells with
    # that tag '1'.
    cells, types, x = plot.create_vtk_mesh(msh, entities=cell_tags.find(1))

    # Add this grid to the second plotter window
    sub_grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 1)
    subplotter.add_text(
        "Subset of mesh", font_size=14, color="black", position="upper_edge"
    )
    subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")

    if pyvista.OFF_SCREEN:
        subplotter.screenshot(
            "2D_markers.png",
            transparent_background=transparent,
            window_size=[2 * figsize, figsize],
        )
    else:
        subplotter.show()


plot_meshtags()
