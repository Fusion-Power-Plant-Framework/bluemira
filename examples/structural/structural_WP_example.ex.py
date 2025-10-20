from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path, make_bluemira_path
from bluemira.base.logs import set_log_level
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfinx
from mpi4py import MPI
import pyvista


set_log_level("DEBUG")

def plot_dolfinx_mesh(mesh, show: bool = True):
    """
    Plots the mesh structure, including nodes and faces.

    Parameters
    ----------
    show : bool, optional
        Flag to display the plot immediately (default is True).

    Returns
    -------
    pyvista.Plotter
        The PyVista plotter object with the mesh visualization.
    """
    from dolfinx.plot import vtk_mesh

    plotter = pyvista.Plotter()
    tdim = mesh.topology.dim
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, tdim))
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if show:
        plotter.show()
    return plotter

def create_conductor(x0, y0, dx_core, dy_core, dx_jacket, dy_jacket, dx_ins, dy_ins,
                     name="conductor"):
    p0 = np.array([x0, y0, 0])
    p1 = p0 + np.array([-dx_core/2, -dy_core/2, 0])
    p2 = p0 + np.array([dx_core/2, -dy_core/2, 0])
    p3 = p0 + np.array([dx_core/2, +dy_core/2, 0])
    p4 = p0 + np.array([-dx_core/2, +dy_core/2, 0])
    poly = tools.make_polygon(
        [p1,p2,p3,p4], closed=True, label=name
    )
    return poly

x0 = 0
y0 = 0
dx_core = 0.1
dy_core = 0.1
dx_jacket = 0.1
dy_jacket = 0.1
dx_ins = 0.1
dy_ins = 0.1
poly_core = BluemiraWire([create_conductor(x0, y0, dx_core, dy_core, dx_jacket,
                                           dy_jacket,dx_ins, dy_ins)])
poly_core.mesh_options = {"lcar": 0.01, "physical_group": "poly_core"}
face_core = BluemiraFace(poly_core, label="conductor_face")
face_core.mesh_options = {"lcar": 0.1, "physical_group": "face_core"}

c_all = Component(name="all")
c_core = PhysicalComponent(name="core", shape=face_core, parent=c_all)

directory = make_bluemira_path("structural/structural_WP_example",
                              subfolder="generated_data")

meshfiles = [Path(directory, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]
m = meshing.Mesh(meshfile=meshfiles)
buffer = m(c_all)

msh_to_xdmf(
    "Mesh.msh",
    dimensions=(0, 2),
    directory=directory,
)

from bluemira.mesh.tools import import_mesh_v9
directory = Path(directory)
mesh, cell_markers, facet_markers = dolfinx.io.gmshio.read_from_msh(directory
                                                                    / "Mesh.msh",
                                                            MPI.COMM_WORLD, 0 ,gdim=2)
mesh_in, boundaries_mf_in, subdomains_mf_in, link_dict_in = import_mesh_v9("Mesh",
                                                                 subdomains=True, directory=directory)
plot_dolfinx_mesh(mesh_in, show=True)