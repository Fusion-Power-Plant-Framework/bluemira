# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Converter from MSH to XDMF mesh file formats, derived from F. Loiseau,
R. Delaporte-Mathurin, and C. Weickhmann's https://github.com/floiseau/msh2xdmf

Credit: F. Loiseau, R. Delaporte-Mathurin, and C. Weickhmann
"""

import json
import os
from typing import Tuple, Union

import meshio
import numpy as np
from dolfin.cpp.mesh import MeshFunctionSizet
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import Mesh, MeshValueCollection
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.mesh.error import MeshConversionError

__all__ = ("read_from_msh", "import_mesh")


CELL_TYPE_DIM = {
    2: "line",
    3: "triangle",
    4: "tetra",
}


GMSH_PHYS = "gmsh:physical"
GMSH_BE = "gmsh:bounding_entities"

DOMAIN_SUFFIX = "domain.xdmf"
BOUNDARY_SUFFIX = "boundaries.xdmf"
LINKFILE_SUFFIX = "linkfile.json"


def read_from_msh(filename, dimension=2):
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(filename, gdim=dimension)
    return mesh, cell_markers, facet_markers


def import_mesh(file_prefix="mesh", subdomains=False, directory="."):
    """
    Import a dolfinx mesh.

    Parameters
    ----------
    file_prefix: str
        File prefix to use when importing a mesh (defaults to 'mesh')
    subdomains: bool
        Whether or not to subdomains are present (defaults to False)
    directory: str
        Directory in which the MSH file and XDMF files exist
    Returns
    -------
    mesh: dolfinx.mesh::Mesh
        Dolfin Mesh object containing the domain
    boundaries_mf: dolfin::MeshFunctionSizet
        Dolfin MeshFunctionSizet object containing the geometry
    subdomains_mf: Optional[dolfin::MeshFunctionSizet]
        Dolfin MeshFunctionSizet object containing the geometry
    link_dict: dict
        Link dictionary between MSH and XDMF objects
    """
    domain_file = os.path.join(directory, f"{file_prefix}_{DOMAIN_SUFFIX}")
    boundary_file = os.path.join(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}")
    link_file = os.path.join(directory, f"{file_prefix}_{LINKFILE_SUFFIX}")
    files = [domain_file, boundary_file, link_file]
    exists = [os.path.exists(file) for file in files]

    if not all(exists):
        msg = "\n".join([fn for fn, exist in zip(files, exists) if not exist])
        raise MeshConversionError(f"No mesh file(s) found:\n {msg}")

    with XDMFFile(domain_file) as file:
        mesh = file.read_mesh()

    dimension = mesh.topology().dim()
    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=dimension)

    with XDMFFile(boundary_file) as file:
        file.read_mesh(boundaries_mvc, "boundaries")

    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)

    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=dimension)
        with XDMFFile(domain_file) as file:
            file.read_mesh(subdomains_mvc, "subdomains")
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    else:
        subdomains_mf = None

    with open(link_file, "r") as file:
        link_dict = json.load(file)

    return mesh, boundaries_mf, subdomains_mf, link_dict


def _check_dimensions(dimensions):
    if isinstance(dimensions, int):
        dimensions = tuple(np.arange(dimensions))

    if len(dimensions) not in [2, 3]:
        raise MeshConversionError(
            f"Length of dimensions must be either 2 or 3, not: {len(dimensions)}"
        )
    for dim in dimensions:
        if dim not in [0, 1, 2]:
            raise MeshConversionError(
                f"Dimensions tuple must contain integers 0, 1, or 2, not: {dim}"
            )

    if len(dimensions) != len(set(dimensions)):
        raise MeshConversionError(
            f"Dimensions tuple cannot have repeated integers: {dimensions}"
        )
    return dimensions


def _export_domain(mesh, file_prefix, directory, dimensions):
    """
    Export the domain of a mesh to XDMF.
    """
    dimensions = _check_dimensions(dimensions)

    cell_type = CELL_TYPE_DIM[len(dimensions) + 1]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No domain physical group found in: {file_prefix}")
        return

    if GMSH_PHYS not in mesh.cell_data:
        raise MeshConversionError(f"No domain physical group found in: {file_prefix}")

    cells = _make_cellblocks(data, cell_type)

    subdomains = _get_cells(mesh, cell_type)

    cell_data = {"subdomains": [np.concatenate(subdomains)]}

    domain = _make_mesh(mesh, dimensions, cells, cell_data)

    _write_mesh(
        os.path.join(directory, f"{file_prefix}_{DOMAIN_SUFFIX}"),
        domain,
    )


def _export_boundaries(mesh, file_prefix, directory, dimensions):
    """
    Export the boundaries of a mesh to XDMF.
    """
    dimensions = _check_dimensions(dimensions)

    cell_type = CELL_TYPE_DIM[len(dimensions)]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No boundary physical group found in: {file_prefix}")
        return

    cells = _make_cellblocks(data, cell_type)

    boundaries = _get_cells(mesh, cell_type)

    cell_data = {"boundaries": [np.concatenate(boundaries)]}

    boundaries = _make_mesh(mesh, dimensions, cells, cell_data)

    _write_mesh(
        os.path.join(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}"),
        boundaries,
    )


def _export_link_file(mesh, file_prefix, directory):
    """
    Export the association file between MSH and XDMF objects.
    """
    table = {}
    for key, arrays in mesh.cell_sets.items():
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        if key != GMSH_BE:
            value = mesh.cell_data[GMSH_PHYS][index][0]
            table[key] = int(value)

    bluemira_debug(
        tabulate(
            list(table.items()),
            headers=["GMSH label", "MeshFunction value"],
            tablefmt="simple",
        )
    )

    filename = os.path.join(directory, f"{file_prefix}_{LINKFILE_SUFFIX}")
    with open(filename, "w") as file:
        json.dump(table, file, indent=4)


def _get_data(mesh, cell_type):
    return [array for (typ, array) in mesh.cells if typ == cell_type]


def _make_cellblocks(data, cell_type):
    return [meshio.CellBlock(cell_type, np.concatenate(data))]


def _make_mesh(mesh, dimensions, cells, cell_data):
    return meshio.Mesh(
        mesh.points[:, list(dimensions)], cells=cells, cell_data=cell_data
    )


def _get_cells(mesh, cell_type):
    return [
        mesh.cell_data[GMSH_PHYS][i]
        for i, cell in enumerate(mesh.cells)
        if cell.type == cell_type
    ]


def _write_mesh(filename, obj):
    meshio.write(
        filename,
        obj,
        file_format="xdmf",
    )
