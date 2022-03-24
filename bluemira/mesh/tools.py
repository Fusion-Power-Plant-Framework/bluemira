# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

import os
from configparser import ConfigParser

import meshio
import numpy as np
from dolfin import Mesh, MeshValueCollection, XDMFFile
from dolfin.cpp.mesh import MeshFunctionSizet

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.mesh.error import MeshConversionError

__all__ = ("msh_to_xdmf", "import_mesh")


CELL_TYPE_DIM = {
    2: "line",
    3: "triangle",
    4: "tetra",
}


GMSH_PHYS = "gmsh:physical"

DOMAIN_SUFFIX = "domain.xdmf"
BOUNDARY_SUFFIX = "boundaries.xdmf"
ASS_TAB_SUFFIX = "association_table.ini"


def msh_to_xdmf(mesh_name, dimension=2, directory=".", verbose=False):
    """
    Convert a MSH file to an XMDF file.

    Parameters
    ----------
    mesh_name: str

    dimension: int

    directory: str

    verbose: bool

    Notes
    -----
    Creates the following files:
        * DOMAIN_SUFFIX
        * BOUNDARY_SUFFIX
        * ASS_TAB_SUFFIX
    """
    if dimension not in [2, 3]:
        raise MeshConversionError(f"Dimension must be either 2 or 3, not: {dimension}")

    file_prefix = mesh_name.split(".")[0]
    mesh = meshio.read(os.sep.join([directory, mesh_name]))
    _export_domain(mesh, file_prefix, directory, dimension)
    _export_boundaries(mesh, file_prefix, directory, dimension)
    _export_association_table(mesh, file_prefix, directory, verbose=verbose)


def import_mesh(file_prefix, subdomains=False, dimension=2, directory="."):
    """ """
    domain_file = os.sep.join([directory, f"{file_prefix}_{DOMAIN_SUFFIX}"])
    boundary_file = os.sep.join([directory, f"{file_prefix}_{BOUNDARY_SUFFIX}"])
    association_file = os.sep.join([directory, f"{file_prefix}_{ASS_TAB_SUFFIX}"])

    if not os.path.exists(domain_file) or not os.path.exists(boundary_file):
        msh_to_xdmf(f"{file_prefix}.msh", dimension=dimension, directory=directory)

    mesh = Mesh()

    with XDMFFile(domain_file) as file:
        file.read(mesh)

    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=dimension)

    with XDMFFile(boundary_file) as file:
        file.read(boundaries_mvc, "boundaries")

    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)

    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=dimension)
        with XDMFFile(domain_file) as file:
            file.read(subdomains_mvc, "subdomains")
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    else:
        subdomains_mf = None

    content = ConfigParser()
    content.read(association_file)
    association_table = dict(content["ASSOCIATION TABLE"])

    return mesh, boundaries_mf, subdomains_mf, association_table


def _export_domain(mesh, file_prefix, directory, dimension):
    """ """
    cell_type = CELL_TYPE_DIM[dimension + 1]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No domain physical group found in: {file_prefix}")
        return

    if GMSH_PHYS not in mesh.cell_data:
        raise MeshConversionError(f"No domain physical group found in: {file_prefix}")

    cells = _make_cellblocks(data, cell_type)

    subdomains = _get_cells(mesh, cell_type)

    cell_data = {"subdomains": [np.concatenate(subdomains)]}

    domain = meshio.Mesh(mesh.points[:, :dimension], cells=cells, cell_data=cell_data)

    _write_mesh(
        os.sep.join([directory, f"{file_prefix}_{DOMAIN_SUFFIX}"]),
        domain,
    )


def _export_boundaries(mesh, file_prefix, directory, dimension):
    """ """
    cell_type = CELL_TYPE_DIM[dimension]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No boundary physical group found in: {file_prefix}")
        return

    cells = _make_cellblocks(data, cell_type)

    boundaries = _get_cells(mesh, cell_type)

    cell_data = {"boundaries": [np.concatenate(boundaries)]}

    boundaries = meshio.Mesh(
        points=mesh.points[:, :dimension], cells=cells, cell_data=cell_data
    )

    _write_mesh(
        os.sep.join([directory, f"{file_prefix}_{BOUNDARY_SUFFIX}"]),
        boundaries,
    )


def _export_association_table(mesh, file_prefix, directory, verbose=False):
    """ """
    table = {}
    for key, arrays in mesh.cell_sets.items():
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        if key != "gmsh:bounding_entities":
            value = mesh.cell_data[GMSH_PHYS][index][0]
            table[key] = int(value)

    content = ConfigParser()
    content["ASSOCIATION TABLE"] = table

    filename = os.sep.join([directory, f"{file_prefix}_{ASS_TAB_SUFFIX}"])
    with open(filename, "w") as file:
        content.write(file)


def _get_data(mesh, cell_type):
    return [array for (typ, array) in mesh.cells if typ == cell_type]


def _make_cellblocks(data, cell_type):
    return [meshio.CellBlock(cell_type, np.concatenate(data))]


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
