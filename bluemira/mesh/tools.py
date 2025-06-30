# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Converter from MSH to XDMF mesh file formats, derived from F. Loiseau,
R. Delaporte-Mathurin, and C. Weickhmann's https://github.com/floiseau/msh2xdmf

Credit: F. Loiseau, R. Delaporte-Mathurin, and C. Weickhmann
"""

from __future__ import annotations

import json
from pathlib import Path

import meshio
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.mesh.error import MeshConversionError

__all__ = ("import_mesh", "msh_to_xdmf")


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


def msh_to_xdmf(
    mesh_name: str,
    dimensions: tuple[int, ...] | int = (0, 2),
    directory: str = ".",
):
    """
    Convert a MSH file to an XMDF file.

    Parameters
    ----------
    mesh_name: str
        Name of the MSH file to convert to XDMF
    dimensions: Union[Tuple[int], int]
        Dimensions of the mesh (0: x, 1: y, 2: z), defaults to x-z
        (0, 1, 2) would be a 3-D mesh
    directory: str
        Directory in which the MSH file exists and where the XDMF files will be written

    Raises
    ------
    MeshConversionError
        * If the file does not exist
        * If the dimensionality != [2, 3]
        * If no domain physical groups are found

    Notes
    -----
    Creates the following files:
        * DOMAIN_SUFFIX
        * BOUNDARY_SUFFIX
        * LINKFILE_SUFFIX
    """
    dimensions = _check_dimensions(dimensions)

    file_path = Path(directory, mesh_name)
    if not file_path.exists():
        raise MeshConversionError(f"No such file: {file_path}")

    file_prefix = mesh_name.split(".", 1)[0]
    mesh = meshio.read(file_path.as_posix())
    _export_domain(mesh, file_prefix, directory, dimensions)
    _export_boundaries(mesh, file_prefix, directory, dimensions)
    _export_link_file(mesh, file_prefix, directory)


def import_mesh(
    file_prefix: str = "mesh", *, subdomains: bool = False, directory: str = "."
) -> tuple[Mesh, Mesh, Mesh, dict]:
    """
    Import a dolfin mesh.

    Parameters
    ----------
    file_prefix:
        File prefix to use when importing a mesh (defaults to 'mesh')
    subdomains:
        Whether or not to subdomains are present (defaults to False)
    directory:
        Directory in which the MSH file and XDMF files exist

    Returns
    -------
    mesh:
        Dolfin Mesh object containing the domain
    boundaries_mf:
        Dolfin MeshFunctionSizet object containing the geometry
    subdomains_mf:
        Dolfin MeshFunctionSizet object containing the geometry
    link_dict:
        Link dictionary between MSH and XDMF objects

    Raises
    ------
    FileNotFoundError
        no mesh file(s) found
    """
    domain_file = Path(directory, f"{file_prefix}_{DOMAIN_SUFFIX}")
    boundary_file = Path(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}")
    link_file = Path(directory, f"{file_prefix}_{LINKFILE_SUFFIX}")
    files = [domain_file, boundary_file, link_file]
    exists = [file.exists() for file in files]

    if not all(exists):
        msg = "\n".join([
            fn.as_posix() for fn, exist in zip(files, exists, strict=False) if not exist
        ])
        raise FileNotFoundError(f"No mesh file(s) found:\n {msg}")

    mesh = Mesh()

    with XDMFFile(domain_file.as_posix()) as file:
        file.read(mesh)

    boundaries_mvc = None

    with XDMFFile(boundary_file.as_posix()) as file:
        file.read(boundaries_mvc, "boundaries")

    boundaries_mf = None

    if subdomains:
        subdomains_mvc = None
        with XDMFFile(domain_file.as_posix()) as file:
            file.read(subdomains_mvc, "subdomains")
    else:
        subdomains_mf = None

    with open(link_file) as file:
        link_dict = json.load(file)

    return mesh, boundaries_mf, subdomains_mf, link_dict


def _check_dimensions(dimensions: int | list[int]) -> tuple[int]:
    if isinstance(dimensions, int):
        dimensions = tuple(np.arange(dimensions))

    if len(dimensions) not in {2, 3}:
        raise MeshConversionError(
            f"Length of dimensions must be either 2 or 3, not: {len(dimensions)}"
        )
    for dim in dimensions:
        if dim not in {0, 1, 2}:
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

    Raises
    ------
    MeshConversionError
        No domain physical group found
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
        Path(directory, f"{file_prefix}_{DOMAIN_SUFFIX}").as_posix(),
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
        Path(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}").as_posix(),
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

    with open(Path(directory, f"{file_prefix}_{LINKFILE_SUFFIX}"), "w") as file:
        json.dump(table, file, indent=4)


def _get_data(mesh, cell_type):
    return [cell.data for cell in mesh.cells if cell.type == cell_type]


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
