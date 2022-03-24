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

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.mesh.error import MeshConversionError

__all__ = "msh_to_xdmf"


DOMAIN_CELL_DIM = {
    2: "triangle",
    3: "tetra",
}


BOUNDARY_CELL_DIM = {
    2: "line",
    3: "triangle",
}


def msh_to_xdmf(mesh_name, dimension=2, directory="."):
    """
    Convert a MSH file to an XMDF file.

    Parameters
    ----------
    mesh_name: str

    dimension: int

    directory: str

    Notes
    -----
    Creates the following files:
        * domain.xdmf
        * boundaries.xdmf
    """
    if dimension not in [2, 3]:
        raise MeshConversionError(f"Dimension must be either 2 or 3, not: {dimension}")

    file_prefix = mesh_name.split(".")[0]
    mesh = meshio.read(os.sep.join([directory, mesh_name]))
    _export_domain(mesh, file_prefix, directory, dimension)
    _export_boundaries(mesh, file_prefix, directory, dimension)
    _export_association_table(mesh, file_prefix, directory)


def _export_domain(mesh, file_prefix, directory, dimension):
    cell_type = DOMAIN_CELL_DIM[dimension]
    data = [arr for (t, arr) in mesh.cells if t == cell_type]

    if len(data) == 0:
        raise MeshConversionError("No domain physical group found in: {file_prefix}")
    if "gmsh:physical" not in mesh.cell_data:
        raise MeshConversionError("No domain physical group found in: {file_prefix}")

    cells = meshio.CellBlock(cell_type, np.concatenate(data))

    subdomains = [
        mesh.cell_data["gmsh:physical"][i]
        for i, cell in enumerate(mesh.cells)
        if cell.type == cell_type
    ]

    cell_data = {"subdomains": np.concatenate(subdomains)}

    domain = meshio.Mesh(mesh.points[:, :dimension], cells, cell_data=cell_data)

    meshio.write(
        os.sep.join([directory, f"{file_prefix}_domain.xdmf"]),
        domain,
        file_format="xdmf",
    )


def _export_boundaries(mesh, file_prefix, directory, dimension):
    pass


def _export_association_table(mesh, file_prefix, directory):
    pass
