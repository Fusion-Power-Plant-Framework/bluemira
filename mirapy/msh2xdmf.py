# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

# from https://github.com/floiseau/msh2xdmf.git
import argparse
import meshio
import os
import numpy as np
from configparser import ConfigParser
try:
    from dolfin import XDMFFile, Mesh, MeshValueCollection
    from dolfin.cpp.mesh import MeshFunctionSizet
except ImportError:
    print("Could not import dolfin. Continuing without Dolfin support.")


def msh2xdmf(mesh_name, dim=2, directory="."):
    """
    Function converting a MSH mesh into XDMF files.
    The XDMF files are:
    - "domain.xdmf": the domain;
    - "boundaries.xdmf": the boundaries physical groups from GMSH;
    """
    
    # Get the mesh name has prefix
    prefix = mesh_name.split('.')[0]
    # Read the input mesh
    msh = meshio.read("{}/{}".format(directory, mesh_name))
    # Generate the domain XDMF file
    export_domain(msh, dim, directory, prefix)
    # Generate the boundaries XDMF file
    export_boundaries(msh, dim, directory, prefix)
    # Export association table
    export_association_table(msh, prefix, directory)


def export_domain(msh, dim, directory, prefix):
    """
    Export the domain XDMF file as well as the subdomains values.
    """

    # Set cell type
    if dim == 2:
        cell_type = "triangle"
    elif dim == 3:
        cell_type = "tetra"
    # Generate the cell block for the domain cells
    data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    if len(data_array) == 0:
        print("WARNING: No domain physical group found.")
        return
    else:
        data = np.concatenate(data_array)
    cells = [
        meshio.CellBlock(
            type=cell_type,
            data=data,
            )
        ]
    # Generate the domain cells data (for the subdomains)
    try:
        cell_data = {
            "subdomains": [
                np.concatenate(
                    [
                        msh.cell_data["gmsh:physical"][i]
                        for i, cellBlock in enumerate(msh.cells)
                        if cellBlock.type == cell_type
                        ]
                    )
                ]
            }
    except KeyError:
        raise ValueError("No physical group found for the domain."
                         "Define the domain physical group."
                         " - if dim=2, the domain is a surface"
                         "- if dim=3, the domain is a volume"
            )

    # Generate a meshio Mesh for the domain
    domain = meshio.Mesh(
        points=msh.points, #[:, :dim],
        cells=cells,
        cell_data=cell_data,
    )
    # Export the XDMF mesh of the domain
    meshio.write(
        "{}/{}_{}".format(directory, prefix, "domain.xdmf"),
        domain,
        file_format="xdmf"
        )


def export_boundaries(msh, dim, directory, prefix):
    """
    Export the boundaries XDMF file.
    """

    # Set the cell type
    if dim == 2:
        cell_type = "line"
    elif dim == 3:
        cell_type = "triangle"
    # Generate the cell block for the boundaries cells
    data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    if len(data_array) == 0:
        print("WARNING: No boundary physical group found.")
        return
    else:
        data = np.concatenate(data_array)
    boundaries_cells = [
        meshio.CellBlock(
            type=cell_type,
            data=data,
            )
        ]
    # Generate the boundaries cells data
    cell_data = {
        "boundaries": [
            np.concatenate(
                [
                    msh.cell_data["gmsh:physical"][i]
                    for i, cellBlock in enumerate(msh.cells)
                    if cellBlock.type == cell_type
                    ]
                )
            ]
        }
    # Generate the meshio Mesh for the boundaries physical groups
    boundaries = meshio.Mesh(
        points=msh.points, #[:, :dim],
        cells=boundaries_cells,
        cell_data=cell_data,
    )

    # Export the XDMF mesh of the lines boundaries
    meshio.write(
        "{}/{}_{}".format(directory, prefix, "boundaries.xdmf"),
        boundaries,
        file_format="xdmf"
        )


def export_association_table(msh, prefix='mesh', directory='.', verbose=True):
    """
    Display the association between the physical group label and the mesh
    value.
    """
    # Create association table
    association_table = {}

    # Display the correspondance
    formatter = "|{:^20}|{:^20}|"
    topbot = "+{:-^41}+".format("")
    separator = "+{:-^20}+{:-^20}+".format("", "")

    # Display
    if verbose:
        print('\n' + topbot)
        print(formatter.format("GMSH label", "MeshFunction value"))
        print(separator)

    for label, arrays in msh.cell_sets.items():
        # Get the index of the array in arrays
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        # Get the value in cell_data for the corresponding array
        value = msh.cell_data["gmsh:physical"][index][0]
        # Store the association table in a dictionnary
        association_table[label] = value

        # Display the association
        if verbose:
            print(formatter.format(label, value))
    if verbose:
        print(topbot)
    # Export the association table
    file_content = ConfigParser()
    file_content["ASSOCIATION TABLE"] = association_table
    file_name = "{}/{}_{}".format(directory, prefix, "association_table.ini")
    with open(file_name, 'w') as f:
        file_content.write(f)


def import_mesh_from_xdmf(
        prefix="mesh",
        subdomains=False,
        dim=2,
        directory=".",
        ):
    """
    Function importing a dolfin mesh.

    Arguments:
        - domain (str): name of the domain XDMF file;
        - boundaries (str): name of the boundaries XDMF file;
        - dim (int): dimension of the domain;
        - subdomains (bool): true if there are subdomains, else false
        - directory (str): (optional) directory of the mesh;

    Output:
        - dolfin Mesh object containing the domain;
        - dolfin MeshFunction object containing the physical lines (dim=2) or
            surfaces (dim=3) defined in the msh file and the sub-domains;
        - association table
    """

    print("import_mesh_from_xdmf dim = {}".format(dim))

    # Set the file name
    domain = "{}_domain.xdmf".format(prefix)
    boundaries = "{}_boundaries.xdmf".format(prefix)
    # Import the converted domain
    mesh = Mesh()
    with XDMFFile("{}/{}".format(directory, domain)) as infile:
        infile.read(mesh)
    # Import the boundaries
    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=dim)
    with XDMFFile("{}/{}".format(directory, boundaries)) as infile:
        infile.read(boundaries_mvc, 'boundaries')
    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
    # Import the subdomains
    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=dim)
        with XDMFFile("{}/{}".format(directory, domain)) as infile:
            infile.read(subdomains_mvc, 'subdomains')
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    # Import the association table
    association_table_name = "{}/{}_{}".format(directory, prefix, "association_table.ini")
    file_content = ConfigParser()
    file_content.read(association_table_name)
    association_table = dict(file_content["ASSOCIATION TABLE"])
    # Convert the value from string to int
    for key, value in association_table.items():
        association_table[key] = int(value)
    # Return the Mesh and the MeshFunction objects
    if not subdomains:
        return mesh, boundaries_mf, association_table
    else:
        return mesh, boundaries_mf, subdomains_mf, association_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "msh_file",
        help="input .msh file",
        type=str,
        )
    parser.add_argument(
        "-d",
        "--dimension",
        help="dimension of the domain",
        type=int,
        default=2,
        )
    args = parser.parse_args()
    # Get current directory
    current_directory = os.getcwd()
    # Conert the mesh
    msh2xdmf(args.msh_file, args.dimension, directory=current_directory)
