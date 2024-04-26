# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Functions for creating the openmc tallies."""

from dataclasses import asdict
from itertools import chain

import numpy as np
import openmc

from bluemira.neutronics.make_csg import BlanketCellArray
from bluemira.neutronics.make_geometry import Cells
from bluemira.neutronics.make_materials import MaterialsLibrary


def filter_cells(
    cells: Cells,
    material_lib: MaterialsLibrary,
) -> tuple[
    openmc.CellFilter,
    openmc.MaterialFilter,
    openmc.CellFilter,
    openmc.ParticleFilter,
    openmc.ParticleFilter,
    openmc.MeshFilter,
]:
    """
    Requests cells for scoring.

    Parameters
    ----------
    cells_and_cell_lists:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    material_lib:
        A dataclass with all of the material definitions stored.
    """
    mats = (
        "inb_fw_mat",
        "outb_fw_mat",
        "inb_bz_mat",
        "outb_bz_mat",
        "inb_mani_mat",
        "outb_mani_mat",
        "inb_vv_mat",
        "outb_vv_mat",
        "divertor_mat",
        "div_fw_mat",
        "tf_coil_mat",
        "inb_sf_mat",
        "outb_sf_mat",
        "div_sf_mat",
    )

    cell_filter = openmc.CellFilter((
        cells.tf_coil,
        *cells.plasma.get_cells(),
        cells.divertor.fw,
        cells.divertor.fw_sf,
        *cells.inboard.get_cells(),
        *cells.outboard.get_cells(),
        *cells.divertor.regions,
    ))

    mat_filter = openmc.MaterialFilter([getattr(material_lib, mat) for mat in mats])

    fw_surf_filter = openmc.CellFilter((
        *cells.inboard.sf,
        *cells.outboard.sf,
        cells.divertor.fw_sf,
        *cells.inboard.fw,
        *cells.outboard.fw,
        cells.divertor.fw,
    ))

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    cyl_mesh = openmc.CylindricalMesh(mesh_id=1)
    cyl_mesh.r_grid = np.linspace(400, 1400, 100 + 1)
    cyl_mesh.z_grid = np.linspace(-800.0, 800.0, 160 + 1)
    cyl_mesh_filter = openmc.MeshFilter(cyl_mesh)

    return (
        cell_filter,
        mat_filter,
        fw_surf_filter,
        neutron_filter,
        photon_filter,
        cyl_mesh_filter,
    )


def filter_new_cells(
    material_library: MaterialsLibrary,
    blanket_cell_array: BlanketCellArray,
    divertor_cells=None,  # noqa: ARG001
    plasma_void=None,  # noqa: ARG001
):
    """Bodge method to filter new cells. To be fixed later."""
    # TODO: make prettier! Delete unwanted parts.
    cells = chain.from_iterable(blanket_cell_array)
    fw_surf_cells = [stack[0] for stack in blanket_cell_array] + [
        stack[1] for stack in blanket_cell_array
    ]
    cell_filter = openmc.CellFilter(list(cells))
    fw_surf_filter = openmc.CellFilter(fw_surf_cells)
    mat_filter = openmc.MaterialFilter(list(asdict(material_library).values()))
    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    return (
        cell_filter,
        mat_filter,
        fw_surf_filter,
        neutron_filter,
        photon_filter,
        None,
    )


def _create_tallies_from_filters(
    cell_filter: openmc.CellFilter,  # noqa: ARG001
    mat_filter: openmc.MaterialFilter,
    fw_surf_filter: openmc.CellFilter,
    neutron_filter: openmc.ParticleFilter,
    photon_filter: openmc.ParticleFilter,
    cyl_mesh_filter: openmc.MeshFilter | None = None,  # noqa: ARG001
) -> None:
    """
    Produces tallies for OpenMC scoring.

    Parameters
    ----------
    cell_filter:
        tally binned by cell
    mat_filter:
        tally binned by materials
        # wait you should provide cells, not materials??!
    fw_surf_filter:
        tally binned by first wall surface
    neutron_filter:
        tally binned by neutron
    photon_filter:
        tally binned by photon
    cyl_mesh_filter:
        tally binned spatially: the tokamak is cut into stacks of concentric rings

    Returns
    -------
    Exports the tallies to an xml file.

    """
    tallies_list = []
    for name, scores, filters in (
        ("TBR", "(n,Xt)", []),
        # this is only the THEORETICAL TBR, does not account for extraction losses.
        # we can use the (n,Xt) score because the Lithium produces a maximum of 1 Tritium
        # per reaction, so there won't be issue of under-counting.
        ("material heating", "heating", [mat_filter]),  # eV per sp
        ("neutron wall load", "damage-energy", [fw_surf_filter, neutron_filter]),
        ("photon heating", "heating", [fw_surf_filter, photon_filter]),
        # skipped
        # ("neutron flux in every cell", "flux", [cell_filter, neutron_filter]),
        # ("neutron flux in 2d mesh", "flux", [cyl_mesh_filter, neutron_filter]),
    ):
        tally = openmc.Tally(name=name)
        tally.scores = [scores]
        tally.filters = filters
        tallies_list.append(tally)

    tallies = openmc.Tallies(tallies_list)
    tallies.export_to_xml()


def create_tallies(
    cells: Cells,
    material_lib: MaterialsLibrary,
) -> None:
    """First create the filters (list of cells to be tallied),
    then create create the tallies from those filters.
    """
    _create_tallies_from_filters(*filter_cells(cells, material_lib))
