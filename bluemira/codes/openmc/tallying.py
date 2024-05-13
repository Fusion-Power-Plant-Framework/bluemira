# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Functions for creating the openmc tallies."""

from itertools import chain

import openmc

from bluemira.codes.openmc.make_csg import BlanketCellArray


def filter_cells(
    material_list,
    blanket_cell_array: BlanketCellArray,
    divertor_cells=None,  # noqa: ARG001
    plasma_void=None,  # noqa: ARG001
):
    """Bodge method to filter new cells. To be fixed later."""
    # TODO: make prettier! Delete unwanted parts.
    cells = chain.from_iterable(blanket_cell_array)
    fw_surf_cells = [
        *(stack[0] for stack in blanket_cell_array),
        *(stack[1] for stack in blanket_cell_array),
    ]
    cell_filter = openmc.CellFilter(list(cells))
    fw_surf_filter = openmc.CellFilter(fw_surf_cells)
    mat_filter = openmc.MaterialFilter(material_list)
    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # name, scores, filters
    return (
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
    )
