# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""MCIO-based Tallies"""

import openmc
from mcio.tally import (
    EnergyDepositionTally,
    VolumeFluenceTally,
)


def add_default_tallies(
    model: openmc.Model,
    material_name_mapping: dict[str, str],
):
    """Add default tallies to the OpenMC model."""

    def add_mcio_tally(
        tally_class,
        name: str,
        filters: list,
        particles: tuple[str, ...] = ("neutron",),
    ) -> None:
        mcio_tally = tally_class(
            comment=name,
            particles=list(particles),
        )

        tally = openmc.Tally(name=name)
        tally.scores = mcio_tally.openmc_tally_score
        tally.filters = [
            openmc.ParticleFilter(list(particles)),
            *filters,
        ]
        model.tallies.append(tally)

    # All cells in the OpenMC model
    all_cells = list(model.geometry.get_all_cells().values())

    # -----------------------------------------------
    # Cell filters
    # -----------------------------------------------
    # Map each component name to its OpenMC cells.
    # Components remain distinct even when they share a material.
    component_cells = {
        component_name: [cell for cell in all_cells if cell.name == component_name]
        for component_name in material_name_mapping
    }

    # Create a filter for each component with matching cells.
    cell_filters = {
        component_name: openmc.CellFilter(cells)
        for component_name, cells in component_cells.items()
        if cells
    }

    # Filter containing all cells
    cell_filter = openmc.CellFilter(all_cells)

    # -----------------------------------------------
    # Powers
    # -----------------------------------------------
    for component_name, component_filter in cell_filters.items():
        add_mcio_tally(
            EnergyDepositionTally,
            f"{component_name} power",
            [component_filter],
        )

    add_mcio_tally(
        EnergyDepositionTally,
        "total power",
        [cell_filter],
    )

    # -----------------------------------------------
    # Neutron flux
    # -----------------------------------------------
    add_mcio_tally(
        VolumeFluenceTally,
        "neutron flux in every cell",
        [cell_filter],
    )

    # -----------------------------------------------
    # Tritium breeding ratio (TBR)
    # -----------------------------------------------
    tbr_tally = openmc.Tally(name="TBR")
    tbr_tally.scores = ["(n,Xt)"]
    model.tallies.append(tbr_tally)
