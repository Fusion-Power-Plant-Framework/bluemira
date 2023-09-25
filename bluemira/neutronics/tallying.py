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
"""Functions for creating the openmc tallies."""
from typing import Tuple

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.neutronics.make_geometry import Cells
from bluemira.neutronics.make_materials import MaterialsLibrary


def filter_cells(
    cells: Cells,
    material_lib: MaterialsLibrary,
    src_rate: float,
) -> Tuple[
    openmc.CellFilter,
    openmc.MaterialFilter,
    openmc.CellFilter,
    openmc.ParticleFilter,
    openmc.ParticleFilter,
    openmc.EnergyFunctionFilter,
    openmc.EnergyFunctionFilter,
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
    src_rate:
        number of neutrons produced by the source (plasma) per second.
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

    cell_filter = openmc.CellFilter(
        (
            cells.tf_coil,
            *cells.plasma.get_cells(),
            cells.divertor.fw,
            cells.divertor.fw_sf,
            *cells.inboard.get_cells(),
            *cells.outboard.get_cells(),
            *cells.divertor.regions,
        )
    )

    mat_filter = openmc.MaterialFilter([getattr(material_lib, mat) for mat in mats])

    fw_surf_filter = openmc.CellFilter(
        (
            *cells.inboard.sf,
            *cells.outboard.sf,
            cells.divertor.fw_sf,
            *cells.inboard.fw,
            *cells.outboard.fw,
            cells.divertor.fw,
        )
    )

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # eV per source particle to MW coefficients
    # Need to ask @ JAMES HAGUE (original file line L.313)
    eV_per_sp_to_MW = raw_uc(src_rate, "eV/s", "MW")

    MW_energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    MW_dose_coeffs = [eV_per_sp_to_MW, eV_per_sp_to_MW]
    # makes a flat line function
    MW_mult_filter = openmc.EnergyFunctionFilter(MW_energy_bins, MW_dose_coeffs)

    # photon heat flux coefficients (cm per source particle to MW cm)
    # Tally heat flux
    energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    dose_coeffs = [0.0 * eV_per_sp_to_MW, 100.0e6 * eV_per_sp_to_MW]
    # simply modify the energy by multiplying by the constant
    energy_mult_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)

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
        MW_mult_filter,
        energy_mult_filter,
        cyl_mesh_filter,
    )


def _create_tallies_from_filters(
    cell_filter: openmc.CellFilter,
    mat_filter: openmc.MaterialFilter,
    fw_surf_filter: openmc.CellFilter,
    neutron_filter: openmc.ParticleFilter,
    photon_filter: openmc.ParticleFilter,
    MW_mult_filter: openmc.EnergyFunctionFilter,
    energy_mult_filter: openmc.EnergyFunctionFilter,
    cyl_mesh_filter: openmc.MeshFilter,
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
    MW_mult_filter:
        tally binned by energy so that it can be used to obtain the MW rate
    energy_mult_filter:
        tally binned by energy so that it can calculate the spectrum
    cyl_mesh_filter:
        tally binned spatially: the tokamak is cut into stacks of concentric rings

    Returns
    -------
    Exports the tallies to an xml file.

    """
    tallies_list = []
    for name, scores, filters in (
        ("TBR", "(n,Xt)", []),
        ("heating", "heating", [mat_filter]),  # eV per sp
        ("MW heating", "heating", [mat_filter, MW_mult_filter]),  # MW
        ("neutron wall load", "damage-energy", [fw_surf_filter, neutron_filter]),
        (
            "photon heat flux",
            "flux",
            [fw_surf_filter, photon_filter, energy_mult_filter],
        ),
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
    src_rate: float,  # [1/s]
) -> None:
    """First create the filters (list of cells to be tallied),
    then create create the tallies from those filters.
    """
    _create_tallies_from_filters(*filter_cells(cells, material_lib, src_rate))
