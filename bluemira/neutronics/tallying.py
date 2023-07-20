from typing import Dict, List, Tuple, Union

import numpy as np
import openmc

import bluemira.neutronics.make_materials as mm
from bluemira.base.constants import raw_uc


def filter_cells(
    cells_and_cell_lists: Dict[str, Union[List[openmc.Cell], openmc.Cell]],
    material_lib: mm.MaterialsLibrary,
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
    cell_filter = openmc.CellFilter(
        # the single cells
        [
            cells_and_cell_lists["tf_coil_cell"],
            cells_and_cell_lists["plasma_inner1"],
            cells_and_cell_lists["plasma_inner2"],
            cells_and_cell_lists["plasma_outer1"],
            cells_and_cell_lists["plasma_outer2"],
            cells_and_cell_lists["divertor_fw"],
            cells_and_cell_lists["divertor_fw_sf"],  # sf = surface
        ]
        # the cell lists
        + cells_and_cell_lists["inb_vv_cells"]
        + cells_and_cell_lists["inb_mani_cells"]
        + cells_and_cell_lists["inb_bz_cells"]
        + cells_and_cell_lists["inb_fw_cells"]
        + cells_and_cell_lists["inb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["outb_vv_cells"]
        + cells_and_cell_lists["outb_mani_cells"]
        + cells_and_cell_lists["outb_bz_cells"]
        + cells_and_cell_lists["outb_fw_cells"]
        + cells_and_cell_lists["outb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["divertor_cells"],
    )

    mat_filter = openmc.MaterialFilter(
        [
            material_lib.inb_fw_mat,
            material_lib.outb_fw_mat,
            material_lib.inb_bz_mat,
            material_lib.outb_bz_mat,
            material_lib.inb_mani_mat,
            material_lib.outb_mani_mat,
            material_lib.inb_vv_mat,
            material_lib.outb_vv_mat,
            material_lib.divertor_mat,
            material_lib.div_fw_mat,
            material_lib.tf_coil_mat,
            material_lib.inb_sf_mat,  # sf = surface
            material_lib.outb_sf_mat,  # sf = surface
            material_lib.div_sf_mat,  # sf = surface
        ]
    )

    fw_surf_filter = openmc.CellFilter(
        cells_and_cell_lists["inb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["outb_sf_cells"]  # sf = surface
        + [cells_and_cell_lists["divertor_fw_sf"]]  # sf = surface
        + cells_and_cell_lists["inb_fw_cells"]
        + cells_and_cell_lists["outb_fw_cells"]
        + [cells_and_cell_lists["divertor_fw"]]
    )

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # eV per source particle to MW coefficients
    # SOMETHING SEEMS WRONG @ JAMES HAGUE (original file line L.313)
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
    tally_tbr = openmc.Tally(name="TBR")
    tally_tbr.scores = ["(n,Xt)"]

    tally_heating = openmc.Tally(name="heating")  # eV per sp
    tally_heating.scores = ["heating"]
    tally_heating.filters = [mat_filter]

    tally_heating_MW = openmc.Tally(name="MW heating")  # MW
    tally_heating_MW.scores = ["heating"]
    tally_heating_MW.filters = [mat_filter, MW_mult_filter]

    tally_n_wall_load = openmc.Tally(name="neutron wall load")
    tally_n_wall_load.scores = ["damage-energy"]
    tally_n_wall_load.filters = [fw_surf_filter, neutron_filter]

    tally_p_heat_flux = openmc.Tally(name="photon heat flux")
    tally_p_heat_flux.scores = ["flux"]
    tally_p_heat_flux.filters = [fw_surf_filter, photon_filter, energy_mult_filter]

    tally_n_flux = openmc.Tally(name="neutron flux in every cell")
    tally_n_flux.scores = ["flux"]
    tally_n_flux.filters = [cell_filter, neutron_filter]

    tally_n_flux_mesh = openmc.Tally(name="neutron flux in 2d mesh")
    tally_n_flux_mesh.scores = ["flux"]
    tally_n_flux_mesh.filters = [cyl_mesh_filter, neutron_filter]

    tallies = openmc.Tallies(
        [
            tally_tbr,
            tally_heating,
            tally_heating_MW,
            tally_n_wall_load,
            tally_p_heat_flux,
            # tally_n_flux, # skipped
            # tally_n_flux_mesh, # skipped
        ]
    )
    tallies.export_to_xml()


def create_tallies(
    cells_and_cell_lists: Dict[str, Union[List[openmc.Cell], openmc.Cell]],
    material_lib: mm.MaterialsLibrary,
    src_rate: float,
) -> None:
    """First create the filters (list of cells to be tallied),
    then create create the tallies from those filters."""
    _create_tallies_from_filters(
        *filter_cells(cells_and_cell_lists, material_lib, src_rate)
    )
