"""Functions related to the calculation of volumes of openmc cells."""
import os
from typing import List, Union

import openmc
from numpy import pi

from bluemira.neutronics.params import TokamakGeometry


def get_vol_of_truncated_cone(cone, top_z, bot_z):
    """Calculates the volume of a truncated cone from an OpenMC cone surface and z bounds
    Assumes it is centred on the origin.
    """
    # Get cone parameters
    top_r = (cone.r2 * (top_z - cone.z0) ** 2) ** 0.5
    bot_r = (cone.r2 * (bot_z - cone.z0) ** 2) ** 0.5
    height = top_z - bot_z

    volume = pi / 3.0 * height * (top_r**2 + top_r * bot_r + bot_r**2)

    return volume


def get_fw_vol(outer_cone, inner_cone, top_plane, bottom_plane):
    """Takes the bounding OpenMC surface objects and calculates the first wall volume
    This is only approximate but accurate as long as the first wall is thin
    """
    top_z = top_plane.z0
    bottom_z = bottom_plane.z0

    outer_cone_vol = get_vol_of_truncated_cone(outer_cone, top_z, bottom_z)
    inner_cone_vol = get_vol_of_truncated_cone(inner_cone, top_z, bottom_z)

    volume = outer_cone_vol - inner_cone_vol

    return volume


def get_div_fw_section_vol(outer_cone, inner_cone, outer_r, inner_r):
    """Takes the bounding OpenMC surface objects and calculates the first wall volume
    This is only approximate but accurate as long as the first wall is thin

    Getting z coord at radial limits for outer zone
    Squareroot in function so have two roots
    """
    z1a = outer_cone.z0 - inner_r / outer_cone.r2**0.5
    z1b = outer_cone.z0 + inner_r / outer_cone.r2**0.5
    z2a = outer_cone.z0 - outer_r / outer_cone.r2**0.5
    z2b = outer_cone.z0 + outer_r / outer_cone.r2**0.5

    # Select correct roots
    z1 = z1a if z1a < 0.0 and ((z1a > z1b) or z1b > 0.0) else z1b
    z2 = z2a if z2a < 0.0 and ((z2a > z2b) or z2b > 0.0) else z2b

    top_z = max(z1, z2)
    bottom_z = min(z1, z2)

    outer_cone_vol = get_vol_of_truncated_cone(outer_cone, top_z, bottom_z)
    inner_cone_vol = get_vol_of_truncated_cone(inner_cone, top_z, bottom_z)

    volume = outer_cone_vol - inner_cone_vol

    return volume


def get_div_fw_vol(outer_cones, inner_cones, rs):
    """Calculates the volume for each section of the divertor first wall
    This is only approximate but accurate as long as the first wall is thin
    """
    volumes = []
    for i, outer_cone in enumerate(outer_cones):
        vol = get_div_fw_section_vol(outer_cone, inner_cones[i], rs[i + 1], rs[i])

        volumes.append(vol)

    volume = sum(volumes)

    return volume


def stochastic_volume_calculation(
    tokamak_geometry: TokamakGeometry,
    cells_and_cell_lists: Union[List[openmc.Cell], openmc.Cell],
    particles: int = int(4e7),
) -> None:
    """
    Performs a stochastic volume calculation for the cells.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
    cells_and_cell_lists: dict
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    particles: int
        how many randomly generated particle to use
        for the stochastic volume calculation.
    """
    # quietly delete the unused .hf files: bad practice, fix later?
    try:
        os.remove("summary.h5")
        os.remove("statepoint.1.h5")
    except OSError:
        pass

    # maximum radii and heigth reached by all of the tokamak's breeder zone component
    maxr = (
        tokamak_geometry.major_r
        + tokamak_geometry.minor_r
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
    )
    maxz = (
        # height of plasma = 2 * elong * minor
        tokamak_geometry.elong * tokamak_geometry.minor_r
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
        + tokamak_geometry.outb_mnfld_thick
        + tokamak_geometry.outb_vv_thick
    )
    # draw the bounding box for the simulation.
    lower_left = (-maxr, -maxr, -maxz)
    upper_right = (maxr, maxr, maxz)
    cell_vol_calc = openmc.VolumeCalculation(
        cells_and_cell_lists["inb_fw_cells"]
        + [cells_and_cell_lists["divertor_fw"]]
        + cells_and_cell_lists["outb_fw_cells"],
        int(particles),
        lower_left,
        upper_right,
    )
    settings = openmc.Settings()
    settings.volume_calculations = [cell_vol_calc]

    settings.export_to_xml()
    # within this bounding box, use naive Monte Carlo to find
    # the volumes of the cells representing the tokamak components.
    openmc.calculate_volumes()
