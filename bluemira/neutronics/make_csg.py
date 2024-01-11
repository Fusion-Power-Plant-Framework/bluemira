# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Make the entire tokamak from scratch using user-provided variables.
All units used in this file are either [m] or dimensionless.
"""

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import openmc

import bluemira.neutronics.make_materials as mm
from bluemira.base.constants import raw_uc
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.params import TokamakGeometry


def _create_rz_CSG_polygon(points: Coordinates) -> openmc.model.Polygon:
    points = raw_uc(points.xz.T, "m", "cm")
    return openmc.model.Polygon(points, basis="rz")


def _join_lists(lists):
    return list(itertools.chain(*[a if isinstance(a, list) else [a] for a in lists]))


@dataclass
class GraveyardSurface:
    """Graveyard suface CSG sections"""

    top: openmc.ZPlane
    bottom: openmc.ZPlane
    cylinder: openmc.ZCylinder


@dataclass
class Surfaces:
    """CGS surfaces"""

    bore: Optional[openmc.ZCylinder] = None
    graveyard: Optional[GraveyardSurface] = None
    inb_bot: Optional[openmc.ZPlane] = None
    inb_top: Optional[openmc.ZPlane] = None
    outer_surface_cyl: Optional[openmc.ZCylinder] = None
    tf_coil: Optional[openmc.ZCylinder] = None


@dataclass
class LayerCells:
    """Reactor layer CSG cells"""

    breeding_zone: Optional[List[openmc.Cell]] = None
    first_wall: Optional[List[openmc.Cell]] = None
    manifold: Optional[List[openmc.Cell]] = None
    scoring: Optional[List[openmc.Cell]] = None
    vacuum_vessel: Optional[List[openmc.Cell]] = None

    def get_cells(self):
        """Get cells from dataclass"""
        return _join_lists(
            a
            for a in (
                self.breeding_zone,
                self.first_wall,
                self.manifold,
                self.scoring,
                self.vacuum_vessel,
            )
        )


@dataclass
class Cells:
    """Universe CSG cells"""

    bore: openmc.Cell
    tf_coil: openmc.Cell
    # outer_vessel: openmc.Cell
    outer_container: openmc.Cell
    inboard: LayerCells
    divertor: openmc.Cell
    plasma: openmc.Cell
    # outboard: LayerCells

    def get_cells(self):
        """Get cells from dataclass"""
        return [
            self.bore,
            self.tf_coil,
            self.outer_container,
            self.divertor,
            self.plasma,
            *self.inboard.get_cells(),
        ]


def make_geometry(
    tokamak_geometry: TokamakGeometry,
    fw_deconstruction: BluemiraWire,
    material_lib: mm.MaterialsLibrary,
    *,
    fw_surf_score_depth: float = 0.0001,
    clearance_r: float = 50.0,  # [cm]
    container_steel_thick: float = 200.0,  # [cm]
    clear_div_to_shell=5.0,  # [cm]
) -> Tuple[Cells, openmc.Universe]:
    """
    Creates an OpenMC CSG geometry for a reactor

    Parameters
    ----------
    tokamak_geometry:

    """
    surfaces = Surfaces()

    z_min = raw_uc(-10.0, "m", "cm")
    z_max = raw_uc(10.0, "m", "cm")
    r_cs = raw_uc(2.0, "m", "cm")
    r_tf = raw_uc(3.0, "m", "cm")
    r_max = raw_uc(15.0, "m", "cm")
    surfaces.bore = openmc.ZCylinder(r=r_cs)
    surfaces.tf_coil = openmc.ZCylinder(r=r_tf)
    surfaces.inb_top = openmc.ZPlane(
        z0=z_max + clear_div_to_shell, boundary_type="vacuum"
    )
    surfaces.inb_bot = openmc.ZPlane(
        z0=z_min - clear_div_to_shell, boundary_type="vacuum"
    )

    # Outboard surfaces
    # Currently it is not possible to tally on boundary_type='vacuum' surfaces
    surfaces.outer_surface_cyl = openmc.ZCylinder(r=r_max + clearance_r)
    surfaces.graveyard = GraveyardSurface(
        openmc.ZPlane(z0=z_max + container_steel_thick),
        openmc.ZPlane(z0=z_min - container_steel_thick),
        openmc.ZCylinder(
            r=r_max + clearance_r + container_steel_thick, boundary_type="vacuum"
        ),
    )

    # Inboard cells
    bore = openmc.Cell(
        name="Inner bore",
        region=-surfaces.bore & -surfaces.inb_top & +surfaces.inb_bot,
    )
    tf_coil = openmc.Cell(
        name="TF Coils",
        fill=material_lib.tf_coil_mat,
        region=-surfaces.tf_coil
        & +surfaces.bore
        & -surfaces.inb_top
        & +surfaces.inb_bot,
    )

    outer_container = openmc.Cell(
        name="Container steel",
        fill=material_lib.container_mat,
        region=~(-surfaces.outer_surface_cyl & -surfaces.inb_top & +surfaces.inb_bot)
        & -surfaces.graveyard.cylinder
        & -surfaces.graveyard.top
        & +surfaces.graveyard.bottom,
    )

    # Simple offsetting of blanket in onion layers
    # TODO: Process TRegions and handle inboard/outboard
    # TODO: Pipe in proper thicknesses and offsets for outboard
    fw = _create_rz_CSG_polygon(fw_deconstruction.coordinates)
    fw_score = fw.offset(fw_surf_score_depth)
    fw_tungsten = fw_score.offset(tokamak_geometry.cgs.inb_fw_thick)
    fw_breeding = fw_tungsten.offset(tokamak_geometry.cgs.inb_bz_thick)
    fw_manifold = fw_breeding.offset(tokamak_geometry.cgs.inb_mnfld_thick)
    fw_vv = fw_manifold.offset(tokamak_geometry.cgs.inb_vv_thick)

    # Create divertor region and use cut from blanket
    div_thickness = 1.5
    fw_div = fw.offset(raw_uc(div_thickness, "m", "cm"))
    inboard_point = fw_deconstruction.coordinates.points[
        fw_deconstruction.tregions[0].index
    ]
    outboard_point = fw_deconstruction.coordinates.points[
        fw_deconstruction.tregions[2].index
    ]
    inboard_proj = (
        inboard_point + fw_deconstruction.tregions[0].direction * div_thickness
    )
    # TODO: Lazy fudge multiplier for boolean operation...
    ib_proj_down = inboard_proj + np.array([0, 0, -1]) * 3 * div_thickness
    outboard_proj = (
        outboard_point + fw_deconstruction.tregions[2].direction * div_thickness
    )
    ob_proj_down = outboard_proj + np.array([0, 0, -1]) * 3 * div_thickness
    div_coords = Coordinates([
        inboard_proj,
        inboard_point,
        outboard_point,
        outboard_proj,
        ob_proj_down,
        ib_proj_down,
    ])

    divertor_cut = _create_rz_CSG_polygon(div_coords)
    divertor_region = divertor_cut.region & fw_div.region & ~fw.region
    divertor_cell = openmc.Cell(
        name="Divertor", fill=material_lib.divertor_mat, region=divertor_region
    )

    scoring_cell = openmc.Cell(
        name="Blanket scoring",
        fill=material_lib.inb_sf_mat,
        region=fw_score.region & ~fw.region & ~divertor_cut.region,
    )
    tungsten_cell = openmc.Cell(
        name="Blanket FW",
        fill=material_lib.inb_fw_mat,
        region=fw_tungsten.region & ~fw_score.region & ~divertor_cut.region,
    )
    bz_cell = openmc.Cell(
        name="test",
        fill=material_lib.inb_bz_mat,
        region=fw_breeding.region & ~fw_tungsten.region & ~divertor_cut.region,
    )
    manifold_cell = openmc.Cell(
        name="Blanket manifold",
        fill=material_lib.inb_mani_mat,
        region=fw_manifold.region & ~fw_breeding.region & ~divertor_cut.region,
    )
    vv_cell = openmc.Cell(
        name="Vacuum vessel",
        fill=material_lib.inb_vv_mat,
        region=fw_vv.region & ~fw_manifold.region & ~divertor_cut.region,
    )
    inboard = LayerCells(
        [bz_cell], [tungsten_cell], [manifold_cell], [scoring_cell], [vv_cell]
    )

    plasma_cell = openmc.Cell(
        name="Plasma",
        region=fw.region,
    )

    cells = Cells(
        bore=bore,
        tf_coil=tf_coil,
        plasma=plasma_cell,
        outer_container=outer_container,
        divertor=divertor_cell,
        inboard=inboard,
    )

    # Create universe
    universe = openmc.Universe(cells=cells.get_cells())
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return cells, universe
