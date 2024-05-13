# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC CSG neutronics materials"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum, auto
from typing import TYPE_CHECKING

import openmc

if TYPE_CHECKING:
    from pathlib import Path

    from bluemira.radiation_transport.neutronics.materials import NeutronicsMaterials


class CellType(Enum):
    """
    The five layers of the blanket as used in the neutronics simulation.

    Variables
    ---------
    Surface
        The surface layer of the first wall.
    First wall
        Typically made of tungsten or Eurofer
    BreedingZone
        Where tritium is bred
    Manifold
        The pipe works and supporting structure
    VacuumVessel
        The vacuum vessel keeping the plasma from mixing with outside air.
    RadiationShield
        The radiation shield surrounding the reactor, also called a bio shield.

    """

    BlanketSurface = auto()
    BlanketFirstWall = auto()
    BlanketBreedingZone = auto()
    BlanketManifold = auto()
    VacuumVessel = auto()
    DivertorBulk = auto()
    DivertorFirstWall = auto()
    DivertorSurface = auto()
    TFCoil = auto()
    CSCoil = auto()
    RadiationShield = auto()


@dataclass
class MaterialsLibrary:
    """A dictionary of materials according to the type of blanket used"""

    inb_vv_mat: openmc.Material
    inb_fw_mat: openmc.Material
    inb_bz_mat: openmc.Material
    inb_mani_mat: openmc.Material
    divertor_mat: openmc.Material
    div_fw_mat: openmc.Material
    outb_fw_mat: openmc.Material
    outb_bz_mat: openmc.Material
    outb_mani_mat: openmc.Material
    outb_vv_mat: openmc.Material
    tf_coil_mat: openmc.Material
    container_mat: openmc.Material
    inb_sf_mat: openmc.Material
    outb_sf_mat: openmc.Material
    div_sf_mat: openmc.Material

    @classmethod
    def from_neutronics_materials(cls, materials_lib: NeutronicsMaterials):
        """Initialise from neutronics materials"""
        return cls(**{
            field.name: getattr(materials_lib, field.name).to_openmc_material()
            for field in fields(materials_lib)
        })

    def match_material(self, cell_type: CellType, *, inboard: bool = False):
        """Choose the appropriate blanket material for the given blanket cell type."""
        match cell_type:
            case CellType.BlanketSurface:
                return self.inb_sf_mat if inboard else self.outb_sf_mat
            case CellType.BlanketFirstWall:
                return self.inb_fw_mat if inboard else self.outb_fw_mat
            case CellType.BlanketBreedingZone:
                return self.inb_bz_mat if inboard else self.outb_bz_mat
            case CellType.BlanketManifold:
                return self.inb_mani_mat if inboard else self.outb_mani_mat
            case CellType.VacuumVessel:
                return self.inb_vv_mat if inboard else self.outb_vv_mat
            case CellType.CSCoil:
                return self.container_mat
            case CellType.TFCoil:
                return self.tf_coil_mat
            case CellType.DivertorBulk:
                return self.divertor_mat
            case CellType.DivertorFirstWall:
                return self.div_fw_mat
            case CellType.DivertorSurface:
                return self.div_sf_mat

    def export_to_xml(self, path: str | Path = "materials.xml"):
        """Exports material defintions to xml"""
        return openmc.Materials(asdict(self).values()).export_to_xml(path)
