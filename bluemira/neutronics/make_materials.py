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
"""Create the material sets for each type of reactor."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Union

import openmc
from openmc import Material

import bluemira.neutronics.materials_definition as md

if TYPE_CHECKING:
    from pathlib import Path


def duplicate_mat_as(mat_to_clone, new_id, new_name) -> Material:
    """Clones and renames an OpenMC material"""
    new_mat = mat_to_clone.clone()
    new_mat.id = new_id
    new_mat.name = new_name

    return new_mat


@dataclass
class ReactorBaseMaterials:
    """Minimum set of materials that can create a tokamak.
    The rest can be populated by duplication using a priori knowledge,
    e.g. inobard material = outboard material etc.
    """

    inb_vv_mat: Material
    inb_fw_mat: Material
    inb_bz_mat: Material
    inb_mani_mat: Material
    divertor_mat: Material
    div_fw_mat: Material


def _make_dcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a dcll blanket.
    Parameter
    ---------
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        (to be parsed as argument to openmc.Material.add_element)
    Divertor definition from Neutronic analyses of the preliminary
    design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    Using Eurofer instead of SS316LN
    """
    inb_vv_mat = Material.mix_materials(
        name="inb_vacuum_vessel",
        materials=[md.eurofer_mat, md.water_mat],
        fracs=[0.8, 0.2],
        percent_type="vo",
    )
    inb_vv_mat.id = 104

    # Making first wall
    inb_fw_mat = Material.mix_materials(
        name="inb_first_wall",
        materials=[
            md.tungsten_mat,
            md.eurofer_mat,
            md.he_cool_mat,
            md.lined_euro_mat,
        ],
        fracs=[2.0 / 27.0, 1.5 / 27.0, 12.0 / 27.0, 11.5 / 27.0],
        percent_type="vo",
    )
    inb_fw_mat.id = 101

    # Making blanket
    _PbLi_mat = md.make_PbLi_mat(li_enrich_ao)
    inb_bz_mat = Material.mix_materials(
        name="inb_breeder_zone",
        materials=[md.lined_euro_mat, _PbLi_mat],
        fracs=[0.0605 + 0.9395 * 0.05, 0.9395 * 0.95],
        percent_type="vo",
    )
    inb_bz_mat.id = 102

    inb_mani_mat = Material.mix_materials(
        name="inb_manifold",
        materials=[md.eurofer_mat, inb_bz_mat],
        fracs=[0.573, 0.426],
        percent_type="vo",
    )
    inb_mani_mat.id = 103

    # Making divertor
    divertor_mat = duplicate_mat_as(inb_vv_mat, 301, "divertor")
    div_fw_mat = duplicate_mat_as(inb_fw_mat, 302, "div_first_wall")
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


def _make_hcpb_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for an hcpb blanket.
    Parameter
    ---------
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        (to be parsed as argument to openmc.Material.add_element)
    HCPB Design Report, 26/07/2019
    WPBB-DEL-BB-1.2.1-T005-D001
    """
    inb_vv_mat = Material.mix_materials(
        name="inb_vacuum_vessel",  # optional name of homogeneous material
        materials=[md.eurofer_mat, md.water_mat],
        fracs=[0.6, 0.4],
        percent_type="vo",
    )
    inb_vv_mat.id = 104

    # Making first wall
    inb_fw_mat = Material.mix_materials(
        name="inb_first_wall",  # optional name of homogeneous material
        materials=[md.tungsten_mat, md.eurofer_mat, md.he_cool_mat],
        fracs=[2.0 / 27.0, 25.0 * 0.573 / 27.0, 25.0 * 0.427 / 27.0],
        percent_type="vo",
    )
    inb_fw_mat.id = 101

    # Making blanket
    structural_fraction_vo = 0.128
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103  # 0.163
    helium_fraction_vo = 0.276  # 0.062

    inb_bz_mat = Material.mix_materials(
        name="inb_breeder_zone",
        materials=[
            md.eurofer_mat,
            md.Be12Ti_mat,
            md.make_KALOS_ACB_mat(li_enrich_ao),
            md.he_cool_mat,
        ],
        fracs=[
            structural_fraction_vo,
            multiplier_fraction_vo,
            breeder_fraction_vo,
            helium_fraction_vo,
        ],
        percent_type="vo",
    )
    inb_bz_mat.id = 102

    inb_mani_mat = Material.mix_materials(
        name="inb_manifold",
        materials=[
            md.eurofer_mat,
            md.make_KALOS_ACB_mat(li_enrich_ao),
            md.he_cool_mat,
        ],
        fracs=[0.4724, 0.0241, 0.5035],
        percent_type="vo",
    )
    inb_mani_mat.id = 103

    # Making divertor
    divertor_mat = duplicate_mat_as(inb_vv_mat, 301, "divertor")
    div_fw_mat = Material.mix_materials(
        name="div_first_wall",
        materials=[md.tungsten_mat, md.water_mat, md.eurofer_mat],
        fracs=[16.0 / 25.0, 4.5 / 25.0, 4.5 / 25.0],
        percent_type="vo",
    )
    div_fw_mat.id = 302
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


def _make_wcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a wcll blanket
    Parameter
    ---------
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        (to be parsed as argument to openmc.Material.add_element)
    Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion,
        WPBB-DEL-BB-3.2.1-T005-D001, June 2019.
    """
    _PbLi_mat = md.make_PbLi_mat(li_enrich_ao)

    # Divertor definition from Neutronic analyses of the preliminary
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    # Using Eurofer instead of SS316LN
    inb_vv_mat = Material.mix_materials(
        name="inb_vacuum_vessel",
        materials=[md.eurofer_mat, md.water_mat],
        fracs=[0.6, 0.4],
        percent_type="vo",
    )
    inb_vv_mat.id = 104

    # Making first wall
    inb_fw_mat = Material.mix_materials(
        name="inb_first_wall",
        materials=[md.tungsten_mat, md.water_mat, md.eurofer_mat],
        fracs=[0.0766, 0.1321, 0.7913],
        percent_type="vo",
    )
    inb_fw_mat.id = 101

    # Making blanket
    inb_bz_mat = Material.mix_materials(
        name="inb_breeder_zone",
        materials=[
            md.tungsten_mat,
            _PbLi_mat,
            md.water_mat,
            md.eurofer_mat,
        ],
        fracs=[0.0004, 0.8238, 0.0176, 0.1582],
        percent_type="vo",
    )
    inb_bz_mat.id = 102

    inb_mani_mat = Material.mix_materials(
        name="inb_manifold",
        materials=[_PbLi_mat, md.water_mat, md.eurofer_mat],
        fracs=[0.2129, 0.2514, 0.5357],
        percent_type="vo",
    )
    inb_mani_mat.id = 103

    # Making divertor
    divertor_mat = duplicate_mat_as(md.eurofer_mat, 301, "divertor")
    div_fw_mat = duplicate_mat_as(inb_fw_mat, 302, "div_first_wall")
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


class BlanketType(Enum):
    """Types of allowed blankets, named by their acronyms."""

    DCLL = auto()
    HCPB = auto()
    WCLL = auto()


@dataclass
class MaterialsLibrary:
    """A dictionary of materials according to the type of blanket used"""

    inb_vv_mat: Material
    inb_fw_mat: Material
    inb_bz_mat: Material
    inb_mani_mat: Material
    divertor_mat: Material
    div_fw_mat: Material
    outb_fw_mat: Material
    outb_bz_mat: Material
    outb_mani_mat: Material
    outb_vv_mat: Material
    tf_coil_mat: Material
    container_mat: Material
    inb_sf_mat: Material
    outb_sf_mat: Material
    div_sf_mat: Material

    @classmethod
    def create_from_blanket_type(
        cls, blanket_type: BlanketType, li_enrich_ao: float
    ) -> MaterialsLibrary:
        """Create from blanket type"""
        if blanket_type is BlanketType.DCLL:
            base_materials = _make_dcll_mats(li_enrich_ao)
        elif blanket_type is BlanketType.HCPB:
            base_materials = _make_hcpb_mats(li_enrich_ao)
        elif blanket_type is BlanketType.WCLL:
            base_materials = _make_wcll_mats(li_enrich_ao)
        return cls(
            inb_vv_mat=base_materials.inb_vv_mat,
            inb_fw_mat=base_materials.inb_fw_mat,
            inb_bz_mat=base_materials.inb_bz_mat,
            inb_mani_mat=base_materials.inb_mani_mat,
            divertor_mat=base_materials.divertor_mat,
            div_fw_mat=base_materials.div_fw_mat,
            outb_fw_mat=duplicate_mat_as(
                base_materials.inb_fw_mat, 201, "outb_first_wall"
            ),
            outb_bz_mat=duplicate_mat_as(
                base_materials.inb_bz_mat, 202, "outb_breeder_zone"
            ),
            outb_mani_mat=duplicate_mat_as(
                base_materials.inb_mani_mat, 203, "outb_manifold"
            ),
            outb_vv_mat=duplicate_mat_as(
                base_materials.inb_vv_mat, 204, "outb_vacuum_vessel"
            ),
            tf_coil_mat=duplicate_mat_as(md.eurofer_mat, 401, "tf_coil"),
            container_mat=duplicate_mat_as(base_materials.inb_vv_mat, 501, "container"),
            # surfaces
            inb_sf_mat=duplicate_mat_as(md.eurofer_mat, 601, "inb_sf"),
            outb_sf_mat=duplicate_mat_as(md.eurofer_mat, 602, "outb_sf"),
            div_sf_mat=duplicate_mat_as(md.eurofer_mat, 603, "div_sf"),
        )

    def export(self, path: Union[str, Path] = "materials.xml"):
        """Exports material defintions to xml"""
        material_list = openmc.Materials(dataclasses.asdict(self).values())
        return material_list.export_to_xml(path)
