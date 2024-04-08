# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create the material sets for each type of reactor."""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from openmc import Materials

import bluemira.neutronics.materials_definition as md
from bluemira.materials.mixtures import HomogenisedMixture, MixtureFraction

if TYPE_CHECKING:
    from pathlib import Path

    from openmc import Material

    from bluemira.materials.material import MassFractionMaterial


def duplicate_mat_as(
    mat_to_clone: MassFractionMaterial | HomogenisedMixture,
    new_id: int,
    new_name: str,
) -> MassFractionMaterial | HomogenisedMixture:
    """Clones and renames an OpenMC material"""
    new_mat = deepcopy(mat_to_clone)
    new_mat.material_id = new_id
    new_mat.name = new_name

    return new_mat


@dataclass
class ReactorBaseMaterials:
    """Minimum set of materials that can create a tokamak.
    The rest can be populated by duplication using a priori knowledge,
    e.g. inobard material = outboard material etc.
    """

    inb_vv_mat: HomogenisedMixture
    inb_fw_mat: HomogenisedMixture
    inb_bz_mat: HomogenisedMixture
    inb_mani_mat: HomogenisedMixture
    divertor_mat: HomogenisedMixture
    div_fw_mat: HomogenisedMixture


def _make_dcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a dcll blanket.

    Parameters
    ----------
    --_-------
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    Divertor definition from Neutronic analyses of the preliminary
    design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    Using Eurofer instead of SS316LN
    """
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",
        material_id=104,
        materials=[
            MixtureFraction(md.eurofer_mat, 0.8),
            MixtureFraction(md.water_mat, 0.2),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",
        material_id=101,
        materials=[
            MixtureFraction(md.tungsten_mat, 2.0 / 27.0),
            MixtureFraction(md.eurofer_mat, 1.5 / 27.0),
            MixtureFraction(md.he_cool_mat, 12.0 / 27.0),
            MixtureFraction(md.lined_euro_mat, 11.5 / 27.0),
        ],
        percent_type="vo",
    )

    # Making blanket
    _PbLi_mat = md.make_PbLi_mat(li_enrich_ao)
    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(md.lined_euro_mat, 0.0605 + 0.9395 * 0.05),
            MixtureFraction(_PbLi_mat, 0.9395 * 0.95),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(md.eurofer_mat, 0.573),
            MixtureFraction(inb_bz_mat, 0.426),
        ],  # 1% void
        percent_type="vo",
    )

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

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    HCPB Design Report, 26/07/2019
    WPBB-DEL-BB-1.2.1-T005-D001
    """
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",  # optional name of homogeneous material
        material_id=104,
        materials=[
            MixtureFraction(md.eurofer_mat, 0.6),
            MixtureFraction(md.water_mat, 0.4),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",  # optional name of homogeneous material
        material_id=101,
        materials=[
            MixtureFraction(md.tungsten_mat, 2.0 / 27.0),
            MixtureFraction(md.eurofer_mat, 25.0 * 0.573 / 27.0),
            MixtureFraction(md.he_cool_mat, 25.0 * 0.427 / 27.0),
        ],
        percent_type="vo",
    )

    # Making blanket
    structural_fraction_vo = 0.128
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103  # 0.163
    helium_fraction_vo = 0.276  # 0.062

    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(md.eurofer_mat, structural_fraction_vo),
            MixtureFraction(md.Be12Ti_mat, multiplier_fraction_vo),
            MixtureFraction(md.make_KALOS_ACB_mat(li_enrich_ao), breeder_fraction_vo),
            MixtureFraction(md.he_cool_mat, helium_fraction_vo),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(md.eurofer_mat, 0.4724),
            MixtureFraction(md.make_KALOS_ACB_mat(li_enrich_ao), 0.0241),
            MixtureFraction(md.he_cool_mat, 0.5035),
        ],
        percent_type="vo",
    )

    # Making divertor
    divertor_mat = duplicate_mat_as(inb_vv_mat, 301, "divertor")
    div_fw_mat = HomogenisedMixture(
        name="div_first_wall",
        material_id=302,
        materials=[
            MixtureFraction(md.tungsten_mat, 16.0 / 25.0),
            MixtureFraction(md.water_mat, 4.5 / 25.0),
            MixtureFraction(md.eurofer_mat, 4.5 / 25.0),
        ],
        percent_type="vo",
    )
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

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion,
    WPBB-DEL-BB-3.2.1-T005-D001, June 2019.
    """
    _PbLi_mat = md.make_PbLi_mat(li_enrich_ao)

    # Divertor definition from Neutronic analyses of the preliminary
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    # Using Eurofer instead of SS316LN
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",
        material_id=104,
        materials=[
            MixtureFraction(md.eurofer_mat, 0.6),
            MixtureFraction(md.water_mat, 0.4),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",
        material_id=101,
        materials=[
            MixtureFraction(md.tungsten_mat, 0.0766),
            MixtureFraction(md.water_mat, 0.1321),
            MixtureFraction(md.eurofer_mat, 0.7913),
        ],
        percent_type="vo",
    )

    # Making blanket
    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(md.tungsten_mat, 0.0004),
            MixtureFraction(_PbLi_mat, 0.8238),
            MixtureFraction(md.water_mat, 0.0176),
            MixtureFraction(md.eurofer_mat, 0.1582),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(_PbLi_mat, 0.2129),
            MixtureFraction(md.water_mat, 0.2514),
            MixtureFraction(md.eurofer_mat, 0.5357),
        ],
        percent_type="vo",
    )

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
        """Create Materials Library by specifying just the blanket type

        Parameters
        ----------
        blanket_type:
            the blanket type
        li_enrich_ao:
            PERCENTAGE enrichment of Li6 (float between 0 - 100)
        """
        if blanket_type is BlanketType.DCLL:
            base_materials = _make_dcll_mats(li_enrich_ao)
        elif blanket_type is BlanketType.HCPB:
            base_materials = _make_hcpb_mats(li_enrich_ao)
        elif blanket_type is BlanketType.WCLL:
            base_materials = _make_wcll_mats(li_enrich_ao)
        return cls(
            inb_vv_mat=base_materials.inb_vv_mat.to_openmc_material(),
            inb_fw_mat=base_materials.inb_fw_mat.to_openmc_material(),
            inb_bz_mat=base_materials.inb_bz_mat.to_openmc_material(),
            inb_mani_mat=base_materials.inb_mani_mat.to_openmc_material(),
            divertor_mat=base_materials.divertor_mat.to_openmc_material(),
            div_fw_mat=base_materials.div_fw_mat.to_openmc_material(),
            outb_fw_mat=duplicate_mat_as(
                base_materials.inb_fw_mat, 201, "outb_first_wall"
            ).to_openmc_material(),
            outb_bz_mat=duplicate_mat_as(
                base_materials.inb_bz_mat, 202, "outb_breeder_zone"
            ).to_openmc_material(),
            outb_mani_mat=duplicate_mat_as(
                base_materials.inb_mani_mat, 203, "outb_manifold"
            ).to_openmc_material(),
            outb_vv_mat=duplicate_mat_as(
                base_materials.inb_vv_mat, 204, "outb_vacuum_vessel"
            ).to_openmc_material(),
            tf_coil_mat=duplicate_mat_as(
                md.eurofer_mat, 401, "tf_coil"
            ).to_openmc_material(),
            container_mat=duplicate_mat_as(
                base_materials.inb_vv_mat, 501, "container"
            ).to_openmc_material(),
            # surfaces
            inb_sf_mat=duplicate_mat_as(
                md.eurofer_mat, 601, "inb_sf"
            ).to_openmc_material(),
            outb_sf_mat=duplicate_mat_as(
                md.eurofer_mat, 602, "outb_sf"
            ).to_openmc_material(),
            div_sf_mat=duplicate_mat_as(
                md.eurofer_mat, 603, "div_sf"
            ).to_openmc_material(),
        )

    def export(self, path: str | Path = "materials.xml"):
        """Exports material defintions to xml"""
        material_list = Materials(dataclasses.asdict(self).values())
        return material_list.export_to_xml(path)
