# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Create specific materials from known blanket data."""

from dataclasses import dataclass

from bluemira.base.constants import raw_uc
from bluemira.materials.neutronics import (
    BlanketType,
    BreederTypeParameters,
    _make_dcll_mats,
    _make_hcpb_mats,
    _make_wcll_mats,
    duplicate_mat_as,
    eurofer_mat,
)
from bluemira.radiation_transport.neutronics.materials import NeutronicsMaterials


@dataclass(frozen=True)
class TokamakGeometry:
    """The thickness measurements for all of the generic components of the tokamak.

    Parameters
    ----------
    inb_fw_thick:
        inboard first wall thickness [m]
    inb_bz_thick:
        inboard breeding zone thickness [m]
    inb_mnfld_thick:
        inboard manifold thickness [m]
    inb_vv_thick:
        inboard vacuum vessel thickness [m]
    tf_thick:
        toroidal field coil thickness [m]
    outb_fw_thick:
        outboard first wall thickness [m]
    outb_bz_thick:
        outboard breeding zone thickness [m]
    outb_mnfld_thick:
        outboard manifold thickness [m]
    outb_vv_thick:
        outboard vacuum vessel thickness [m]
    inb_gap:
        inboard gap [m]
    """

    inb_fw_thick: float
    inb_bz_thick: float
    inb_mnfld_thick: float
    inb_vv_thick: float
    tf_thick: float
    outb_fw_thick: float
    outb_bz_thick: float
    outb_mnfld_thick: float
    outb_vv_thick: float
    inb_gap: float


def get_preset_physical_properties(
    blanket_type: str | BlanketType,
) -> tuple[BreederTypeParameters, TokamakGeometry]:
    """
    Works as a switch-case for choosing the tokamak geometry
    and blankets for a given blanket type.
    The allowed list of blanket types are specified in BlanketType.
    Currently, the blanket types with pre-populated data in this function are:
    {'wcll', 'dcll', 'hcpb'}

    Returns
    -------
    breeder_materials:
        breeder blanket materials
    tokamak_geometry:
        tokamak geometry parameters
    """
    blanket_type = BlanketType(blanket_type)

    breeder_materials = BreederTypeParameters(
        blanket_type=blanket_type,
        enrichment_fraction_Li6=0.60,
    )

    # Geometry variables

    # Break down from here.
    # Paper inboard build ---
    # Nuclear analyses of solid breeder blanket options for DEMO:
    # Status,challenges and outlook,
    # Pereslavtsev, 2019
    #
    # 0.400,      # TF Coil inner
    # 0.200,      # gap                  from figures
    # 0.060,       # VV steel wall
    # 0.480,      # VV
    # 0.060,       # VV steel wall
    # 0.020,       # gap                  from figures
    # 0.350,      # Back Supporting Structure
    # 0.060,       # Back Wall and Gas Collectors   Back wall = 3.0
    # 0.350,      # breeder zone
    # 0.022        # fw and armour

    shared_building_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": 0.2,  # [m]
        "inb_vv_thick": 0.6,  # [m]
        "tf_thick": 0.4,  # [m]
        "outb_vv_thick": 0.6,  # [m]
    }
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.378,  # [m]
            inb_mnfld_thick=0.435,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.538,  # [m]
            outb_mnfld_thick=0.429,  # [m]
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.022,  # [m]
            inb_bz_thick=0.300,  # [m]
            inb_mnfld_thick=0.178,  # [m]
            outb_fw_thick=0.022,  # [m]
            outb_bz_thick=0.640,  # [m]
            outb_mnfld_thick=0.248,  # [m]
        )
    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.460,  # [m]
            inb_mnfld_thick=0.560,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.460,  # [m]
            outb_mnfld_thick=0.560,  # [m]
        )
    return breeder_materials, tokamak_geometry


def create_materials(
    breeder_materials: BreederTypeParameters,
) -> NeutronicsMaterials:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'enrichment_fraction_Li6'

    Returns
    -------
    :
        Materials used along with those blankets.
    """
    return _create_from_blanket_type(
        breeder_materials.blanket_type,
        raw_uc(breeder_materials.enrichment_fraction_Li6, "", "%"),
    )


def _create_from_blanket_type(
    blanket_type: BlanketType, li_enrich_ao: float
) -> NeutronicsMaterials:
    """Create Materials Library by specifying just the blanket type

    Parameters
    ----------
    blanket_type:
        the blanket type
    li_enrich_ao:
        PERCENTAGE enrichment of Li6 (float between 0 - 100)

    Returns
    -------
    :
        Set of materials used for that specific type of reactor.
    """
    if blanket_type is BlanketType.DCLL:
        base_materials = _make_dcll_mats(li_enrich_ao)
    elif blanket_type is BlanketType.HCPB:
        base_materials = _make_hcpb_mats(li_enrich_ao)
    elif blanket_type is BlanketType.WCLL:
        base_materials = _make_wcll_mats(li_enrich_ao)
    return NeutronicsMaterials(
        inb_vv_mat=base_materials.inb_vv_mat,
        inb_fw_mat=base_materials.inb_fw_mat,
        inb_bz_mat=base_materials.inb_bz_mat,
        inb_mani_mat=base_materials.inb_mani_mat,
        divertor_mat=base_materials.divertor_mat,
        div_fw_mat=base_materials.div_fw_mat,
        outb_fw_mat=duplicate_mat_as(base_materials.inb_fw_mat, "outb_first_wall", 201),
        outb_bz_mat=duplicate_mat_as(
            base_materials.inb_bz_mat, "outb_breeder_zone", 202
        ),
        outb_mani_mat=duplicate_mat_as(
            base_materials.inb_mani_mat, "outb_manifold", 203
        ),
        outb_vv_mat=duplicate_mat_as(
            base_materials.inb_vv_mat, "outb_vacuum_vessel", 204
        ),
        tf_coil_mat=duplicate_mat_as(eurofer_mat, "tf_coil", 401),
        container_mat=duplicate_mat_as(base_materials.inb_vv_mat, "container", 501),
        # surfaces
        inb_sf_mat=duplicate_mat_as(eurofer_mat, "inb_sf", 601),
        outb_sf_mat=duplicate_mat_as(eurofer_mat, "outb_sf", 602),
        div_sf_mat=duplicate_mat_as(eurofer_mat, "div_sf", 603),
        # TODO @OceanNuclear: get shield material
        # 3659
        rad_shield=duplicate_mat_as(eurofer_mat, "div_sf", 604),
    )
