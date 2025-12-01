# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Create specific materials from known blanket data."""

from dataclasses import dataclass

from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.materials.neutronics import (
    EUROFER_MAT,
    BlanketType,
    _make_dcll_mats,
    _make_hcpb_mats,
    _make_wcll_mats,
    duplicate_mat_as,
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


def get_preset_geometry(params: ParameterFrame) -> TokamakGeometry:
    """
    Get the tokamak geometry.

    Returns
    -------
    tokamak_geometry:
        tokamak geometry parameters

    Raises
    ------
    ValueError
        If the thickness of the sub-layers is incompatible with the totoal
        blanket thickness.
    """
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
    r_vv_ib_in = params.get_values("r_vv_ib_in")[0]
    r_tf_in = params.get_values("r_tf_in")[0]
    r_tf_inboard_out = params.get_values("r_tf_inboard_out")[0]
    tk_bb_ib = params.get_values("tk_bb_ib")[0]
    tk_bb_ob = params.get_values("tk_bb_ob")[0]

    ib_fw_thick = params.get_values("tk_bb_fw_ib")[0]
    ib_bz_thick = params.get_values("tk_bb_bz_ib")[0]
    ob_fw_thick = params.get_values("tk_bb_fw_ob")[0]
    ob_bz_thick = params.get_values("tk_bb_bz_ob")[0]

    ib_mnfld_thick = tk_bb_ib - ib_fw_thick - ib_bz_thick
    ob_mnfld_thick = tk_bb_ob - ob_fw_thick - ob_bz_thick

    if ib_mnfld_thick < 0.0:
        raise ValueError(
            f"Inboard manifold thickness is negative: {ib_mnfld_thick:.3f} m. "
            "Please check blanket thickness parameters."
        )
    if ob_mnfld_thick < 0.0:
        raise ValueError(
            f"Outboard manifold thickness is negative: {ob_mnfld_thick:.3f} m. "
            "Please check blanket thickness parameters."
        )

    return TokamakGeometry(
        inb_fw_thick=ib_fw_thick,
        inb_bz_thick=ib_bz_thick,
        inb_mnfld_thick=ib_mnfld_thick,
        inb_vv_thick=params.get_values("tk_vv_in")[0],
        outb_vv_thick=params.get_values("tk_vv_out")[0],
        tf_thick=r_tf_inboard_out - r_tf_in,
        outb_fw_thick=ob_fw_thick,
        outb_bz_thick=ob_bz_thick,
        outb_mnfld_thick=ob_mnfld_thick,
        inb_gap=r_tf_in - r_vv_ib_in,
    )


def create_materials(blanket_type: BlanketType) -> NeutronicsMaterials:
    """
    Create Materials Library by specifying just the blanket type

    Parameters
    ----------
    blanket_type:
        the blanket type

    Returns
    -------
    :
        Set of materials used for that specific type of reactor.
    """
    match blanket_type:
        case BlanketType.HCPB:
            li_enrich_ao = 0.6
        case _:
            li_enrich_ao = 0.9

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
        tf_coil_mat=duplicate_mat_as(EUROFER_MAT, "tf_coil", 401),
        container_mat=duplicate_mat_as(base_materials.inb_vv_mat, "container", 501),
        # surfaces
        inb_sf_mat=duplicate_mat_as(EUROFER_MAT, "inb_sf", 601),
        outb_sf_mat=duplicate_mat_as(EUROFER_MAT, "outb_sf", 602),
        div_sf_mat=duplicate_mat_as(EUROFER_MAT, "div_sf", 603),
        # TODO @OceanNuclear: get shield material
        # 3659
        rad_shield=duplicate_mat_as(EUROFER_MAT, "rad_shield", 604),
    )
