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


def _scale_blanket_thicknesses(
    blanket_tk: dict[str, float], total_thickness: float
) -> dict[str, float]:
    """
    Normalize the values of a blanket thicnkess dictionary to sum to 1, then scale
    by `total_thickness`.

    Returns
    -------
    :
        The scaled blanket thickness dictionary

    Raises
    ------
    ValueError
        If the original blanket values sum to 0.0
    """
    s = sum(blanket_tk.values())
    if s == 0:
        raise ValueError("Sum of dictionary values is zero; cannot normalize.")
    return {k: (v / s) * total_thickness for k, v in blanket_tk.items()}


def get_preset_geometry(
    params: ParameterFrame,
    blanket_type: BlanketType,
) -> TokamakGeometry:
    """
    Get the tokamak geometry according to a specified blanket type.

    Returns
    -------
    tokamak_geometry:
        tokamak geometry parameters

    Notes
    -----
    Blanket sub-component thicknesses are scaled according to design data from
    various sources.
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
    shared_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": r_tf_in - r_vv_ib_in,  # [m]
        "inb_vv_thick": params.get_values("tk_vv_in")[0],  # [m]
        "tf_thick": r_tf_inboard_out - r_tf_in,  # [m]
        "outb_vv_thick": params.get_values("tk_vv_out")[0],  # [m]
    }

    if blanket_type is BlanketType.WCLL:
        ib_blanket_geometry = {
            "inb_fw_thick": 0.027,
            "inb_bz_thick": 0.378,
            "inb_mnfld_thick": 0.435,
        }
        ob_blanket_geometry = {
            "outb_fw_thick": 0.027,
            "outb_bz_thick": 0.538,
            "outb_mnfld_thick": 0.429,
        }

    elif blanket_type is BlanketType.DCLL:
        ib_blanket_geometry = {
            "inb_fw_thick": 0.022,
            "inb_bz_thick": 0.300,
            "inb_mnfld_thick": 0.178,
        }
        ob_blanket_geometry = {
            "outb_fw_thick": 0.022,
            "outb_bz_thick": 0.640,
            "outb_mnfld_thick": 0.248,
        }

    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        # MC: This does not look right @Ocean... please put findable references
        # for these data
        ib_blanket_geometry = {
            "inb_fw_thick": 0.027,
            "inb_bz_thick": 0.460,
            "inb_mnfld_thick": 0.560,
        }

        ob_blanket_geometry = {
            "outb_fw_thick": 0.027,
            "outb_bz_thick": 0.460,
            "outb_mnfld_thick": 0.560,
        }
    ib_blanket_geometry = _scale_blanket_thicknesses(ib_blanket_geometry, tk_bb_ib)
    ob_blanket_geometry = _scale_blanket_thicknesses(ob_blanket_geometry, tk_bb_ob)
    return TokamakGeometry(
        **shared_geometry, **ib_blanket_geometry, **ob_blanket_geometry
    )


def create_materials(
    blanket_type: BlanketType,
) -> NeutronicsMaterials:
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
            li_enrich_ao = 60.0  # [%]
        case _:
            li_enrich_ao = 90.0  # [%]
>>>>>>> 47eb01aa4 (LAR fix neutronics radii (#4116))

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
