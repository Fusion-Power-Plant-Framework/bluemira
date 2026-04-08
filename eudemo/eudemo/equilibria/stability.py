# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.vertical_stability import (
    calculate_rzip_stability_criterion,
    make_coils_along_wire,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, offset_wire
from bluemira.geometry.wire import BluemiraWire


def run_vertical_stability_calculation(
    params: dict | ParameterFrame,
    build_config: dict,
    eq: Equilibrium,
    vv_outer_wire: BluemiraWire,
    vv_inner_wire: BluemiraWire,
    keep_out_zones: list[BluemiraFace] | None = None,
):
    bluemira_print("Running RZIp vertical stability calculation.")
    tk_shell = params.tk_vv_single_wall.value
    outer_shell_centreline = offset_wire(vv_outer_wire, -tk_shell)
    inner_shell_centreline = offset_wire(vv_inner_wire, tk_shell)
    # keep_out_zones = None  # TODO
    if keep_out_zones is not None:
        outer_shell_segments = boolean_cut(outer_shell_centreline, keep_out_zones)
        inner_shell_segments = boolean_cut(inner_shell_centreline, keep_out_zones)
    else:
        outer_shell_segments = [outer_shell_centreline]
        inner_shell_segments = [inner_shell_centreline]

    all_passives = []
    d_thickness = 2 * tk_shell  # TODO
    for segment in outer_shell_segments:
        all_passives.extend(
            make_coils_along_wire(
                segment, d_thickness, name_prefix="VV_outer_passive"
            )._coils
        )
    for segment in inner_shell_segments:
        all_passives.extend(
            make_coils_along_wire(
                segment, d_thickness, name_prefix="VV_inner_passive"
            )._coils
        )

    # new_coils = []
    # for coil in coilset.get_control_coils()._coils:
    #     n_quad = len(coil._quad_x)
    #     for i, (xi, zi) in enumerate(zip(coil._quad_x, coil._quad_z)):
    #         new_coils.append(
    #             Coil(
    #                 xi,
    #                 zi,
    #                 dx = coil._quad_dx[i],
    #                 dz = coil._quad_dz[i],
    #                 current=coil.current / n_quad,
    #                 ctype=coil.ctype,
    #                 name=f"{coil.name}_{i}",
    #                 discretisation=np.nan,
    #             )
    #         )
    # new_coilset = CoilSet(*new_coils)
    # new_coilset.control = [coil.name for coil in new_coils]
    eq.coilset.add_coil(*all_passives)
    eq._remap_greens()
    f_s = calculate_rzip_stability_criterion(eq)
    params.update({"m_s": {"value": f_s - 1.0, "source": "BLUEMIRA"}})
