# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil structure stuff
"""

from dataclasses import dataclass

from bluemira.base.components import Component
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.builders.coil_supports import (
    ITERGravitySupportBuilder,
    OISBuilder,
    PFCoilSupportBuilder,
    StraightOISDesigner,
)


@dataclass
class CoilStructuresParameters(ParameterFrame):
    """
    Parameters for the coil structures
    """

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    tf_wp_width: Parameter[float]

    # OIS
    tk_ois: Parameter[float]
    g_ois_tf_edge: Parameter[float]
    min_OIS_length: Parameter[float]

    # PF
    pf_s_tk_plate: Parameter[float]
    pf_s_n_plate: Parameter[int]
    pf_s_g: Parameter[float]

    # GS
    x_g_support: Parameter[float]
    z_gs: Parameter[float]
    tf_gs_tk_plate: Parameter[float]
    tf_gs_g_plate: Parameter[float]
    tf_gs_base_depth: Parameter[float]


def build_coil_structures_component(
    params, build_config, tf_coil_xz_face, pf_coil_xz_wires, pf_coil_keep_out_zones
):
    """
    Build the coil structures super-component.
    """
    params = make_parameter_frame(params, CoilStructuresParameters)
    ois_designer = StraightOISDesigner(
        params, build_config, tf_coil_xz_face, pf_coil_keep_out_zones
    )
    ois_xz_profiles = ois_designer.run()
    ois_builder = OISBuilder(params, build_config, ois_xz_profiles)
    ois_component = ois_builder.build()

    tf_koz = tf_coil_xz_face.boundary[0]
    support_components = []
    for i, pf_coil in enumerate(pf_coil_xz_wires):
        bc = {**build_config, "support_number": str(i)}
        pf_support_builder = PFCoilSupportBuilder(params, bc, tf_koz, pf_coil)
        support_components.append(pf_support_builder.build())

    pf_support_component = Component("PF supports", children=support_components)

    gs_builder = ITERGravitySupportBuilder(params, build_config, tf_koz)
    gs_component = gs_builder.build()

    return Component(
        "Coil Structures", children=[ois_component, pf_support_component, gs_component]
    )
