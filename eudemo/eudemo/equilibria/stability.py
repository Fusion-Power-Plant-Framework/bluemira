# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from copy import deepcopy
from bluemira.base.parameter_frame._frame import ParameterFrame
from bluemira.equilibria.coils._grouping import CoilGroup
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.tools import offset_wire
from bluemira.geometry.wire import BluemiraWire


def run_vertical_stability_calculation(
    params: dict | ParameterFrame,
    build_config: dict,
    eq: Equilibrium,
    vv_outer_wire: BluemiraWire,
    vv_inner_wire: BluemiraWire,
):
    tk_shell = params.tk_vv_single_wall.value
    outer_shell_centreline = offset_wire(vv_outer_wire, -tk_shell)
    inner_shell_centreline = offset_wire(vv_inner_wire, tk_shell)
    outer_passives = make_coils_along_wire(outer_shell_centreline, tk_shell)
    inner_passives = make_coils_along_wire(inner_shell_centreline, tk_shell)
    eq = deepcopy(eq)
    eq.coilset.add_coil(CoilGroup([inner_passives, outer_passives]))
    m_s = calculate_rzip_stability_criterion(eq)
    params.update("m_s", m_s)