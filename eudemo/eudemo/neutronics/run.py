# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.neutronics.solver import (
    OpenMCNeutronicsSolver,
)
from bluemira.neutronics.sources import make_pps_source

if TYPE_CHECKING:
    from collections.abc import Callable

    import openmc.source

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.geometry.coordinates import Coordinates
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.params import PlasmaSourceParameters


def run_neutronics(
    params: dict | ParameterFrame,
    build_config: dict,
    blanket_panel_points: Coordinates,
    blanket_outer_boundary: BluemiraWire,
    divertor_wire: BluemiraWire,
    vv_wire: BluemiraWire,
    source: Callable[[PlasmaSourceParameters], openmc.source.SourceBase] | None = None,
):
    """Runs the neutronics model"""
    obj = OpenMCNeutronicsSolver(
        params,
        build_config,
        blanket_wire=blanket_outer_boundary,
        divertor_wire=divertor_wire,
        vv_wire=vv_wire,
        source=source or make_pps_source,
    )
    res = obj.execute()
    return res
