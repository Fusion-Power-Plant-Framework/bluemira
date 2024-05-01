# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from __future__ import annotations

from os import close
from typing import TYPE_CHECKING

from scipy.spatial.distance import euclidean

from bluemira.display import plot_2d, plot_3d, show_cad  # noqa: F401
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.solver import (
    OpenMCNeutronicsSolver,
)
from bluemira.neutronics.sources import make_pps_source

if TYPE_CHECKING:
    from collections.abc import Callable

    import openmc.source

    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.neutronics.params import PlasmaSourceParameters


def dedup_coordinates(
    a: Coordinates, b: Coordinates, dist_tol: float
) -> tuple[Coordinates, Coordinates]:
    """Merge two sets of coordinates."""
    new_coords = a.points
    close_points = []
    for point in b.points:
        tmp = [euclidean(point, p) for p in a.points]
        pass
        if all(d > dist_tol for d in tmp):
            new_coords.append(point)
        else:
            close_points.append(point)
    return Coordinates(new_coords), Coordinates(close_points)


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
    blanket_wire_pts = dedup_coordinates(
        blanket_ib_boundary.vertexes, blanket_ob_boundary.vertexes, 1
    )[0]
    blanket_wire = make_polygon(blanket_wire_pts)
    obj = OpenMCNeutronicsSolver(
        params,
        build_config,
        blanket_wire=blanket_wire,
        divertor_wire=divertor_wire,
        vv_wire=vv_wire,
        source=source or make_pps_source,
    )
    res = obj.execute()
    return res
