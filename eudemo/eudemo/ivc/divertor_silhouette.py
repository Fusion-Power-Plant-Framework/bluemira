# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Define builder for divertor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.base.reactor import ComponentManager
from bluemira.builders.divertor import DivertorBuilder, DivertorDesigner, LegPosition

if TYPE_CHECKING:
    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.equilibria import Equilibrium
    from bluemira.geometry.wire import BluemiraWire


class Divertor(ComponentManager):
    """
    Wrapper around a divertor component tree.
    """

    def silhouette(self) -> BluemiraWire:
        """
        Returns
        -------
        :
            A wire representing the divertor poloidal silhouette.
        """
        return (
            self.component()
            .get_component("xz")
            .get_component(DivertorBuilder.BODY)
            .shape.boundary[0]
        )


class DivertorSilhouetteDesigner(DivertorDesigner):
    """
    Designs the divertor silhouette to help design the divertor keep out zone

    Parameters
    ----------
    params:
        Divertor silhouette designer parameters
    equilibrium:
        The equilibrium to design around
    wall:
        wall boundary keep out zone (cut at divertor)

    """

    def __init__(
        self,
        params: dict | ParameterFrame,
        equilibrium: Equilibrium,
        wall: BluemiraWire,
    ):
        x_limits = (wall.start_point().x[0], wall.end_point().x[0])
        z_limits = (wall.start_point().z[0], wall.end_point().z[0])
        super().__init__(params, equilibrium, x_limits=x_limits, z_limits=z_limits)
        if self.equilibrium.is_double_null:
            raise NotImplementedError("Double Null divertor not implemented")

    def run(self) -> tuple[BluemiraWire, ...]:
        """
        Run method of DivertorSilhouetteDesigner


        Returns
        -------
        :
            Inner baffle
        :
            Inner target
        :
            Dome
        :
            Outer target
        :
            Outer baffle
        """
        # Build the targets for each separatrix leg
        inner_target = self._make_target(LegPosition.INNER, self.INNER_TARGET)
        outer_target = self._make_target(LegPosition.OUTER, self.OUTER_TARGET)

        # The inner target ends inside the private flux region (lower psi_norm)
        # The outer target ends outside the private flux region (higher psi_norm)
        inner_target_start, inner_target_end = self._get_wire_ends_by_psi(inner_target)
        outer_target_end, outer_target_start = self._get_wire_ends_by_psi(outer_target)

        # Build the dome based on target positions
        dome = self.make_dome(inner_target_end, outer_target_start, label=self.DOME)

        # Build the baffles
        inner_baffle = self.make_baffle(
            self.INNER_BAFFLE,
            target_baffle_join_point=inner_target_start,
            target_dome_join_point=inner_target_end,
        )
        outer_baffle = self.make_baffle(
            self.OUTER_BAFFLE,
            target_baffle_join_point=outer_target_end,
            target_dome_join_point=outer_target_start,
        )

        return inner_baffle, inner_target, dome, outer_target, outer_baffle
