# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
EUDEMO builders related to blankets.
"""
from typing import Any, Dict

from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.builders.EUDEMO.tools import varied_offset
from bluemira.geometry.wire import BluemiraWire


class BlanketThicknessBuilder(Builder):
    """
    Build a blanket for the given wall shape using a variable thickness.

    .. code-block::

        blanket (Component)
        └── xz (Component)
            └── blanket_boundary (PhysicalComponent)
    """

    _required_params = [
        "tk_bb_ib",  # Inboard blanket thickness
        "tk_bb_ob",  # Outboard blanket thickness
    ]
    COMPONENT_BOUNDARY = "blanket_boundary"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        wall_shape: BluemiraWire,
        z_min: float,
    ):
        if not wall_shape.is_closed():
            raise BuilderError("Wall shape must be closed.")
        super().__init__(params, build_config)
        self._wall_shape = wall_shape
        self._z_min = z_min

    def reinitialise(self, params, **kwargs) -> None:
        """Reinitialise the parameters on this builder."""
        return super().reinitialise(params, **kwargs)

    def build(self) -> Component:
        """
        Build the component.
        """
        blanket = Component(self.name)
        blanket.add_child(self._build_xz())
        return blanket

    def _build_xz(self) -> Component:
        """Build the components in the xz-plane."""
        ib_offset_angle = 45  # degrees
        ob_offset_angle = 175  # degrees
        boundary_wire = varied_offset(
            self._wall_shape,
            self._params.tk_bb_ib,
            self._params.tk_bb_ob,
            ib_offset_angle,
            ob_offset_angle,
        )
        xz_component = Component("xz")
        boundary = PhysicalComponent(self.COMPONENT_BOUNDARY, boundary_wire)
        xz_component.add_child(boundary)
        return xz_component
