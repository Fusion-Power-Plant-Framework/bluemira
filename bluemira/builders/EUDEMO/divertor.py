# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Define builder for divertor
"""

from typing import List, Optional

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder, Component
from bluemira.base.components import PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.builders.EUDEMO.tools import (
    circular_pattern_component,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire


class DivertorBuilder(Builder):
    """
    Build an EUDEMO divertor.
    """

    _required_params: List[str] = [
        "n_TF",
        "n_div_cassettes",
        "c_rm",
    ]

    _params: Configuration
    _silhouette: Optional[BluemiraWire] = None

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        divertor_silhouette: BluemiraFace,
    ):
        super().__init__(
            params,
            build_config,
            divertor_silhouette=divertor_silhouette,
        )

    def reinitialise(self, params, divertor_silhouette) -> None:
        """
        Reinitialise the parameters and boundary.

        Parameters
        ----------
        params: dict
            The new parameter values to initialise this builder against.
        """
        super().reinitialise(params)

        self._silhouette = divertor_silhouette

    def build(self) -> Component:
        """
        Build the divertor component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build()

        component = Component(name=self.name)
        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xz(self):
        """
        Build the x-z components of the divertor.
        """
        body = PhysicalComponent("body", self._silhouette)
        body.plot_options.face_options["color"] = BLUE_PALETTE["DIV"][0]
        component = Component("xz", children=[body])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the divertor.
        """
        component = Component("xy", children=[])
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0):
        """
        Build the x-y-z components of the divertor.
        """
        shapes = pattern_revolved_silhouette(
            self._silhouette,
            self._params.n_div_cassettes.value,
            self._params.n_TF.value,
            self._params.c_rm.value,
        )

        n_sector_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        degree = (360.0 / self._params.n_TF.value) * n_sector_draw

        segments = []
        for i, shape in enumerate(shapes):
            segment = PhysicalComponent(f"segment_{i}", shape)
            segment.display_cad_options.color = BLUE_PALETTE["DIV"][i]
            segments.append(segment)

        sector = Component("cassettes", children=segments)
        sectors = circular_pattern_component(sector, n_sector_draw, degree=degree)

        component = Component("xyz", children=sectors)
        return component
