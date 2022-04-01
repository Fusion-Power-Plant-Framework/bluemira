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
Define builder for blanket
"""

from typing import List, Optional

import numpy as np

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
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.tools import boolean_cut, make_polygon, slice_shape


class BlanketBuilder(Builder):
    """
    Build an EUDEMO blanket.
    """

    _required_params: List[str] = [
        "n_TF",
        "n_bb_inboard",
        "n_bb_outboard",
        "c_rm",
    ]

    _params: Configuration
    _silhouette: Optional[BluemiraFace] = None

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        blanket_silhouette: BluemiraFace,
    ):
        super().__init__(
            params,
            build_config,
            blanket_silhouette=blanket_silhouette,
        )

    def reinitialise(self, params, blanket_silhouette) -> None:
        """
        Reinitialise the parameters and boundary.

        Parameters
        ----------
        params: dict
            The new parameter values to initialise this builder against.
        """
        super().reinitialise(params)

        self._silhouette = blanket_silhouette

    def build(self) -> Component:
        """
        Build the blanket component.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        super().build()
        self._segment_blanket()

        component = Component(name=self.name)
        component.add_child(self.build_xz())
        component.add_child(self.build_xyz())
        component.add_child(self.build_xy())
        return component

    def build_xz(self):
        """
        Build the x-z components of the blanket.
        """
        ibs = PhysicalComponent("IBS", self._ibs_silhouette)
        ibs.plot_options.face_options["color"] = BLUE_PALETTE["BB"][0]
        obs = PhysicalComponent("OBS", self._obs_silhouette)
        obs.plot_options.face_options["color"] = BLUE_PALETTE["BB"][1]
        component = Component("xz", children=[ibs, obs])
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xy(self):
        """
        Build the x-y components of the blanket.
        """
        component = Component("xy")

        xy_plane = BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])

        slices = []
        for i, segment in enumerate(self._segments):
            slice = PhysicalComponent(
                segment.name, BluemiraFace(slice_shape(segment.shape, xy_plane)[0])
            )
            slice.plot_options.face_options["color"] = BLUE_PALETTE["BB"][i]
            slices.append(slice)

        sector = Component("sector", children=slices)

        sectors = circular_pattern_component(sector, self._params.n_TF.value)
        component.add_children(sectors, merge_trees=True)

        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xyz(self, degree=360.0):
        """
        Build the x-y-z components of the blanket.
        """
        n_sector_draw = max(1, int(degree // (360 // self._params.n_TF.value)))
        degree = (360.0 / self._params.n_TF.value) * n_sector_draw

        ibs_shapes = pattern_revolved_silhouette(
            self._ibs_silhouette,
            self._params.n_bb_inboard.value,
            self._params.n_TF.value,
            self._params.c_rm.value,
        )

        segments = []
        for i, shape in enumerate(ibs_shapes):
            segment = PhysicalComponent(f"IBS_{i}", shape)
            segment.display_cad_options.color = BLUE_PALETTE["BB"][i]
            segments.append(segment)

        obs_shapes = pattern_revolved_silhouette(
            self._obs_silhouette,
            self._params.n_bb_outboard.value,
            self._params.n_TF.value,
            self._params.c_rm.value,
        )
        for i, shape in enumerate(obs_shapes):
            segment = PhysicalComponent(f"OBS_{i}", shape)
            segment.display_cad_options.color = BLUE_PALETTE["BB"][
                self._params.n_bb_inboard.value + i + 1
            ]
            segments.append(segment)
        self._segments = segments
        # TODO: Add blanket cuts properly in 3-D

        sector = Component("segments", children=segments)
        sectors = circular_pattern_component(sector, n_sector_draw, degree=degree)

        component = Component("xyz", children=sectors)
        return component

    def _segment_blanket(self):
        """
        Cut the blanket silhouette into segment silhouettes. Simple vertical cut for now.
        """
        bb = self._silhouette.bounding_box
        x_mid = 0.5 * (bb.x_min + bb.x_max)
        delta = 0.5 * self._params.c_rm.value
        x = np.array([x_mid - delta, x_mid + delta, x_mid + delta, x_mid - delta])
        off = 0.1
        z = np.array([0, 0, bb.z_max + off, bb.z_max + off])
        cut_wire = make_polygon({"x": x, "y": 0, "z": z}, closed=True)
        cut_face = BluemiraFace(cut_wire)

        segments = boolean_cut(self._silhouette, cut_face)
        segments.sort(key=lambda seg: seg.bounding_box.x_min)
        ibs, obs = segments
        self._ibs_silhouette = ibs
        self._obs_silhouette = obs
