# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
EUDEMO builder for blanket
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder, Component, ComponentManager
from bluemira.base.components import PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    circular_pattern_component,
    get_n_sectors,
    pattern_revolved_silhouette,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.geometry.tools import boolean_cut, make_polygon, slice_shape
from eudemo.tools import get_inner_cut_point


class Blanket(ComponentManager):
    """
    Wrapper around a Blanket component tree.
    """


@dataclass
class BlanketBuilderParams(ParameterFrame):
    """
    Blanket builder parameters
    """

    n_TF: Parameter[int]
    n_bb_inboard: Parameter[int]
    n_bb_outboard: Parameter[int]
    c_rm: Parameter[float]


class BlanketBuilder(Builder):
    """
    Blanket builder

    Parameters
    ----------
    params
        the parameter frame
    build_config
        the build config
    blanket_silhouette
        breeding blanket silhouette
    r_inner_cut
        Cut radius on the plasma-facing surface
    cut_angle
        Cut plane angle (off from vertical) [degrees]

    """

    BB = "BB"
    IBS = "IBS"
    OBS = "OBS"
    param_cls: Type[BlanketBuilderParams] = BlanketBuilderParams

    def __init__(
        self,
        params: Union[BlanketBuilderParams, Dict],
        build_config: Dict,
        blanket_silhouette: BluemiraFace,
        r_inner_cut: float,
        cut_angle: float,
    ):
        super().__init__(params, build_config)
        self.silhouette = blanket_silhouette
        self.r_inner_cut = r_inner_cut
        self.cut_angle = cut_angle

    def build(self) -> Component:
        """
        Build the blanket component.
        """
        ibs_silhouette, obs_silhouette = self.segment_blanket()
        segments = self.get_segments(ibs_silhouette, obs_silhouette)

        return self.component_tree(
            xz=[self.build_xz(ibs_silhouette, obs_silhouette)],
            xy=self.build_xy(segments),
            xyz=self.build_xyz(segments),
        )

    def build_xz(self, ibs_silhouette: BluemiraFace, obs_silhouette: BluemiraFace):
        """
        Build the x-z components of the blanket.
        """
        ibs = PhysicalComponent(self.IBS, ibs_silhouette)
        ibs.plot_options.face_options["color"] = BLUE_PALETTE[self.BB][0]
        obs = PhysicalComponent(self.OBS, obs_silhouette)
        obs.plot_options.face_options["color"] = BLUE_PALETTE[self.BB][1]
        return Component(self.BB, children=[ibs, obs])

    def build_xy(self, segments: List[PhysicalComponent]):
        """
        Build the x-y components of the blanket.
        """
        xy_plane = BluemiraPlacement.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])

        slices = []
        for i, segment in enumerate(segments):
            single_slice = PhysicalComponent(
                segment.name, BluemiraFace(slice_shape(segment.shape, xy_plane)[0])
            )
            single_slice.plot_options.face_options["color"] = BLUE_PALETTE[self.BB][i]
            slices.append(single_slice)

        return circular_pattern_component(
            Component(self.BB, children=slices), self.params.n_TF.value
        )

    def build_xyz(self, segments: List[PhysicalComponent], degree: float = 360.0):
        """
        Build the x-y-z components of the blanket.
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        # TODO: Add blanket cuts properly in 3-D
        return circular_pattern_component(
            Component(self.BB, children=segments),
            n_sectors,
            degree=sector_degree * n_sectors,
        )

    def get_segments(self, ibs_silhouette: BluemiraFace, obs_silhouette: BluemiraFace):
        """
        Create segments of the blanket from inboard and outboard silhouettes
        """
        ibs_shapes = pattern_revolved_silhouette(
            ibs_silhouette,
            self.params.n_bb_inboard.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        obs_shapes = pattern_revolved_silhouette(
            obs_silhouette,
            self.params.n_bb_outboard.value,
            self.params.n_TF.value,
            self.params.c_rm.value,
        )

        segments = []
        for name, base_no, bs_shape in [
            [self.IBS, 0, ibs_shapes],
            [self.OBS, self.params.n_bb_inboard.value + 1, obs_shapes],
        ]:
            for no, shape in enumerate(bs_shape):
                segment = PhysicalComponent(f"{name}_{no}", shape)
                segment.display_cad_options.color = BLUE_PALETTE[self.BB][base_no + no]
                segments.append(segment)
        return segments

    def segment_blanket(self) -> Tuple[BluemiraFace]:
        """
        Segment the breeding blanket poloidal cross-section into inboard and outboard
        segment silhouettes.

        Returns
        -------
        Inboard blanket segment and Outboard blanket segment silhouette
        """
        # Make cutting geometry
        p0 = get_inner_cut_point(self.silhouette, self.r_inner_cut)
        p1 = [p0[0], 0, p0[2] + VERY_BIG]
        p2 = [p0[0] - self.params.c_rm.value, 0, p1[2]]
        p3 = [p2[0], 0, p0[2] - np.sqrt(2) * self.params.c_rm.value]
        cut_zone = BluemiraFace(make_polygon([p0, p1, p2, p3], closed=True))
        if self.cut_angle != 0.0:
            cut_zone.rotate(base=p0, direction=(0, -1, 0), degree=self.cut_angle)

        # Do cut
        cut_result = boolean_cut(self.silhouette, cut_zone)
        if len(cut_result) < 2:
            raise BuilderError(
                f"BB poloidal segmentation only returning {len(cut_result)} faces."
            )
        if len(cut_result) > 2:
            bluemira_warn(
                f"The BB poloidal segmentation operation returned more than 2 faces ({len(cut_result)}); only taking the first two..."
            )
        return sorted(cut_result, key=lambda x: x.center_of_mass[0])[:2]
