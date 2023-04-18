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
"""Designer for EUDEMO blankets."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.ivc.panelling import PanellingDesigner
from eudemo.tools import get_inner_cut_point


@dataclass
class BlanketDesignerParams(ParameterFrame):
    """EUDEMO blanket designer parameters for :class:`BlanketDesigner`."""

    n_TF: Parameter[int]
    """Number of TF coils."""
    n_bb_inboard: Parameter[int]
    """Number of inboard blanket segments."""
    n_bb_outboard: Parameter[int]
    """Number of outboard blanket segments."""
    c_rm: Parameter[float]
    """Remote maintenance clearance [m]."""
    fw_a_max: Parameter[float]
    """Maximum angle of rotation between adjacent panels [degrees]."""
    fw_dL_min: Parameter[float]  # noqa: N815
    """Minimum length for an individual panel [m]."""


class BlanketDesigner(Designer[Tuple[BluemiraFace, BluemiraFace]]):
    """Designer for an EUDEMO-style blanket."""

    param_cls = BlanketDesignerParams
    params: BlanketDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        blanket_boundary: BluemiraWire,
        blanket_silhouette: BluemiraFace,
        r_inner_cut: float,
        cut_angle: float,
        build_config: Optional[Dict] = None,
    ):
        super().__init__(params, build_config)
        self.boundary = blanket_boundary
        self.silhouette = blanket_silhouette
        self.r_inner_cut = r_inner_cut
        if abs(cut_angle) >= 90:
            raise ValueError(
                "Cannot cut boundary silhouette at an angle greater than 90°."
            )
        self.cut_angle = cut_angle

    def run(self) -> Tuple[BluemiraFace, BluemiraFace]:
        inboard, outboard, ib_boundary, ob_boundary = self.segment_blanket()
        # Inboard
        ib_panels = self.panel_boundary(ib_boundary)
        ib_panels_face = BluemiraFace(ib_panels)
        cut_ib = boolean_cut(inboard, [ib_panels_face])[0]
        # Outboard
        ob_panels = self.panel_boundary(ob_boundary)
        ob_panels_face = BluemiraFace(ob_panels)
        cut_ob = boolean_cut(outboard, [ob_panels_face])[0]
        return cut_ib, cut_ob

    def segment_blanket(
        self,
    ) -> Tuple[BluemiraFace, BluemiraFace, BluemiraWire, BluemiraWire]:
        """
        Segment the breeding blanket's poloidal cross-section.

        Segment it into inboard and outboard silhouettes.

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

        # Cut the silhouette
        cut_result = boolean_cut(self.silhouette, cut_zone)
        if len(cut_result) < 2:
            raise BuilderError(
                f"BB poloidal segmentation only returned {len(cut_result)} face(s)."
            )
        if len(cut_result) > 2:
            bluemira_warn(
                "The BB poloidal segmentation operation returned more than 2 faces "
                f"({len(cut_result)}); only taking the first two..."
            )
        ib_face, ob_face = sorted(cut_result, key=lambda x: x.center_of_mass[0])[:2]

        # Cut the boundary
        boundary_cut = boolean_cut(self.boundary, cut_zone)
        if len(cut_result) < 2:
            raise BuilderError(
                f"BB poloidal boundary segmentation only returned '{len(cut_result)}' "
                "wire(s)."
            )
        if len(cut_result) > 2:
            bluemira_warn(
                "The BB poloidal boundary segmentation operation returned more than 2 "
                f"wires ({len(cut_result)}); only taking the first two..."
            )
        ib_bound, ob_bound = sorted(boundary_cut, key=lambda x: x.center_of_mass[0])[:2]
        return ib_face, ob_face, ib_bound, ob_bound

    def panel_boundary(self, boundary: BluemiraWire) -> BluemiraWire:
        panel_coords = PanellingDesigner(self.params, boundary).run()
        return make_polygon(
            {"x": panel_coords[0], "z": panel_coords[1]}, label="panels", closed=True
        )
