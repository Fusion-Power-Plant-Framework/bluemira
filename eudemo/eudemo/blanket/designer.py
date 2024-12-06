# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Designer for EUDEMO blankets."""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from scipy.spatial.distance import euclidean

from bluemira.base.designer import Designer
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.blanket.panelling import PanellingDesigner
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


@dataclass
class BlanketSegments:
    """Container for parts of a blanket."""

    inboard: BluemiraFace
    outboard: BluemiraFace
    inboard_boundary: BluemiraWire
    outboard_boundary: BluemiraWire


class BlanketDesigner(Designer[tuple[BluemiraFace, BluemiraFace, Coordinates]]):
    """
    Designer for an EUDEMO-style blanket.

    This takes a blanket boundary and silhouette, cuts them into inboard
    and outboard segments, and panels each segment.

    Parameters
    ----------
    params:
        The parameters for the designer.
        See :class:`.BlanketDesignerParams` for more details.
    blanket_boundary:
        The wire defining the inner boundary of the blanket.
        This should be an open wire, where the start and end points
        share the same z-coordinate (i.e., an open loop).
    blanket_silhouette:
        A face defining the poloidal shape of the blanket.
        The inner boundary of this face *must* be the same as the
        :obj:`blanket_boundary`. It's difficult to reverse engineer the
        wire from the face, so both are required.
    r_inner_cut:
        The x coordinate at which to cut the blanket into segments.
        Note that this is the coordinate of the x-most end of the cut on
        the inner wire of the boundary, not the center.
    cut_angle:
        The angle at which to segment the blanket [degrees].
        A positive angle will result in a downward top-to-bottom slope
        on the inboard.
    """

    param_cls = BlanketDesignerParams
    params: BlanketDesignerParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        blanket_boundary: BluemiraWire,
        blanket_silhouette: BluemiraFace,
        r_inner_cut: float,
        cut_angle: float,
        build_config: dict | None = None,
    ):
        super().__init__(params, build_config)
        self.boundary = blanket_boundary
        self.silhouette = blanket_silhouette
        self.r_inner_cut = r_inner_cut
        if abs(cut_angle) >= 90:  # noqa: PLR2004
            raise ValueError(
                "Cannot cut boundary silhouette at an angle greater than 90°."
            )
        self.cut_angle = cut_angle

    @staticmethod
    def _remove_gap_and_merge(
        ib_panels: Coordinates, ob_panels: Coordinates
    ) -> Coordinates:
        """Merge two sets of coordinates.

        Returns
        -------
        :
            Merged coordinates
        """
        sm_dist = np.inf
        closest_points = None
        for ib_pt in ib_panels.points:
            dists = [euclidean(ib_pt, ob_pt) for ob_pt in ob_panels.points]
            sm_dist_idx = np.argmin(dists)
            d = dists[sm_dist_idx]
            if d < sm_dist:
                sm_dist = d
                ob_pt = ob_panels.points[sm_dist_idx]
                closest_points = (ib_pt, ob_pt)

        new_coords = [
            p
            for p in ib_panels.points + ob_panels.points
            if not (
                np.array_equal(p, closest_points[0])
                or np.array_equal(p, closest_points[1])
            )
        ]
        return Coordinates(new_coords)

    def run(self) -> tuple[BluemiraFace, BluemiraFace, Coordinates]:
        """
        Run the blanket design problem.

        Returns
        -------
        cut_ib:
            Inboard blanket face
        cut_ob:
            Outboard blanket face
        panel_points:
            The panel points of the blanket
        """
        segments = self.segment_blanket()
        return segments.inboard, segments.outboard, None
        # Inboard
        ib_panels = self.panel_boundary(segments.inboard_boundary)
        ib_panels_face = BluemiraFace(ib_panels)
        cut_ib = boolean_cut(segments.inboard, [ib_panels_face])[0]
        # Outboard
        ob_panels = self.panel_boundary(segments.outboard_boundary)
        ob_panels_face = BluemiraFace(ob_panels)
        cut_ob = boolean_cut(segments.outboard, [ob_panels_face])[0]

        # to get the panel points, we must remove the gap points
        # introduced in the ib <-> ob slice
        ib_verts = ib_panels.vertexes.points
        ob_verts = ob_panels.vertexes.points
        ib_verts[0] = cut_ib.vertexes.points[cut_ib.vertexes.argmin(ib_verts[0])]
        ob_verts[-1] = cut_ob.vertexes.points[cut_ob.vertexes.argmin(ob_verts[-1])]

        panel_points = self._remove_gap_and_merge(
            Coordinates(ib_verts), Coordinates(ob_verts)
        )

        return cut_ib, cut_ob, panel_points

    def segment_blanket(self) -> BlanketSegments:
        """
        Segment the breeding blanket's poloidal cross-section.

        Segment it into inboard and outboard silhouettes.

        Returns
        -------
        :
            An instance of :class:`.BlanketSegments` containing the
            blanket segment geometries.
        """
        cut_zone = self._make_cutting_face()
        ib_face, ob_face = self._cut_geom(self.silhouette, cut_zone)
        ib_bound, ob_bound = self._cut_geom(self.boundary, cut_zone)
        return BlanketSegments(
            inboard=ib_face,
            outboard=ob_face,
            inboard_boundary=ib_bound,
            outboard_boundary=ob_bound,
        )

    def panel_boundary(self, boundary: BluemiraWire) -> BluemiraWire:
        """
        Create the panel shapes for the given boundary.

        Returns
        -------
        :
            The panel boundary
        """
        panel_coords = PanellingDesigner(self.params, boundary).run()
        return make_polygon(
            {"x": panel_coords[0], "z": panel_coords[1]}, label="panels", closed=True
        )

    def _make_cutting_face(self) -> BluemiraFace:
        """
        Make a face that can be used to cut the blanket into inboard & outboard.

        Returns
        -------
        cut_zone:
            The face cut for the inboard outboard split
        """
        p0 = get_inner_cut_point(self.silhouette, self.r_inner_cut)
        p1 = [p0[0], 0, p0[2] + VERY_BIG]
        p2 = [p0[0] - self.params.c_rm.value, 0, p1[2]]
        p3 = [p2[0], 0, p0[2] - np.sqrt(2) * self.params.c_rm.value]
        cut_zone = BluemiraFace(make_polygon([p0, p1, p2, p3], closed=True))
        if self.cut_angle != 0.0:
            cut_zone.rotate(base=p0, direction=(0, -1, 0), degree=self.cut_angle)
        return cut_zone

    _GeomT = TypeVar("_GeomT", BluemiraFace, BluemiraWire)

    @staticmethod
    def _cut_geom(geom: _GeomT, cut_tool: BluemiraFace) -> tuple[_GeomT, _GeomT]:
        """Cut the given geometry into two using the given cutting tool.

        Returns
        -------
        inboard:
            Inboard geometry
        outboard:
            Outboard geometry

        Raises
        ------
        BuilderError
            boolean cut returns too few segments
        """
        parts = boolean_cut(geom, cut_tool)
        if len(parts) < 2:  # noqa: PLR2004
            raise BuilderError(
                f"BB poloidal segmentation only returned {len(parts)} part(s), expected "
                "2."
            )
        if len(parts) > 2:  # noqa: PLR2004
            bluemira_warn(
                "The BB poloidal segmentation operation returned more than 2 parts "
                f"({len(parts)}); only taking the first two..."
            )
        inboard, outboard = sorted(parts, key=lambda x: x.center_of_mass[0])[:2]
        return inboard, outboard
