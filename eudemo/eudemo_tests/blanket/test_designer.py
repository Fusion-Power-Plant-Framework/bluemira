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
from unittest import mock

import numpy as np
import pytest

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.blanket import BlanketDesigner
from eudemo.blanket.designer import BlanketSegments


class TestBlanketDesigner:
    @classmethod
    def setup_class(cls):
        cls.params = {
            "n_bb_inboard": {"value": 2, "unit": "m"},
            "n_bb_outboard": {"value": 3, "unit": "m"},
            "c_rm": {"value": 0.4, "unit": "m"},
            "n_TF": {"value": 12, "unit": ""},
            "fw_a_max": {"value": 25, "unit": "degrees"},
            "fw_dL_min": {"value": 0.15, "unit": "m"},
        }
        # makes a rectangular 'horseshoe' with the open end at the bottom
        cls.boundary = make_polygon(
            [
                [1, 0, -2],
                [1, 0, 10],
                [5, 0, 10],
                [5, 0, -2],
                [4, 0, -2],
                [4, 0, 9],
                [2, 0, 9],
                [2, 0, -2],
            ],
            closed=True,
        )
        # note that the area of this face is 26 m2.
        cls.silhouette = BluemiraFace(cls.boundary)

    def test_segment_blanket_0_angle_returns_two_faces_with_correct_area(self):
        r_inner_cut = 3
        cut_angle = 0
        designer = BlanketDesigner(
            self.params,
            self.boundary,
            self.silhouette,
            r_inner_cut=r_inner_cut,
            cut_angle=cut_angle,
        )

        segments = designer.segment_blanket()

        assert segments.inboard.area == pytest.approx(13 - self.params["c_rm"]["value"])
        assert segments.outboard.area == pytest.approx(13)
        assert segments.inboard.center_of_mass[0] < segments.outboard.center_of_mass[0]

    @pytest.mark.parametrize("cut_angle", [1, 25, 58])
    def test_segment_blanket_returns_two_faces_with_correct_area(self, cut_angle):
        r_inner_cut = 3.5
        designer = BlanketDesigner(
            self.params,
            self.boundary,
            self.silhouette,
            r_inner_cut=r_inner_cut,
            cut_angle=cut_angle,
        )

        segments = designer.segment_blanket()

        c_rm = self.params["c_rm"]["value"]
        expected_cut_area = c_rm / np.cos(np.deg2rad(cut_angle))
        cut_area = self.silhouette.area - (
            segments.inboard.area + segments.outboard.area
        )
        assert cut_area == pytest.approx(expected_cut_area)
        expected_ib_area = 13.5 - expected_cut_area - np.tan(np.deg2rad(cut_angle)) / 2
        assert segments.inboard.area == pytest.approx(expected_ib_area)
        expected_ob_area = 12.5 + np.tan(np.deg2rad(cut_angle)) / 2
        assert segments.outboard.area == pytest.approx(expected_ob_area)
        assert segments.inboard.center_of_mass[0] < segments.outboard.center_of_mass[0]

    @pytest.mark.parametrize("cut_angle", [90, 90.01, 100])
    def test_ValueError_given_cut_angle_ge_90(self, cut_angle):
        with pytest.raises(ValueError):
            BlanketDesigner(
                self.params,
                self.boundary,
                self.silhouette,
                r_inner_cut=3,
                cut_angle=cut_angle,
            )

    @mock.patch("eudemo.blanket.designer.PanellingDesigner")
    def test_segments_cut_using_panels(self, paneller_mock):
        # Make a pre-cut blanket that's just two quarter-circles and
        # cook up some panel coordinates to cut into the two faces with.
        # We're testing that the panelling is called, and the resulting
        # coordinates are used to cut into the inboard/outboard
        # silhouettes. We check this using the expected areas of the
        # newly-cut shapes.

        # There's a fair bit of mocking going on in this test. Ideally,
        # we'd be passing a fake PanellingDesigner using dependency
        # injection. But, because the PanellingDesigner takes its
        # boundary in the constructor, and the blanket boundary is cut
        # within the BlanketDesigner, we can't initialise the
        # PanellingDesigner first. This probably speaks to a bit of a
        # design issue.
        # TODO(hsaunders1904): open issue for refactor of
        # PanellingDesigner into Bluemira-proper, make it not a designer
        # and allow dependency injection here. We could also add a
        # BlanketSegmentor class to do the segmenting, as that would now
        # need to happen outside of the BlanketDesigner.
        blanket = make_simple_blanket()
        d = 3 * np.sqrt(2) / 2
        # fmt: off
        ib_panel_coords = np.array([
            [2, 2, 5 - d, 8 - 2 * d, 5, 2],
            [4, 1 + 2 * d, 4 + d, 7, 7, 4],
        ])
        ob_panel_coords = np.array([
            [9, 9, 6 + d, 3 + 2 * d, 6],
            [4, 1 + 2 * d, 4 + d, 7, 7],
        ])
        # fmt: on
        paneller_mock.return_value.run.side_effect = [ib_panel_coords, ob_panel_coords]
        r_inner_cut = 3.5
        cut_angle = 3
        designer = BlanketDesigner(
            self.params,
            self.boundary,
            self.silhouette,
            r_inner_cut=r_inner_cut,
            cut_angle=cut_angle,
        )

        with mock.patch.object(designer, "segment_blanket") as sb_mock:
            sb_mock.return_value = blanket
            ib, ob = designer.run()

        # These areas were (painstakingly) worked out by hand
        panel_trapezium_area = 9 / 2 * (4 * np.sqrt(2) - 5)
        circle_segment_area = 9 / 2 * (np.pi / 2 - 1)
        area_removed = panel_trapezium_area - circle_segment_area
        assert ib.area == pytest.approx(blanket.inboard.area - area_removed)
        assert ob.area == pytest.approx(blanket.outboard.area - area_removed)
        assert ib.center_of_mass[0] < ob.center_of_mass[0]


def make_simple_blanket() -> BlanketSegments:
    """
    Make semi-circular blanket segments.

    The inboard and outboard are symmetrical quarter-circles with radius
    3. Where the centre of the inboard quarter-circle is [5, 4] and the
    centre of the outboard is [6, 4] (in the xz-plane).
    """
    # Inboard
    ib_arc_inner = make_circle(
        radius=3, center=[5, 0, 4], start_angle=180, end_angle=270, axis=(0, 1, 0)
    )
    ib_arc_outer = make_circle(
        radius=4, center=[5, 0, 4], start_angle=180, end_angle=270, axis=(0, 1, 0)
    )
    lower_join = make_polygon({"x": [1, 2], "z": [4, 4]})
    upper_join = make_polygon({"x": [5, 5], "z": [7, 8]})
    inboard = BluemiraFace(
        BluemiraWire([ib_arc_inner, lower_join, ib_arc_outer, upper_join])
    )
    # Outboard
    ob_arc_inner = make_circle(
        radius=3, center=[6, 0, 4], start_angle=270, end_angle=360, axis=(0, 1, 0)
    )
    ob_arc_outer = make_circle(
        radius=4, center=[6, 0, 4], start_angle=270, end_angle=360, axis=(0, 1, 0)
    )
    lower_join = make_polygon({"x": [9, 10], "z": [4, 4]})
    upper_join = make_polygon({"x": [6, 6], "z": [7, 8]})
    outboard = BluemiraFace(
        BluemiraWire([ob_arc_inner, upper_join, ob_arc_outer, lower_join])
    )
    return BlanketSegments(
        inboard=inboard,
        outboard=outboard,
        inboard_boundary=ib_arc_inner,
        outboard_boundary=ob_arc_inner,
    )
