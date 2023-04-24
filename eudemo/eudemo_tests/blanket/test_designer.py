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
import numpy as np
import pytest

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from eudemo.blanket import BlanketDesigner


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

        ib_face, ob_face, _, _ = designer.segment_blanket()

        assert ib_face.area == pytest.approx(13 - self.params["c_rm"]["value"])
        assert ob_face.area == pytest.approx(13)
        assert ib_face.center_of_mass[0] < ob_face.center_of_mass[0]

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

        ib_face, ob_face, _, _ = designer.segment_blanket()

        c_rm = self.params["c_rm"]["value"]
        expected_cut_area = c_rm / np.sin(np.deg2rad(90 - cut_angle))
        cut_area = self.silhouette.area - (ib_face.area + ob_face.area)
        assert cut_area == pytest.approx(expected_cut_area)
        expected_ib_area = 13.5 - expected_cut_area - np.tan(np.deg2rad(cut_angle)) / 2
        assert ib_face.area == pytest.approx(expected_ib_area)
        expected_ob_area = 12.5 + np.tan(np.deg2rad(cut_angle)) / 2
        assert ob_face.area == pytest.approx(expected_ob_area)
        assert ib_face.center_of_mass[0] < ob_face.center_of_mass[0]

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
