# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from dataclasses import dataclass
from unittest.mock import patch

import pytest

import bluemira.codes._polyscope as poly


@dataclass
class DummyPart:
    """A lightweight mock object to carry a predetermined bounding box."""

    bb: tuple[float, float, float, float, float, float]


class TestPolyscopeAPI:
    # ---- _compute_default_camera -----------------------------------------------

    @pytest.mark.parametrize(
        ("bbox", "expected_cam", "expected_target"),
        [
            # flat in xz plane: looks from -y
            (
                (0.0, 0.0, 0.0, 10.0, 0.0, 10.0),
                (5.0, -15.0, 5.0),
                (5.0, 0.0, 5.0),
            ),
            # flat in yz plane: looks from +x
            (
                (0.0, 0.0, 0.0, 0.0, 10.0, 10.0),
                (15.0, 5.0, 5.0),
                (0.0, 5.0, 5.0),
            ),
            # flat in xy plane: looks from +z
            (
                (0.0, 0.0, 0.0, 10.0, 10.0, 0.0),
                (5.0, 5.0, 15.0),
                (5.0, 5.0, 0.0),
            ),
            # 3D object: falls back to xz side view (-y)
            (
                (0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
                (5.0, -10.0, 5.0),
                (5.0, 5.0, 5.0),
            ),
            # single point: acts like flat in y
            (
                (2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
                (2.0, 0.5, 2.0),
                (2.0, 2.0, 2.0),
            ),
            # near flat geometry: triggers flat in y.
            (
                (0.0, 0.0, 0.0, 100.0, 0.5, 100.0),
                (50.0, 0.25 - 150.0, 50.0),
                (50.0, 0.25, 50.0),
            ),
        ],
    )
    @patch("bluemira.codes._geometryapi.bounding_box")
    def test_compute_default_camera_orientations(
        self, mock_bb, bbox, expected_cam, expected_target
    ):
        """Different camera placements and orientation thresholds."""
        mock_bb.side_effect = lambda p: p.bb

        cam, target = poly._compute_default_camera([DummyPart(bbox)])

        assert cam == pytest.approx(expected_cam)
        assert target == pytest.approx(expected_target)

    @patch("bluemira.codes._geometryapi.bounding_box")
    def test_compute_default_camera_multipart_bounding_box(self, mock_bb):
        """Multiple parts merge into a single global bounding box."""
        mock_bb.side_effect = lambda p: p.bb

        # part 1 (0, 0, 0) to (2, 2, 2)
        part1 = DummyPart((0.0, 0.0, 0.0, 2.0, 2.0, 2.0))
        # part 2 (8, 8, 8) to (10, 10, 10)
        part2 = DummyPart((8.0, 8.0, 8.0, 10.0, 10.0, 10.0))

        # global bounding box (0, 0, 0) to (10, 10, 10).
        cam, target = poly._compute_default_camera([part1, part2])

        # global center (5, 5, 5)
        assert target == pytest.approx((5.0, 5.0, 5.0))

        # expected camera is 3D XZ fallback
        assert cam == pytest.approx((5.0, -10.0, 5.0))
