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

import numpy as np
import pytest

from bluemira.builders.EUDEMO.tools import (
    pattern_lofted_silhouette,
    pattern_revolved_silhouette,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import distance_to, make_polygon


class TestPatterning:
    @pytest.mark.parametrize(
        "n_segments, n_sectors, gap",
        [
            (3, 16, 0.05),
        ],
    )
    def test_revolved_silhouette(self, n_segments, n_sectors, gap):
        p = make_polygon({"x": [9, 10, 10, 9], "y": 0, "z": [-1, -1, 1, 1]}, closed=True)
        face = BluemiraFace(p)

        shapes = pattern_revolved_silhouette(face, n_segments, n_sectors, gap)

        assert len(shapes) == n_segments

        volume = shapes[0].volume
        for i in range(n_segments) - 1:
            # Check distances between shapes is correct
            np.testing.assert_almost_equal(distance_to(shapes[i], shapes[i + 1]), gap)
            # Check volumes
            np.testing.assert_almost_equal(shapes[i + 1].volume, volume)
