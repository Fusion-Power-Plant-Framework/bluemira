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

from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.parameterisations import tapered_picture_frame


class TestTaperedPictureFrame:
    def test_zero_radius(self):
        x1 = 1
        x2 = 2
        x3 = 8
        z1_frac = 0.2
        z2 = 3
        z3 = 5
        r = 0
        x, z = tapered_picture_frame(x1, x2, x3, z1_frac, z2, z3, r, npoints=200)
        points = np.vstack((x, z)).T
        # Check if we have a corner point
        assert [x3, z3] in points

    def test_min_points(self):
        x1 = 1
        x2 = 2
        x3 = 8
        z1_frac = 0.2
        z2 = 3
        z3 = 5
        r = 1
        x, z = tapered_picture_frame(x1, x2, x3, z1_frac, z2, z3, r, npoints=10)

        # Check if the 5th to 8th points represent the chamfered corners as we'd expect
        assert (x[4], z[4]) == (x3 - r, z3)
        assert (x[5], z[5]) == (x3, z3 - r)
        assert (x[6], z[6]) == (x3, -z3 + r)
        assert (x[7], z[7]) == (x3 - r, -z3)

    def test_invalid_inputs(self):
        with pytest.raises(GeometryError):

            # x1 > x2:
            points = tapered_picture_frame(2, 1, 8, 0.2, 3, 5, 1, npoints=200)

        with pytest.raises(GeometryError):

            # x2 > x3:
            points = tapered_picture_frame(1, 8, 3, 0.2, 3, 5, 1, npoints=200)

        with pytest.raises(GeometryError):

            # z2 > z3:
            points = tapered_picture_frame(1, 2, 8, 0.5, 8, 2, 1, npoints=200)

        with pytest.raises(GeometryError):

            # z1_frac > 1:
            points = tapered_picture_frame(1, 2, 8, 2, 5, 8, 1, npoints=200)
        with pytest.raises(GeometryError):

            # r > x3-x2
            points = tapered_picture_frame(1, 2, 5, 0.2, 3, 7, 4, npoints=200)

        with pytest.raises(ValueError):

            # npoints<10:
            points = tapered_picture_frame(1, 2, 8, 0.2, 3, 5, 1, npoints=9)
