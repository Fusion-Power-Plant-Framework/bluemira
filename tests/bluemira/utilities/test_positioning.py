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

from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.utilities.error import PositionerError
from bluemira.utilities.positioning import PathInterpolator, RegionInterpolator


class TestPathInterpolator:
    def test_open(self):
        x = [0, 2, 4]
        z = [0, 1, 0]
        # Note that we're making a CW polygon, which will get reversed to a CCW one
        polygon = make_polygon({"x": x, "z": z})
        interpolator = PathInterpolator(polygon)

        x0, z0 = interpolator.to_xz(0)
        assert np.isclose(x0, 4)
        assert np.isclose(z0, 0)
        assert np.isclose(interpolator.to_L(4, 0), 0)

        x05, z05 = interpolator.to_xz(0.5)
        assert np.isclose(x05, 2)
        assert np.isclose(z05, 1)
        assert np.isclose(interpolator.to_L(2, 1), 0.5)

        x1, z1 = interpolator.to_xz(1)
        assert np.isclose(x1, 0)
        assert np.isclose(z1, 0)
        assert np.isclose(interpolator.to_L(0, 0), 1)

    def test_closed(self):
        circle = make_circle(center=(0, 0, 0), axis=(0, 1, 0), radius=10)
        interpolator = PathInterpolator(circle)
        x0, z0 = interpolator.to_xz(0)
        assert np.isclose(x0, 10)
        assert np.isclose(z0, 0)
        assert np.isclose(interpolator.to_L(10, 0), 0)

        x90, z90 = interpolator.to_xz(0.25)
        assert np.isclose(x90, 0)
        assert np.isclose(z90, 10)
        assert np.isclose(interpolator.to_L(0, 10), 0.25)

        x180, z180 = interpolator.to_xz(0.5)
        assert np.isclose(x180, -10)
        assert np.isclose(z180, 0)
        assert np.isclose(interpolator.to_L(-10, 0), 0.5)


class TestRegionInterpolator:
    def test_bad_shapes(self):
        x = [0, 4, 2, 4, 0, 2]
        z = [0, 0, 1, 2, 2, 1]
        concave_polygon = make_polygon({"x": x, "z": z}, closed=True)
        with pytest.raises(PositionerError):
            RegionInterpolator(concave_polygon)

        open_polygon = make_polygon({"x": x, "z": z})
        with pytest.raises(PositionerError):
            RegionInterpolator(open_polygon)

    def test_circle(self):
        circle = make_circle(center=(0, 0, 0), axis=(0, 1, 0), radius=10)
        interpolator = RegionInterpolator(circle)

        xc, zc = interpolator.to_xz((0.5, 0.5))
        assert np.isclose(xc, 0)
        assert np.isclose(zc, 0)
        l0, l1 = interpolator.to_L(0, 0)
        assert np.isclose(l0, 0.5)
        assert np.isclose(l1, 0.5)
