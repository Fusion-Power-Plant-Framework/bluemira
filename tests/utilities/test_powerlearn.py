# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

import pytest
import numpy as np
from pandas import DataFrame
from BLUEPRINT.base.typebase import TypeFrameworkError
from BLUEPRINT.utilities.powerlearn import PowerLaw, LinearLaw, surface_fit


def power_law_func1(x, y, z):
    return 0.5 * x ** 2.3 * y ** -0.6 * z ** 4.44


def power_law_func2(x, y, z):
    return 4.3 * x ** 1.1 * y ** 4.6 * z ** 0


def linear_law_func1(x, y, z):
    return 1.1 * x + 3.6 * y + 6.67 * z


def linear_law_func2(x, y, z):
    return -0.5 * x + 0 * y + 2.34 * z


class TestPowerLaw:
    def test_powerlaw(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        target1 = power_law_func1(x, y, z)
        target2 = power_law_func2(x, y, z)
        array = np.array([x, y, z, target1, target2]).T
        df = DataFrame(array, columns=["x", "y", "z", "target1", "target2"])

        law1 = PowerLaw(df, targets=["target1", "target2"], target="target1")
        law2 = PowerLaw(df, targets=["target1", "target2"], target="target2")

        assert law1.r_2 == 1
        assert law2.r_2 == 1


class TestLinearLaw:
    def test_linearlaw(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        target1 = linear_law_func1(x, y, z)
        target2 = linear_law_func2(x, y, z)
        array = np.array([x, y, z, target1, target2]).T
        df = DataFrame(array, columns=["x", "y", "z", "target1", "target2"])

        law1 = LinearLaw(df, targets=["target1", "target2"], target="target1")
        law2 = LinearLaw(df, targets=["target1", "target2"], target="target2")

        assert law1.r_2 == 1
        assert law2.r_2 == 1


class TestSurfaceFit:
    def test_linear_fit(self):
        def linear_surface(x, y):
            # z = 34 plane
            c1, c2, c3 = 0, 0, 34
            return c1 * x + c2 * y + c3

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = linear_surface(x, y)
        x2d, y2d, zz, coeffs, r2 = surface_fit(x, y, z, order=1)

        assert np.allclose(np.array([0, 0, 34]), coeffs)

        # Introduce some noise
        z[1] += 0.003
        z[3] += 0.05
        z[5] -= 0.00235
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=1)
        assert np.all(np.abs(np.array([0, 0, 34]) - coeffs) < 0.12)

    def test_quadratic_fit(self):

        coeffs_true = np.array([34, 2, 1.4, 5, 0.4, 5])

        def quadratic_surface(x, y):
            c1, c2, c3, c4, c5, c6 = coeffs_true
            return c1 * x ** 2 + c2 * y ** 2 + c3 * x * y + c4 * x + c5 * y + c6

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = quadratic_surface(x, y)
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=2)

        assert np.allclose(coeffs_true, coeffs)

    def test_cubic_fit(self):
        coeffs_true = np.array([34, 2, 1.4, 5, 0.4, 5, 23, 53, 1.4, 1.2])

        def cubic_surface(x, y):
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = coeffs_true
            return (
                c1 * x ** 3
                + c2 * y ** 3
                + c3 * x ** 2 * y
                + c4 * y ** 2 * x
                + c5 * x ** 2
                + c6 * y ** 2
                + c7 * x * y
                + c8 * x
                + c9 * y
                + c10
            )

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = cubic_surface(x, y)
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=3)

        assert np.allclose(coeffs_true, coeffs)

    def test_order_wrong(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = np.random.rand(len(x))
        bad_orders = [4, 1.3, 1.0, 0, 0.0, "2", np.linspace(1, 3, 40)]
        for bad in bad_orders:
            with pytest.raises((ValueError, TypeFrameworkError)):
                surface_fit(x, y, z, order=bad)

    def test_bad_inputs(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = np.random.rand(len(x))
        with pytest.raises(ValueError):
            surface_fit(x, y, z, order=1)

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2])
        z = np.random.rand(len(x))
        with pytest.raises(ValueError):
            surface_fit(x, y, z, order=1)

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = np.random.rand(len(x) - 1)
        with pytest.raises(ValueError):
            surface_fit(x, y, z, order=1)
