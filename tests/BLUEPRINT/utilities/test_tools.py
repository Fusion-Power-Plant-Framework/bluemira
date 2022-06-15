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

from collections import OrderedDict

import numpy as np

from BLUEPRINT.geometry.geomtools import polyarea
from BLUEPRINT.utilities.tools import _apply_rule  # noqa
from BLUEPRINT.utilities.tools import _apply_rules  # noqa
from BLUEPRINT.utilities.tools import _split_rule  # noqa
from BLUEPRINT.utilities.tools import (
    PowerLawScaling,
    ellipse,
    expand_nested_list,
    maximum,
    nested_dict_search,
)


class TestPowerLaw:
    def test_powerlaw_cerr(self):
        law = PowerLawScaling(c=0.9, cerr=0.2, exponents=[2.5, 3.5], err=[0.2, 0.3])
        result = np.array(law.error(2.1, 3.4))
        shouldbe = np.array(
            [
                0.9 * 2.1**2.5 * 3.4**3.5,
                0.7 * 2.1**2.3 * 3.4**3.2,
                1.1 * 2.1**2.7 * 3.4**3.8,
            ]
        )
        assert np.allclose(result, shouldbe), f"{result} != {shouldbe}"

    def test_powerlaw_cexperr(self):
        law = PowerLawScaling(
            c=2.15, cerr=0, cexperr=0.1, exponents=[2.5, 3.5], err=[0.2, 0.3]
        )
        result = np.array(law.error(5.1, 6.3))
        shouldbe = np.array(
            [
                2.15 * np.exp(0) * 5.1**2.5 * 6.3**3.5,
                2.15 * np.exp(-0.1) * 5.1**2.3 * 6.3**3.2,
                2.15 * np.exp(0.1) * 5.1**2.7 * 6.3**3.8,
            ]
        )
        assert np.allclose(result, shouldbe), f"{result} != {shouldbe}"


coil = OrderedDict(
    [
        (
            "Coil5",
            {
                "I": -30050735.995012265,
                "dx": 1.0964879342989633,
                "dz": 1.0964879342989633,
                "rc": 0.70710678118654757,
                "x": 4.709153066745877,
                "z": -9.630529940091678,
            },
        ),
        (
            "Coil6",
            {
                "I": 7669303.0674889563,
                "dx": 0.67165535111396835,
                "dz": 0.67165535111396835,
                "rc": 0.70710678118654757,
                "x": 10.708497068813323,
                "z": -9.81261770717519,
            },
        ),
        (
            "Coil7",
            {
                "I": -13756744.599736707,
                "dx": 0.75114041812607302,
                "dz": 0.75114041812607302,
                "rc": 0.70710678118654757,
                "x": 17.226050539612455,
                "z": -3.927632865391808,
            },
        ),
        (
            "Coil8",
            {
                "I": -10711691.250734033,
                "dx": 0.65457381287107474,
                "dz": 0.65457381287107474,
                "rc": 0.70710678118654757,
                "x": 17.029308827626245,
                "z": 5.236564569827094,
            },
        ),
        (
            "Coil9",
            {
                "I": 1166318.0419358762,
                "dx": 0.70221089303011219,
                "dz": 0.70221089303011219,
                "rc": 0.70710678118654757,
                "x": 5.6692134494009725,
                "z": 9.149472461847667,
            },
        ),
        (
            "Coil11",
            {
                "I": 23301095.856578674,
                "dx": 0.55189999999999995,
                "dz": 2.987638311382979,
                "rc": 1.5019760588917015,
                "x": 2.5094499999999997,
                "z": -6.6046754268622729,
            },
        ),
        (
            "Coil12",
            {
                "I": -40488377.302747779,
                "dx": 0.55189999999999995,
                "dz": 2.9344719915019231,
                "rc": 1.5019760588917015,
                "x": 2.5094499999999997,
                "z": -3.5436202754198227,
            },
        ),
        (
            "Coil13",
            {
                "I": -40541704.115436234,
                "dx": 0.55189999999999995,
                "dz": 2.9694326773057678,
                "rc": 1.5019760588917015,
                "x": 2.5094499999999997,
                "z": -0.49166794101597677,
            },
        ),
        (
            "Coil14",
            {
                "I": -43114969.95519124,
                "dx": 0.55189999999999995,
                "dz": 3.4609796721568684,
                "rc": 1.5019760588917015,
                "x": 2.5094499999999997,
                "z": 2.8235382337153414,
            },
        ),
        (
            "Coil15",
            {
                "I": -24594884.248026744,
                "dx": 0.55189999999999995,
                "dz": 1.7825609166897443,
                "rc": 1.5019760588917015,
                "x": 2.5094499999999997,
                "z": 5.5453085281386478,
            },
        ),
        (
            "Coil10",
            {
                "I": 3643842.6710308534,
                "dx": 0.73430032715205951,
                "dz": 0.73430032715205951,
                "rc": 0.70710678118654757,
                "x": 3.030663999877272,
                "z": 6.978529376138229,
            },
        ),
    ]
)


class Testfindcoil:
    def test_split_rule(self):
        assert _split_rule("z<=-3") == ["z", "<=", "-3"]
        assert _split_rule("z>=-3") == ["z", ">=", "-3"]
        assert _split_rule("z<-3") == ["z", "<", "-3"]
        assert _split_rule("z<=-3") == ["z", "<=", "-3"]

    def test_apply(self):
        assert _apply_rule(4, ">", 3)
        assert not _apply_rule(5, ">=", 6)
        assert _apply_rule(6, "=", 6)
        assert not _apply_rule(6.0000000000001, "=", 6.00000000000002)
        assert _apply_rule(17.029308827626245, ">", 4)

    def test_apply_multiple(self):
        assert not _apply_rules([4, ">", 3, ">", 4])
        assert not _apply_rules([2, ">", 3, ">", 4])
        assert _apply_rules([4, ">", 3, ">", 2])
        assert _apply_rules([3, ">=", 3, ">=", 3])
        assert _apply_rules([3, "=", 3, "=", 3])
        assert not _apply_rules([2, "<=", 3, "<", 3])

    def test_find(self):
        n = nested_dict_search(coil, ["z>0", "x>4"])
        assert list(n.keys()) == ["Coil8", "Coil9"]
        n = nested_dict_search(coil, ["z<-10"])
        assert n is None
        n = nested_dict_search(coil, ["z>0", "5<x<10"])
        assert list(n.keys()) == ["Coil9"]

    def test_str(self):
        n = nested_dict_search(coil, "z>0")
        assert len(n.keys()) == 5


class TestNestedList:
    def test_nested_list(self):
        a = [1, 2, [3, 4, [4, [3, 3, [4, 4]]]], 5]
        n = expand_nested_list(a)
        res = [1, 2, 3, 4, 4, 3, 3, 4, 4, 5]
        assert n == res
        assert expand_nested_list(a, a, a) == res * 3


class TestMaximum:
    def test_maximum_array(self):
        test_array = [0.1234, 1.0, 0.3, 1, 0.0, 0.756354, 1e-8, 0]
        test_array = np.array(test_array)
        test_array = maximum(test_array, 1e-8)
        expected_array = [0.1234, 1.0, 0.3, 1, 1e-8, 0.756354, 1e-8, 1e-8]
        expected_array = np.array(expected_array)
        assert np.allclose(test_array, expected_array)

    def test_maximum_float(self):
        test_float = 0.1234
        test_float = maximum(test_float, 1e-8)
        expected_float = 0.1234
        assert np.allclose(test_float, expected_float)

        test_float = 0.0
        test_float = maximum(test_float, 1e-8)
        expected_float = 1e-8
        assert np.allclose(test_float, expected_float)

        test_float = 1.0
        test_float = maximum(test_float, 1e-8)
        expected_float = 1.0
        assert np.allclose(test_float, expected_float)


class TestEllipse:
    def test_ellipse_area(self):
        a = 5
        b = 4
        x, y = ellipse(a, b, n=200)
        area = polyarea(x, y)
        assert np.isclose(area, np.pi * a * b, 3)
