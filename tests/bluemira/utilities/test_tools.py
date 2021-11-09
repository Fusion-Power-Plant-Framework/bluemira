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

import pytest
import os
import numpy as np
import json
from io import StringIO

from bluemira.base.file import get_bluemira_path
from bluemira.utilities.tools import (
    asciistr,
    cartesian_to_polar,
    clip,
    CommentJSONDecoder,
    compare_dicts,
    cross,
    dot,
    get_module,
    is_num,
    levi_civita_tensor,
    norm,
    NumpyJSONEncoder,
    polar_to_cartesian,
)


class TestCommentJSONDecoder:
    def test_decoder(self):
        loaded = json.load(
            StringIO(
                """{
                "reference_data_root": "",
                "generated_data_root": "",
                "plot_flag": {"abgc": false},
                "process_mode": "run input",
                "process_indat": "IN.DAT",
                "plasma_mode": "mock",  //Thisis a comment @#$%^&*()_+|'
                // hellloo
                }
            """
            ),
            cls=CommentJSONDecoder,
        )

        result = {
            "reference_data_root": "",
            "generated_data_root": "",
            "plot_flag": {"abgc": False},
            "process_mode": "run input",
            "process_indat": "IN.DAT",
            "plasma_mode": "mock",
        }

        assert loaded == result


class TestNumpyJSONEncoder:
    def test_childclass(self):
        fp = get_bluemira_path("bluemira/utilities/test_data", subfolder="tests")
        fn = os.sep.join([fp, "testJSONEncoder.json"])
        d = {"x": np.array([1, 2, 3.4, 4]), "y": [1, 3], "z": 3, "a": "aryhfdhsdf"}
        with open(fn, "w") as file:
            json.dump(d, file, cls=NumpyJSONEncoder)
        with open(fn, "r") as file:
            dd = json.load(file)
        for k, v in d.items():
            for kk, vv in dd.items():
                if k == kk:
                    if isinstance(v, np.ndarray):
                        assert v.tolist() == vv
                    else:
                        assert v == vv


def test_is_num():
    vals = [0, 34.0, 0.0, -0.0, 34e183, 28e-182, np.pi, np.inf]
    for v in vals:
        assert is_num(v) is True

    vals = [True, False, np.nan, object()]
    for v in vals:
        assert is_num(v) is False


class TestAsciiStr:
    def test_asciistr(self):
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(52):
            assert asciistr(i + 1) == alphabet[: i + 1]

        with pytest.raises(ValueError):
            asciistr(53)


class TestLeviCivitaTensor:
    def test_lct_creation(self):
        d1 = np.array(1)
        d2 = np.array([[0, 1], [-1, 0]])
        d3 = np.zeros((3, 3, 3))
        d3[0, 1, 2] = d3[1, 2, 0] = d3[2, 0, 1] = 1
        d3[0, 2, 1] = d3[2, 1, 0] = d3[1, 0, 2] = -1
        d4 = np.zeros((4, 4, 4, 4))

        min1 = (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
            np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
            np.array([3, 1, 2, 2, 3, 0, 3, 0, 1, 1, 2, 0]),
            np.array([2, 3, 1, 3, 0, 2, 1, 3, 0, 2, 0, 1]),
        )
        plus1 = (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
            np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
            np.array([2, 3, 1, 3, 0, 2, 1, 3, 0, 2, 0, 1]),
            np.array([3, 1, 2, 2, 3, 0, 3, 0, 1, 1, 2, 0]),
        )

        d4[min1] = -1
        d4[plus1] = 1

        for i, arr in enumerate([d1, d2, d3, d4], start=1):
            np.testing.assert_equal(levi_civita_tensor(i), arr)


class TestEinsumNorm:
    def test_norm(self):
        val = np.random.rand(999, 3)
        np.testing.assert_allclose(norm(val, axis=1), np.linalg.norm(val, axis=1))
        np.testing.assert_allclose(norm(val, axis=0), np.linalg.norm(val, axis=0))

    def test_raise(self):
        val = np.random.rand(999, 3)

        with pytest.raises(ValueError):
            norm(val, axis=3)


class TestEinsumDot:
    def test_dot(self):
        val3 = np.random.rand(999, 3, 3)
        val2 = np.random.rand(999, 3)
        val = np.random.rand(3)

        # ab, bc -> ac
        np.testing.assert_allclose(dot(val2, val2.T), np.dot(val2, val2.T))

        # abc, acd -> abd
        dv = dot(val3, val3)
        for no, i in enumerate(val3):
            np.testing.assert_allclose(dv[no], np.dot(i, i))

        # abc, c -> ab
        np.testing.assert_allclose(dot(val3, val), np.dot(val3, val))

        # a, abc -> ac | ab, abc -> ac | abc, bc -> ac -- undefined behaviour
        for (a, b) in [(val, val3.T), (val2, val3), (val3, val3[1:])]:
            with pytest.raises(ValueError):
                dot(a, b)

        # ab, b -> a
        np.testing.assert_allclose(dot(val2, val), np.dot(val2, val))

        # a, ab -> b
        np.testing.assert_allclose(dot(val, val2.T), np.dot(val, val2.T))

        # 'a, a -> ...'
        np.testing.assert_allclose(dot(val, val), np.dot(val, val))


class TestEinsumCross:
    def test_cross(self):
        val3 = np.random.rand(999, 3)
        val2 = np.random.rand(999, 2)
        val = np.random.rand(999)

        for i, v in enumerate([val2, val3], start=2):
            np.testing.assert_allclose(cross(v, v), np.cross(v, v))

        np.testing.assert_allclose(cross(val, val), val ** 2)

    def test_raises(self):
        val = np.random.rand(5, 4)

        with pytest.raises(ValueError):
            cross(val, val)


class TestCompareDicts:
    def test_equal(self):
        a = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        b = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        assert compare_dicts(a, b, almost_equal=False, verbose=False)
        c = {"a": 1.111111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        assert compare_dicts(a, c, almost_equal=False, verbose=False) is False
        c = {
            "a": 1.11111111,
            "b": np.array([1.00001, 2.09, 2.3000000000001]),
            "c": "test",
        }
        assert compare_dicts(a, c, almost_equal=False, verbose=False) is False
        c = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "ttest"}
        assert compare_dicts(a, c, almost_equal=False, verbose=False) is False
        c = {
            "a": 1.11111111,
            "b": np.array([1, 2.09, 2.3000000000001]),
            "c": "test",
            "extra_key": 1,
        }
        assert compare_dicts(a, c, almost_equal=False, verbose=False) is False

        # This will work, because it is an array of length 1
        c = {
            "a": np.array([1.11111111]),
            "b": np.array([1, 2.09, 2.3000000000001]),
            "c": "test",
        }
        assert compare_dicts(a, c, almost_equal=False, verbose=False)

    def test_almost_equal(self):
        a = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        b = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        assert compare_dicts(a, b, almost_equal=True, verbose=False)
        c = {"a": 1.111111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "test"}
        assert compare_dicts(a, c, almost_equal=True, verbose=False)
        c = {
            "a": 1.11111111,
            "b": np.array([1.00001, 2.09, 2.3000000000001]),
            "c": "test",
        }
        assert compare_dicts(a, c, almost_equal=True, verbose=False)
        c = {"a": 1.11111111, "b": np.array([1, 2.09, 2.3000000000001]), "c": "ttest"}
        assert compare_dicts(a, c, almost_equal=True, verbose=False) is False
        c = {
            "a": 1.11111111,
            "b": np.array([1, 2.09, 2.3000000000001]),
            "c": "test",
            "extra_key": 1,
        }
        assert compare_dicts(a, c, almost_equal=True, verbose=False) is False

        # This will work, because it is an array of length 1
        c = {
            "a": np.array([1.11111111]),
            "b": np.array([1, 2.09, 2.3000000000001]),
            "c": "test",
        }
        assert compare_dicts(a, c, almost_equal=True, verbose=False)

        c = {
            "a": np.array([1.111111111]),
            "b": np.array([1, 2.09, 2.3000000000001]),
            "c": "test",
        }
        assert compare_dicts(a, c, almost_equal=True, verbose=False)


class TestClip:
    def test_clip_array(self):
        test_array = [0.1234, 1.0, 0.3, 1, 0.0, 0.756354, 1e-8, 0]
        test_array = np.array(test_array)
        test_array = clip(test_array, 1e-8, 1 - 1e-8)
        expected_array = [0.1234, 1 - 1e-8, 0.3, 1 - 1e-8, 1e-8, 0.756354, 1e-8, 1e-8]
        expected_array = np.array(expected_array)
        assert np.allclose(test_array, expected_array)

    def test_clip_float(self):
        test_float = 0.1234
        test_float = clip(test_float, 1e-8, 1 - 1e-8)
        expected_float = 0.1234
        assert np.allclose(test_float, expected_float)

        test_float = 0.0
        test_float = clip(test_float, 1e-8, 1 - 1e-8)
        expected_float = 1e-8
        assert np.allclose(test_float, expected_float)

        test_float = 1.0
        test_float = clip(test_float, 1e-8, 1 - 1e-8)
        expected_float = 1 - 1e-8
        assert np.allclose(test_float, expected_float)


def test_polar_cartesian():
    x = np.random.rand(100)
    z = np.random.rand(100)
    x_ref = np.random.rand()
    z_ref = np.random.rand()
    r, phi = cartesian_to_polar(x, z, x_ref, z_ref)
    xx, zz = polar_to_cartesian(r, phi, x_ref, z_ref)
    assert np.allclose(x, xx)
    assert np.allclose(z, zz)


class TestGetModule:
    def test_getmodule(self):
        test_mod = "bluemira.utilities.tools"
        test_mod_loc = get_bluemira_path("utilities") + "/tools.py"

        for mod in [test_mod, test_mod_loc]:
            module = get_module(mod)
            assert module.__name__.rsplit(".", 1)[-1] == test_mod.rsplit(".", 1)[-1]

    def test_getmodule_failures(self):

        # Path doesn't exist
        with pytest.raises(FileNotFoundError):
            get_module("/This/file/doesnt/exist.py")

        # Directory exists but not file
        with pytest.raises(FileNotFoundError):
            get_module(get_bluemira_path() + "/README.md")

        # Not a python module
        with pytest.raises(ImportError):
            get_module(get_bluemira_path() + "../README.md")


if __name__ == "__main__":
    pytest.main([__file__])
