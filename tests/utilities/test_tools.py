# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import copy
import filecmp
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.utilities.tools import (
    NumpyJSONEncoder,
    asciistr,
    cartesian_to_polar,
    clip,
    compare_dicts,
    consec_repeat_elem,
    cross,
    cylindrical_to_toroidal,
    deprecation_wrapper,
    dot,
    get_class_from_module,
    get_module,
    is_num,
    levi_civita_tensor,
    norm,
    polar_to_cartesian,
    toroidal_to_cylindrical,
    write_csv,
)


class TestNumpyJSONEncoder:
    def test_save_and_load_returns_original_dict(self):
        original_dict = {
            "x": np.array([1, 2, 3.4, 4]),
            "y": [1, 3],
            "z": 3,
            "a": "aryhfdhsdf",
        }

        json_out = json.dumps(original_dict, cls=NumpyJSONEncoder)
        loaded_dict = json.loads(json_out)

        expected = copy.deepcopy(original_dict)
        expected["x"] = original_dict["x"].tolist()
        assert loaded_dict == expected


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

        with pytest.raises(ValueError):  # noqa: PT011
            asciistr(53)


class TestCSVWriter:
    @pytest.mark.parametrize(("ext", "comment_char"), [(".csv", "#"), (".txt", "!")])
    def test_csv_writer(self, tmp_path, ext, comment_char):
        # Some dummy data to write to file
        x_vals = [0, 1, 2]
        z_vals = [-1, 0, 1]
        flux_vals = [10, 15, 20]
        data = np.array([x_vals, z_vals, flux_vals]).T
        header = "This is a test\nThis is a second line"
        col_names = ["x", "z", "heat_flux"]

        # Write the data to csv, using default extension and comment style
        expected_file = f"test_csv_writer{ext}"

        expected_path = Path(tmp_path, expected_file).as_posix()
        write_csv(data, expected_path, col_names, header, ext, comment_char)

        # Retrieve data file to compare
        test_output_path = Path(Path(__file__).parent, "test_data", f"{expected_file}")

        # Compare
        assert filecmp.cmp(expected_path, test_output_path)


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
    rng = np.random.default_rng()

    def test_norm(self):
        val = self.rng.random((999, 3))
        np.testing.assert_allclose(norm(val, axis=1), np.linalg.norm(val, axis=1))
        np.testing.assert_allclose(norm(val, axis=0), np.linalg.norm(val, axis=0))

    def test_raise(self):
        val = self.rng.random((999, 3))

        with pytest.raises(ValueError):  # noqa: PT011
            norm(val, axis=3)


class TestEinsumDot:
    rng = np.random.default_rng()

    def test_dot(self):
        val3 = self.rng.random((999, 3, 3))
        val2 = self.rng.random((999, 3))
        val = self.rng.random(3)

        # ab, bc -> ac
        np.testing.assert_allclose(dot(val2, val2.T), np.dot(val2, val2.T))

        # abc, acd -> abd
        dv = dot(val3, val3)
        for no, i in enumerate(val3):
            np.testing.assert_allclose(dv[no], np.dot(i, i))

        # abc, c -> ab
        np.testing.assert_allclose(dot(val3, val), np.dot(val3, val))

        # a, abc -> ac | ab, abc -> ac | abc, bc -> ac -- undefined behaviour
        for a, b in [(val, val3.T), (val2, val3), (val3, val3[1:])]:
            with pytest.raises(ValueError):  # noqa: PT011
                dot(a, b)

        # ab, b -> a
        np.testing.assert_allclose(dot(val2, val), np.dot(val2, val))

        # a, ab -> b
        np.testing.assert_allclose(dot(val, val2.T), np.dot(val, val2.T))

        # 'a, a -> ...'
        np.testing.assert_allclose(dot(val, val), np.dot(val, val))


class TestEinsumCross:
    rng = np.random.default_rng()

    def test_cross(self):
        val3 = self.rng.random((999, 3))
        val2 = self.rng.random((999, 2))
        val = self.rng.random(999)

        for _i, v in enumerate([val2, val3], start=2):
            np.testing.assert_allclose(cross(v, v), np.cross(v, v))

        np.testing.assert_allclose(cross(val, val), val**2)

    def test_raises(self):
        val = self.rng.random((5, 4))

        with pytest.raises(ValueError):  # noqa: PT011
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


def test_consec_repeat_elem():
    arr = np.array([0, 1, 1, 2, 2, 2, 3, 3, 4])

    np.testing.assert_array_equal(consec_repeat_elem(arr, 2), np.array([1, 3, 4, 6]))
    np.testing.assert_array_equal(consec_repeat_elem(arr, 3), np.array([3]))

    with pytest.raises(NotImplementedError):
        consec_repeat_elem(arr, 1)


def test_polar_cartesian():
    rng = np.random.default_rng()

    x = rng.random(100)
    z = rng.random(100)
    x_ref = rng.random()
    z_ref = rng.random()
    r, phi = cartesian_to_polar(x, z, x_ref, z_ref)
    xx, zz = polar_to_cartesian(r, phi, x_ref, z_ref)
    assert np.allclose(x, xx)
    assert np.allclose(z, zz)


class TestGetModule:
    test_mod = "bluemira.utilities.tools"
    test_mod_loc = Path(get_bluemira_path("utilities"), "tools.py").as_posix()
    test_class_name = "NumpyJSONEncoder"

    def test_getmodule(self):
        for mod in [self.test_mod, self.test_mod_loc]:
            module = get_module(mod)
            assert module.__name__.rsplit(".", 1)[-1] == self.test_mod.rsplit(".", 1)[-1]

    def test_getmodule_failures(self):
        # Path doesn't exist
        with pytest.raises(FileNotFoundError):
            get_module("/This/file/doesnt/exist.py")

        # Directory exists but not file
        with pytest.raises(FileNotFoundError):
            get_module(Path(get_bluemira_path(), "README.md").as_posix())

        # Not a python module
        with pytest.raises(ImportError):
            get_module(Path(get_bluemira_path(), "../README.md").as_posix())

    def test_get_weird_ext_python_file(self, tmp_path):
        path1 = tmp_path / "file"
        path2 = tmp_path / "file.hello"
        function = """def f():
    return True"""
        for path in [path1, path2]:
            with open(path, "w") as file:
                file.writelines(function)

            mod = get_module(str(path))
            assert mod.f()

    def test_get_class(self):
        for mod in [self.test_mod, self.test_mod_loc]:
            the_class = get_class_from_module(f"{mod}::{self.test_class_name}")
            assert the_class.__name__ == self.test_class_name

    def test_get_class_default(self):
        class_name = "NumpyJSONEncoder"
        for mod in [self.test_mod, self.test_mod_loc]:
            the_class = get_class_from_module(class_name, default_module=mod)
            assert the_class.__name__ == class_name

    def test_get_class_default_override(self):
        class_name = "NumpyJSONEncoder"
        for mod in [self.test_mod, self.test_mod_loc]:
            the_class = get_class_from_module(
                f"{mod}::{self.test_class_name}", default_module="a_module"
            )
            assert the_class.__name__ == class_name

    def test_get_class_failure(self):
        # Class not in module
        with pytest.raises(ImportError):
            get_class_from_module("Spam", default_module=self.test_mod)


class TestDeprecationWrapper:
    def test_no_message_wrap(self):
        with patch("warnings.warn") as w_patch:
            deprecation_wrapper(lambda x: 1)(1)  # noqa: ARG005

        assert w_patch.call_args_list[0][0][1] is DeprecationWarning

        @deprecation_wrapper
        def func(x, *, xx):  # noqa: ARG001
            return

        with patch("warnings.warn") as w_patch:
            func(1, xx=1)

        assert w_patch.call_args_list[0][0][1] is DeprecationWarning

    def test_message_wrap(self):
        message = "message"
        with patch("warnings.warn") as w_patch:
            deprecation_wrapper(message)(lambda x: 1)(1)  # noqa: ARG005

        assert w_patch.call_args_list[0][0][0] == message
        assert w_patch.call_args_list[0][0][1] is DeprecationWarning

        @deprecation_wrapper(message)
        def func(x, *, xx):  # noqa: ARG001
            return

        with patch("warnings.warn") as w_patch:
            func(1, xx=1)

        assert w_patch.call_args_list[0][0][0] == message
        assert w_patch.call_args_list[0][0][1] is DeprecationWarning


def test_toroidal_coordinate_transform():
    # set values to be used in the tests
    R_0_test = 1.0  # noqa: N806
    z_0_test = 0.0
    # set tau and sigma isosurfaces to be used in the tests and the min & max r & z
    # points for each isosurface
    # fmt: off
    tau_test_isosurface_rz_points = np.array([
        [
            0.24491866, 0.24497605, 0.24514832, 0.24543577, 0.24583895, 0.24635857,
            0.24699559, 0.24775117, 0.24862671, 0.24962383, 0.25074437, 0.25199043,
            0.25336434, 0.25486872, 0.25650642, 0.2582806, 0.2601947, 0.26225246,
            0.26445795, 0.26681557, 0.26933009, 0.27200665, 0.27485079, 0.27786846,
            0.28106608, 0.28445053, 0.28802921, 0.29181007, 0.29580161, 0.30001299,
            0.30445402, 0.30913522, 0.31406787, 0.3192641, 0.32473692, 0.33050029,
            0.33656921, 0.3429598, 0.34968939, 0.35677662, 0.36424155, 0.37210577,
            0.38039257, 0.38912702, 0.39833621, 0.40804935, 0.41829804, 0.42911641,
            0.44054143, 0.45261309, 0.46537477, 0.47887352, 0.4931604, 0.50829089,
            0.52432531, 0.54132932, 0.55937441, 0.57853847, 0.5989065, 0.62057122,
            0.64363392, 0.66820523, 0.69440612, 0.72236886, 0.75223812, 0.78417213,
            0.81834398, 0.85494285, 0.89417545, 0.93626739, 0.9814645, 1.03003417,
            1.08226639, 1.1384746, 1.19899595, 1.26419094, 1.33444201, 1.41015062,
            1.49173242, 1.57960973, 1.67420047, 1.77590249, 1.88507207, 2.00199523,
            2.12685026, 2.2596603, 2.40023496, 2.54810089, 2.70242316, 2.86192128,
            3.02478749, 3.18861912, 3.3503814, 3.50642039, 3.65254676, 3.78420667,
            3.89674489, 3.9857473, 4.0474279, 4.07900554, 4.07900554, 4.0474279,
            3.9857473, 3.89674489, 3.78420667, 3.65254676, 3.50642039, 3.3503814,
            3.18861912, 3.02478749, 2.86192128, 2.70242316, 2.54810089, 2.40023496,
            2.2596603, 2.12685026, 2.00199523, 1.88507207, 1.77590249, 1.67420047,
            1.57960973, 1.49173242, 1.41015062, 1.33444201, 1.26419094, 1.19899595,
            1.1384746, 1.08226639, 1.03003417, 0.9814645, 0.93626739, 0.89417545,
            0.85494285, 0.81834398, 0.78417213, 0.75223812, 0.72236886, 0.69440612,
            0.66820523, 0.64363392, 0.62057122, 0.5989065, 0.57853847, 0.55937441,
            0.54132932, 0.52432531, 0.50829089, 0.4931604, 0.47887352, 0.46537477,
            0.45261309, 0.44054143, 0.42911641, 0.41829804, 0.40804935, 0.39833621,
            0.38912702, 0.38039257, 0.37210577, 0.36424155, 0.35677662, 0.34968939,
            0.3429598, 0.33656921, 0.33050029, 0.32473692, 0.3192641, 0.31406787,
            0.30913522, 0.30445402, 0.30001299, 0.29580161, 0.29181007, 0.28802921,
            0.28445053, 0.28106608, 0.27786846, 0.27485079, 0.27200665, 0.26933009,
            0.26681557, 0.26445795, 0.26225246, 0.2601947, 0.2582806, 0.25650642,
            0.25486872, 0.25336434, 0.25199043, 0.25074437, 0.24962383, 0.24862671,
            0.24775117, 0.24699559, 0.24635857, 0.24583895, 0.24543577, 0.24514832,
            0.24497605, 0.24491866,
        ],
        [
            -5.75593088e-17, -1.48409294e-02, -2.96879267e-02, -4.45470685e-02,
            -5.94244485e-02, -7.43261856e-02, -8.92584334e-02, -1.04227388e-01,
            -1.19239298e-01, -1.34300472e-01, -1.49417289e-01, -1.64596205e-01,
            -1.79843766e-01, -1.95166616e-01, -2.10571504e-01, -2.26065300e-01,
            -2.41654997e-01, -2.57347730e-01, -2.73150780e-01, -2.89071589e-01,
            -3.05117768e-01, -3.21297113e-01, -3.37617611e-01, -3.54087457e-01,
            -3.70715065e-01, -3.87509079e-01, -4.04478388e-01, -4.21632140e-01,
            -4.38979752e-01, -4.56530928e-01, -4.74295671e-01, -4.92284297e-01,
            -5.10507448e-01, -5.28976111e-01, -5.47701626e-01, -5.66695703e-01,
            -5.85970433e-01, -6.05538306e-01, -6.25412214e-01, -6.45605469e-01,
            -6.66131806e-01, -6.87005394e-01, -7.08240836e-01, -7.29853170e-01,
            -7.51857868e-01, -7.74270822e-01, -7.97108333e-01, -8.20387082e-01,
            -8.44124101e-01, -8.68336728e-01, -8.93042543e-01, -9.18259297e-01,
            -9.44004810e-01, -9.70296851e-01, -9.97152984e-01, -1.02459038e00,
            -1.05262557e00, -1.08127418e00, -1.11055056e00, -1.14046736e00,
            -1.17103501e00, -1.20226110e00, -1.23414956e00, -1.26669979e00,
            -1.29990549e00, -1.33375335e00, -1.36822134e00, -1.40327678e00,
            -1.43887397e00, -1.47495132e00, -1.51142792e00, -1.54819945e00,
            -1.58513330e00, -1.62206275e00, -1.65878011e00, -1.69502858e00,
            -1.73049277e00, -1.76478773e00, -1.79744633e00, -1.82790504e00,
            -1.85548819e00, -1.87939109e00, -1.89866258e00, -1.91218826e00,
            -1.91867604e00, -1.91664670e00, -1.90443309e00, -1.88019283e00,
            -1.84194047e00, -1.78760601e00, -1.71512655e00, -1.62257652e00,
            -1.50833783e00, -1.37130478e00, -1.21110868e00, -1.02833543e00,
            -8.24698777e-01, -6.03126370e-01, -3.67720331e-01, -1.23570809e-01,
            1.23570809e-01, 3.67720331e-01, 6.03126370e-01, 8.24698777e-01,
            1.02833543e00, 1.21110868e00, 1.37130478e00, 1.50833783e00, 1.62257652e00,
            1.71512655e00, 1.78760601e00, 1.84194047e00, 1.88019283e00, 1.90443309e00,
            1.91664670e00, 1.91867604e00, 1.91218826e00, 1.89866258e00, 1.87939109e00,
            1.85548819e00, 1.82790504e00, 1.79744633e00, 1.76478773e00, 1.73049277e00,
            1.69502858e00, 1.65878011e00, 1.62206275e00, 1.58513330e00, 1.54819945e00,
            1.51142792e00, 1.47495132e00, 1.43887397e00, 1.40327678e00, 1.36822134e00,
            1.33375335e00, 1.29990549e00, 1.26669979e00, 1.23414956e00, 1.20226110e00,
            1.17103501e00, 1.14046736e00, 1.11055056e00, 1.08127418e00, 1.05262557e00,
            1.02459038e00, 9.97152984e-01, 9.70296851e-01, 9.44004810e-01,
            9.18259297e-01, 8.93042543e-01, 8.68336728e-01, 8.44124101e-01,
            8.20387082e-01, 7.97108333e-01, 7.74270822e-01, 7.51857868e-01,
            7.29853170e-01, 7.08240836e-01, 6.87005394e-01, 6.66131806e-01,
            6.45605469e-01, 6.25412214e-01, 6.05538306e-01, 5.85970433e-01,
            5.66695703e-01, 5.47701626e-01, 5.28976111e-01, 5.10507448e-01,
            4.92284297e-01, 4.74295671e-01, 4.56530928e-01, 4.38979752e-01,
            4.21632140e-01, 4.04478388e-01, 3.87509079e-01, 3.70715065e-01,
            3.54087457e-01, 3.37617611e-01, 3.21297113e-01, 3.05117768e-01,
            2.89071589e-01, 2.73150780e-01, 2.57347730e-01, 2.41654997e-01,
            2.26065300e-01, 2.10571504e-01, 1.95166616e-01, 1.79843766e-01,
            1.64596205e-01, 1.49417289e-01, 1.34300472e-01, 1.19239298e-01,
            1.04227388e-01, 8.92584334e-02, 7.43261856e-02, 5.94244485e-02,
            4.45470685e-02, 2.96879267e-02, 1.48409294e-02, 5.75593088e-17,
            ]
        ])
    # fmt: on
    tau_test_r_max = np.max(tau_test_isosurface_rz_points[0])
    tau_test_r_min = np.min(tau_test_isosurface_rz_points[0])
    tau_test_z_max = np.max(tau_test_isosurface_rz_points[1])
    tau_test_z_min = np.min(tau_test_isosurface_rz_points[1])
    # fmt: off
    sigma_test_isosurface_rz_points = np.array([
        [
            0.0, 0.20473915, 0.40647062, 0.60233523, 0.78975568, 0.96654224, 1.13096272,
            1.28177399, 1.41821761, 1.53998554, 1.64716442, 1.74016738, 1.81966123,
            1.88649575, 1.94163941, 1.98612427, 2.02100119, 2.04730509, 2.06602971,
            2.07811025, 2.08441281, 2.08572888, 2.08277395, 2.07618884, 2.06654303,
            2.05433923, 2.04001863, 2.02396646, 2.00651753, 1.98796162, 1.96854858,
            1.948493, 1.92797854, 1.90716178, 1.88617571, 1.86513282, 1.84412781,
            1.82324002, 1.8025355, 1.78206886, 1.76188489, 1.7420199, 1.72250299,
            1.70335705, 1.68459969, 1.666244, 1.64829924, 1.63077143, 1.61366383,
            1.5969774, 1.58071114, 1.56486244, 1.54942733, 1.53440075, 1.51977671,
            1.5055485, 1.49170882, 1.47824992, 1.46516368, 1.45244174, 1.44007556,
            1.42805649, 1.4163758, 1.4050248, 1.39399478, 1.38327713, 1.37286332,
            1.36274493, 1.35291368, 1.34336144, 1.33408024, 1.32506227, 1.31629991,
            1.30778571, 1.29951241, 1.29147293, 1.28366039, 1.27606809, 1.26868951,
            1.26151832, 1.25454838, 1.24777371, 1.24118853, 1.23478721, 1.2285643,
            1.22251453, 1.21663276, 1.21091403, 1.20535352, 1.19994657, 1.19468867,
            1.18957542, 1.18460259, 1.17976607, 1.17506188, 1.17048615, 1.16603515,
            1.16170526, 1.15749296, 1.15339487, 1.14940769, 1.14552822, 1.14175338,
            1.13808018, 1.13450569, 1.13102713, 1.12764175, 1.12434691, 1.12114006,
            1.1180187, 1.11498044, 1.11202293, 1.10914391, 1.10634119, 1.10361264,
            1.10095619, 1.09836984, 1.09585165, 1.09339973, 1.09101225, 1.08868744,
            1.08642357, 1.08421898, 1.08207204, 1.07998118, 1.07794487, 1.07596162,
            1.07403, 1.0721486, 1.07031607, 1.06853108, 1.06679236, 1.06509866,
            1.06344877, 1.06184152, 1.06027577, 1.0587504, 1.05726434, 1.05581654,
            1.05440599, 1.0530317, 1.05169269, 1.05038805, 1.04911686, 1.04787824,
            1.04667134, 1.0454953, 1.04434933, 1.04323264, 1.04214445, 1.04108402,
            1.04005063, 1.03904356, 1.03806214, 1.03710568, 1.03617355, 1.03526511,
            1.03437975, 1.03351686, 1.03267587, 1.0318562, 1.0310573, 1.03027865,
            1.02951971, 1.02877997, 1.02805895, 1.02735616, 1.02667113, 1.0260034,
            1.02535253, 1.02471809, 1.02409966, 1.02349682, 1.02290918, 1.02233635,
            1.02177794, 1.0212336, 1.02070295, 1.02018566, 1.01968139, 1.01918979,
            1.01871055, 1.01824335, 1.01778789, 1.01734387, 1.01691099, 1.01648898,
            1.01607756, 1.01567646, 1.01528542, 1.01490418, 1.0145325, 1.01417013,
            1.01381684, 1.0134724, 1.01313659, 1.01280918, 1.01248997, 1.01217875,
            1.01187531,
        ],
        [
            3.91631736, 3.90624473, 3.8763291, 3.82745481, 3.76102424, 3.67886035,
            3.58308826, 3.47600914, 3.35997917, 3.23730322, 3.11014965, 2.98048893,
            2.85005542, 2.72032988, 2.5925387, 2.46766561, 2.34647174, 2.22952052,
            2.11720458, 2.00977261, 1.90735476, 1.80998574, 1.71762512, 1.63017495,
            1.54749441, 1.46941206, 1.39573575, 1.32626053, 1.26077489, 1.1990656,
            1.14092134, 1.08613543, 1.03450778, 0.98584622, 0.93996737, 0.89669713,
            0.85587092, 0.81733368, 0.78093969, 0.74655236, 0.71404384, 0.68329464,
            0.65419322, 0.62663555, 0.60052464, 0.57577013, 0.55228785, 0.52999941,
            0.50883181, 0.48871707, 0.46959184, 0.45139713, 0.43407795, 0.41758304,
            0.40186459, 0.38687801, 0.37258166, 0.35893668, 0.34590674, 0.3334579,
            0.32155841, 0.31017855, 0.29929052, 0.28886824, 0.27888731, 0.2693248,
            0.26015922, 0.25137039, 0.24293934, 0.23484825, 0.22708035, 0.21961986,
            0.21245192, 0.20556254, 0.19893852, 0.1925674, 0.18643745, 0.18053756,
            0.17485726, 0.16938663, 0.16411631, 0.15903743, 0.15414159, 0.14942084,
            0.14486764, 0.14047484, 0.13623567, 0.13214367, 0.12819274, 0.12437708,
            0.12069116, 0.11712973, 0.11368779, 0.1103606, 0.10714362, 0.10403255,
            0.10102326, 0.09811184, 0.09529455, 0.09256782, 0.08992825, 0.08737258,
            0.0848977, 0.08250065, 0.08017857, 0.07792875, 0.0757486, 0.07363562,
            0.07158742, 0.06960173, 0.06767635, 0.06580919, 0.06399824, 0.06224157,
            0.06053732, 0.05888371, 0.05727905, 0.05572169, 0.05421006, 0.05274264,
            0.05131798, 0.04993468, 0.0485914, 0.04728684, 0.04601975, 0.04478893,
            0.04359324, 0.04243155, 0.0413028, 0.04020595, 0.03914001, 0.03810401,
            0.03709704, 0.03611819, 0.03516661, 0.03424147, 0.03334196, 0.03246731,
            0.03161676, 0.03078961, 0.02998514, 0.02920269, 0.0284416, 0.02770124,
            0.026981, 0.0262803, 0.02559856, 0.02493524, 0.02428979, 0.02366171,
            0.02305049, 0.02245564, 0.02187671, 0.02131324, 0.02076479, 0.02023093,
            0.01971125, 0.01920535, 0.01871285, 0.01823337, 0.01776654, 0.01731202,
            0.01686946, 0.01643853, 0.01601891, 0.01561029, 0.01521236, 0.01482484,
            0.01444744, 0.01407987, 0.01372188, 0.0133732, 0.01303358, 0.01270278,
            0.01238055, 0.01206667, 0.0117609, 0.01146304, 0.01117288, 0.01089019,
            0.01061479, 0.01034649, 0.01008508, 0.00983039, 0.00958224, 0.00934046,
            0.00910488, 0.00887533, 0.00865166, 0.00843371, 0.00822133, 0.00801437,
            0.0078127, 0.00761617, 0.00742464, 0.007238, 0.00705611, 0.00687884,
            0.00670608, 0.00653771,
            ],
    ])
    # fmt: on
    sigma_test_r_max = np.max(sigma_test_isosurface_rz_points[0])
    sigma_test_r_min = np.min(sigma_test_isosurface_rz_points[0])
    sigma_test_z_max = np.max(sigma_test_isosurface_rz_points[1])
    sigma_test_z_min = np.min(sigma_test_isosurface_rz_points[1])

    # test that the coordinate transform functions are inverses of each other
    # use the test tau isosurface r,z points for this
    toroidal_conversion = cylindrical_to_toroidal(
        R_0=R_0_test,
        z_0=z_0_test,
        R=tau_test_isosurface_rz_points[0],
        Z=tau_test_isosurface_rz_points[1],
    )
    cylindrical_conversion = toroidal_to_cylindrical(
        R_0=R_0_test,
        z_0=z_0_test,
        tau=toroidal_conversion[0],
        sigma=toroidal_conversion[1],
    )
    # assert that the converted coordinates match the original coordinates
    np.testing.assert_almost_equal(cylindrical_conversion, tau_test_isosurface_rz_points)

    # generate tau and sigma isosurfaces and test that the max and min r&z points are as
    # expected
    # tau isosurface test:
    tau_input = [0.5]
    sigma_input = np.linspace(-np.pi, np.pi, 200)
    rs, zs = toroidal_to_cylindrical(
        R_0=R_0_test, z_0=z_0_test, sigma=sigma_input, tau=tau_input
    )
    rs_max = np.max(rs)
    rs_min = np.min(rs)
    zs_max = np.max(zs)
    zs_min = np.min(zs)
    np.testing.assert_almost_equal(rs_max, tau_test_r_max)
    np.testing.assert_almost_equal(rs_min, tau_test_r_min)
    np.testing.assert_almost_equal(zs_max, tau_test_z_max)
    np.testing.assert_almost_equal(zs_min, tau_test_z_min)

    # sigma isosurface test
    sigma_input = [0.5]
    tau_input = np.linspace(0, 5, 200)
    rs, zs = toroidal_to_cylindrical(
        R_0=R_0_test, z_0=z_0_test, sigma=sigma_input, tau=tau_input
    )
    rs_max = np.max(rs)
    rs_min = np.min(rs)
    zs_max = np.max(zs)
    zs_min = np.min(zs)
    np.testing.assert_almost_equal(rs_max, sigma_test_r_max)
    np.testing.assert_almost_equal(rs_min, sigma_test_r_min)
    np.testing.assert_almost_equal(zs_max, sigma_test_z_max)
    np.testing.assert_almost_equal(zs_min, sigma_test_z_min)
