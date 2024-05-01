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

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.tools import (
    NumpyJSONEncoder,
    asciistr,
    cartesian_to_polar,
    clip,
    compare_dicts,
    consec_repeat_elem,
    cross,
    deprecation_wrapper,
    dot,
    get_class_from_module,
    get_module,
    is_num,
    levi_civita_tensor,
    norm,
    polar_to_cartesian,
    write_csv,
    write_geometry_to_csv,
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
    def test_csv_writer(self, tmp_path):
        # Some dummy data to write to file
        x_vals = [0, 1, 2]
        z_vals = [-1, 0, 1]
        flux_vals = [10, 15, 20]
        data = np.array([x_vals, z_vals, flux_vals]).T
        header = "This is a test\nThis is a second line"
        col_names = ["x", "z", "heat_flux"]

        # Write the data to csv, using default extension and comment style
        test_output_base = tmp_path.as_posix() + "csv_write_dummy_data"
        write_csv(data, test_output_base, col_names, header)

        # Retrieve data file to compare
        expected_csv_output =Path(__file__, "test_data","test_csv_writer.csv")

        # Compare
        test_csv_output = test_output_base + ".csv"

        assert filecmp.cmp(test_csv_output, expected_csv_output)

        # Clean up
        Path(test_csv_output).unlink()

        # Write the data to csv, using modified extension and comment style
        ext = ".txt"
        comment_char = "!"
        write_csv(data, test_output_base, col_names, header, ext, comment_char)

        # Retrieve data file to compare
        expected_txt_output = (
            get_bluemira_root() + "/tests/utilities/test_data/test_csv_writer.txt"
        )

        # Compare
        test_txt_output = test_output_base + ".txt"
        assert filecmp.cmp(test_txt_output, expected_txt_output)

    def test_write_geometry_to_csv(self, tmp_path):
        # Define simple loop
        loop = make_polygon([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True)

        # Write out the loop
        test_output_base = tmp_path.as_posix() + "loop_test_write"
        metadata = "Metadata string"
        write_geometry_to_csv(loop, test_output_base, metadata)

        # Fetch comparison data file
        expected_csv_output = (
            get_bluemira_root() + "/tests/utilities/test_data/loop_test_data.csv"
        )

        # Compare generated data to data file
        test_file = test_output_base + ".csv"
        assert filecmp.cmp(test_file, expected_csv_output)


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
