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

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.codes.error import FreeCADError
from bluemira.display.plotter import plot_coordinates
from bluemira.geometry._deprecated_tools import (
    convert_coordinates_to_face,
    convert_coordinates_to_wire,
    coords_plane_intersect,
    get_area,
    get_intersect,
    join_intersect,
    make_face,
    make_mixed_face,
    make_mixed_wire,
    make_wire,
    offset,
    rotation_matrix,
)
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates, on_polygon
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import extrude_shape, revolve_shape

TEST_PATH = get_bluemira_path("geometry/test_data", subfolder="tests")


class TestArea:
    def test_area(self):
        """
        Checked with:
        https://www.analyzemath.com/Geometry_calculators/irregular_polygon_area.html
        """
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        assert get_area(x, y) == 29.5
        coords = Coordinates({"x": x, "y": y})
        coords.rotate(base=[3, 2, 1], direction=[42, 2, 1], degree=43)
        assert np.isclose(get_area(*coords.xyz), 29.5)

    def test_error(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        with pytest.raises(GeometryError):
            get_area(x, y[:-1])


class TestCoordinatesPlane:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_simple(self):
        coords = Coordinates({"x": [0, 1, 2, 2, 0, 0], "z": [-1, -1, -1, 1, 1, -1]})
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        e = np.array([[0, 0, 0], [2, 0, 0]])
        assert np.allclose(intersect, e)

    def test_complex(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "z": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 2

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)

        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 2.7], [1, 0, 2.7], [0, 1, 2.7]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 4

        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4], [1, 0, 4], [0, 1, 4]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 1
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")

            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert intersect is None

    def test_other_dims(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "y": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 0, 1])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 2

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
        plt.show()

        plane = BluemiraPlane.from_3_points([0, 10, 0], [1, 10, 0], [0, 10, 1])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert intersect is None

    def test_xyzplane(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "y": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        coords.translate([-2, 0, 0])
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = coords_plane_intersect(coords, plane)

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

    def test_flat_intersect(self):
        # test that a shared segment with plane only gives two intersects
        coords = Coordinates({"x": [0, 2, 2, 0, 0], "z": [-1, -1, 1, 1, -1]})
        plane = BluemiraPlane.from_3_points([0, 0, 1], [0, 1, 1], [1, 0, 1])
        inter = coords_plane_intersect(coords, plane)
        assert np.allclose(inter, np.array([[0, 0, 1], [2, 0, 1]]))


class TestRotationMatrix:
    def test_axes(self):
        axes = ["x", "y", "z"]
        axes2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for a1, a2 in zip(axes, axes2):
            r_1 = rotation_matrix(np.pi / 6, a1)
            r_2 = rotation_matrix(np.pi / 6, a2)
            assert np.allclose(r_1, r_2), a1

        axes = ["fail", "somthing", "1"]
        for axis in axes:
            with pytest.raises(GeometryError):
                rotation_matrix(30, axis)

    def test_ccw(self):
        p1 = [9, 0, 0]

        r_matrix = rotation_matrix(np.pi / 2, axis="z")
        p2 = r_matrix @ p1

        assert np.isclose(p2[1], 9), p2


class TestOffset:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_rectangle(self):
        # Rectangle - positive offset
        x = [1, 3, 3, 1, 1, 3]
        y = [1, 1, 3, 3, 1, 1]
        o = offset(x, y, 0.25)
        assert sum(o[0] - np.array([0.75, 3.25, 3.25, 0.75, 0.75])) == 0
        assert sum(o[1] - np.array([0.75, 0.75, 3.25, 3.25, 0.75])) == 0

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*o, "r", marker="o")
        ax.set_aspect("equal")

    def test_triangle(self):
        x = [1, 2, 1.5, 1, 2]
        y = [1, 1, 4, 1, 1]
        t = offset(x, y, -0.25)
        assert (
            abs(sum(t[0] - np.array([1.29511511, 1.70488489, 1.5, 1.29511511])) - 0)
            < 1e-3
        )
        assert abs(sum(t[1] - np.array([1.25, 1.25, 2.47930937, 1.25])) - 0) < 1e-3

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*t, "r", marker="o")
        ax.set_aspect("equal")

    def test_complex_open(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1]
        # fmt:on

        c = offset(x, y, 1)

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*c, "r", marker="o")
        ax.set_aspect("equal")

    def test_complex_closed(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -3]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 1, 0, 2]
        # fmt:on

        c = offset(x, y, 1)

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*c, "r", marker="o")
        ax.set_aspect("equal")


class TestIntersections:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    @pytest.mark.parametrize("c1, c2", [["x", "z"], ["x", "y"], ["y", "z"]])
    def test_get_intersect(self, c1, c2):
        loop1 = Coordinates({c1: [0, 0.5, 1, 2, 3, 4, 0], c2: [1, 1, 1, 1, 2, 5, 5]})
        loop2 = Coordinates({c1: [1.5, 1.5, 2.5, 2.5, 2.5], c2: [4, -4, -4, -4, 5]})
        shouldbe = [[1.5, 1], [2.5, 1.5], [2.5, 5]]
        intersect = np.array(
            get_intersect(
                getattr(loop1, "".join([c1, c2])), getattr(loop2, "".join([c1, c2]))
            )
        )
        correct = np.array(shouldbe).T
        np.testing.assert_allclose(intersect, correct)

    def test_join_intersect(self):
        loop1 = Coordinates(
            {"x": [0, 0.5, 1, 2, 3, 5, 4.5, 4, 0], "z": [1, 1, 1, 1, 2, 4, 4.5, 5, 5]}
        )
        loop2 = Coordinates({"x": [1.5, 1.5, 2.5, 2.5], "z": [4, -4, -4, 5]})
        join_intersect(loop1, loop2)

        np.testing.assert_allclose(loop1.points[3], [1.5, 0, 1])
        np.testing.assert_allclose(loop1.points[5], [2.5, 0, 1.5])
        np.testing.assert_allclose(loop1.points[10], [2.5, 0, 5])

    def test_join_intersect_arg1(self):
        tf = Coordinates.from_json(os.sep.join([TEST_PATH, "test_TF_intersect.json"]))
        lp = Coordinates.from_json(os.sep.join([TEST_PATH, "test_LP_intersect.json"]))
        eq = Coordinates.from_json(os.sep.join([TEST_PATH, "test_EQ_intersect.json"]))
        up = Coordinates.from_json(os.sep.join([TEST_PATH, "test_UP_intersect.json"]))

        _, ax = plt.subplots()
        for coords in [tf, up, eq, lp]:
            plot_coordinates(coords, ax=ax, fill=False)

        args = []
        intx, intz = [], []
        for coords in [lp, eq, up]:
            i = get_intersect(tf.xz, coords.xz)
            a = join_intersect(tf, coords, get_arg=True)
            args.extend(a)
            intx.extend(i[0])
            intz.extend(i[1])

        for coords in [tf, up, eq, lp]:
            plot_coordinates(coords, ax=ax, fill=False, points=True)
        ax.plot(*tf.xz.T[args].T, marker="o", color="r")
        ax.plot(intx, intz, marker="^", color="k")

        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args]))
        assert np.allclose(np.sort(intz), np.sort(tf.z[args]))

    def test_join_intersect_arg2(self):
        tf = Coordinates.from_json(os.path.join(TEST_PATH, "test_TF_intersect2.json"))
        lp = Coordinates.from_json(os.path.join(TEST_PATH, "test_LP_intersect2.json"))
        eq = Coordinates.from_json(os.path.join(TEST_PATH, "test_EQ_intersect2.json"))
        up = Coordinates.from_json(os.path.join(TEST_PATH, "test_UP_intersect2.json"))

        _, ax = plt.subplots()
        for coords in [tf, up, eq, lp]:
            plot_coordinates(coords, ax=ax, fill=False)

        args = []
        intx, intz = [], []
        for coords in [lp, eq, up]:
            i = get_intersect(tf.xz, coords.xz)
            a = join_intersect(tf, coords, get_arg=True)
            args.extend(a)
            intx.extend(i[0])
            intz.extend(i[1])

        ax.plot(*tf.xz.T[args].T, marker="o", color="r")
        ax.plot(intx, intz, marker="^", color="k")

        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args])), f"{intx} != {tf.x[args]}"
        assert np.allclose(np.sort(intz), np.sort(tf.z[args])), f"{intz} != {tf.z[args]}"


class TestMixedFaces:
    """
    Various tests of the MixedFaceMaker functionality. Checks the 3-D geometric
    properties of the results with some regression results done when everything was
    working correctly.
    """

    def assert_properties(self, true_props: Dict[str, Any], part: BluemiraGeo):
        """
        Helper function to pull out the properties to be compared, and to make the
        comparison in an output-friendly way.
        """
        error = False
        kwargs = {"atol": 1e-8, "rtol": 1e-5}
        keys, expected, actual = [], [], []
        for key, value in true_props.items():
            comp_method = np.allclose if isinstance(value, tuple) else np.isclose
            result = getattr(part, key, None)
            assert result is not None, f"Attribute {key} not defined on part {part}."
            if not comp_method(value, result, **kwargs):
                error = True
                keys.append(key)
                expected.append(value)
                actual.append(result)
        if error:
            assert False, list(zip(keys, expected, actual))

    @pytest.mark.parametrize(
        "filename,degree,true_props",
        [
            (
                "IB_test.json",
                100,
                {
                    "center_of_mass": (
                        3.50440,
                        4.17634,
                        1.17870,
                    ),
                    "volume": 106.080,
                    "area": 348.296,
                },
            ),
            (
                "OB_test.json",
                15,
                {
                    "center_of_mass": (
                        11.583014,
                        1.524777,
                        -0.186182,
                    ),
                    "volume": 43.0233,
                    "area": 121.5713,
                },
            ),
        ],
    )
    def test_face_revolve(self, filename, degree, true_props):
        """
        Tests some blanket faces that combine splines and polygons.
        """
        coords = Coordinates.from_json(os.sep.join([TEST_PATH, filename]))
        face = make_mixed_face(*coords.xyz)
        part = revolve_shape(face, degree=degree)
        self.assert_properties(true_props, part)

    @pytest.mark.parametrize(
        "filename,vec,true_props",
        [
            (
                "TF_case_in_test.json",
                (0, 1, 0),
                {
                    "center_of_mass": (
                        9.45877,
                        0.5,
                        -2.1217e-5,
                    ),
                    "volume": 185.185,
                    "area": 423.998,
                },
            ),
            (
                "div_test_mfm.json",
                (0, 2, 0),
                {
                    "center_of_mass": (
                        8.03233,
                        0.990000,
                        -6.44430,
                    ),
                    "volume": 4.58653,
                    "area": 29.2239,
                },
            ),
            (
                "div_test_mfm2.json",
                (0, 2, 0),
                {
                    "center_of_mass": (
                        8.03265,
                        0.9900,
                        -6.44432,
                    ),
                    "volume": 4.58959,
                    "area": 29.1868,
                },
            ),
        ],
    )
    def test_face_extrude(self, filename, vec, true_props):
        """
        Tests TF and divertor faces that combine splines and polygons.
        """
        fn = os.sep.join([TEST_PATH, filename])
        coords = Coordinates.from_json(fn)
        face = make_mixed_face(*coords.xyz)
        part = extrude_shape(face, vec=vec)
        self.assert_properties(true_props, part)

    def test_face_seg_fault(self):
        """
        Tests a particularly tricky face that can result in a seg fault...
        """
        fn = os.path.join(TEST_PATH, "divertor_seg_fault_LDS.json")
        coords = Coordinates.from_json(fn)
        face = make_mixed_face(*coords.xyz)
        true_props = {
            "area": 2.26163,
        }
        self.assert_properties(true_props, face)

    @pytest.mark.parametrize(
        "name,true_props",
        [
            (
                "shell_mixed_test",
                {
                    "area": 6.35215,
                },
            ),
            (
                "failing_mixed_shell",
                {
                    "area": 31.4998,
                },
            ),
            (
                "tf_wp_tricky",
                {
                    "area": 31.0914,
                },
            ),
        ],
    )
    def test_shell(self, name, true_props):
        """
        Tests some shell mixed faces
        """
        inner = Coordinates.from_json(os.path.join(TEST_PATH, f"{name}_inner.json"))
        outer = Coordinates.from_json(os.path.join(TEST_PATH, f"{name}_outer.json"))
        inner_wire = make_mixed_wire(*inner.xyz)
        outer_wire = make_mixed_wire(*outer.xyz)
        face = BluemiraFace([outer_wire, inner_wire])
        self.assert_properties(true_props, face)

    def test_coordinate_cleaning(self):
        fn = os.path.join(TEST_PATH, "bb_ob_bss_test.json")
        coords = Coordinates.from_json(fn)
        make_mixed_wire(*coords.xyz, allow_fallback=False)

        with pytest.raises(FreeCADError):
            make_mixed_wire(*coords.xyz, allow_fallback=False, cleaning_atol=1e-8)


class TestCoordsConversion:
    def generate_face_polygon(self, x, y, z):
        face = make_face(x, y, z, spline=False)
        converted_face = convert_coordinates_to_face(x, y, z, method="polygon")
        return face, converted_face

    def generate_face_spline(self, x, y, z):
        face = make_face(x, y, z, spline=True)
        converted_face = convert_coordinates_to_face(x, y, z, method="spline")
        return face, converted_face

    def generate_face_mixed(self, x, y, z):
        face = make_mixed_face(x, y, z)
        converted_face = convert_coordinates_to_face(x, y, z)
        return face, converted_face

    def generate_wire_polygon(self, x, y, z):
        wire = make_wire(x, y, z, spline=False)
        converted_wire = convert_coordinates_to_wire(x, y, z, method="polygon")
        return wire, converted_wire

    def generate_wire_spline(self, x, y, z):
        wire = make_wire(x, y, z, spline=True)
        converted_wire = convert_coordinates_to_wire(x, y, z, method="spline")
        return wire, converted_wire

    def generate_wire_mixed(self, x, y, z):
        wire = make_mixed_wire(x, y, z)
        converted_wire = convert_coordinates_to_wire(x, y, z)
        return wire, converted_wire

    @pytest.mark.parametrize(
        "filename,method",
        [
            ("IB_test.json", generate_face_polygon),
            ("IB_test.json", generate_face_spline),
            ("IB_test.json", generate_face_mixed),
        ],
    )
    def test_coordinates_to_face(self, filename, method):
        fn = os.sep.join([TEST_PATH, filename])
        coords = Coordinates.from_json(fn)
        face, converted_face = method(self, *coords.xyz)
        assert face.area == converted_face.area
        assert face.volume == converted_face.volume
        np.testing.assert_equal(face.center_of_mass, converted_face.center_of_mass)

    @pytest.mark.parametrize(
        "filename,method",
        [
            ("IB_test.json", generate_wire_polygon),
            ("IB_test.json", generate_wire_spline),
            ("IB_test.json", generate_wire_mixed),
        ],
    )
    def test_coordinates_to_wire_polygon(self, filename, method):
        fn = os.sep.join([TEST_PATH, filename])
        coords = Coordinates.from_json(fn)
        wire, converted_wire = method(self, *coords.xyz)
        assert wire.area == converted_wire.area
