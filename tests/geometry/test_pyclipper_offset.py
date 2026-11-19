# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial import KDTree

from bluemira.base.file import get_bluemira_path
from bluemira.geometry._pyclipr_offset import offset_clipper
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError


class TestClipperOffset:
    options = ("square", "round", "miter")

    # fmt: off
    x = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -4)
    y = (0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 0)
    # fmt: on

    @pytest.mark.parametrize("method", options)
    @pytest.mark.parametrize("delta", [1.0, 1e-3, 1e6, -1.0, -1e-3])
    @pytest.mark.parametrize(("x", "y"), [(x, y), (x[::-1], y[::-1])])
    def test_complex_polygon(self, x, y, delta, method):
        rng = np.random.default_rng()
        coordinates = Coordinates({"x": x, "y": y, "z": rng.random()})
        c = offset_clipper(coordinates, delta, method=method)

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(c.x, c.y, "r", marker="o")
        ax.set_aspect("equal")

        distance, _ = self._offset_distance_range(coordinates, c)
        np.testing.assert_allclose(
            distance,
            abs(delta),
            rtol=1e-5,  # ~5 significant digits
            atol=1e-5,  # ~ 0.01 mm tolerance
        )

    @pytest.mark.parametrize("method", options)
    def test_complex_polygon_overoffset_raises_error(self, method):
        coordinates = Coordinates({"x": self.x, "y": self.y, "z": 0})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, -30, method=method)

    def test_blanket_offset(self):
        fp = get_bluemira_path("geometry/test_data", subfolder="tests")
        with open(Path(fp, "bb_offset_test.json"), "rb") as file:
            data = json.load(file)
        coordinates = Coordinates(data)
        offsets = []
        for m in ["miter", "square", "round"]:
            offset_coordinates = offset_clipper(coordinates, 1.5, method=m)
            offsets.append(offset_coordinates)
            distance, _ = self._offset_distance_range(coordinates, offset_coordinates)
            np.testing.assert_almost_equal(distance, 1.5, decimal=5)

        _, ax = plt.subplots()
        ax.plot(coordinates.x, coordinates.z, color="k")
        colors = ["r", "g", "y"]
        for offset_coordinates, c in zip(offsets, colors, strict=False):
            ax.plot(offset_coordinates.x, offset_coordinates.z, color=c)
        ax.set_aspect("equal")

    @pytest.mark.parametrize("method", options)
    @pytest.mark.parametrize("delta", [0.5, -0.4, 1e3])
    def test_finely_discretised_wavy_polygon(self, method, delta):
        """
        Benchmark: 1000-vertex curve with alternating convex/concave curvature.
        -0.4 forces many local self-intersections to be resolved, 1e3 is a huge
        offset relative to the feature size (many ROUND arc segments).
        """
        t = np.linspace(0, 2 * np.pi, 1001)[:-1]
        r = 3 + 0.8 * np.sin(12 * t)
        coordinates = Coordinates({"x": 9 + r * np.cos(t), "y": 0, "z": r * np.sin(t)})
        coordinates.close()

        t0 = time.perf_counter()
        c = offset_clipper(coordinates, delta, method=method)
        elapsed = time.perf_counter() - t0

        assert c.closed
        distance, max_distance = self._offset_distance_range(coordinates, c)
        if method == "miter":
            # At |delta| >> edge length, neighbouring miter tips can cover an edge's
            # offset segment entirely, so only check that the offset never undershoots
            assert distance >= abs(delta) * (1 - 1e-5) - 1e-5
        else:
            np.testing.assert_allclose(distance, abs(delta), rtol=1e-5, atol=1e-5)
        if method == "round":
            # A round offset must not overshoot anywhere (arc tolerance check)
            assert max_distance <= abs(delta) * (1 + 1e-5) + 1e-5
        assert elapsed < 0.25  # pyclipper (Clipper1): ~0.9 s for round 1e3

    def test_wrong_method(self):
        coordinates = Coordinates({"x": [0, 1, 2, 0], "y": [0, 1, -1, 0]})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method="fail")

    @pytest.mark.parametrize("method", options)
    def test_open_polygon_raises_error(self, method):
        coordinates = Coordinates({"x": [0, 1, 2], "y": [0, 1, -1]})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method=method)

    @pytest.mark.parametrize("method", options)
    def test_non_planar_polygon_raises_error(self, method):
        coordinates = Coordinates({
            "x": [0, 1, 2, 0],
            "y": [0, 1, -1, 0],
            "z": [1, 0, 1, 0],
        })
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method=method)

    @staticmethod
    def _offset_distance_range(coordinates, offset_coordinates, n=10, k=8):
        """
        Min and max distance from points along the offset polygon to the original
        polyline (exact point-to-segment, on the k nearest segments via a KDTree).
        """

        def densify(p):
            s = np.linspace(0, 1, n, endpoint=False)[None, :, None]
            return (p[:-1, None] + s * (p[1:] - p[:-1])[:, None]).reshape(-1, 3)

        a = coordinates.xyz.T
        a0, ab = a[:-1], np.diff(a, axis=0)
        p = densify(offset_coordinates.xyz.T)
        seg = KDTree(densify(a)).query(p, k=k)[1] // n
        ap, ab = p[:, None] - a0[seg], ab[seg]
        t = np.clip((ap * ab).sum(-1) / np.maximum((ab**2).sum(-1), 1e-300), 0, 1)
        d = np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(axis=1)
        return d.min(), d.max()
