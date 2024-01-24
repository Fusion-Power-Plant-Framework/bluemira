# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.geometry.tools import Coordinates
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.polyhedral_prism import (
    Bottura,
    Fabbri,
    PolyhedralPrismCurrentSource,
)
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource
from tests.magnetostatics.setup_methods import make_xs_from_bd, plane_setup


class TestPolyhedralInstantiation:
    def test_diff_angle_error(self):
        with pytest.raises(MagnetostaticsError):
            PolyhedralPrismCurrentSource(
                [10, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                make_xs_from_bd(0.5, 0.5),
                40,
                39.5,
                current=1,
            )


class TestPolyhedralMaths:
    same_angle_1 = (
        TrapezoidalPrismCurrentSource(
            [10, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.5, 0.5, 40, 40, current=1
        ),
        PolyhedralPrismCurrentSource(
            [10, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            make_xs_from_bd(0.5, 0.5),
            40,
            40,
            current=1,
        ),
    )

    same_angle_2 = (
        TrapezoidalPrismCurrentSource(
            [10, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.5, 0.5, 0, 0, current=1
        ),
        PolyhedralPrismCurrentSource(
            [10, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            make_xs_from_bd(0.5, 0.5),
            0,
            0,
            current=1,
        ),
    )
    test_cases = (same_angle_1, same_angle_2)

    @pytest.mark.parametrize("kernel", ["Fabbri", "Bottura"])
    @pytest.mark.parametrize(("trap", "poly"), test_cases)
    def test_geometry(
        self,
        kernel: str,
        trap: TrapezoidalPrismCurrentSource,
        poly: PolyhedralPrismCurrentSource,
    ):
        poly._kernel = Fabbri() if kernel == "Fabbri" else Bottura()
        poly.plot()
        ax = plt.gca()
        trap.plot(ax)
        colors = ["r", "g", "b", "pink", "cyan", "yellow"]
        for i, normal in enumerate(poly._face_normals):
            points = poly._face_points[i]
            centre = np.sum(points[:3], axis=0) / 3
            ax.quiver(*centre, *normal, color=colors[i])

        for i, points in enumerate(poly._face_points):
            for point in points:
                ax.plot(*point, marker="o", ms=int(50 / (i + 1)), color=colors[i])

        for i in range(len(trap._points)):
            np.testing.assert_allclose(trap._points[i], poly._points[i])

    @pytest.mark.parametrize("kernel", ["Fabbri", "Bottura"])
    @pytest.mark.parametrize(("trap", "poly"), test_cases)
    def test_xz_field(
        self,
        kernel: str,
        trap: TrapezoidalPrismCurrentSource,
        poly: PolyhedralPrismCurrentSource,
    ):
        poly._kernel = Fabbri() if kernel == "Fabbri" else Bottura()
        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        n = 50
        x = np.linspace(8, 12, n)
        z = np.linspace(-2, 2, n)
        xx, zz = np.meshgrid(x, z)
        yy = 0.0 * np.ones_like(xx)

        trap.plot(ax)
        Bx, By, Bz = trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, B, zz, zdir="y", offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title(f"PolyhedralPrism {kernel}")
        poly.plot(ax)
        Bx, By, Bz = poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, B_new, zz, zdir="y", offset=0)
        f.colorbar(cm)

        np.testing.assert_allclose(B_new, B)

    @pytest.mark.parametrize("kernel", ["Fabbri", "Bottura"])
    @pytest.mark.parametrize(("trap", "poly"), test_cases)
    def test_xy_field(
        self,
        kernel: str,
        trap: TrapezoidalPrismCurrentSource,
        poly: PolyhedralPrismCurrentSource,
    ):
        poly._kernel = Fabbri() if kernel == "Fabbri" else Bottura()
        n = 50
        x = np.linspace(8, 12, n)
        y = np.linspace(-2, 2, n)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        trap.plot(ax)

        Bx, By, Bz = trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, yy, B, zdir="z", offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title(f"PolyhedralPrism {kernel}")
        poly.plot(ax)
        Bx, By, Bz = poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, yy, B_new, zdir="z", offset=0)
        f.colorbar(cm)

        np.testing.assert_allclose(B_new, B)

    @pytest.mark.parametrize("kernel", ["Fabbri", "Bottura"])
    @pytest.mark.parametrize(("trap", "poly"), test_cases)
    def test_yz_field(
        self,
        kernel: str,
        trap: TrapezoidalPrismCurrentSource,
        poly: PolyhedralPrismCurrentSource,
    ):
        poly._kernel = Fabbri() if kernel == "Fabbri" else Bottura()
        n = 50
        y = np.linspace(-2, 2, n)
        z = np.linspace(-2, 2, n)
        yy, zz = np.meshgrid(y, z)
        xx = 10 * np.ones_like(yy)

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        trap.plot(ax)
        Bx, By, Bz = trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(B, yy, zz, zdir="x", offset=10)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title(f"PolyhedralPrism {kernel}")
        poly.plot(ax)
        Bx, By, Bz = poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(B_new, yy, zz, zdir="x", offset=10)
        f.colorbar(cm)

        np.testing.assert_allclose(B_new, B)


class TestPolyhedralCoordinates:
    @classmethod
    def setup_class(cls):
        coords = Coordinates(
            {
                "x": [1, 0.5, -0.5, -1, -0.5, 0.5],
                "z": [
                    0,
                    0.5 * np.sqrt(3),
                    0.5 * np.sqrt(3),
                    0,
                    -0.5 * np.sqrt(3),
                    -0.5 * np.sqrt(3),
                ],
            }
        )
        cls.hexagon = PolyhedralPrismCurrentSource(
            [0, 0, 0],
            [10, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            coords,
            10,
            10,
            1e6,
        )
        coords = Coordinates(
            {
                "x": [-1, 1, 0],
                "z": [-0.5, -0.5, 0.25],
            }
        )
        cls.triangle = PolyhedralPrismCurrentSource(
            [0, 0, 0],
            [10, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            coords,
            10,
            10,
            1e6,
        )

    @pytest.mark.parametrize("plane", ["y", "z"])
    def test_hexagon(self, plane):
        n = 50
        xx, yy, zz, i, j, k = plane_setup(plane, x_min=-10, x_max=10, n=n)

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
        self.hexagon.plot(ax)
        ax.set_title("HexagonPrism")
        Bx, By, Bz = self.hexagon.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)

        if plane == "y":
            np.testing.assert_allclose(B[:, : n // 2], B[:, n // 2 :][::-1][::-1, ::-1])
        else:
            np.testing.assert_allclose(B[: n // 2, :], B[n // 2 :, :][::-1])

    @pytest.mark.parametrize("plane", ["y", "z"])
    def test_triangle(self, plane):
        n = 50
        xx, yy, zz, i, j, k = plane_setup(plane, x_min=-10, x_max=10, n=n)

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
        self.triangle.plot(ax)
        ax.set_title("TrianglePrism")
        Bx, By, Bz = self.triangle.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)
        if plane == "z":
            np.testing.assert_allclose(B[:, : n // 2], B[:, n // 2 :][::-1][::-1, ::-1])
        else:
            np.testing.assert_allclose(B[:, : n // 2], B[:, n // 2 :][::-1][::-1, ::-1])


class TestCombinedShapes:
    @classmethod
    def setup_class(cls):
        current = 1e6
        coords = Coordinates({"x": [-1, -1, 1, 1], "z": [1, -1, -1, 1]})
        cls.square = PolyhedralPrismCurrentSource(
            [0, 0, 0],
            [10, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            coords,
            10,
            10,
            current,
        )
        coords = Coordinates({"x": [-1, -1, 1], "z": [1, -1, -1]})
        cls.triangle1 = PolyhedralPrismCurrentSource(
            [0, 0, 0],
            [10, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            coords,
            10,
            10,
            current / 2,
        )
        coords = Coordinates({"x": [-1, 1, 1], "z": [1, -1, 1]})
        cls.triangle2 = PolyhedralPrismCurrentSource(
            [0, 0, 0],
            [10, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            coords,
            10,
            10,
            current / 2,
        )

    @pytest.mark.parametrize("plane", ["x", "y", "z"])
    def test_plot(self, plane):
        xx, yy, zz, i, j, k = plane_setup(plane)

        f = plt.figure()
        ax = f.add_subplot(1, 3, 1, projection="3d")
        ax.set_title("Sqaure")
        self.square.plot(ax)
        Bx, By, Bz = self.square.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        args = [xx, yy, zz, B]
        cm = ax.contourf(args[i], args[j], args[k], zdir=plane, offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 3, 2, projection="3d")
        ax.set_title("CombinedTriangles")
        self.triangle1.plot(ax)
        self.triangle2.plot(ax)
        Bx, By, Bz = self.triangle1.field(xx, yy, zz) + self.triangle2.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 3, 3, projection="3d")
        ax.set_title("difference [%]")
        args_diff = [xx, yy, zz, 100 * (B - B_new) / B]
        self.triangle1.plot(ax)
        self.triangle2.plot(ax)
        cm = ax.contourf(args_diff[i], args_diff[j], args_diff[k], zdir=plane, offset=0)
        f.colorbar(cm)
        np.testing.assert_allclose(B_new, B)
