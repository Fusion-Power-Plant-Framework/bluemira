# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.constants import EPS, raw_uc
from bluemira.geometry.tools import Coordinates
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def make_xs_from_bd(b, d):
    return Coordinates(
        {
            "x": [-b, b, b, -b],
            "y": 0,
            "z": [-d, -d, d, d],
        }
    )


class TestPolyhedralMaths:
    @classmethod
    def setup_class(cls):
        cls.trap = TrapezoidalPrismCurrentSource(
            [10, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.5, 0.5, 40, 40, current=1
        )
        cls.poly = PolyhedralPrismCurrentSource(
            [10, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            make_xs_from_bd(0.5, 0.5),
            40,
            40,
            current=1,
        )

    def test_geometry(self):
        self.poly.plot()
        ax = plt.gca()
        self.trap.plot(ax)
        colors = ["r", "g", "b", "pink", "cyan", "yellow"]
        for i, normal in enumerate(self.poly.face_normals):
            points = self.poly.face_points[i]
            centre = np.sum(points[:3], axis=0) / 3
            ax.quiver(*centre, *normal, color=colors[i])

        for i, points in enumerate(self.poly.face_points):
            for point in points:
                ax.plot(*point, marker="o", ms=int(50 / (i + 1)), color=colors[i])

        plt.show()

    def test_xz_field(self):
        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        n = 50
        x = np.linspace(8, 12, n)
        z = np.linspace(-2, 2, n)
        xx, zz = np.meshgrid(x, z)
        yy = 0.0 * np.ones_like(xx)

        self.trap.plot(ax)
        Bx, By, Bz = self.trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, B, zz, zdir="y", offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title("PolyhedralPrism")
        self.poly.plot(ax)

        Bx, By, Bz = self.poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, B_new, zz, zdir="y", offset=0)
        f.colorbar(cm)
        plt.show()
        np.testing.assert_allclose(B_new, B)

    def test_xy_field(self):
        n = 50
        x = np.linspace(8, 12, n)
        y = np.linspace(-2, 2, n)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        self.trap.plot(ax)

        Bx, By, Bz = self.trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, yy, B, zdir="z", offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title("PolyhedralPrism")
        self.poly.plot(ax)
        Bx, By, Bz = self.poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, yy, B_new, zdir="z", offset=0)
        f.colorbar(cm)
        plt.show()
        np.testing.assert_allclose(B_new, B)

    def test_yz_field(self):
        n = 50
        y = np.linspace(-2, 2, n)
        z = np.linspace(-2, 2, n)
        yy, zz = np.meshgrid(y, z)
        xx = 10 * np.ones_like(yy)

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        self.trap.plot(ax)
        Bx, By, Bz = self.trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(B, yy, zz, zdir="x", offset=10)
        f.colorbar(cm)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        ax.set_title("PolyhedralPrism")
        self.poly.plot(ax)
        Bx, By, Bz = self.poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(B_new, yy, zz, zdir="x", offset=10)
        f.colorbar(cm)
        plt.show()
        np.testing.assert_allclose(B_new, B)


class TestPolyhedralPrismBabicAykel:
    @classmethod
    def setup_class(cls):
        """
        Verification test.

        Babic and Aykel example

        https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594
        """
        # Babic and Aykel example (single trapezoidal prism)
        cls.trap = TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([2 * 2.154700538379251, 0, 0]),  # This gives b=1
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            1,
            1,
            60.0,
            30.0,
            4e5,
        )
        cls.poly = PolyhedralPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([2 * 2.154700538379251, 0, 0]),  # This gives b=1
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            make_xs_from_bd(1, 1),
            60.0,
            30.0,
            4e5,
        )

    @pytest.mark.parametrize("plane", ["x", "y", "z"])
    def test_plot(self, plane):
        n = 50
        x1, x2 = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx3 = np.zeros_like(xx1)

        if plane == "x":
            xx, yy, zz = xx3, xx1, xx2
            i, j, k = 3, 1, 2
        elif plane == "y":
            xx, yy, zz = xx1, xx3, xx2
            i, j, k = 0, 3, 2
        elif plane == "z":
            xx, yy, zz = xx1, xx2, xx3
            i, j, k = 0, 1, 3

        f = plt.figure()
        ax = f.add_subplot(1, 3, 1, projection="3d")
        ax.set_title("TrapezoidalPrism")
        self.trap.plot(ax)
        Bx, By, Bz = self.trap.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        args = [xx, yy, zz, B]

        cm = ax.contourf(args[i], args[j], args[k], zdir=plane, offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 3, 2, projection="3d")
        ax.set_title("PolyhedralPrism")
        self.poly.plot(ax)
        Bx, By, Bz = self.poly.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)

        ax = f.add_subplot(1, 3, 3, projection="3d")
        ax.set_title("difference [%]")
        args_diff = [xx, yy, zz, 100 * (B - B_new) / B]
        self.poly.plot(ax)
        cm = ax.contourf(args_diff[i], args_diff[j], args_diff[k], zdir=plane, offset=0)
        f.colorbar(cm)
        plt.show()
        np.testing.assert_allclose(B_new, B)

    @pytest.mark.parametrize(
        ("point", "value", "precision"),
        [((2, 2, 2), 15.5533805, 7), ((1, 1, 1), 53.581000397, 9)],
    )
    def test_paper_values(self, point, value, precision):
        field = self.poly.field(*point)
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**precision) / 10**precision
        assert field_ndecimals == pytest.approx(value, rel=0, abs=EPS)


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

    @pytest.mark.parametrize("plane", ["x", "y", "z"])
    def test_hexagon(self, plane):
        n = 50
        x1, x2 = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx3 = np.zeros_like(xx1)

        if plane == "x":
            xx, yy, zz = xx3, xx1, xx2
            i, j, k = 3, 1, 2
        elif plane == "y":
            xx, yy, zz = xx1, xx3, xx2
            i, j, k = 0, 3, 2
        elif plane == "z":
            xx, yy, zz = xx1, xx2, xx3
            i, j, k = 0, 1, 3

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
        self.hexagon.plot(ax)
        ax.set_title("HexagonPrism")
        Bx, By, Bz = self.hexagon.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)
        plt.show()

    @pytest.mark.parametrize("plane", ["x", "y", "z"])
    def test_triangle(self, plane):
        n = 50
        x1, x2 = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx3 = np.zeros_like(xx1)

        if plane == "x":
            xx, yy, zz = xx3, xx1, xx2
            i, j, k = 3, 1, 2
        elif plane == "y":
            xx, yy, zz = xx1, xx3, xx2
            i, j, k = 0, 3, 2
        elif plane == "z":
            xx, yy, zz = xx1, xx2, xx3
            i, j, k = 0, 1, 3

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection="3d")
        self.triangle.plot(ax)
        ax.set_title("TrianglePrism")
        Bx, By, Bz = self.triangle.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)
        plt.show()
