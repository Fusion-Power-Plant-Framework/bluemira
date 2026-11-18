# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad

from bluemira.base.constants import MU_0, MU_0_4PI
from bluemira.display.auto_config import plot_defaults
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.magnetostatics.biot_savart import BiotSavartFilament, Bz_coil_axis
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.greens import (
    circular_coil_inductance_elliptic,
    circular_coil_inductance_kirchhoff,
    greens_all,
)


def subdivide(points: np.ndarray, n: int) -> np.ndarray:
    """Split every segment of a polyline into n collinear pieces."""
    pieces = [np.linspace(p0, p1, n + 1)[:-1] for p0, p1 in pairwise(points)]
    return np.vstack([*pieces, points[-1:]])


def circle(radius: float, n: int) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[radius * np.cos(theta), radius * np.sin(theta), np.zeros(n + 1)]


def square(side: float) -> np.ndarray:
    h = 0.5 * side
    return np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h], [-h, 0, -h]])


def regular_polygon(circumradius: float, n: int, z: float = 0.0) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[
        circumradius * np.cos(theta), circumradius * np.sin(theta), np.full(n + 1, z)
    ]


def random_points(n: int, scale: float = 1.5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-scale, scale, (n, 3))


def numerical_curl(func, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    jac = np.zeros((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        jac[:, k] = (func(*(point + step)) - func(*(point - step))) / (2 * h)
    return np.array([
        jac[2, 1] - jac[1, 2],
        jac[0, 2] - jac[2, 0],
        jac[1, 0] - jac[0, 1],
    ])


def line_integral(func, path: np.ndarray, n_sub: int = 20) -> float:
    """Gauss-Legendre line integral of a vector function along a polyline."""
    nodes, weights = np.polynomial.legendre.leggauss(n_sub)
    total = 0.0
    for p0, p1 in pairwise(path):
        d = p1 - p0
        pts = p0 + 0.5 * (nodes[:, None] + 1) * d
        total += 0.5 * np.sum(weights * (d @ func(*pts.T)))
    return total


def plot_errors(x, z, analytical, numerical, errors, error_labels):
    """
    Plot the analytical and Biot-Savart fields and their errors, one row per
    component. Not a test.
    """
    plot_defaults()
    f, ax = plt.subplots(3, 4)
    for row, name in enumerate(["x", "z", "p"]):
        ax[row, 0].contourf(x, z, analytical[row])
        ax[row, 0].set_title(f"$B_{{{name}}}$ Green's function")
        ax[row, 1].contourf(x, z, numerical[row])
        ax[row, 1].set_title(f"$B_{{{name}}}$ Biot-Savart")
        cm = ax[row, 2].contourf(x, z, errors[row])
        ax[row, 2].set_title(error_labels[row])
        f.colorbar(cm, ax[row, 3])
        for col in range(3):
            ax[row, col].set_aspect("equal")
        ax[row, 3].set_aspect(20)


class TestCircularLoopField:
    """
    Verify the field of a discretised circular BiotSavartFilament against the
    analytical Green's functions of a circular coil.
    """

    x_coil = 4.0

    @classmethod
    def setup_class(cls):
        circle = make_circle(radius=cls.x_coil, center=(0, 0, 0), axis=(0, 0, 1))
        cls.filament = BiotSavartFilament(circle.discretise(ndiscr=2000), radius=1e-13)
        cls.filament.plot()

    def test_poloidal_field(self):
        x, z = np.meshgrid(
            np.linspace(0.1, 10, 100), np.linspace(-5, 5, 100), indexing="ij"
        )
        _, bx, bz = greens_all(self.x_coil, 0, x, z)
        bx2, by2, bz2 = self.filament.field(x, np.zeros_like(x), z)

        # Bx and Bz change sign in the domain, so their errors are measured relative
        # to the peak field rather than point-wise
        b_max = np.max(np.hypot(bx, bz))
        bp, bp2 = np.hypot(bx, bz), np.hypot(bx2, bz2)
        plot_errors(
            x,
            z,
            (bx, bz, bp),
            (bx2, bz2, bp2),
            (100 * (bx2 - bx) / b_max, 100 * (bz2 - bz) / b_max, 100 * (bp2 - bp) / bp),
            ("Error / max $B_p$ [%]", "Error / max $B_p$ [%]", "Relative error [%]"),
        )
        np.testing.assert_allclose(bx2, bx, rtol=0, atol=5e-5 * b_max)
        np.testing.assert_allclose(bz2, bz, rtol=0, atol=2e-4 * b_max)
        np.testing.assert_allclose(bp2, bp, rtol=1e-4)
        np.testing.assert_allclose(by2, 0, atol=1e-12 * b_max)

    def test_on_axis_field(self):
        z = np.linspace(-10, 10, 100)
        _, _, bz = self.filament.field(np.zeros_like(z), np.zeros_like(z), z)
        expected = MU_0 * self.x_coil**2 / (2 * (z**2 + self.x_coil**2) ** 1.5)
        np.testing.assert_allclose(bz, expected, rtol=1e-5)


class TestCircularLoop:
    loop_radius = 4.0
    filament = BiotSavartFilament(circle(loop_radius, 720), 1e-6)

    def test_field_matches_greens(self):
        x, z = np.meshgrid(
            np.linspace(0.1, 10, 40), np.linspace(-5, 5, 40), indexing="ij"
        )
        # Stay away from the conductor itself
        mask = np.hypot(x - self.loop_radius, z) > 0.2
        _, bx, bz = greens_all(self.loop_radius, 0, x, z)
        bx2, by2, bz2 = self.filament.field(x, np.zeros_like(x), z)
        scale = np.hypot(bx, bz)[mask].max()
        assert np.max(np.abs(bx2 - bx)[mask]) < 2e-4 * scale
        assert np.max(np.abs(bz2 - bz)[mask]) < 2e-4 * scale
        assert np.max(np.abs(by2)) < 1e-12

    def test_field_on_axis(self):
        z = np.linspace(-10, 10, 51)
        _, _, bz = self.filament.field(np.zeros_like(z), np.zeros_like(z), z)
        np.testing.assert_allclose(bz, Bz_coil_axis(self.loop_radius, pz=z), rtol=2e-5)

    def test_polygon_convergence_is_second_order(self):
        """Curved geometry is approximated by a polygon; error ~ (segment length)^2."""
        errors = []
        for n in [20, 40, 80]:
            f = BiotSavartFilament(circle(self.loop_radius, n), 1e-6)
            errors.append(abs(f.field(0, 0, 0)[2] / Bz_coil_axis(self.loop_radius) - 1))
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        np.testing.assert_allclose(ratios, 4, rtol=0.05)

    def test_curl_potential_equals_field(self):
        theta = np.linspace(0, 2 * np.pi, 41)
        coords = np.c_[2 * np.cos(theta), 2 * np.sin(theta), 0.3 * np.sin(3 * theta)]
        f = BiotSavartFilament(coords, 0.05)
        for point in np.array([[1.2, 0.4, 0.5], [0.0, 0.0, 0.0], [3.0, -1.0, -0.7]]):
            np.testing.assert_allclose(
                numerical_curl(f.potential, point),
                f.field(*point),
                rtol=1e-6,
                atol=1e-6 * np.linalg.norm(f.field(*point)),
            )


class TestStraightSegment:
    radius = 0.01
    start = np.array([0.1, -0.2, -1.0])
    end = np.array([-0.1, 0.3, 1.0])

    @pytest.mark.parametrize(
        "point", [[0.3, 0.2, 0.4], [1.0, -1.0, 2.5], [0.0, 0.5, 0.0]]
    )
    def test_field_matches_numerical_biot_savart(self, point):
        """Outside the conductor the exact thin filament result is recovered."""
        point = np.array(point)
        length = np.linalg.norm(self.end - self.start)
        e_hat = (self.end - self.start) / length

        def integrand(s, k):
            r = point - (self.start + s * e_hat)
            return np.cross(e_hat, r)[k] / np.linalg.norm(r) ** 3

        expected = [
            MU_0_4PI * quad(integrand, 0, length, args=(k,), epsabs=1e-14)[0]
            for k in range(3)
        ]
        filament = BiotSavartFilament(np.array([self.start, self.end]), self.radius)
        np.testing.assert_allclose(
            filament.field(*point), expected, rtol=1e-10, atol=1e-20
        )

    @pytest.mark.parametrize("point", [[0.3, 0.2, 0.4], [1.0, -1.0, 2.5]])
    def test_potential_matches_numerical_integral(self, point):
        point = np.array(point)
        length = np.linalg.norm(self.end - self.start)
        e_hat = (self.end - self.start) / length

        def integrand(s):
            return 1 / np.linalg.norm(point - (self.start + s * e_hat))

        expected = MU_0_4PI * quad(integrand, 0, length, epsabs=1e-14)[0] * e_hat
        filament = BiotSavartFilament(np.array([self.start, self.end]), self.radius)
        np.testing.assert_allclose(filament.potential(*point), expected, rtol=1e-10)

    def test_current_direction(self):
        """Current flows from the first point to the second (right-hand rule)."""
        up = BiotSavartFilament(np.array([[0, 0, -1], [0, 0, 1]]), self.radius)
        down = BiotSavartFilament(np.array([[0, 0, 1], [0, 0, -1]]), self.radius)
        assert up.field(1, 0, 0)[1] > 0
        assert up.potential(1, 0, 0)[2] > 0
        np.testing.assert_allclose(down.field(1, 0, 0), -up.field(1, 0, 0))


class TestLongWire:
    """A long segment behaves as an infinite wire with uniform current density."""

    radius = 0.01
    current = 1e3
    wire = BiotSavartFilament(
        np.array([[0, 0, -1e4], [0, 0, 1e4]]), radius, current=current
    )

    @pytest.mark.parametrize("rho", [0.0, 0.002, 0.005, 0.01, 0.03, 1.0])
    def test_field(self, rho):
        inside = rho < self.radius
        expected = (
            MU_0
            * self.current
            / (2 * np.pi)
            * (rho / self.radius**2 if inside else 1 / rho)
        )
        b = self.wire.field(rho, 0, 0)
        np.testing.assert_allclose(b, [0, expected, 0], rtol=1e-8, atol=1e-15)

    @pytest.mark.parametrize("rho", [0.0, 0.004, 0.008])
    def test_potential_inside(self, rho):
        """A_z(rho) - A_z(a) = mu_0 I / (4 pi) (1 - rho^2 / a^2) inside the wire."""
        a_rho = self.wire.potential(rho, 0, 0)[2]
        a_edge = self.wire.potential(self.radius, 0, 0)[2]
        expected = MU_0_4PI * self.current * (1 - rho**2 / self.radius**2)
        np.testing.assert_allclose(a_rho - a_edge, expected, rtol=1e-8)

    def test_potential_outside(self):
        """A_z(r1) - A_z(r2) = mu_0 I / (2 pi) ln(r2 / r1) outside the wire."""
        r1, r2 = 0.05, 2.0
        diff = self.wire.potential(r1, 0, 0)[2] - self.wire.potential(r2, 0, 0)[2]
        expected = MU_0 * self.current / (2 * np.pi) * np.log(r2 / r1)
        np.testing.assert_allclose(diff, expected, rtol=1e-8)

    def test_field_continuous_at_surface(self):
        eps = 1e-9
        np.testing.assert_allclose(
            self.wire.field(self.radius - eps, 0, 0),
            self.wire.field(self.radius + eps, 0, 0),
            rtol=1e-6,
        )


class TestDiscretisationInsensitivity:
    radius = 0.01
    straight = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    # Points everywhere, including inside the conductor and on its axis extension
    points = np.vstack([
        random_points(200),
        np.c_[np.full(10, 0.3 * radius), np.zeros(10), np.linspace(-1.5, 1.5, 10)],
        np.c_[np.zeros(5), np.zeros(5), [-1.2, -0.5, 0.0, 0.7, 1.4]],
    ])

    @pytest.mark.parametrize("method", ["field", "potential"])
    @pytest.mark.parametrize("n", [2, 7, 50])
    def test_straight_wire(self, method, n):
        coarse = BiotSavartFilament(self.straight, self.radius)
        fine = BiotSavartFilament(subdivide(self.straight, n), self.radius)
        expected = getattr(coarse, method)(*self.points.T)
        result = getattr(fine, method)(*self.points.T)
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-13 * np.abs(expected).max()
        )

    @pytest.mark.parametrize("method", ["field", "potential"])
    def test_polygon_loop(self, method):
        coarse = BiotSavartFilament(square(2.0), self.radius)
        fine = BiotSavartFilament(subdivide(square(2.0), 13), self.radius)
        expected = getattr(coarse, method)(*self.points.T)
        result = getattr(fine, method)(*self.points.T)
        np.testing.assert_allclose(
            result, expected, rtol=0, atol=1e-13 * np.abs(expected).max()
        )

    def test_polygon_inductance(self):
        coarse = BiotSavartFilament(square(2.0), 0.05)
        fine = BiotSavartFilament(subdivide(square(2.0), 5), 0.05)
        np.testing.assert_allclose(fine.inductance(), coarse.inductance(), rtol=1e-8)

    def test_repeated_points_ignored(self):
        repeated = np.vstack([self.straight[:1], self.straight])
        a = BiotSavartFilament(self.straight, self.radius)
        b = BiotSavartFilament(repeated, self.radius)
        np.testing.assert_allclose(b.field(*self.points.T), a.field(*self.points.T))


class TestSelfInductance:
    @staticmethod
    def _circle(radius: float) -> Coordinates:
        circle = make_circle(radius, center=(0, 0, 0), axis=(0, 1, 0))
        return circle.discretise(dl=circle.length / 50)

    def test_circular_inductance_map(self):
        """
        Overview of the inductance of circular coils over a range of coil and wire
        radii, compared with the elliptic and Kirchhoff formulae. Includes the small
        coils affected by the near-segment cutoff (see the xfail test below).
        """
        n = 25
        radii = np.linspace(1, 100, n)
        rci = np.linspace(0.0001, 1, n)
        ind, ind2, ind3 = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))

        for i, r in enumerate(radii):
            filament = self._circle(r)
            for j, rc in enumerate(rci):
                ind[i, j] = circular_coil_inductance_elliptic(r, rc)
                ind2[i, j] = circular_coil_inductance_kirchhoff(r, rc)
                ind3[i, j] = BiotSavartFilament(filament, rc).inductance()

        levels = np.linspace(np.amin(ind), np.amax(ind), 20)
        f, ax = plt.subplots(1, 4)
        xx, yy = np.meshgrid(radii, rci, indexing="ij")
        for a, values, title in zip(
            ax[:3],
            (ind, ind2, ind3),
            ("Elliptic", "Kirchhoff", "Biot-Savart"),
            strict=True,
        ):
            a.contourf(xx, yy, values, levels=levels)
            a.set_title(title)
        for a in ax:
            a.set_xlabel("radius [m]")
            a.set_ylabel("filament radius [m]")

        diff = 100 * (ind2 - ind3) / ind2
        cm = ax[3].contourf(xx, yy, diff, levels=np.linspace(-5, 5, 100))
        ax[3].set_title("(K - BS) / K")
        f.colorbar(cm).set_label("%")
        plt.subplots_adjust(wspace=0.5)

        r2 = 1 - np.sum((ind2 - ind3) ** 2) / np.sum((ind2 - np.average(ind2)) ** 2)
        assert r2 > 0.998

    @pytest.mark.parametrize("radius", [5, 10, 100])
    @pytest.mark.parametrize(
        ("r_over_a", "rtol"), [(10, 0.05), (100, 0.03), (1000, 0.02)]
    )
    def test_circular_inductance(self, radius: float, r_over_a: float, rtol: float):
        """
        Compare with Kirchhoff's formula, mu_0 R (ln(8R/a) - 7/4), for a uniform
        current density. The filament systematically underestimates it by a few %.
        """
        a = radius / r_over_a
        inductance = BiotSavartFilament(self._circle(radius), a).inductance()
        expected = circular_coil_inductance_kirchhoff(radius, a)
        np.testing.assert_allclose(inductance, expected, rtol=rtol)

    @pytest.mark.xfail(
        strict=True,
        reason="Segments shorter than ~0.5 m: the near-segment cutoff compares a "
        "squared distance with a length, and excludes neighbouring segments",
    )
    def test_circular_inductance_small_coil(self):
        inductance = BiotSavartFilament(self._circle(1.0), 0.01).inductance()
        expected = circular_coil_inductance_kirchhoff(1.0, 0.01)
        np.testing.assert_allclose(inductance, expected, rtol=0.03)

    @pytest.mark.parametrize(
        ("c", "discretise", "rtol"),
        [
            (10, False, 0.13),
            (20, False, 0.11),
            (40, False, 0.09),
            (80, False, 0.075),
            (160, False, 0.065),
            # dl = 12 a is longer than the sides, so this is not discretised further
            (10, True, 0.13),
            (20, True, 0.04),
            (40, True, 0.01),
            (80, True, 0.02),
            (160, True, 0.025),
        ],
    )
    def test_square_inductance(self, c: float, discretise: bool, rtol: float):  # noqa: FBT001
        """
        Compare with eq. (8) of https://arxiv.org/abs/1204.1486 for a square of side
        c and wire radius a, with Y = 1/2 for a uniform current density. Eq. (8)
        neglects the corners, and is itself accurate only to O(mu_0 a).
        """
        a, d, y = 1.0, c, 0.5
        square = Coordinates({
            "x": [-c / 2, c / 2, c / 2, -c / 2],
            "y": 0,
            "z": [-d / 2, -d / 2, d / 2, d / 2],
        })
        square.close()
        if discretise:
            square = make_polygon(square, closed=True).discretise(
                dl=12 * a, byedges=True
            )
        inductance = BiotSavartFilament(square, radius=a).inductance()
        expected = (
            MU_0
            / np.pi
            * (
                c * np.log(2 * c / a)
                + d * np.log(2 * d / a)
                - (c + d) * (2 - y / 2)
                + 2 * np.hypot(c, d)
                - c * np.arcsinh(c / d)
                - d * np.arcsinh(d / c)
                + a
            )
        )
        np.testing.assert_allclose(inductance, expected, rtol=rtol)


class TestInductance:
    @pytest.mark.parametrize("length", [1.0, 10.0])
    def test_straight_wire(self, length):
        """Round wire with uniform current (geometric mean distance a e^(-1/4))."""
        a = 0.01
        gmd = a * np.exp(-0.25)
        wire = BiotSavartFilament(np.array([[0, 0, 0], [0, 0, length]]), a)
        exact = MU_0_4PI * (
            2 * (length * np.arcsinh(length / gmd) - np.hypot(length, gmd) + gmd)
        )
        np.testing.assert_allclose(wire.inductance(), exact, rtol=1e-12)
        # Classical long wire limit
        classical = MU_0 * length / (2 * np.pi) * (np.log(2 * length / a) - 0.75)
        np.testing.assert_allclose(wire.inductance(), classical, rtol=2e-3)

    @pytest.mark.parametrize(("loop_radius", "a"), [(1.0, 0.01), (5.0, 0.05)])
    def test_circular_loop(self, loop_radius, a):
        f = BiotSavartFilament(circle(loop_radius, 150), a)
        np.testing.assert_allclose(
            f.inductance(), circular_coil_inductance_kirchhoff(loop_radius, a), rtol=2e-3
        )

    def test_circular_loop_converges_with_fine_discretisation(self):
        """Keeps converging when segments are shorter than the conductor radius."""
        loop_radius, a = 2.0, 0.01
        exact = circular_coil_inductance_kirchhoff(loop_radius, a)
        errors = [
            abs(BiotSavartFilament(circle(loop_radius, n), a).inductance() / exact - 1)
            for n in (200, 800, 2000)
        ]
        assert errors[0] > errors[1]
        assert errors[2] < 1e-5

    @pytest.mark.parametrize("seed", range(5))
    def test_skew_segments_match_numerical_neumann(self, seed):
        """Closed form mutual inductance of two arbitrary segments."""
        rng = np.random.default_rng(seed)
        a0, da, b0, db = rng.normal(size=(4, 3))
        b0 += 3 * np.sign(b0)  # Keep the segments apart
        nodes, weights = np.polynomial.legendre.leggauss(100)
        s, w = 0.5 * (nodes + 1), 0.5 * weights
        r = np.linalg.norm(
            (a0 + s[:, None] * da)[:, None] - (b0 + s[:, None] * db)[None], axis=-1
        )
        a = 1e-3
        gmd2 = (a * np.exp(-0.25)) ** 2
        expected = (
            MU_0_4PI * (da @ db) * np.sum(w[:, None] * w[None] / np.sqrt(r**2 + gmd2))
        )
        mutual = (
            BiotSavartFilament(
                [np.array([a0, a0 + da]), np.array([b0, b0 + db])], a
            ).inductance()
            - BiotSavartFilament(np.array([a0, a0 + da]), a).inductance()
        )
        np.testing.assert_allclose(mutual, expected, rtol=1e-10)

    @pytest.mark.parametrize("angle", [1e-4, 1e-6, 1e-7, 1e-9])
    def test_nearly_parallel_segments(self, angle):
        """Continuous through the switch to the parallel formula."""
        a, length, d = 0.01, 2.0, 0.3
        w1 = np.array([[0, 0, 0], [0, 0, length]])
        w2 = np.array([[d, 0, 0], [d + length * np.sin(angle), 0, length]])
        w2_par = w1 + np.array([d, 0, 0])
        single = BiotSavartFilament(w1, a).inductance()
        mutual = BiotSavartFilament([w1, w2], a).inductance() - single
        mutual_par = BiotSavartFilament([w1, w2_par], a).inductance() - single
        np.testing.assert_allclose(mutual, mutual_par, rtol=10 * angle + 1e-10)

    @pytest.mark.parametrize(("side", "rtol"), [(20, 6e-3), (80, 1e-3), (160, 4e-4)])
    def test_square_loop(self, side, rtol):
        """Compare with https://arxiv.org/pdf/1204.1486.pdf, error ~ O(a / side)."""
        a = 1.0
        c = d = side
        exact = (
            MU_0
            / np.pi
            * (
                c * np.log(2 * c / a)
                + d * np.log(2 * d / a)
                - (c + d) * (2 - 0.5 / 2)
                + 2 * np.hypot(c, d)
                - c * np.arcsinh(c / d)
                - d * np.arcsinh(d / c)
                + a
            )
        )
        f = BiotSavartFilament(square(side), a)
        np.testing.assert_allclose(f.inductance(), exact, rtol=rtol)

    def test_includes_other_filaments(self):
        a = 0.01
        loop_1 = circle(1.0, 60)
        loop_2 = loop_1 + np.array([0, 0, 0.5])
        single = BiotSavartFilament(loop_1, a).inductance()
        double = BiotSavartFilament([loop_1, loop_2], a).inductance()
        assert double > single
        # Mutual inductance via the flux of loop 2 through loop 1
        f2 = BiotSavartFilament(loop_2, a)
        mids = 0.5 * (loop_1[1:] + loop_1[:-1])
        d_l = np.diff(loop_1, axis=0)
        mutual = np.sum(f2.potential(*mids.T).T * d_l)
        np.testing.assert_allclose(double - single, mutual, rtol=1e-3)


class TestElementaryPhysics:
    """Checks of the field, potential and inductance against basic physics."""

    radius = 0.01
    loop = regular_polygon(1.0, 12)
    tilted = np.c_[
        1.2 * np.cos(np.linspace(0, 2 * np.pi, 31)),
        0.8 * np.sin(np.linspace(0, 2 * np.pi, 31)),
        0.3 * np.cos(2 * np.linspace(0, 2 * np.pi, 31)),
    ]

    # Field

    @pytest.mark.parametrize("n", [3, 4, 6, 17])
    def test_regular_polygon_centre_field(self, n):
        """B = mu_0 I n tan(pi / n) / (2 pi R) at the centre of a regular n-gon."""
        circumradius = 1.5
        f = BiotSavartFilament(regular_polygon(circumradius, n), self.radius)
        expected = MU_0 * n * np.tan(np.pi / n) / (2 * np.pi * circumradius)
        np.testing.assert_allclose(f.field(0, 0, 0), [0, 0, expected], atol=1e-18)

    @pytest.mark.parametrize("linked", [True, False])
    def test_amperes_law(self, linked):
        """The circulation of B is mu_0 I around a path linking the current, else 0."""
        current = 3.0
        f = BiotSavartFilament(self.tilted, self.radius, current=current)
        # The loop crosses the y = 0 plane at (1.2, 0, 0.3), travelling along y
        x0 = 1.2 if linked else 2.5
        theta = np.linspace(0, 2 * np.pi, 41)
        path = np.c_[
            x0 + 0.2 * np.cos(theta), np.zeros_like(theta), 0.3 + 0.2 * np.sin(theta)
        ]
        expected = MU_0 * current if linked else 0.0
        result = abs(line_integral(f.field, path))
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-12)

    @pytest.mark.parametrize("method", ["field", "potential"])
    @pytest.mark.parametrize("point", [[0.3, 0.2, 0.1], [1.5, -0.4, 0.6], [0.1, 0.5, 2]])
    def test_divergence_free(self, method, point):
        """div B = 0, and div A = 0 (Coulomb gauge) for a closed loop."""
        f = BiotSavartFilament(self.tilted, self.radius)
        func = getattr(f, method)
        point, h = np.array(point), 1e-5
        terms = [
            (func(*(point + h * e))[k] - func(*(point - h * e))[k]) / (2 * h)
            for k, e in enumerate(np.eye(3))
        ]
        scale = np.linalg.norm(func(*point)) / np.linalg.norm(point) + np.sum(
            np.abs(terms)
        )
        assert abs(sum(terms)) < 1e-6 * scale

    def test_dipole_far_field(self):
        """Far from a small loop, B and A are those of a magnetic dipole."""
        side, current = 0.2, 2.0
        f = BiotSavartFilament(square(side), self.radius, current=current)
        m = current * side**2 * np.array([0, -1, 0])  # Right-hand rule
        for r in np.array([[30, 10, -20], [0, 50, 0], [-40, 0, 5]], dtype=float):
            r_mag = np.linalg.norm(r)
            r_hat = r / r_mag
            b_dip = MU_0_4PI * (3 * (m @ r_hat) * r_hat - m) / r_mag**3
            a_dip = MU_0_4PI * np.cross(m, r) / r_mag**3
            np.testing.assert_allclose(
                f.field(*r), b_dip, rtol=1e-3, atol=1e-4 * np.abs(b_dip).max()
            )
            np.testing.assert_allclose(
                f.potential(*r), a_dip, rtol=1e-3, atol=1e-4 * np.abs(a_dip).max()
            )

    def test_helmholtz_pair(self):
        """Centre field of a Helmholtz pair is (4/5)^(3/2) mu_0 I / R."""
        loop_radius = 2.0
        lower = circle(loop_radius, 400) - [0, 0, 0.5 * loop_radius]
        upper = lower + np.array([0, 0, loop_radius])
        f = BiotSavartFilament([lower, upper], 1e-4)
        expected = (4 / 5) ** 1.5 * MU_0 / loop_radius
        np.testing.assert_allclose(f.field(0, 0, 0)[2], expected, rtol=1e-4)
        # The field is uniform to fourth order near the centre
        np.testing.assert_allclose(f.field(0, 0, 0.1)[2], expected, rtol=1e-5)

    def test_long_solenoid(self):
        """Centre of a finite solenoid: B = mu_0 n I L / sqrt(L^2 + 4 R^2)."""
        loop_radius, length, turns = 0.2, 4.0, 200
        theta = np.linspace(0, 2 * np.pi * turns, 40 * turns + 1)
        helix = np.c_[
            loop_radius * np.cos(theta),
            loop_radius * np.sin(theta),
            length * (theta / theta[-1] - 0.5),
        ]
        f = BiotSavartFilament(helix, 1e-3)
        expected = MU_0 * turns / length * length / np.hypot(length, 2 * loop_radius)
        np.testing.assert_allclose(f.field(0, 0, 0)[2], expected, rtol=2e-3)

    def test_mirror_symmetry(self):
        """A planar loop's field is mirror symmetric through its plane."""
        f = BiotSavartFilament(self.loop, self.radius)
        b_up = f.field(0.3, 0.4, 0.5)
        b_down = f.field(0.3, 0.4, -0.5)
        np.testing.assert_allclose(b_down, [-b_up[0], -b_up[1], b_up[2]], atol=1e-20)
        # In the plane of the loop, B is perpendicular to it
        np.testing.assert_allclose(f.field(0.3, 0.4, 0)[:2], 0, atol=1e-20)

    def test_scaling(self):
        """Scaling all lengths by s scales B by 1/s, and A and L are unchanged / x s."""
        s = 3.7
        f = BiotSavartFilament(self.tilted, self.radius)
        g = BiotSavartFilament(s * self.tilted, s * self.radius)
        p = np.array([0.2, 0.3, -0.4])
        np.testing.assert_allclose(g.field(*(s * p)), f.field(*p) / s, rtol=1e-12)
        np.testing.assert_allclose(g.potential(*(s * p)), f.potential(*p), rtol=1e-12)
        np.testing.assert_allclose(g.inductance(), s * f.inductance(), rtol=1e-8)

    def test_translation(self):
        shift = np.array([1.0, -2.0, 3.0])
        f = BiotSavartFilament(self.tilted, self.radius)
        g = BiotSavartFilament(self.tilted + shift, self.radius)
        p = np.array([0.2, 0.3, -0.4])
        np.testing.assert_allclose(g.field(*(p + shift)), f.field(*p), rtol=1e-12)
        np.testing.assert_allclose(g.inductance(), f.inductance(), rtol=1e-10)

    # Potential

    def test_stokes_flux(self):
        """The circulation of A around a loop is the flux of B through it."""
        f = BiotSavartFilament(self.tilted, self.radius)
        probe_radius, z0 = 0.4, 0.5
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        a_vec = f.potential(
            probe_radius * np.cos(theta),
            probe_radius * np.sin(theta),
            np.full_like(theta, z0),
        )
        tangent = probe_radius * np.array([-np.sin(theta), np.cos(theta), 0 * theta])
        circulation = np.sum(a_vec * tangent) * (2 * np.pi / len(theta))
        # Flux through the disc by polar Gauss-Legendre quadrature
        r_nodes, r_weights = np.polynomial.legendre.leggauss(30)
        r = 0.5 * probe_radius * (r_nodes + 1)
        phi = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        rr, pp = np.meshgrid(r, phi, indexing="ij")
        bz = f.field(rr * np.cos(pp), rr * np.sin(pp), np.full_like(rr, z0))[2]
        flux = np.sum(
            0.5 * probe_radius * r_weights[:, None] * rr * bz * (2 * np.pi / len(phi))
        )
        np.testing.assert_allclose(circulation, flux, rtol=1e-8)

    def test_potential_parallel_to_current(self):
        """A straight segment's potential points along the current."""
        f = BiotSavartFilament(np.array([[0, 0, 0], [1, 1, 1]]), self.radius)
        a = f.potential(*random_points(20, scale=3).T)
        np.testing.assert_allclose(np.cross(a.T, [1, 1, 1]), 0, atol=1e-20)
        assert np.all(a @ np.ones(3) > 0) if a.ndim == 1 else np.all(a.sum(axis=0) > 0)

    # Inductance

    def test_parallel_wires_mutual(self):
        """Neumann mutual inductance of two parallel equal segments."""
        length, d, a = 2.0, 0.3, 0.01
        w1 = np.array([[0, 0, 0], [0, 0, length]])
        w2 = w1 + np.array([d, 0, 0])
        single = BiotSavartFilament(w1, a).inductance()
        mutual = BiotSavartFilament([w1, w2], a).inductance() - single
        anti = BiotSavartFilament([w1, w2[::-1]], a).inductance() - single

        def neumann(dist):
            # MU_0_4PI, as MU_0 differs from 4 pi 1e-7 by ~1e-10
            return (
                2
                * MU_0_4PI
                * (length * np.arcsinh(length / dist) - np.hypot(length, dist) + dist)
            )

        # Exact with the geometric mean distance correction ...
        gmd = a * np.exp(-0.25)
        np.testing.assert_allclose(mutual, neumann(np.hypot(d, gmd)), rtol=1e-12)
        # ... which is O(gmd^2 / d^2) from the thin filament result
        np.testing.assert_allclose(mutual, neumann(d), rtol=(gmd / d) ** 2)
        # Antiparallel return conductor gives the opposite mutual inductance
        np.testing.assert_allclose(anti, -mutual, rtol=1e-12)

    def test_reciprocity(self):
        """M_12 = M_21."""
        a = 0.02
        loop_1 = self.tilted
        loop_2 = 0.7 * regular_polygon(1.0, 9) + [0.3, 0.1, 0.8]
        m_12 = (
            BiotSavartFilament([loop_1, loop_2], a).inductance()
            - BiotSavartFilament(loop_1, a).inductance()
        )
        m_21 = (
            BiotSavartFilament([loop_2, loop_1], a).inductance()
            - BiotSavartFilament(loop_2, a).inductance()
        )
        np.testing.assert_allclose(m_12, m_21, rtol=1e-8)

    def test_mutual_bounded_by_self(self):
        """|M| <= sqrt(L1 L2) (positive definite magnetic energy)."""
        a = 0.02
        loop_1 = self.tilted
        loop_2 = loop_1 + np.array([0, 0, 0.1])
        l_1 = BiotSavartFilament(loop_1, a).inductance()
        l_2 = BiotSavartFilament(loop_2, a).inductance()
        m = BiotSavartFilament([loop_1, loop_2], a).inductance() - l_1
        assert 0 < m < np.sqrt(l_1 * l_2)

    def test_coincident_turns(self):
        """Two coincident turns: total flux linkage per turn doubles."""
        a = 0.02
        single = BiotSavartFilament(self.loop, a).inductance()
        double = BiotSavartFilament([self.loop, self.loop.copy()], a).inductance()
        np.testing.assert_allclose(double, 2 * single, rtol=1e-12)

    def test_inductance_flux_linkage(self):
        """L I = flux through the loop, computed from the far-from-wire potential."""
        a = 1e-3
        f = BiotSavartFilament(square(1.0), a)
        # Flux through a slightly smaller concentric square: circulation of A on it
        inner = square(1.0 - 0.2)
        flux_inner = line_integral(f.potential, inner)
        assert 0 < flux_inner < f.inductance()

    def test_inductance_independent_of_current(self):
        f = BiotSavartFilament(self.loop, self.radius, current=1.0)
        g = BiotSavartFilament(self.loop, self.radius, current=123.0)
        np.testing.assert_allclose(g.inductance(), f.inductance(), rtol=1e-14)

    def test_inductance_increases_as_radius_decreases(self):
        values = [
            BiotSavartFilament(self.loop, a).inductance() for a in (0.1, 0.03, 0.01)
        ]
        assert np.all(np.diff(values) > 0)
        # d L / d ln(a) -> -mu_0 P / (2 pi) for a thin wire of perimeter P
        perimeter = np.sum(np.linalg.norm(np.diff(self.loop, axis=0), axis=1))
        slope = (values[2] - values[1]) / np.log(0.01 / 0.03)
        np.testing.assert_allclose(slope, -MU_0 * perimeter / (2 * np.pi), rtol=2e-2)


class TestAPI:
    radius = 0.05
    coords = circle(1.0, 30)

    def test_coordinates_input(self):
        a = BiotSavartFilament(self.coords, self.radius)
        b = BiotSavartFilament(Coordinates(self.coords.T), self.radius)
        np.testing.assert_allclose(b.field(0.2, 0.1, 0.3), a.field(0.2, 0.1, 0.3))

    def test_list_input_is_superposition(self):
        other = self.coords + np.array([0, 0, 1])
        both = BiotSavartFilament([self.coords, other], self.radius)
        a = BiotSavartFilament(self.coords, self.radius)
        b = BiotSavartFilament(other, self.radius)
        p = (0.2, 0.1, 0.3)
        np.testing.assert_allclose(both.field(*p), a.field(*p) + b.field(*p))
        np.testing.assert_allclose(both.potential(*p), a.potential(*p) + b.potential(*p))
        np.testing.assert_allclose(both.length, a.length)

    def test_current_scaling(self):
        f = BiotSavartFilament(self.coords, self.radius, current=1.0)
        b1 = f.field(0.2, 0.1, 0.3)
        f.set_current(7.5)
        np.testing.assert_allclose(f.field(0.2, 0.1, 0.3), 7.5 * b1)

    def test_output_shapes(self):
        f = BiotSavartFilament(self.coords, self.radius)
        assert f.field(0.1, 0.2, 0.3).shape == (3,)
        assert f.field(np.array([0.1]), np.array([0.2]), np.array([0.3])).shape == (3,)
        x = np.linspace(0, 1, 4)
        assert f.field(x, x, x).shape == (3, 4)
        xx = np.ones((3, 5))
        assert f.field(xx, xx, xx).shape == (3, 3, 5)
        assert f.potential(xx, xx, xx).shape == (3, 3, 5)
        # Array results agree with point-wise results
        np.testing.assert_allclose(f.field(x, x, x)[:, 2], f.field(x[2], x[2], x[2]))

    def test_mismatched_inputs_raise(self):
        f = BiotSavartFilament(self.coords, self.radius)
        with pytest.raises(MagnetostaticsError):
            f.field(np.zeros(3), np.zeros(2), np.zeros(3))
        with pytest.raises(MagnetostaticsError):
            f.field(*np.zeros((3, 2, 2, 2)))

    @pytest.mark.parametrize("radius", [0.0, -1.0])
    def test_bad_radius_raises(self, radius):
        with pytest.raises(MagnetostaticsError):
            BiotSavartFilament(self.coords, radius)

    def test_single_point_raises(self):
        with pytest.raises(MagnetostaticsError):
            BiotSavartFilament(np.array([[0, 0, 0], [0, 0, 0]]), self.radius)

    def test_rotate(self):
        f = BiotSavartFilament(self.coords + np.array([0.5, 0, 0]), self.radius)
        rotated = f.copy()
        rotated.rotate(90, "z")
        # A 90 degree rotation about z maps (x, y, z) -> (-y, x, z)
        p = np.array([0.3, -0.2, 0.4])
        p_rot = np.array([-p[1], p[0], p[2]])
        b = f.field(*p)
        np.testing.assert_allclose(
            rotated.field(*p_rot), [-b[1], b[0], b[2]], atol=1e-15
        )
        np.testing.assert_allclose(
            rotated.ref_mid_points[:, 0], -f.ref_mid_points[:, 1], atol=1e-14
        )
        np.testing.assert_allclose(rotated.inductance(), f.inductance(), rtol=1e-10)

    def test_plot(self):
        f = BiotSavartFilament(
            [self.coords, self.coords + np.array([0, 0, 1])], self.radius
        )
        f.plot(show_coord_sys=True)
