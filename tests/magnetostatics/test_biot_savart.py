# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.constants import MU_0
from bluemira.display.auto_config import plot_defaults
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.greens import (
    circular_coil_inductance_elliptic,
    circular_coil_inductance_kirchhoff,
    greens_all,
)


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
