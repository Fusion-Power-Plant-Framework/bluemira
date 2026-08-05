# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pytest

from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from tests.magnetostatics.setup_methods import _plot_verification_test


class TestCircularArcCurrentSource:
    @classmethod
    def setup_class(cls):
        cls.xc, cls.zc = 4, 4
        cls.dx = 0.5
        cls.dz = 1.0
        cls.current = 1e6

        cls.arc = CircularArcCurrentSource(
            [0, 0, cls.zc],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            cls.dx,
            cls.dz,
            cls.xc,
            360,
            cls.current,
        )

    def test_2D_vs_3D_circular(self):
        nx, nz = 50, 60
        x = np.linspace(self.xc - 2, self.xc + 2, nx)
        z = np.linspace(self.zc - 2, self.zc + 2, nz)
        xx, zz = np.meshgrid(x, z, indexing="ij")

        Bx, _, Bz = self.arc.field(xx, np.zeros_like(xx), zz)
        Bp = np.hypot(Bx, Bz)

        cBx = semianalytic_Bx(self.xc, self.zc, xx, zz, self.dx, self.dz)
        cBz = semianalytic_Bz(self.xc, self.zc, xx, zz, self.dx, self.dz)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bp_coil = np.hypot(Bx_coil, Bz_coil)

        # Because this is a circular calculation, we expect them to be almost identical
        assert np.allclose(Bx_coil, Bx)
        assert np.allclose(Bz_coil, Bz)

        self.arc.plot()
        _plot_verification_test(
            self.xc,
            self.zc,
            self.dx,
            self.dz,
            xx,
            zz,
            Bx_coil,
            Bz_coil,
            Bp_coil,
            Bx,
            Bz,
            Bp,
        )

    def test_singularities(self):
        """
        Trigger singularities (ZeroDivisionErrors and such should not come up)
        """
        self.arc.field(self.arc.radius - self.arc.breadth, 0, 0)
        self.arc.field(self.arc.radius - self.arc.breadth, 0, self.arc._depth)
        self.arc.field(self.arc.radius - self.arc.breadth, 0, -self.arc._depth)
        self.arc.field(self.arc.radius + self.arc.breadth, 0, self.arc._depth)
        self.arc.field(self.arc.radius + self.arc.breadth, 0, -self.arc._depth)
        self.arc.field(self.arc.radius, 0, 1)
        self.arc.field(self.arc.radius, 0, 0)
        self.arc.field(0, 0, -1)


class TestCircularArcCurrentSourceSuperposition:
    """Tests that arc decomposition preserves the magnetic field."""

    origin = np.zeros(3)

    ds = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    t_vec = np.array([0.0, 0.0, 1.0])

    radius = 1.0
    breadth = 0.05
    depth = 0.05
    current = 1000.0

    test_points = [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.05),
        (0.2, -0.1, 0.0),
        (-0.15, 0.25, -0.1),
        (0.3, 0.2, 0.15),
    ]

    @classmethod
    def make_arc(cls, start_angle_deg: float, dtheta_deg: float):
        """Construct an arc starting at start_angle_deg."""

        theta = np.deg2rad(start_angle_deg)

        ds = np.array(
            [
                np.cos(theta),
                np.sin(theta),
                0.0,
            ]
        )

        normal = np.array(
            [
                -np.sin(theta),
                np.cos(theta),
                0.0,
            ]
        )

        return CircularArcCurrentSource(
            origin=cls.origin,
            ds=ds,
            normal=normal,
            t_vec=cls.t_vec,
            breadth=cls.breadth,
            depth=cls.depth,
            radius=cls.radius,
            dtheta=dtheta_deg,
            current=cls.current,
        )

    @pytest.mark.parametrize("point", test_points)
    def test_full_ring_equals_two_half_rings(self, point):
        """360° ring == 180° + 180°."""

        full_ring = self.make_arc(0.0, 360.0)

        half_rings = SourceGroup([
            self.make_arc(0.0, 180.0),
            self.make_arc(180.0, 180.0),
        ])
        b_full = full_ring.field(*point)

        b_halves = half_rings.field(*point)

        np.testing.assert_allclose(
            b_full,
            b_halves,
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.mark.parametrize("point", test_points)
    def test_full_ring_equals_four_quarter_rings(self, point):
        """360° ring == 4 × 90° arcs."""

        full_ring = self.make_arc(0.0, 360.0)

        quarter_rings = SourceGroup([
            self.make_arc(0.0, 90.0),
            self.make_arc(90.0, 90.0),
            self.make_arc(180.0, 90.0),
            self.make_arc(270.0, 90.0),
        ])

        b_quarters = quarter_rings.field(*point)

        b_full = full_ring.field(*point)

        np.testing.assert_allclose(
            b_full,
            b_quarters,
            rtol=1e-10,
            atol=1e-12,
        )