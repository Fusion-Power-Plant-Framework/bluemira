# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from itertools import starmap

import numpy as np
import pytest
from matplotlib import pyplot as plt

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
    """Tests that circular arc decomposition preserves the magnetic field."""

    origin = np.zeros(3)

    ds = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    t_vec = np.array([0.0, 0.0, 1.0])

    radius = 1.0
    breadth = 0.05
    depth = 0.05
    current = 1000.0

    rtol = 1e-10
    atol = 1e-11

    angle_eps = 1e-11
    length_eps = 1e-11

    @staticmethod
    def r1() -> float:
        """Inner radius of the rectangular arc source."""
        return (
            TestCircularArcCurrentSourceSuperposition.radius
            - TestCircularArcCurrentSourceSuperposition.breadth
        )

    @staticmethod
    def r2() -> float:
        """Outer radius of the rectangular arc source."""
        return (
            TestCircularArcCurrentSourceSuperposition.radius
            + TestCircularArcCurrentSourceSuperposition.breadth
        )

    @staticmethod
    def cylindrical_point(
        radius: float,
        theta: float,
        z: float,
    ) -> tuple[float, float, float]:
        """Return a Cartesian point from cylindrical coordinates."""
        return (
            radius * np.cos(theta),
            radius * np.sin(theta),
            z,
        )

    @staticmethod
    def make_arc(start_angle_deg: float, dtheta_deg: float):
        """Construct an arc starting at start_angle_deg."""
        cls = TestCircularArcCurrentSourceSuperposition

        theta = np.deg2rad(start_angle_deg)

        _ds = np.array([
            np.cos(theta),
            np.sin(theta),
            0.0,
        ])

        _normal = np.array([
            -np.sin(theta),
            np.cos(theta),
            0.0,
        ])

        return CircularArcCurrentSource(
            origin=cls.origin,
            ds=_ds,
            normal=_normal,
            t_vec=cls.t_vec,
            breadth=cls.breadth,
            depth=cls.depth,
            radius=cls.radius,
            dtheta=dtheta_deg,
            current=cls.current,
        )

    @staticmethod
    def make_decomposed_arc(
        start_angle_deg: float,
        dtheta_deg: float,
        n_segments: int,
    ) -> SourceGroup:
        """Construct an arc as a group of n equivalent smaller arcs."""
        cls = TestCircularArcCurrentSourceSuperposition

        segment_angle = dtheta_deg / n_segments

        return SourceGroup([
            cls.make_arc(
                start_angle_deg + i * segment_angle,
                segment_angle,
            )
            for i in range(n_segments)
        ])

    @staticmethod
    def singularity_probe_points() -> tuple[tuple[float, float, float], ...]:
        """
        Points chosen to exercise circular-arc singularity handling.

        The primitive singularity criteria include combinations of:
          - z_k == 0, corresponding to z_p == +/- depth
          - r_j <= r_pc, especially around r1 and r2
          - phi_pc at the start, end, or inside the arc

        These points deliberately hit:
          - arc seams,
          - just inside and outside seams,
          - inner and outer radial surfaces,
          - upper and lower axial surfaces,
          - inside-coil points.
        """
        cls = TestCircularArcCurrentSourceSuperposition

        points: list[tuple[float, float, float]] = []

        r1 = cls.r1()
        r2 = cls.r2()

        # Benign off-source points.
        points.extend([
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.05),
            (0.2, -0.1, 0.0),
            (-0.15, 0.25, -0.1),
            (0.3, 0.2, 0.15),
            (1.3, -0.4, 0.2),
            (-1.2, 0.7, -0.15),
        ])

        # Seams used by 2, 4, and 8 segment decompositions.
        seam_angles = np.deg2rad([
            0.0,
            45.0,
            90.0,
            135.0,
            180.0,
            225.0,
            270.0,
            315.0,
        ])

        for theta in seam_angles:
            for dtheta in (0.0, -cls.angle_eps, cls.angle_eps):
                phi = theta + dtheta

                # Exactly on radial and axial surfaces.
                points.extend([
                    cls.cylindrical_point(r1, phi, -cls.depth),
                    cls.cylindrical_point(r1, phi, cls.depth),
                    cls.cylindrical_point(r2, phi, -cls.depth),
                    cls.cylindrical_point(r2, phi, cls.depth),
                ])

                # Just inside / outside the radial surfaces.
                points.extend([
                    cls.cylindrical_point(r1 - cls.length_eps, phi, -cls.depth),
                    cls.cylindrical_point(r1 + cls.length_eps, phi, -cls.depth),
                    cls.cylindrical_point(r2 - cls.length_eps, phi, cls.depth),
                    cls.cylindrical_point(r2 + cls.length_eps, phi, cls.depth),
                ])

                # Mid-radius points on and near z surfaces.
                points.extend([
                    cls.cylindrical_point(cls.radius, phi, 0.0),
                    cls.cylindrical_point(cls.radius, phi, -cls.depth),
                    cls.cylindrical_point(cls.radius, phi, cls.depth),
                    cls.cylindrical_point(
                        cls.radius,
                        phi,
                        -cls.depth + cls.length_eps,
                    ),
                    cls.cylindrical_point(
                        cls.radius,
                        phi,
                        cls.depth - cls.length_eps,
                    ),
                ])

        return tuple(points)

    @staticmethod
    def assert_sources_close_at_probe_points(source_a, source_b):
        """Compare two sources at all singularity probe points."""
        cls = TestCircularArcCurrentSourceSuperposition

        for point in cls.singularity_probe_points():
            b_a = source_a.field(*point)
            b_b = source_b.field(*point)

            np.testing.assert_allclose(
                b_a,
                b_b,
                rtol=cls.rtol,
                atol=cls.atol,
                err_msg=f"Field mismatch at point {point}",
            )

    @staticmethod
    def assert_source_finite_at_probe_points(source):
        """Assert that a source has finite field values at all probe points."""
        cls = TestCircularArcCurrentSourceSuperposition

        for point in cls.singularity_probe_points():
            b = source.field(*point)

            assert np.all(np.isfinite(b)), f"Non-finite field at point {point}: {b}"

    @pytest.mark.parametrize("n_segments", [2, 4, 8])
    @pytest.mark.parametrize("start_angle_deg", [0.0, 13.0, 90.0])
    def test_full_ring_equals_decomposed_ring(
        self,
        n_segments,
        start_angle_deg,
    ):
        """A 360 degree ring should equal N equivalent circular arcs."""
        full_ring = self.make_arc(start_angle_deg, 360.0)

        decomposed_ring = self.make_decomposed_arc(
            start_angle_deg=start_angle_deg,
            dtheta_deg=360.0,
            n_segments=n_segments,
        )

        self.assert_sources_close_at_probe_points(
            full_ring,
            decomposed_ring,
        )

    @pytest.mark.parametrize(
        ("start_angle_deg", "dtheta_deg", "n_segments"),
        [
            (0.0, 180.0, 2),
            (0.0, 90.0, 2),
            (45.0, 90.0, 3),
            (13.0, 137.0, 3),
            (120.0, 210.0, 5),
            (270.0, 90.0, 3),
        ],
    )
    def test_finite_arc_equals_decomposed_finite_arc(
        self,
        start_angle_deg,
        dtheta_deg,
        n_segments,
    ):
        """
        A finite arc should equal the sum of smaller equivalent arcs.

        This is stricter than the full-ring test because missing tangential
        components and bad endpoint treatment are not hidden by full-circle
        symmetry.
        """
        arc = self.make_arc(start_angle_deg, dtheta_deg)

        decomposed_arc = self.make_decomposed_arc(
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
            n_segments=n_segments,
        )

        self.assert_sources_close_at_probe_points(
            arc,
            decomposed_arc,
        )

    @pytest.mark.parametrize(
        ("start_angle_deg", "dtheta_deg"),
        [
            (0.0, 45.0),
            (0.0, 90.0),
            (0.0, 180.0),
            (0.0, 270.0),
            (13.0, 137.0),
            (270.0, 90.0),
            (0.0, 360.0),
        ],
    )
    def test_field_is_finite_at_singularity_probe_points(
        self,
        start_angle_deg,
        dtheta_deg,
    ):
        """
        The field should remain finite at points that exercise singularity
        branches in the primitive integral functions.
        """
        arc = self.make_arc(start_angle_deg, dtheta_deg)

        self.assert_source_finite_at_probe_points(arc)

    @pytest.mark.parametrize("n_segments", [2, 4, 8])
    def test_vectorised_field_matches_pointwise_field(self, n_segments):
        """
        Vectorised field evaluation should match pointwise evaluation.

        This catches shape/order regressions in process_xyz_array and SourceGroup
        handling while using singularity-heavy points.
        """
        cls = TestCircularArcCurrentSourceSuperposition

        source = self.make_decomposed_arc(
            start_angle_deg=0.0,
            dtheta_deg=360.0,
            n_segments=n_segments,
        )

        points = np.array(cls.singularity_probe_points())

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        b_vectorised = source.field(x, y, z)

        b_pointwise = np.array(list(starmap(source.field, points))).T

        np.testing.assert_allclose(
            b_vectorised,
            b_pointwise,
            rtol=cls.rtol,
            atol=cls.atol,
        )

    @pytest.mark.parametrize(
        ("start_angle_deg", "dtheta_deg", "n_segments"),
        [
            (0.0, 360.0, 2),
            (0.0, 360.0, 4),
            (0.0, 360.0, 8),
            (13.0, 360.0, 4),
            (0.0, 180.0, 2),
            (45.0, 90.0, 3),
            (13.0, 137.0, 3),
        ],
    )
    def test_decomposed_source_field_is_finite_at_probe_points(
        self,
        start_angle_deg,
        dtheta_deg,
        n_segments,
    ):
        """
        Decomposed sources should also remain finite at singularity probe points.

        This catches cases where individual segment endpoints introduce NaNs
        even if the equivalent larger source evaluates cleanly.
        """
        source = self.make_decomposed_arc(
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
            n_segments=n_segments,
        )

        self.assert_source_finite_at_probe_points(source)

    def _plot_source_comp(self, source_1, source_2):
        n_points = 50
        dim = 1.6
        x = np.linspace(-dim, dim, n_points)
        y = np.linspace(-dim, dim, n_points)

        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = 0.3 * np.ones_like(xx)

        b_full = np.linalg.norm(source_1.field(xx, yy, zz), axis=0)
        b_halves = np.linalg.norm(source_2.field(xx, yy, zz), axis=0)

        fig = plt.figure(figsize=(9, 4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        levels = np.linspace(
            min(b_full.min(), b_halves.min()),
            max(b_full.max(), b_halves.max()),
            30,
        )

        ax0.contourf(xx, yy, b_full, levels=levels, cmap="magma")
        cm = ax1.contourf(xx, yy, b_halves, levels=levels, cmap="magma")

        for a in (ax0, ax1):
            a.set_aspect("equal")
            a.set_xlabel("x [m]")
            a.set_ylabel("y [m]")

        cb = fig.colorbar(cm, cax=cax)
        cb.set_label(r"$|\mathbf{B}|$ [T]")
        plt.show()
