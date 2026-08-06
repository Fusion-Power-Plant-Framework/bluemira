# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest
from matplotlib import pyplot as plt

from bluemira.base.constants import EPS
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

        Bx, By, Bz = self.arc.field(xx, np.zeros_like(xx), zz)
        Bp = np.hypot(Bx, Bz)

        cBx = semianalytic_Bx(self.xc, self.zc, xx, zz, self.dx, self.dz)
        cBz = semianalytic_Bz(self.xc, self.zc, xx, zz, self.dx, self.dz)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bp_coil = np.hypot(Bx_coil, Bz_coil)

        # Because this is a circular calculation, we expect them to be almost identical
        assert np.allclose(By, np.zeros_like(By), rtol=0.0, atol=EPS)
        # np.testing.assert_allclose(Bx_coil, Bx)
        # np.testing.assert_allclose(Bz_coil, Bz)

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

    radius = 1.0
    breadth = 0.05
    depth = 0.05
    current = 1000.0

    rtol = 0.0
    atol = 1e-9

    angle_eps = 1e-9
    length_eps = 1e-9

    test_points = [  # noqa: RUF012
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.05),
        (0.2, -0.1, 0.0),
        (-0.15, 0.25, -0.1),
        (0.3, 0.2, 0.15),
        (np.linspace(-2, 2), np.linspace(-2, 2), np.linspace(-2, 2)),
    ]

    axis_bases = [  # noqa: RUF012
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            id="axis-z",
        ),
        pytest.param(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 1.0, 0.0]),
            id="axis-y",
        ),
    ]

    @staticmethod
    def rotate_arc_basis(
        ds0: np.ndarray,
        normal0: np.ndarray,
        t_vec: np.ndarray,
        start_angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate the local arc start basis about the ring axis.

        The returned basis preserves:
            ds x normal == t_vec
        """
        theta = np.deg2rad(start_angle_deg)

        ds = np.cos(theta) * ds0 + np.sin(theta) * normal0
        normal = -np.sin(theta) * ds0 + np.cos(theta) * normal0

        return ds, normal, t_vec

    @staticmethod
    def make_arc(
        ds0: np.ndarray,
        normal0: np.ndarray,
        t_vec: np.ndarray,
        start_angle_deg: float,
        dtheta_deg: float,
    ):
        """Construct an arc starting at start_angle_deg."""
        cls = TestCircularArcCurrentSourceSuperposition

        ds, normal, t_vec = cls.rotate_arc_basis(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=start_angle_deg,
        )

        return CircularArcCurrentSource(
            origin=cls.origin,
            ds=ds,
            normal=normal,
            t_vec=t_vec,
            breadth=cls.breadth,
            depth=cls.depth,
            radius=cls.radius,
            dtheta=dtheta_deg,
            current=cls.current,
        )

    @staticmethod
    def make_decomposed_arc(
        ds0: np.ndarray,
        normal0: np.ndarray,
        t_vec: np.ndarray,
        start_angle_deg: float,
        dtheta_deg: float,
        n_segments: int,
    ) -> SourceGroup:
        """Construct an arc as a group of n equivalent smaller arcs."""
        cls = TestCircularArcCurrentSourceSuperposition

        segment_angle = dtheta_deg / n_segments

        return SourceGroup([
            cls.make_arc(
                ds0=ds0,
                normal0=normal0,
                t_vec=t_vec,
                start_angle_deg=start_angle_deg + i * segment_angle,
                dtheta_deg=segment_angle,
            )
            for i in range(n_segments)
        ])

    @staticmethod
    def basis_point(
        ds0: np.ndarray,
        normal0: np.ndarray,
        t_vec: np.ndarray,
        radius: float,
        angle_deg: float,
        z: float,
    ) -> tuple[float, float, float]:
        """Construct a global point from local cylindrical coordinates."""
        theta = np.deg2rad(angle_deg)

        point = (
            TestCircularArcCurrentSourceSuperposition.origin
            + radius * np.cos(theta) * ds0
            + radius * np.sin(theta) * normal0
            + z * t_vec
        )

        return tuple(point)

    @staticmethod
    def singularity_points(
        ds0: np.ndarray,
        normal0: np.ndarray,
        t_vec: np.ndarray,
        start_angle_deg: float,
        dtheta_deg: float,
        n_segments: int = 1,
        *,
        exact: bool,
    ) -> tuple[tuple[float, float, float], ...]:
        """
        Generate points that probe likely singularity locations.

        Exact points are useful for finite-value checks.
        Near points are useful for equality checks.
        """
        cls = TestCircularArcCurrentSourceSuperposition

        r1 = cls.radius - cls.breadth
        r2 = cls.radius + cls.breadth

        segment_angle = dtheta_deg / n_segments

        seam_angles = [
            start_angle_deg + i * segment_angle for i in range(n_segments + 1)
        ]

        mid_angles = [
            start_angle_deg + (i + 0.5) * segment_angle for i in range(n_segments)
        ]

        if exact:
            angle_offsets = [0.0]
            radius_offsets = [0.0]
            z_offsets = [0.0]
        else:
            # TODO: Address near-singularity integration errors
            angle_offsets = [-np.rad2deg(cls.angle_eps), np.rad2deg(cls.angle_eps)]
            radius_offsets = [-cls.length_eps, cls.length_eps]
            z_offsets = [-cls.length_eps, cls.length_eps]

        points = []

        for angle_deg in [*seam_angles, *mid_angles]:
            for angle_offset in angle_offsets:
                shifted_angle = angle_deg + angle_offset

                for base_radius in (r1, cls.radius, r2):
                    for radius_offset in radius_offsets:
                        radius = base_radius + radius_offset

                        for base_z in (-cls.depth, 0.0, cls.depth):
                            for z_offset in z_offsets:
                                z = base_z + z_offset

                                points.append(
                                    cls.basis_point(
                                        ds0=ds0,
                                        normal0=normal0,
                                        t_vec=t_vec,
                                        radius=radius,
                                        angle_deg=shifted_angle,
                                        z=z,
                                    )
                                )

        return tuple(points)

    @staticmethod
    def assert_sources_close(source_a, source_b, point):
        """Assert that two sources produce the same field at point."""
        cls = TestCircularArcCurrentSourceSuperposition

        b_a = source_a.field(*point)
        b_b = source_b.field(*point)

        np.testing.assert_allclose(
            b_a,
            b_b,
            rtol=cls.rtol,
            atol=cls.atol,
        )

    @staticmethod
    def assert_sources_close_at_points(source_a, source_b, points):
        """Assert that two sources produce the same field at many points."""
        cls = TestCircularArcCurrentSourceSuperposition

        for point in points:
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
    def assert_source_finite_at_points(source, points):
        """Assert that a source has finite field values at many points."""
        for point in points:
            b = source.field(*point)

            assert np.all(np.isfinite(b)), f"Non-finite field at point {point}: {b}"

    @pytest.mark.parametrize("point", test_points)
    @pytest.mark.parametrize(("ds0", "normal0", "t_vec"), axis_bases)
    def test_full_ring_equals_two_half_rings(
        self,
        point,
        ds0,
        normal0,
        t_vec,
    ):
        """360 degree ring == 180 degree + 180 degree, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
        )

        half_rings = self.make_decomposed_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
            n_segments=2,
        )

        self.assert_sources_close(full_ring, half_rings, point)

    @pytest.mark.parametrize("point", test_points)
    @pytest.mark.parametrize(("ds0", "normal0", "t_vec"), axis_bases)
    def test_full_ring_equals_four_quarter_rings(
        self,
        point,
        ds0,
        normal0,
        t_vec,
    ):
        """360 degree ring == 4 x 90 degree arcs, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
        )

        quarter_rings = self.make_decomposed_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
            n_segments=4,
        )

        self.assert_sources_close(full_ring, quarter_rings, point)

    @pytest.mark.parametrize("point", test_points)
    @pytest.mark.parametrize(("ds0", "normal0", "t_vec"), axis_bases)
    def test_full_ring_equals_eight_octant_rings(
        self,
        point,
        ds0,
        normal0,
        t_vec,
    ):
        """360 degree ring == 8 x 45 degree arcs, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
        )

        octant_rings = self.make_decomposed_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=0.0,
            dtheta_deg=360.0,
            n_segments=8,
        )

        self.assert_sources_close(full_ring, octant_rings, point)

    @pytest.mark.parametrize(("ds0", "normal0", "t_vec"), axis_bases)
    @pytest.mark.parametrize(
        ("start_angle_deg", "dtheta_deg"),
        [
            (0.0, 45.0),
            (0.0, 90.0),
            (0.0, 180.0),
            (0.0, 270.0),
            (0.0, 360.0),
            (13.0, 137.0),
            (90.0, 180.0),
            (270.0, 90.0),
        ],
    )
    def test_arc_field_is_finite_at_exact_singularity_points(
        self,
        ds0,
        normal0,
        t_vec,
        start_angle_deg,
        dtheta_deg,
    ):
        """
        Arc field should remain finite at exact singularity probe points.

        This checks exact seam, surface, and mid-arc points.
        """
        arc = self.make_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
        )

        points = self.singularity_points(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
            exact=True,
        )

        self.assert_source_finite_at_points(arc, points)

    @pytest.mark.parametrize(("ds0", "normal0", "t_vec"), axis_bases)
    @pytest.mark.parametrize(
        ("start_angle_deg", "dtheta_deg", "n_segments"),
        [
            (0.0, 360.0, 2),
            (0.0, 360.0, 4),
            (0.0, 360.0, 8),
            (13.0, 360.0, 4),
            (0.0, 180.0, 2),
            (0.0, 90.0, 2),
            (45.0, 90.0, 3),
            (13.0, 137.0, 3),
            (120.0, 210.0, 5),
            (270.0, 90.0, 3),
        ],
    )
    def test_decomposed_arc_field_is_finite_at_exact_singularity_points(
        self,
        ds0,
        normal0,
        t_vec,
        start_angle_deg,
        dtheta_deg,
        n_segments,
    ):
        """
        Decomposed sources should remain finite at exact singularity points.

        This catches segment endpoint singularities introduced by decomposition.
        """
        decomposed_arc = self.make_decomposed_arc(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
            n_segments=n_segments,
        )

        points = self.singularity_points(
            ds0=ds0,
            normal0=normal0,
            t_vec=t_vec,
            start_angle_deg=start_angle_deg,
            dtheta_deg=dtheta_deg,
            n_segments=n_segments,
            exact=True,
        )

        self.assert_source_finite_at_points(decomposed_arc, points)

    def _plot_source_comp(self, full_ring, segmented_ring):
        n_points = 50
        dim = 1.6
        x = np.linspace(-dim, dim, n_points)
        y = np.linspace(-dim, dim, n_points)

        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = 0.3 * np.ones_like(xx)

        b_full = np.linalg.norm(full_ring.field(xx, yy, zz), axis=0)
        b_segmented = np.linalg.norm(segmented_ring.field(xx, yy, zz), axis=0)

        fig = plt.figure(figsize=(9, 4))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        levels = np.linspace(
            min(b_full.min(), b_segmented.min()),
            max(b_full.max(), b_segmented.max()),
            30,
        )

        ax0.contourf(xx, yy, b_full, levels=levels, cmap="magma")
        cm = ax1.contourf(xx, yy, b_segmented, levels=levels, cmap="magma")

        for ax in (ax0, ax1):
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

        cb = fig.colorbar(cm, cax=cax)
        cb.set_label(r"$|\mathbf{B}|$ [T]")
        plt.show()
