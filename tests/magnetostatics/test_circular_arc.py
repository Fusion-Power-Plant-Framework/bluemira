# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy

import numpy as np
import pytest
from matplotlib import pyplot as plt

from bluemira.base.constants import EPS, MU_0_4PI
from bluemira.geometry.coordinates import Coordinates, rotation_matrix
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.greens import greens_psi
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from bluemira.utilities.plot_tools import Plot3D
from tests.magnetostatics.setup_methods import _plot_verification_test


def test_gauss_legendre_fallback_no_longer_required():
    """
    10 nm above the inner top edge of a full ring, the Bz integration is difficult
    to converge.
    """
    radius, breadth, depth, current = 2.0, 0.25, 0.5, 1e6
    coil = CircularArcCurrentSource(
        [0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], breadth, depth, radius, 360, current
    )
    angle = np.deg2rad(15.0)
    rp, zp = radius - breadth, depth + 10.0e-9

    field = coil.field(zp, rp * np.cos(angle), rp * np.sin(angle))

    br = current * semianalytic_Bx(radius, 0.0, rp, zp, breadth, depth)
    bz = current * semianalytic_Bz(radius, 0.0, rp, zp, breadth, depth)
    expected = np.array([bz, br * np.cos(angle), br * np.sin(angle)])
    # The fallback averages over a ~1 um cube, so does not match exactly
    np.testing.assert_allclose(
        field, expected, rtol=0, atol=1e-5 * np.linalg.norm(expected)
    )


@pytest.mark.parametrize(
    ("rp", "zp", "expected"),
    [
        pytest.param(
            1.75,
            0.5 + 1e-6,
            [0.27561318358041387, 0.07562127556128435, 0.27978880839924103],
            id="1um-above-inner-top-edge",
        ),
        pytest.param(
            2.25,
            0.5 + 1e-6,
            [0.2623655711286338, 0.06715898584330263, -0.17991927003096395],
            id="1um-above-outer-top-edge",
        ),
        pytest.param(
            3.0,
            0.5,
            [0.033814046759571885, 0.01421576273771245, -0.06803763762499697],
            id="top-face-plane-outside",
        ),
    ],
)
def test_field_near_singularities_matches_high_precision_reference(rp, zp, expected):
    """
    Field next to the edges of the cross-section, 10 degrees into a 90 degree arc.

    Around psi = 0 the logarithm arguments of the integrands are differences of
    nearly equal numbers here, and lose all their digits in float64. Without the
    stable forms, Bz 1 um above the top edges is wrong in the fifth significant
    digit.

    The references are the same primitives evaluated with mpmath at 50 significant
    digits (70 digits give identical float64 values).
    """
    arc = CircularArcCurrentSource(
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.25, 0.5, 2.0, 90, 1e6
    )
    angle = np.deg2rad(10.0)
    field = arc.field(rp * np.cos(angle), rp * np.sin(angle), zp)
    np.testing.assert_allclose(
        field, expected, rtol=0, atol=1e-9 * np.linalg.norm(expected)
    )


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
        nx, nz = 250, 260
        x = np.linspace(self.xc - 3.99999999, self.xc + 5, nx)
        z = np.linspace(self.zc - 5, self.zc + 5, nz)
        xx, zz = np.meshgrid(x, z, indexing="ij")

        Bx, By, Bz = self.arc.field(xx, np.zeros_like(xx), zz)
        Bp = np.hypot(Bx, Bz)

        cBx = semianalytic_Bx(self.xc, self.zc, xx, zz, self.dx, self.dz)
        cBz = semianalytic_Bz(self.xc, self.zc, xx, zz, self.dx, self.dz)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bp_coil = np.hypot(Bx_coil, Bz_coil)

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

        # Because this is a circular calculation, we expect them to be almost identical
        np.testing.assert_allclose(By, np.zeros_like(By), rtol=0.0, atol=EPS)
        np.testing.assert_allclose(Bx_coil, Bx, rtol=0.0, atol=4.8e-6)
        np.testing.assert_allclose(Bz_coil, Bz, rtol=0.0, atol=2.2e-5)

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


def _debug_plot(test_class, source):
    source_obj = getattr(test_class, source)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Reference solutions
    ax.plot(test_class.Bx_ref, color="navy", lw=6, label="Bx reference")

    ax.plot(test_class.Bz_ref, color="darkred", lw=6, label="Bz reference")

    bx_colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(test_class.ANGLES)))
    by_colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(test_class.ANGLES)))
    bz_colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(test_class.ANGLES)))

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for i, angle in enumerate(test_class.ANGLES):
        Bx, By, Bz = test_class._field_for_angle(source_obj, angle)

        ls = linestyles[i % len(linestyles)]

        ax.plot(Bx, color=bx_colors[i], linestyle=ls, label=f"Bx {angle}°")

        ax.plot(By, color=by_colors[i], linestyle=ls, label=f"By {angle}°")

        ax.plot(Bz, color=bz_colors[i], linestyle=ls, label=f"Bz {angle}°")

    ax.set_title(f"{source} field comparison")
    ax.set_xlabel("Boundary index")
    ax.set_ylabel("Field [T]")
    ax.grid(visible=True)
    ax.legend(ncol=3, fontsize="small")

    fig.tight_layout()
    plt.show()


class TestCircularArcCurrentSourceTwoHalfCirclesXZ:
    ANGLES = (0, 35, 45, 90, 135, 180, 270, 315, 360)

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

        arc_1 = CircularArcCurrentSource(
            [0, 0, cls.zc],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            cls.dx,
            cls.dz,
            cls.xc,
            180,
            cls.current,
        )

        arc_2 = CircularArcCurrentSource(
            [0, 0, cls.zc],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            cls.dx,
            cls.dz,
            cls.xc,
            180,
            cls.current,
        )

        cls.two_halves = SourceGroup([arc_1, arc_2])

        cls.border_rectangle = cls._make_boundary_points()

        x = cls.border_rectangle.x
        z = cls.border_rectangle.z

        cls.Bx_ref = cls.current * semianalytic_Bx(cls.xc, cls.zc, x, z, cls.dx, cls.dz)

        cls.Bz_ref = cls.current * semianalytic_Bz(cls.xc, cls.zc, x, z, cls.dx, cls.dz)

    @classmethod
    def _make_boundary_points(cls, n_edge=50):
        x_horizontal = np.linspace(cls.xc - cls.dx, cls.xc + cls.dx, n_edge)

        z_vertical = np.linspace(cls.zc - cls.dz, cls.zc + cls.dz, n_edge)

        x = np.concatenate([
            x_horizontal,
            np.full(n_edge, cls.xc + cls.dx),
            x_horizontal[::-1],
            np.full(n_edge, cls.xc - cls.dx),
        ])

        z = np.concatenate([
            np.full(n_edge, cls.zc - cls.dz),
            z_vertical,
            np.full(n_edge, cls.zc + cls.dz),
            z_vertical[::-1],
        ])

        return Coordinates({"x": x, "y": np.zeros_like(x), "z": z})

    def _field_for_angle(self, source, angle):
        points = deepcopy(self.border_rectangle)
        points.rotate(degree=angle)

        Bx_rot, By_rot, Bz = source.field(*points.xyz)

        theta = np.deg2rad(-angle)

        Bx = np.cos(theta) * Bx_rot - np.sin(theta) * By_rot
        By = np.sin(theta) * Bx_rot + np.cos(theta) * By_rot

        return Bx, By, Bz

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_Bx_matches_axisymmetric_solution(self, source, angle):
        Bx, _, _ = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            Bx, self.Bx_ref, rtol=0.0, atol=5e-7, err_msg=f"{source=}, {angle=}"
        )

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_By_is_zero(self, source, angle):
        _, By, _ = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            By, 0.0, rtol=0.0, atol=6e-10, err_msg=f"{source=}, {angle=}"
        )

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_Bz_matches_axisymmetric_solution(self, source, angle):
        _, _, Bz = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            Bz, self.Bz_ref, rtol=0.0, atol=6e-10, err_msg=f"{source=}, {angle=}"
        )

    def test_mirror_singularity(self):
        """
        Somewhat counterintuitively, the remaining singularities occur
        with R_j << R_pc and z_k =0.0, phi = 0.0

        Here z_k is rounding noise (~1e-15) and r_j - r_pc * cos(psi) + sqrt_term
        cancels around psi = 0. The integrands should handle this.
        """
        c = deepcopy(self.border_rectangle)
        c.rotate(base=(0, 0, 0), direction=(0, 0, 1), degree=180)
        self.two_halves.field(*c.xyz)

    def test_debug_plot(self):
        _debug_plot(self, "arc")
        _debug_plot(self, "two_halves")


class TestCircularArcCurrentSourceTwoHalfCirclesXY:
    ANGLES = (0, 35, 45, 90, 135, 180, 270, 315, 360)

    @classmethod
    def setup_class(cls):
        cls.xc, cls.zc = 5, 4
        cls.r_0 = 10.0
        cls.dx = 0.75
        cls.dz = 1.0
        cls.current = 1e6

        cls.arc = CircularArcCurrentSource(
            [cls.r_0, 0, cls.zc],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            cls.dx,
            cls.dz,
            cls.xc,
            360,
            cls.current,
        )

        arc_1 = CircularArcCurrentSource(
            [cls.r_0, 0, cls.zc],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            cls.dx,
            cls.dz,
            cls.xc,
            180,
            cls.current,
        )

        arc_2 = CircularArcCurrentSource(
            [cls.r_0, 0, cls.zc],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            cls.dx,
            cls.dz,
            cls.xc,
            180,
            -cls.current,
        )

        cls.two_halves = SourceGroup([arc_1, arc_2])

        cls.border_rectangle = cls._make_boundary_points()

        x = cls.border_rectangle.x
        z = cls.border_rectangle.z

        cls.Bx_ref = cls.current * semianalytic_Bx(cls.xc, cls.zc, x, z, cls.dx, cls.dz)

        cls.Bz_ref = cls.current * semianalytic_Bz(cls.xc, cls.zc, x, z, cls.dx, cls.dz)
        cls.border_rectangle.rotate(base=(0, 0, 0), direction=(1, 0, 0), degree=90)
        cls.border_rectangle.translate((cls.r_0, cls.zc, cls.zc))
        cls.border_rectangle.rotate(
            base=(cls.r_0, 0, cls.zc), direction=(0, 1, 0), degree=-90
        )
        ax = Plot3D()
        cls.two_halves.sources[0].plot(ax=ax)
        ax.scatter(*cls.border_rectangle.xyz)
        plt.show()

    @classmethod
    def _make_boundary_points(cls, n_edge=50):
        x_horizontal = np.linspace(cls.xc - cls.dx, cls.xc + cls.dx, n_edge)

        z_vertical = np.linspace(cls.zc - cls.dz, cls.zc + cls.dz, n_edge)

        x = np.concatenate([
            x_horizontal,
            np.full(n_edge, cls.xc + cls.dx),
            x_horizontal[::-1],
            np.full(n_edge, cls.xc - cls.dx),
        ])

        z = np.concatenate([
            np.full(n_edge, cls.zc - cls.dz),
            z_vertical,
            np.full(n_edge, cls.zc + cls.dz),
            z_vertical[::-1],
        ])

        return Coordinates({"x": x, "y": np.zeros_like(x), "z": z})

    def _field_for_angle(self, source, angle):
        points = deepcopy(self.border_rectangle)
        points.rotate(base=(self.r_0, 0, self.zc), direction=(0, 1, 0), degree=angle)

        Bx_rot, By_rot, Bz_rot = source.field(*points.xyz)

        theta = np.deg2rad(angle + 90)
        Bx = np.cos(theta) * Bx_rot - np.sin(theta) * Bz_rot
        By = np.sin(theta) * Bx_rot + np.cos(theta) * Bz_rot
        Bz = By_rot

        return Bx, By, Bz

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_Bx_matches_axisymmetric_solution(self, source, angle):
        Bx, _, _ = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            Bx, self.Bx_ref, rtol=0.0, atol=5e-7, err_msg=f"{source=}, {angle=}"
        )

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_By_is_zero(self, source, angle):
        _, By, _ = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            By, 0.0, rtol=0.0, atol=6e-10, err_msg=f"{source=}, {angle=}"
        )

    @pytest.mark.parametrize("source", ["arc", "two_halves"])
    @pytest.mark.parametrize("angle", ANGLES)
    def test_Bz_matches_axisymmetric_solution(self, source, angle):
        _, _, Bz = self._field_for_angle(getattr(self, source), angle)

        np.testing.assert_allclose(
            Bz, self.Bz_ref, rtol=0.0, atol=4e-9, err_msg=f"{source=}, {angle=}"
        )

    def test_mirror_singularity(self):
        """
        Somewhat counterintuitively, the remaining singularities occur
        with R_j << R_pc and z_k =0.0, phi = 0.0

        Here z_k is rounding noise (~1e-15) and r_j - r_pc * cos(psi) + sqrt_term
        cancels around psi = 0. The integrands should handle this.
        """
        c = deepcopy(self.border_rectangle)
        c.rotate(base=(0, 0, 0), direction=(0, 0, 1), degree=180)
        self.two_halves.field(*c.xyz)

    def test_debug_plot(self):
        _debug_plot(self, "arc")
        _debug_plot(self, "two_halves")


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
        ds0: np.ndarray, normal0: np.ndarray, t_vec: np.ndarray, start_angle_deg: float
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
            ds0=ds0, normal0=normal0, t_vec=t_vec, start_angle_deg=start_angle_deg
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
            # TODO(CoronelBuendia): https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/4468
            # Address near-singularity integration errors
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

        np.testing.assert_allclose(b_a, b_b, rtol=cls.rtol, atol=cls.atol)

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
    def test_full_ring_equals_two_half_rings(self, point, ds0, normal0, t_vec):
        """360 degree ring == 180 degree + 180 degree, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0, normal0=normal0, t_vec=t_vec, start_angle_deg=0.0, dtheta_deg=360.0
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
    def test_full_ring_equals_four_quarter_rings(self, point, ds0, normal0, t_vec):
        """360 degree ring == 4 x 90 degree arcs, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0, normal0=normal0, t_vec=t_vec, start_angle_deg=0.0, dtheta_deg=360.0
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
    def test_full_ring_equals_eight_octant_rings(self, point, ds0, normal0, t_vec):
        """360 degree ring == 8 x 45 degree arcs, for multiple axes."""
        full_ring = self.make_arc(
            ds0=ds0, normal0=normal0, t_vec=t_vec, start_angle_deg=0.0, dtheta_deg=360.0
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
        self, ds0, normal0, t_vec, start_angle_deg, dtheta_deg
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
        self, ds0, normal0, t_vec, start_angle_deg, dtheta_deg, n_segments
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


class TestCircularArcVectorPotential:
    R, B, D = 4.0, 0.5, 1.0
    ANGLE = 130.0
    CURRENT = 1e6

    @classmethod
    def _make_arc(cls, dtheta, start=0.0, breadth=B, depth=D):
        t = np.deg2rad(start)
        return CircularArcCurrentSource(
            np.zeros(3),
            np.array([np.cos(t), np.sin(t), 0.0]),
            np.array([-np.sin(t), np.cos(t), 0.0]),
            np.array([0.0, 0.0, 1.0]),
            breadth,
            depth,
            cls.R,
            dtheta,
            cls.CURRENT,
        )

    @staticmethod
    def _cylindrical(r, angle, z):
        t = np.deg2rad(angle)
        return np.array([r * np.cos(t), r * np.sin(t), z])

    @classmethod
    def _brute_force(cls, point, n_theta=200, n_xs=30):
        """3-D Gauss-Legendre volume integral; for exterior points only."""
        xt, wt = np.polynomial.legendre.leggauss(n_theta)
        xs, ws = np.polynomial.legendre.leggauss(n_xs)
        half_angle = 0.5 * np.deg2rad(cls.ANGLE)
        t, r, z = np.meshgrid(
            half_angle * (xt + 1), cls.R + cls.B * xs, cls.D * xs, indexing="ij"
        )
        weights = half_angle * wt[:, None, None] * ws[None, :, None] * ws
        # The cross-section Jacobian cancels with the area in J = I / (4 B D)
        x, y = r * np.cos(t), r * np.sin(t)
        dist = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2 + (point[2] - z) ** 2)
        dv = 0.25 * MU_0_4PI * cls.CURRENT * weights * r / dist
        return np.array([-np.sum(dv * np.sin(t)), np.sum(dv * np.cos(t)), 0.0])

    @staticmethod
    def _curl(source, point, h=2e-3):
        def a(p):
            return source.vector_potential(*p)

        # 4th order central difference; grad[i, j] = dA_j/dx_i
        grad = np.array([
            (8 * (a(point + e) - a(point - e)) - (a(point + 2 * e) - a(point - 2 * e)))
            / (12 * h)
            for e in h * np.eye(3)
        ])
        return np.array([
            grad[1, 2] - grad[2, 1],
            grad[2, 0] - grad[0, 2],
            grad[0, 1] - grad[1, 0],
        ])

    @pytest.mark.parametrize(
        "point", [(1, 2, 0.3), (0, 0, 0), (0, 0, 2), (-3, 4, 1), (6, 1, -2), (10, -4, 0)]
    )
    def test_brute_force(self, point):
        np.testing.assert_allclose(
            self._make_arc(self.ANGLE).vector_potential(*point),
            self._brute_force(point),
            rtol=1e-10,
            atol=1e-14,
        )

    @pytest.mark.parametrize(
        "point",
        [
            (1, 2, 0.3),
            (6, 1, -2),
            (-2, 3.2, 0.4),
            # Inside the conductor
            (2.5, 3.1, 0.2),
            (3.9, 0.3, -0.6),
        ],
    )
    def test_curl_is_field(self, point):
        arc = self._make_arc(self.ANGLE)
        point = np.array(point, dtype=float)
        np.testing.assert_allclose(
            self._curl(arc, point), arc.field(*point), rtol=0.0, atol=1e-8
        )

    @pytest.mark.parametrize(
        ("r", "angle", "z"),
        [
            pytest.param(R - B, 45, 0, id="inner face"),
            pytest.param(R, 45, D, id="top face"),
            pytest.param(R - B, 45, D, id="inner edge"),
            pytest.param(R + B, 45, -D, id="outer edge"),
            pytest.param(R, 0, 0, id="start cap"),
            pytest.param(R - B, 0, D, id="start corner"),
            pytest.param(R + B, ANGLE, D, id="end corner"),
            pytest.param(R, 180, 0, id="opposite start cap"),
            pytest.param(R + B, 45 + 180, D, id="opposite edge"),
            pytest.param(R + 2 * B, 45, D, id="top face extension"),
            pytest.param(R - B, 45, 3 * D, id="inner face extension"),
            pytest.param(0, 0, D, id="axis"),
        ],
    )
    def test_singularities_continuous(self, r, angle, z):
        arc = self._make_arc(self.ANGLE)
        point = self._cylindrical(r, angle, z)
        value = arc.vector_potential(*point)
        assert np.all(np.isfinite(value))
        for offset in 1e-7 * np.array([[0.3, 0.5, 0.7], [-0.7, 0.3, -0.5]]):
            np.testing.assert_allclose(
                arc.vector_potential(*(point + offset)), value, rtol=0.0, atol=1e-7
            )

    def test_zero_inner_radius(self):
        arc = self._make_arc(self.ANGLE, breadth=self.R)
        value = arc.vector_potential(0.0, 0.0, 0.0)
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(
            arc.vector_potential(1e-7, 1e-7, 1e-7), value, rtol=0.0, atol=1e-7
        )

    @pytest.mark.parametrize("half_width", [0.5, 0.05, 0.005])
    def test_full_circle_vs_greens(self, half_width):
        circle = self._make_arc(360, breadth=half_width, depth=half_width)
        x, z = 6.0, 0.5
        xs, ws = np.polynomial.legendre.leggauss(8)
        psi = sum(
            0.25 * wi * wj * greens_psi(self.R + half_width * xi, half_width * xj, x, z)
            for xi, wi in zip(xs, ws, strict=True)
            for xj, wj in zip(xs, ws, strict=True)
        )
        np.testing.assert_allclose(
            circle.vector_potential(x, 0.0, z),
            [0.0, self.CURRENT * psi / x, 0.0],
            rtol=1e-9,
            atol=1e-14,
        )

    @pytest.mark.parametrize("angle", [ANGLE, 360.0])
    @pytest.mark.parametrize(
        "point", [(1, 2, 0.3), (2.5, 3.1, 0.2), (-3.5, 0, 0), (4, 0, 1)]
    )
    def test_superposition(self, angle, point):
        split = 0.3 * angle
        parts = (self._make_arc(split), self._make_arc(angle - split, start=split))
        np.testing.assert_allclose(
            sum(p.vector_potential(*point) for p in parts),
            self._make_arc(angle).vector_potential(*point),
            rtol=0.0,
            atol=1e-9,
        )

    def test_rotate_translate(self):
        reference = self._make_arc(self.ANGLE)
        moved = self._make_arc(self.ANGLE)
        moved.rotate(35, "x")
        shift = np.array([1.0, -2.0, 3.0])
        moved.translate(shift)
        r = rotation_matrix(np.deg2rad(35), "x")
        point = np.array([2.5, 3.1, 0.2])
        np.testing.assert_allclose(
            moved.vector_potential(*(r @ point + shift)),
            r @ reference.vector_potential(*point),
            rtol=0.0,
            atol=1e-12,
        )

    def test_array_inputs(self):
        arc = self._make_arc(self.ANGLE)
        x, z = np.meshgrid([1.0, 4.0, 6.0], [0.0, 0.5], indexing="ij")
        y = np.ones_like(x)
        result = arc.vector_potential(x, y, z)
        assert result.shape == (3, 3, 2)
        np.testing.assert_array_equal(
            result[:, 1, 1], arc.vector_potential(4.0, 1.0, 0.5)
        )
        np.testing.assert_array_equal(
            result[:, :, 0], arc.vector_potential(x[:, 0], y[:, 0], z[:, 0])
        )
