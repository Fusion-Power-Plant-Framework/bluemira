# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.constants import EPS, MU_0_4PI, raw_uc
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


class TestTrapezoidalPrismBabicAykel:
    """
    Verification test.

    Babic and Aykel example

    .. doi:: 10.1002/jnm.594
    """

    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
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
    # The paper's local coordinate system is not centred on the prism when
    # alpha != beta; shift the paper's points into the centred coordinate system.
    paper_offset = 0.5 * (np.tan(np.deg2rad(60.0)) - np.tan(np.deg2rad(30.0)))

    def _paper_field(self, point):
        point = np.asarray(point, dtype=float) - self.paper_offset * self.source._dcm[1]
        return self.source.field(*point)

    @pytest.mark.parametrize(
        ("point", "value", "precision"),
        [((2, 2, 2), 15.5533805, 7), ((1, 1, 1), 53.581000397, 9)],
    )
    def test_paper_singularity_values(self, point, value, precision):
        field = self._paper_field(point)
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**precision) / 10**precision
        assert field_ndecimals == pytest.approx(value, rel=0, abs=EPS)

    def test_paper_inside_conductor(self):
        field = self._paper_field((0.5, 0.5, 0.5))
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**7) / 10**7
        assert field_ndecimals == pytest.approx(34.9969156, rel=0, abs=EPS)

    def _paper_curl_vector_potential(self, point, h=1e-6):
        """
        Curl of the vector potential by central differences.
        """
        point = np.asarray(point, dtype=float) - self.paper_offset * self.source._dcm[1]
        jac = np.array([
            (
                self.source.vector_potential(*(point + h * e))
                - self.source.vector_potential(*(point - h * e))
            )
            / (2 * h)
            for e in np.eye(3)
        ])  # jac[i, j] = dA_j / dx_i
        return np.array([
            jac[1, 2] - jac[2, 1],
            jac[2, 0] - jac[0, 2],
            jac[0, 1] - jac[1, 0],
        ])

    @pytest.mark.parametrize(
        ("point", "value"),
        [
            ((2, 2, 2), 15.5533805),
            ((1, 1, 1), 53.581000397),
            ((0.5, 0.5, 0.5), 34.9969156),
        ],
    )
    def test_paper_vector_potential(self, point, value):
        field = self._paper_curl_vector_potential(point)
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Aykel paper. The tolerance is limited by the finite
        # differences at (1, 1, 1), which lies on an edge of the prism, where the
        # field has a kink and central differences only converge linearly in h
        # (error ~2e-5 mT at h = 1e-6).
        assert abs_field == pytest.approx(value, rel=0, abs=1e-4)


class TestTrapezoidalPrismPolygonLoop:
    """
    Verification tests of mitred prisms daisy-chained into a closed convex polygonal
    loop of rectangular cross-section, in the x-z plane.

    With uniform current density, the mitred loop is exactly a superposition of thin
    filament polygons: the filament at in-plane offset x' (outward) from the
    centreline is the centreline polygon offset by x', with its corners on the mitre
    bisectors. The reference field is therefore the analytical Biot-Savart field of
    a finite straight filament, summed over the polygon sides and integrated over the
    cross-section with Gauss-Legendre quadrature.
    """

    breadth = 0.25  # half-width (in-plane)
    depth = 0.15  # half-height (out-of-plane)
    current = 1e6
    # Counter-clockwise in (x, z); normal = -y so that t_vec = ds x normal points
    # outward and the local frame is right-handed
    normal = np.array([0.0, -1.0, 0.0])

    # Square: every corner has a 45 degree mitre, so alpha == beta for all prisms
    square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    # Isosceles trapezoid: the legs join a ~58 degree mitre to a ~32 degree mitre,
    # so alpha != beta for those prisms
    trapezoid = np.array([[-3.0, -1.5], [3.0, -1.5], [1.5, 1.5], [-1.5, 1.5]])

    points = (
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -2.5, 0.0),
        (0.7, 0.4, -0.3),
        (1.2, 0.0, 1.0),
        (3.5, 0.5, 0.2),
        (2.0, 1.0, 2.0),
        (-1.0, -0.8, 1.5),
    )

    @staticmethod
    def _to_3d(polygon, y=0.0):
        return np.column_stack([polygon[:, 0], np.full(len(polygon), y), polygon[:, 1]])

    @staticmethod
    def _outward_normals(polygon):
        """Outward unit normals (x, z) of each side i (from corner i to i + 1)."""
        d = np.roll(polygon, -1, axis=0) - polygon
        n = np.column_stack([d[:, 1], -d[:, 0]])
        return n / np.linalg.norm(n, axis=1)[:, None]

    @classmethod
    def _mitre_angles(cls, polygon):
        """Mitre half-angle [degrees] at each corner i."""
        n = cls._outward_normals(polygon)
        n_prev = np.roll(n, 1, axis=0)
        turn = np.arccos(np.clip(np.sum(n_prev * n, axis=1), -1.0, 1.0))
        return np.rad2deg(0.5 * turn)

    @classmethod
    def _offset_polygon(cls, polygon, offset):
        """Polygon offset outward by offset, with corners on the mitre bisectors."""
        n = cls._outward_normals(polygon)
        n_prev = np.roll(n, 1, axis=0)
        return (
            polygon + offset * (n_prev + n) / (1 + np.sum(n_prev * n, axis=1))[:, None]
        )

    @classmethod
    def _make_sources(cls, polygon):
        corners = cls._to_3d(polygon)
        mitres = cls._mitre_angles(polygon)
        sources = []
        for i, start in enumerate(corners):
            end = corners[(i + 1) % len(corners)]
            ds = end - start
            sources.append(
                TrapezoidalPrismCurrentSource(
                    0.5 * (start + end),
                    ds,
                    cls.normal,
                    np.cross(ds / np.linalg.norm(ds), cls.normal),
                    cls.breadth,
                    cls.depth,
                    mitres[(i + 1) % len(corners)],  # alpha: end corner
                    mitres[i],  # beta: start corner
                    cls.current,
                )
            )
        return sources

    @staticmethod
    def _filament_segment_field(point, start, end, current):
        r1 = point - start
        r2 = point - end
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        return (
            MU_0_4PI
            * current
            * np.cross(r1, r2)
            * (n1 + n2)
            / (n1 * n2 * (n1 * n2 + np.dot(r1, r2)))
        )

    @classmethod
    def _analytical_field(cls, polygon, point, n_quad=12):
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        field = np.zeros(3)
        for xi, wx in zip(nodes, weights, strict=True):
            filament = cls._offset_polygon(polygon, cls.breadth * xi)
            for zi, wz in zip(nodes, weights, strict=True):
                corners = cls._to_3d(filament, y=-cls.depth * zi)  # z' is along -y
                # Weights sum to 2 per dimension, so normalise by 4
                i_fil = 0.25 * wx * wz * cls.current
                for start, end in zip(
                    corners, np.roll(corners, -1, axis=0), strict=True
                ):
                    field += cls._filament_segment_field(point, start, end, i_fil)
        return field

    def _check_field(self, polygon, point):
        point = np.array(point, dtype=float)
        field = sum(s.field(*point) for s in self._make_sources(polygon))
        expected = self._analytical_field(polygon, point)
        np.testing.assert_allclose(
            field, expected, rtol=1e-8, atol=1e-10 * np.linalg.norm(expected)
        )

    @pytest.mark.parametrize("point", points)
    def test_square_field_against_analytical(self, point):
        self._check_field(self.square, point)

    @pytest.mark.parametrize("point", points)
    def test_trapezoid_field_against_analytical(self, point):
        self._check_field(self.trapezoid, point)


class TestTrapezoidalPrismCurrentSource:
    @pytest.mark.parametrize("angle", [54, 45.0001])
    def test_error_on_self_intersect(self, angle):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle,
                angle,
                current=1.0,
            )

    def test_no_error_on_triangle(self):
        TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.5,
            0.1,
            45,
            45,
            current=1.0,
        )

    @pytest.mark.parametrize("angle", [90, 180, 270, 360])
    def test_error_on_angle_limits(self, angle):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle,
                0.25 * np.pi,
                current=1.0,
            )

    @pytest.mark.parametrize(("angle1", "angle2"), [(10, -10), (-20, 30)])
    def test_error_on_mixed_sign_angles(self, angle1, angle2):
        with pytest.raises(MagnetostaticsError):
            TrapezoidalPrismCurrentSource(
                np.array([0, 0, 0]),
                np.array([0, 0, 1]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                0.5,
                0.1,
                angle1,
                angle2,
                current=1.0,
            )

    @pytest.mark.parametrize(("angle1", "angle2"), [(1, 1), (2, 3), (0, 2), (-2, 0)])
    def test_no_error_on_double_sign_angles(self, angle1, angle2):
        TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.5,
            0.1,
            angle1,
            angle2,
            current=1.0,
        )

    @pytest.mark.parametrize(
        "vector", [[], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [[1.0, 2.0, 3.0]]]
    )
    def test_error_on_bad_translation_vector(self, vector):
        source = TrapezoidalPrismCurrentSource(
            np.array([0, 0, 0]),
            np.array([0, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.5,
            0.1,
            10,
            10,
            current=1.0,
        )
        with pytest.raises(MagnetostaticsError):
            source.translate(vector)
