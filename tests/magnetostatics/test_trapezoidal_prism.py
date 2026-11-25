# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.constants import EPS, MU_0, MU_0_4PI, raw_uc
from bluemira.magnetostatics.circuits import ArbitraryPlanarRectangularXSCircuit
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


class TestTrapezoidalPrismBabicAkyel:
    """
    Verification test.

    Babic and Akyel example

    .. doi:: 10.1002/jnm.594
    """

    # Babic and Akyel example (single trapezoidal prism)
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
        # As per Babic and Akyel paper
        # Assume truncated last digit and not rounded...
        field_ndecimals = np.trunc(abs_field * 10**precision) / 10**precision
        assert field_ndecimals == pytest.approx(value, rel=0, abs=EPS)

    def test_paper_inside_conductor(self):
        field = self._paper_field((0.5, 0.5, 0.5))
        abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
        # As per Babic and Akyel paper
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
        # As per Babic and Akyel paper. The tolerance is limited by the finite
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
    # Irregular convex hexagon: every corner has a different mitre, so alpha != beta
    # for every prism and each integration domain is offset from the prism centre
    hexagon = np.array([
        [-3.0, -4.0],
        [2.5, -3.5],
        [4.5, 0.0],
        [2.0, 4.0],
        [-2.0, 3.0],
        [-4.5, -0.5],
    ])

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
    # Hexagon points, including points just outside and inside the loop next to
    # the mitred joints, where the field is most sensitive to the prism positions
    hexagon_points = (
        (0.0, 0.0, 0.0),
        (0.7, 0.4, -0.3),
        (6.0, 0.5, 0.2),
        (2.0, 0.1, 4.7),
        (2.0, 0.0, 3.4),
        (5.0, -0.1, 0.0),
        (-3.2, 0.05, -4.6),
        (-2.0, 0.1, 2.5),
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

    def _check_field(self, polygon, point, n_quad=12):
        point = np.array(point, dtype=float)
        field = sum(s.field(*point) for s in self._make_sources(polygon))
        expected = self._analytical_field(polygon, point, n_quad=n_quad)
        np.testing.assert_allclose(
            field, expected, rtol=1e-8, atol=1e-10 * np.linalg.norm(expected)
        )

    @pytest.mark.parametrize("point", points)
    def test_square_field_against_analytical(self, point):
        self._check_field(self.square, point)

    @pytest.mark.parametrize("point", points)
    def test_trapezoid_field_against_analytical(self, point):
        self._check_field(self.trapezoid, point)

    @pytest.mark.parametrize("point", hexagon_points)
    def test_hexagon_field_against_analytical(self, point):
        self._check_field(self.hexagon, point, n_quad=48)

    @pytest.mark.parametrize("point", hexagon_points)
    def test_hexagon_circuit_field_against_analytical(self, point):
        corners = self._to_3d(self.hexagon)
        circuit = ArbitraryPlanarRectangularXSCircuit(
            np.vstack([corners, corners[:1]]).T, self.breadth, self.depth, self.current
        )
        point = np.array(point, dtype=float)
        expected = self._analytical_field(self.hexagon, point, n_quad=48)
        np.testing.assert_allclose(
            circuit.field(*point),
            expected,
            rtol=1e-8,
            atol=1e-10 * np.linalg.norm(expected),
        )

    def test_hexagon_prisms_meet_on_mitres(self):
        sources = self._make_sources(self.hexagon)
        assert all(abs(s._y_offset) > 1e-3 for s in sources)
        for source, next_source in zip(sources, sources[1:] + sources[:1], strict=True):
            # alpha end-cap of each prism == beta end-cap of the next one
            end_cap = source._local_to_global(source._vertices)[[3, 2, 7, 6]]
            start_cap = next_source._local_to_global(next_source._vertices)[[0, 1, 4, 5]]
            np.testing.assert_allclose(end_cap, start_cap, rtol=0, atol=1e-12)


class TestTrapezoidalPrismInfiniteRectangularConductor:
    """
    Verification test against the classical 2-D analytical solution for an
    infinitely long straight conductor of rectangular cross-section with uniform
    current density. The reference B and A_z are the closed-form integrals of the
    2-D line-current Green's function (B ~ 1/r, A_z ~ -ln r) over the rectangle,
    evaluated as signed sums of their antiderivatives at the four corners.

    At the mid-plane of a very long prism, the field and the vector potential
    differences converge to the 2-D solution with an error O((r / L)^2).
    """

    a, b = 0.3, 0.2  # half-widths of the cross-section [m]
    current = 1e6  # [A]
    length = 1e4  # [m]

    # Current along global z, cross-section in x-y (t_vec = ds x normal)
    source = TrapezoidalPrismCurrentSource(
        np.zeros(3),
        np.array([0, 0, length]),
        np.array([0, -1, 0]),
        np.array([1, 0, 0]),
        a,
        b,
        0.0,
        0.0,
        current,
    )
    # Points outside, inside and far from the conductor (not on edge extensions)
    points = ((0.7, 0.45), (0.1, 0.05), (-0.25, 0.13), (1.3, -2.1), (-0.4, -0.15))

    def _corner_sum(self, func, x, y):
        u1, u2 = x - self.a, x + self.a
        v1, v2 = y - self.b, y + self.b
        return func(u2, v2) - func(u1, v2) - func(u2, v1) + func(u1, v1)

    def _field_2d(self, x, y):
        k = MU_0 * self.current / (4 * self.a * self.b) / (2 * np.pi)

        def f_bx(u, v):
            return 0.5 * u * np.log(u**2 + v**2) + v * np.arctan(u / v)

        def f_by(u, v):
            return 0.5 * v * np.log(u**2 + v**2) + u * np.arctan(v / u)

        return np.array([
            -k * self._corner_sum(f_bx, x, y),
            k * self._corner_sum(f_by, x, y),
            0.0,
        ])

    def _vector_potential_2d(self, x, y):
        k = MU_0 * self.current / (4 * self.a * self.b) / (2 * np.pi)

        def f_az(u, v):
            return 0.5 * (
                u * v * np.log(u**2 + v**2)
                - 3 * u * v
                + u**2 * np.arctan(v / u)
                + v**2 * np.arctan(u / v)
            )

        return -k * self._corner_sum(f_az, x, y)

    @pytest.mark.parametrize("point", points)
    def test_field(self, point):
        field = self.source.field(*point, 0.0)
        expected = self._field_2d(*point)
        np.testing.assert_allclose(
            field, expected, rtol=0, atol=1e-6 * np.linalg.norm(expected)
        )

    @pytest.mark.parametrize("point", points)
    def test_vector_potential_difference(self, point):
        # The 2-D vector potential is only defined up to a constant
        ref_point = (2.0, 1.5)
        potential = self.source.vector_potential(*point, 0.0)
        ref_potential = self.source.vector_potential(*ref_point, 0.0)
        np.testing.assert_allclose(potential[:2], 0.0, rtol=0, atol=EPS)
        np.testing.assert_allclose(
            potential[2] - ref_potential[2],
            self._vector_potential_2d(*point) - self._vector_potential_2d(*ref_point),
            rtol=5e-6,
        )


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
