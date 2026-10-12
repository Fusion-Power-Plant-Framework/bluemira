# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.utilities.error import PositionerError
from bluemira.utilities.positioning import (
    PathInterpolator,
    PositionMapper,
    RegionInterpolator,
)


class TestPathInterpolator:
    @classmethod
    def setup_class(cls):
        x = [0, 2, 4]
        z = [0, 1, 0]
        cls.polygon = make_polygon({"x": x, "z": z})
        cls.circle = make_circle(center=(0, 0, 0), axis=(0, -1, 0), radius=10)

    @pytest.mark.parametrize(
        ("alpha", "point"), [(0.0, [0, 0]), (0.5, [2, 1]), (1.0, [4, 0])]
    )
    def test_open(self, alpha, point):
        interpolator = PathInterpolator(self.polygon)

        x0, z0 = interpolator.to_xz(alpha)
        assert np.isclose(x0, point[0])
        assert np.isclose(z0, point[1])
        assert np.isclose(interpolator.to_L(*point), alpha)

    @pytest.mark.parametrize(
        ("alpha", "point"),
        [(0.0, [10, 0]), (0.25, [0, 10]), (0.5, [-10, 0]), (0.75, [0, -10])],
    )
    def test_closed(self, alpha, point):
        interpolator = PathInterpolator(self.circle)
        x0, z0 = interpolator.to_xz(alpha)
        assert np.isclose(x0, point[0])
        assert np.isclose(z0, point[1])
        assert np.isclose(interpolator.to_L(*point), alpha)

    def test_straight(self):
        line = make_polygon([[5, 10], [0, 0], [-10, 10]])
        interpolator = PathInterpolator(line)
        x05, z05 = interpolator.to_xz(0.5)
        assert np.isclose(x05, 7.5)
        assert np.isclose(z05, 0)
        l05 = interpolator.to_L(7.5, 0)
        assert np.isclose(l05, 0.5)


class TestRegionInterpolator:
    @staticmethod
    def _make_square_with_hole() -> BluemiraFace:
        """
        Construct a square face containing a smaller rectangular hole.

        Outer boundary:
            x in [2.0, 4.0]
            z in [1.0, 2.0]

        Hole:
            x in [2.5, 3.5]
            z in [1.25, 1.75]
        """
        outer = make_polygon(
            {"x": [2.0, 4.0, 4.0, 2.0], "z": [1.0, 1.0, 2.0, 2.0]}, closed=True
        )

        hole = make_polygon(
            {"x": [2.5, 3.5, 3.5, 2.5], "z": [1.25, 1.25, 1.75, 1.75]}, closed=True
        )

        return BluemiraFace([outer, hole])

    def test_bad_shapes(self):
        x = [0, 4, 2, 4, 0, 2]
        z = [0, 0, 1, 2, 2, 1]
        open_polygon = make_polygon({"x": x, "z": z})
        with pytest.raises(PositionerError):
            RegionInterpolator(open_polygon)

    def test_circle(self):
        circle = make_circle(center=(0, 0, 0), axis=(0, 1, 0), radius=10)
        interpolator = RegionInterpolator(circle)

        xc, zc = interpolator.to_xz((0.5, 0.5))
        assert np.isclose(xc, 0)
        assert np.isclose(zc, 0)
        l0, l1 = interpolator.to_L(0, 0)
        assert np.isclose(l0, 0.5)
        assert np.isclose(l1, 0.5)

    def test_square_with_hole_intervals(self):
        """
        Check that slicing through the hole produces two filled intervals.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        intervals = region._intervals_at(1.5)

        np.testing.assert_allclose(
            intervals, np.array([[2.0, 2.5], [3.5, 4.0]]), atol=D_TOLERANCE, rtol=0.0
        )

    def test_square_with_hole_forward_mapping(self):
        """
        Check the mapping on either side of the normalised seam.

        Each valid interval has width 0.5, so each occupies half of the
        normalised horizontal coordinate.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        # Beginning of the first interval.
        assert region.to_xz((0.0, 0.5)) == pytest.approx((2.0, 1.5), abs=D_TOLERANCE)

        # Middle of the first interval.
        assert region.to_xz((0.25, 0.5)) == pytest.approx((2.25, 1.5), abs=D_TOLERANCE)

        # Immediately before the seam remains in the first interval.
        assert region.to_xz((0.4999, 0.5)) == pytest.approx(
            (2.4999, 1.5), abs=D_TOLERANCE
        )

        # The mapping uses a right-continuous seam. Exactly 0.5 therefore maps
        # to the start of the second filled interval.
        assert region.to_xz((0.5, 0.5)) == pytest.approx((3.5, 1.5), abs=D_TOLERANCE)

        # Immediately after the seam lies in the second interval.
        assert region.to_xz((0.5001, 0.5)) == pytest.approx(
            (3.5001, 1.5), abs=D_TOLERANCE
        )

        # Middle of the second interval.
        assert region.to_xz((0.75, 0.5)) == pytest.approx((3.75, 1.5), abs=D_TOLERANCE)

        # End of the final interval.
        assert region.to_xz((1.0, 0.5)) == pytest.approx((4.0, 1.5), abs=D_TOLERANCE)

    def test_square_with_hole_never_maps_inside_hole(self):
        """
        Sample the full normalised scan line and verify that no generated point
        is strictly inside the hole.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        # Tested with 1e4 points
        l_0 = np.linspace(0.0, 1.0, 101)
        result_x, result_z = region.to_xz((l_0, 0.5))

        inside_hole = (result_x > 2.5 + D_TOLERANCE) & (result_x < 3.5 - D_TOLERANCE)

        assert not np.any(inside_hole)

        np.testing.assert_allclose(result_z, 1.5, atol=D_TOLERANCE, rtol=0.0)

    @pytest.mark.parametrize(
        ("x", "expected_x"),
        [
            (1.0, 2.0),  # Left of the face.
            (2.25, 2.25),  # Inside the first material interval.
            (2.75, 2.5),  # Inside the hole and closer to its left boundary.
            (3.25, 3.5),  # Inside the hole and closer to its right boundary.
            (3.75, 3.75),  # Inside the second material interval.
            (5.0, 4.0),  # Right of the face.
        ],
    )
    def test_to_L_projects_onto_nearest_material(self, x, expected_x):
        """
        Check that points outside the material are projected onto the nearest
        filled interval.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        l_0, l_1 = region.to_L(x, 1.5)
        actual_x, actual_z = region.to_xz((l_0, l_1))

        assert actual_x == pytest.approx(expected_x, abs=D_TOLERANCE)
        assert actual_z == pytest.approx(1.5, abs=D_TOLERANCE)

    def test_point_at_centre_of_hole_uses_deterministic_tie_break(self):
        """
        The centre of the hole is equidistant from both boundaries.

        The implementation resolves equal-distance projections towards the
        smaller normalised coordinate, which is the right endpoint of the
        left-hand material interval.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        l_0, l_1 = region.to_L(3.0, 1.5)
        actual_x, actual_z = region.to_xz((l_0, l_1))

        assert actual_x == pytest.approx(2.5, abs=D_TOLERANCE)
        assert actual_z == pytest.approx(1.5, abs=D_TOLERANCE)

    @pytest.mark.parametrize(
        ("x", "z"),
        [(2.0, 1.5), (2.25, 1.5), (2.499, 1.5), (3.501, 1.5), (3.75, 1.5), (4.0, 1.5)],
    )
    def test_valid_material_points_round_trip(self, x, z):
        """
        Physical points strictly within material should round-trip through
        normalised space.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        l_0, l_1 = region.to_L(x, z)
        actual_x, actual_z = region.to_xz((l_0, l_1))

        assert actual_x == pytest.approx(x, abs=D_TOLERANCE)
        assert actual_z == pytest.approx(z, abs=D_TOLERANCE)

    def test_hole_boundary_points_round_trip(self):
        """
        Both physical sides of the normalised seam should round-trip.

        The left side is encoded one floating-point value below the seam, while
        the right side uses the exact seam value.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        left_l_0, left_l_1 = region.to_L(2.5, 1.5)
        right_l_0, right_l_1 = region.to_L(3.5, 1.5)

        left_x, left_z = region.to_xz((left_l_0, left_l_1))
        right_x, right_z = region.to_xz((right_l_0, right_l_1))

        assert left_l_0 < 0.5
        assert right_l_0 == pytest.approx(0.5)

        assert left_x == pytest.approx(2.5, abs=D_TOLERANCE)
        assert left_z == pytest.approx(1.5, abs=D_TOLERANCE)

        assert right_x == pytest.approx(3.5, abs=D_TOLERANCE)
        assert right_z == pytest.approx(1.5, abs=D_TOLERANCE)

    def test_vectorised_round_trip_through_material(self):
        """
        Check vectorised conversion for points on both sides of the hole.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        expected_x = np.array([2.0, 2.1, 2.3, 2.49, 3.51, 3.7, 3.9, 4.0])
        expected_z = np.full_like(expected_x, 1.5)

        l_0, l_1 = region.to_L(expected_x, expected_z)
        actual_x, actual_z = region.to_xz((l_0, l_1))

        np.testing.assert_allclose(actual_x, expected_x, atol=D_TOLERANCE, rtol=0.0)
        np.testing.assert_allclose(actual_z, expected_z, atol=D_TOLERANCE, rtol=0.0)

    def test_hole_is_only_active_over_its_z_range(self):
        """
        Below and above the hole, the complete outer width should be filled.
        """
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        below_hole = region._intervals_at(1.1)
        above_hole = region._intervals_at(1.9)

        expected = np.array([[2.0, 4.0]])

        np.testing.assert_allclose(below_hole, expected, atol=D_TOLERANCE, rtol=0.0)
        np.testing.assert_allclose(above_hole, expected, atol=D_TOLERANCE, rtol=0.0)

        assert region.to_xz((0.5, 0.1)) == pytest.approx((3.0, 1.1), abs=D_TOLERANCE)
        assert region.to_xz((0.5, 0.9)) == pytest.approx((3.0, 1.9), abs=D_TOLERANCE)

    def test_concave_polygon(self):
        """
        Check a concave C-shaped polygon.

        At z=2, the material consists of two disjoint intervals despite the
        geometry having no topological hole.
        """
        polygon = make_polygon(
            {
                "x": [0.0, 4.0, 4.0, 1.0, 1.0, 4.0, 4.0, 0.0],
                "z": [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0],
            },
            closed=True,
        )

        region = RegionInterpolator(polygon)

        # The indentation removes x in (1, 4) at z=2. Depending on whether the
        # polygon is interpreted as a boundary wire or filled face, this slice
        # has the single material interval [0, 1].
        intervals = region._intervals_at(2.0)

        np.testing.assert_allclose(
            intervals, np.array([[0.0, 1.0]]), atol=D_TOLERANCE, rtol=0.0
        )

        assert region.to_xz((0.0, 0.5)) == pytest.approx((0.0, 2.0), abs=D_TOLERANCE)
        assert region.to_xz((0.5, 0.5)) == pytest.approx((0.5, 2.0), abs=D_TOLERANCE)
        assert region.to_xz((1.0, 0.5)) == pytest.approx((1.0, 2.0), abs=D_TOLERANCE)

    def test_open_polygon_is_rejected(self):
        polygon = make_polygon(
            {"x": [0.0, 1.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0, 1.0]}, closed=False
        )

        with pytest.raises(PositionerError, match="closed wires"):
            RegionInterpolator(polygon)

    def test_normalised_coordinates_are_clipped(self):
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        # Vertical coordinate is below the normalised range.
        assert region.to_xz((-1.0, -1.0)) == pytest.approx((2.0, 1.0), abs=D_TOLERANCE)

        # Both coordinates are above the normalised range.
        assert region.to_xz((2.0, 2.0)) == pytest.approx((4.0, 2.0), abs=D_TOLERANCE)

    def test_array_inputs_are_broadcast(self):
        face = self._make_square_with_hole()
        region = RegionInterpolator(face)

        l_0 = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        result_x, result_z = region.to_xz((l_0, 0.5))

        np.testing.assert_allclose(
            result_x, np.array([2.0, 2.25, 3.5, 3.75, 4.0]), atol=D_TOLERANCE, rtol=0.0
        )
        np.testing.assert_allclose(result_z, np.full(5, 1.5), atol=D_TOLERANCE, rtol=0.0)


class TestPositionMapper:
    @classmethod
    def setup_class(cls):
        x = [2, 4, 4, 2]
        z = [1, 1, 2, 2]
        convex_polygon = make_polygon({"x": x, "z": z}, closed=True)
        circle = make_circle(center=(0, 0, 0), axis=(0, -1, 0), radius=10)

        interpolators = {
            "circle_path": PathInterpolator(circle),
            "circle_region": RegionInterpolator(circle),
            "polygon_region": RegionInterpolator(convex_polygon),
        }
        cls.mapper = PositionMapper(interpolators)

    def test_dimensionality(self):
        assert self.mapper.dimension == 5

    def test_to_xz(self):
        l_values = [0.5, 0.5, 0.5, 0.5, 0.5]
        positions = np.array(self.mapper.to_xz(l_values))
        assert positions.shape == (2, 3)

    def test_to_L(self):
        x = [10, 3, 0]
        z = [3, 1.5, 0]
        l_values = self.mapper.to_L(x, z)
        assert len(l_values) == 5
