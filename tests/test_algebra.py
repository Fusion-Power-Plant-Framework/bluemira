#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for mirapy.algebra

@author: ivan
"""
import pytest

import freecad  # noqa: F401 - required to import FreeCAD
from FreeCAD import Base  # type: ignore

from mirapy.algebra import ellipseFit, circle2P, linesIntersection
import math


class TestAlgebraMethods:
    @pytest.mark.parametrize(
        "constraints, expected",
        (
            [[([1, 0, 0],), ([0, 2, 0],), ([-1, 0, 0],), ([0, -2, 0],)], (0, 0, 1, 2)],
            [[([0, 2, 0], math.pi / 2.0), ([1, 0, 0], 0)], (1, 2, 1, 2)],
            [[([1, 0, 0], math.pi / 2.0), ([0, 1, 0], 0)], (0, 0, 1, 1)],
            [
                [(Base.Vector(1, 0, 0), math.pi / 2.0), (Base.Vector(0, 2, 0), 0)],
                (0, 0, 1, 2),
            ],
        ),
    )
    def test_ellipse_fit(self, constraints, expected):
        """
        Test that the ellipse fit generates the expected output for the following cases:

        - A set of four points
        - A set of two points with a pi / 2 rotation on one of the points
        - a circular set of two points with a pi / 2 rotation on one of the points
        - a set of two points using FreeCAD vectors, with a pi / 2 rotation on one of the
          points
        """
        params = ellipseFit(constraints)

        params == pytest.approx(expected)

    @pytest.mark.parametrize(
        "point1, point2, centre_guess, radius, expected",
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], 1, [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0.5, 0], 1, [0, 0, 1]],
        ),
    )
    def test_circle2P(self, point1, point2, centre_guess, radius, expected):
        """
        Test that the two-point circle fit generates the expected output for the
        following cases:

        - Two points with an initial centre guess at the origin
        - Two points with an initial centre guess at a shift of 0.5 from the origin
        """
        params = circle2P(point1, point2, centre_guess, radius)

        params == pytest.approx(expected)

    @pytest.mark.parametrize(
        "point1, angle1, point2, angle2, expected",
        (
            [[1, 0, 0], math.pi / 2.0, [0, 1, 0], 0, [1, 1]],
            [[1, 0, 0], math.pi, [0, 1, 0], 0, None],
        ),
    )
    def test_linesIntersection_simple(self, point1, angle1, point2, angle2, expected):
        """
        Test that the lines intersection generates the expected output for the
        following cases:

        - Lines intersect
        - Lines do not intersect
        """
        intersection = linesIntersection(point1, angle1, point2, angle2)

        intersection == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__])
