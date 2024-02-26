# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools to optimally slice up the neutronics model"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from bluemira.base.constants import EPS


class PoloidalCrossSectionLineBase:
    """An abstract base class for all of the splitting lines used to cut open the
    poloidal cross-section. Includes both variables ones and fixed ones.
    """

    def __init__(
        self,
        z_intercept: Union[float, NDArray[float]],
        slope: Union[float, NDArray[float]],
    ):
        """
        Parameters
        ----------
        z_intercept:
            z_intercept of the line.
        slope:
            Slope of the line.
        """
        self.z_intercept = (
            z_intercept if np.ndim(z_intercept) == 0 else np.array(z_intercept)
        )
        self.slope = slope if np.ndim(slope) == 0 else np.array(slope)

    def shared_viable_config(self, other_line) -> PoloidalCrossSectionLineBase:  # noqa F821
        """Common values that exists within the allowed parameters"""
        if not isinstance(other_line, PoloidalCrossSectionLineBase):
            raise NotImplementedError(
                "Can only find the intersection between neighbouring"
            )
        ...  # write code to account for the slope +/- symmetry
        return PoloidalCrossSectionLineBase()

    def __mul__(self, other_line):
        return self.common(other_line)

    @property
    def phase_space_repr(self) -> AllowedPointSet:  # noqa: F821
        """A property for all PoloidalCrossSectionLineBase class."""
        return self._phase_space_repr


class PoloidalCrossSectionLineFixed(PoloidalCrossSectionLineBase):
    """A slicing line with fixed angle and position."""

    def __init__(self, z_intercept: float, slope: float):
        if np.ndim(z_intercept) != 0 or np.ndim(slope) != 0:
            raise ValueError("Only accepts scalar inputs!")
        super().__init__(z_intercept, slope)
        self._phase_space_repr = [
            AllowedPointSet0D(z_intercept, slope),
            AllowedPointSet0D(z_intercept, -slope),
        ]

    @classmethod
    def from_2points(cls, p1: NDArray[float], p2: NDArray[float]):
        """
        Represent a line fixed by two points.

        Parameters
        ----------
        p1: NDArray of shape (2,) or (3,)
            Contains the x and z coordinates of point 1
        p2: NDArray of shape (2,) or (3,)
            Contains the x and z coordinates of point 2
        """
        slope = (p2[-1] - p1[-1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else np.inf
        return cls.from_point_slope(p1, slope)

    @classmethod
    def from_point_slope(cls, p1: NDArray[float], slope: NDArray[float]):
        """
        Represent a line fixed by one point and an angle.

        Parameters
        ----------
        p1: NDArray of shape (2,) or (3,)
            Contains the x and z coordinates of point 1, which the line must pass through
        slope: float
            slope of the line
        """
        z_intercept = p1[-1] - slope * p1[0]
        return cls(z_intercept, slope)


class PoloidalCrossSectionLineSemiVariable(PoloidalCrossSectionLineBase):
    """A line with variable angle but fixed to one position."""

    def __init__(self, z_intercept: NDArray[float], slope: NDArray[float]):
        """
        A straight line with variable angle, but still passes through a fixed point.

        Parameters
        ----------
        z_intercept: shape (2,)
        slope: shape (2,)
        """
        if not (np.shape(z_intercept) == (2,) and np.shape(slope) == (2,)):
            raise ValueError("Slope, intercept must be provided in a (2,) array each.")
        super().__init__(np.array(z_intercept), np.array(slope))
        self._phase_space_repr = [
            self.AllowedPointSet1D(z_intercept, slope),
            self.AllowedPointSet1D(z_intercept, -slope),
        ]

    @classmethod
    def from_point_variable_slopes(cls, p1: NDArray[float], slope_range: NDArray[float]):
        """
        Parameters
        ----------
        p1: NDArray of shape (2,) or (3,)
            Contains the x and z coordinates of point 1, which the line must pass through
        slope_range: NDArray of shape (2,)
            lower limit, upper limit

        Returns
        -------
        An instance of PoloidalCrossSectionLineSemiVariable
        """
        slope_range = min(slope_range), max(slope_range)
        z_at_min_slope = PoloidalCrossSectionLineFixed.from_point_slope(
            p1, slope_range[0]
        ).z_intercept
        z_at_max_slope = PoloidalCrossSectionLineFixed.from_point_slope(
            p1, slope_range[1]
        ).z_intercept
        return cls((z_at_min_slope, z_at_max_slope), slope_range)


class PoloidalCrossSectionLineVariable(PoloidalCrossSectionLineBase):
    """A line with variable angle and variable position."""

    def __init__(self, z_intercept: NDArray[float], slope: NDArray[float]):
        """
        A straight line that has a variable angle and straight line

        Parameters
        ----------
        z_intercept: shape (4,)
        slope: shape (4,)
        """
        super().__init__(np.array(z_intercept), np.array(slope))
        # post_init check
        if not (
            (np.ndim(self.z_intercept) == (1,) and np.ndim(self.slope) == (1,))
            and (np.shape(self.z_intercept)[0] == np.shape(self.slope)[0] >= 2)
        ):
            raise ValueError(
                "Slope and intercept must be provided in (N,) array each" "where N>=3."
            )
        ...
        # check that it does, in fact, form a quadrilateral with no self-intersection.
        self._phase_space_repr = [
            self.AllowedPointSet2D(z_intercept, slope),
            self.AllowedPointSet2D(z_intercept, -slope),
        ]

    @classmethod
    def from_variable_intercepts_variable_slopes(
        cls, z_intercept_range: NDArray[float], slope_range: NDArray[float]
    ):
        """
        Make a rectangular box in phase space.

        Parameters
        ----------
        z_intercept_range: NDArray (2,)
            minimum z_intercept, maximum z_intercept
        slope_range: NDArray (2,)
            minimum slope, maximum slope
        """
        z_min, z_max = min(z_intercept_range), max(z_intercept_range)
        s_min, s_max = min(slope_range), max(slope_range)
        return cls([z_min, z_min, z_max, z_max], [s_max, s_min, s_min, s_max])

    @classmethod
    def from_variable_points_variable_slopes(
        cls,
        extremum_p1: NDArray[float],
        extremum_p2: NDArray[float],
        slope_range: NDArray[float],
    ):
        """
        Parameters
        ----------
        extremum_p1, extremum_p2: NDArray of shape (2,) or (3,)
            Each contains the x or z coordinates of a point on the edge of the region
            we're allowed into.
        slope_range:
            Both lines shares this same range of slopes.
        """
        line_1 = PoloidalCrossSectionLineSemiVariable.from_point_variable_slopes(
            extremum_p1, slope_range
        )
        line_2 = PoloidalCrossSectionLineSemiVariable.from_point_variable_slopes(
            extremum_p2, slope_range
        )
        z_ = np.flatten([line_1.z_intercept[::-1], line_2.z_intercept])
        s_ = np.flatten([line_1.slope[::-1], line2_.slope])
        return cls(z_, s_)

    @classmethod
    def from_2variable_slope_lines(
        cls,
        line_1: PoloidalCrossSectionLineSemiVariable,
        line_2: PoloidalCrossSectionLineSemiVariable,
    ):
        """
        Make a general quadrilateral in phase space using two
        PoloidalCrossSectionLineSemiVariable
        """
        z_ = np.flatten([line_1.z_intercept[::-1], line_2.z_intercept])
        s_ = np.flatten([line_1.slope[::-1], line_2.slope])
        ...  # Convex hull here
        return cls(z_, s_)


class AllowedPointSet:
    """Base class, representing nothing."""

    def __init__(self, y: NDArray, x: NDArray):
        """Something"""
        self.y = np.array(y)
        self.x = np.array(x)

    def intersection(self, other_set):
        """Strategy used in the code below:
        We will only populate the lower triangle of the interaction matrix between
        various classes of AllowedPointSet. The upper half triangle (minus the main
        diagonal) will simply be a reflection of the lower half.
        """
        pass

    def __bool__(self):
        return True


class AllowedPointSetNull(AllowedPointSet):
    def __init__(self):
        self.y = np.array([])
        self.x = np.array([])

    def intersection(self, other_set):  # noqa: ARG002
        return self

    def __bool__(self):
        return False


class AllowedPointSet0D(AllowedPointSet):
    """A single point"""

    def __init__(self, y, x):
        if np.ndim(y) > 1 or np.ndim(x) > 1:
            raise ValueError("A 0D point requires at least 1 point to specify.")
        super().__init__(np.atleast_1d(y), np.atleast_1d(x))

    def intersection(self, other_set):
        if int(other_set.__class__.__name__[-2]) > 0:
            return other_set.intersection(self)
        # only accept if there is a perfect overlap.
        if np.allclose(
            np.array([self.z_intercept, self.slope]),
            np.array([other_set.z_intercept, other_set.slope]),
            rtol=0,
            atol=EPS,
        ):
            return self

        return AllowedPointSetNull()


class AllowedPointSet1D(AllowedPointSet):
    """A line of finite length, defined by the y and x coordinates of the start and end
    locations.
    """

    def __init__(self, y, x):
        if not (np.shape(y) == (2,) and np.shape(x) == (2,)):
            raise ValueError("A 1D line requires at least 2 points to specify.")
        super().__init__(y, x)

    def intersection(self, other_set):
        if int(other_set.__class__.__name__[-2]) == 2:
            return other_set.intersection(self)
        ...
        return AllowedPointSetNull()


class AllowedPointSet2D(AllowedPointSet):
    """A 2D polygon, defined by the y and x coordinates of the start and end locations"""

    def __init__(self, y, x):
        if not all([
            np.ndim(y) == 1,
            np.ndim(x) == 1,
            np.shape(y)[0] >= 3,
            np.shape(x)[0] >= 3,
        ]):
            raise ValueError("A 2D polgyon requires at least 3 points to specify.")
        super().__init__(y, x)
        coords = np.array([self.y, self.x]).T
        self.edges = [
            AllowedPointSet1D(np.array([yx, next_yx]).T)
            for yx, next_yx in zip(coords, np.roll(coords, -1))
        ]

    def intersection(self, other_set):
        ...
        return AllowedPointSetNull()
