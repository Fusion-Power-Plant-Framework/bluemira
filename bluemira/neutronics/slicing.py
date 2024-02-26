# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tools to optimally slice up the neutronics model"""
from typing import Iterable

import numpy as np

class PoloidalCrossSectionReferenceFrame:
    """
    A common frame of reference, in which two instances of PoloidalCrossSectionLineBase
    Can be compared easily.
    """
    def __init__(self, center_coordinates: Iterable[float]):
        """
        Parameters
        ----------
        center_coordinates: Iterable of shape (2,) or (3,),
            Contains the x and z coordinates of center point used for this frame of
            reference.
        """
        self.x = center_coordinates[0]
        self.z = center_coordinates[-1]

    def __eq__(self, other_frame: PoloidalCrossSectionReferenceFrame):
        return self.x == other_frame.x and self.z == other_frame.z

    def __ne__(self, other_frame: PoloidalCrossSectionReferenceFrame):
        return self.x != other_frame.x or self.z != other_frame.z

    def from_2points(self, p1, p2) -> PoloidalCrossSectionLineFixed:
        """
        Represent a line fixed by two points.

        Parameters
        ----------
        p1: Iterable (2,) or (3,)
            Contains the x and z coordinates of point 1
        p2: Iterable (2,) or (3,)
            Contains the x and z coordinates of point 2
        """
        if p2[0] == p[0]:
            angle = np.pi / 2  # pointing straight up.
        else:
            angle = np.arctan((p2[-1] - p1[-1]) / (p2[0] - p1[0]))
        return self.from_1point_1angle(p1, angle)

    def from_1point_1angle(self, p1, angle) -> PoloidalCrossSectionLineFixed:
        """
        Represent a line fixed by one point and an angle.

        Parameters
        ----------
        p1: Iterable (2,) or (3,)
            Contains the x and z coordinates of point 1, which the line must pass through
        angle: float
            angle of the line in radians [-pi/2, pi/2]
        """
        center_offset = np.dot([-np.sin(angle), np.cos(angle)], self.center)
        return PoloidalCrossSectionLineFixed(self, angle, center_offset)

    def from_point_variable_angle(self, p1: Iterable[float], angle_range: Iterable[float]) -> PoloidalCrossSectionLineSemiVariable:
        """
        Parameters
        ----------
        p1: Iterable (2,) or (3,)
            Contains the x and z coordinates of point 1, which the line must pass through
        angle_range: Iterable[float]
            lower, upper limit
        Returns
        -------
        """
        angle_range = min(angle_range), max(angle_range)
        return PoloidalCrossSectionLineSemiVariable(self, p1, angle_range)

    def from_variable_point_variable_angle(self, extremum_p1: Iterable[float], extremum_p2: Iterable[float], angle_range: Iterable[float]):
        """
        Parameters
        ----------
        extremum_p1, extremum_p2: Iterable (2,) or (3,)
            Each contains the x or z coordinates of a point on the edge of the region we're allowed into.
        angle_range
        """
        angle_range = min(angle_range), max(angle_range)
        extrema_points

class PoloidalCrossSectionLineBase:
    def __init__(self, reference_frame: PoloidalCrossSectionReferenceFrame,
                center_offset, angle):
        """
        Parameters
        ----------
        angle: float
            [-pi/2, pi/2]
        """
        self.reference_center = reference_frame
        self.angle = angle

    def common(self, other_line):
        """Common values that exists within the allowed parameters """
        if not isinstance(other, PoloidalCrossSectionLineBase):
            raise NotImplementedError("Produced to only ")
        if self.reference_center != other_line.reference_center:
            raise VariableSliceLineError()
        if isinstance(self, PoloidalCrossSectionLineVariable) and isinstance(other_line, PoloidalCrossSectionLineVariable):
            raise NotImplementedError("Too complex to determine the viable area.")

class PoloidalCrossSectionLineFixed(PoloidalCrossSectionLineBase):
    def __init__(self, reference_frame: PoloidalCrossSectionReferenceFrame,
                center_offset: Iterable[float], angle: float):
        super().__init__(reference_frame, angle, offset)

class PoloidalCrossSectionLineSemiVariable(PoloidalCrossSectionLineBase):
    def __init__(self, reference_frame: PoloidalCrossSectionReferenceFrame,
                center_offset: Iterable[float], angle: Iterable[float]):
    """
    A straight line with variable angle, but still passes through a fixed point.

    Parameters
    ----------

    """
    pass

class PoloidalCrossSectionLineVariable(PoloidalCrossSectionLineBase):
    def __init__(self, reference_frame: PoloidalCrossSectionReferenceFrame,
                center_offset_points: Iterable[Iterable[float]], angle: Iterable[float]):
    """
    A straight line that has a variable angle and straight line

    Parameters
    ----------
    angle: lower limit and upper limit of the angle
    center_offset: lower limit and upper limit of the center_offset
    """
