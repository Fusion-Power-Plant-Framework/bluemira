# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Base classes for use in magnetostatics.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import numpy as np

from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.coordinates import rotation_matrix
from bluemira.utilities.plot_tools import Plot3D

__all__ = ["CurrentSource", "RectangularCrossSectionCurrentSource", "SourceGroup"]


class CurrentSource(ABC):
    """
    Abstract base class for a current source.
    """

    current: float

    def set_current(self, current):
        """
        Set the current inside each of the circuits.

        Parameters
        ----------
        current: float
            The current of each circuit [A]
        """
        self.current = current

    @abstractmethod
    def field(self, x, y, z):
        """
        Calculate the magnetic field at a set of coordinates.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        field: np.array
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        pass

    @abstractmethod
    def plot(self, ax, **kwargs):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        """
        pass

    @abstractmethod
    def rotate(self, angle, axis):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle: float
            The rotation degree [rad]
        axis: Union[np.array(3), str]
            The axis of rotation
        """
        pass

    def copy(self):
        """
        Get a deepcopy of the CurrentSource.
        """
        return deepcopy(self)


class RectangularCrossSectionCurrentSource(CurrentSource):
    """
    Abstract base class for a current source with a rectangular cross-section.
    """

    origin: np.array
    dcm: np.array
    points: np.array
    breadth: float
    depth: float
    length: float

    def set_current(self, current):
        """
        Set the current inside the source, adjusting current density.

        Parameters
        ----------
        current: float
            The current of the source [A]
        """
        super().set_current(current)
        self.rho = current / (4 * self.breadth * self.depth)

    def rotate(self, angle, axis):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle: float
            The rotation degree [degree]
        axis: Union[np.array(3), str]
            The axis of rotation
        """
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self.origin = self.origin @ r
        self.points = np.array([p @ r for p in self.points], dtype=object)
        self.dcm = self.dcm @ r

    def _local_to_global(self, points):
        """
        Convert local x', y', z' point coordinates to global x, y, z point coordinates.
        """
        return np.array([self.origin + self.dcm.T @ p for p in points])

    def _global_to_local(self, points):
        """
        Convert global x, y, z point coordinates to local x', y', z' point coordinates.
        """
        return np.array([(self.dcm @ (p - self.origin)) for p in points])

    def plot(self, ax=None, show_coord_sys=False):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()
            # If no ax provided, we assume that we want to plot only this source,
            # and thus set aspect ratio equality on this term only
            edge_points = np.concatenate(self.points)

            # Invisible bounding box to set equal aspect ratio plot
            xbox, ybox, zbox = BoundingBox.from_xyz(*edge_points.T).get_box_arrays()
            ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for points in self.points:
            ax.plot(*points.T, color="b", linewidth=1)

        # Plot local coordinate system
        if show_coord_sys:
            ax.scatter(*self.origin, color="k")
            ax.quiver(*self.origin, *self.dcm[0], length=self.breadth, color="r")
            ax.quiver(*self.origin, *self.dcm[1], length=self.length, color="r")
            ax.quiver(*self.origin, *self.dcm[2], length=self.depth, color="r")


class SourceGroup(ABC):
    """
    Abstract base class for multiple current sources.
    """

    sources: List[CurrentSource]
    points: np.array

    def __init__(self, sources):
        self.sources = sources
        self.points = np.vstack([np.vstack(s.points) for s in self.sources])

    def set_current(self, current):
        """
        Set the current inside each of the circuits.

        Parameters
        ----------
        current: float
            The current of each circuit [A]
        """
        for source in self.sources:
            source.set_current(current)

    def field(self, x, y, z):
        """
        Calculate the magnetic field at a point.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        field: np.array
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        return np.sum([source.field(x, y, z) for source in self.sources], axis=0)

    def rotate(self, angle, axis):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle: float
            The rotation degree [rad]
        axis: Union[np.array(3), str]
            The axis of rotation
        """
        for source in self.sources:
            source.rotate(angle, axis)
        self.points = self.points @ rotation_matrix(angle, axis)

    def plot(self, ax=None, show_coord_sys=False):
        """
        Plot the MultiCurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()

        # Invisible bounding box to set equal aspect ratio plot
        xbox, ybox, zbox = BoundingBox.from_xyz(*self.points.T).get_box_arrays()
        ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for source in self.sources:
            source.plot(ax=ax, show_coord_sys=show_coord_sys)

    def copy(self):
        """
        Get a deepcopy of the SourceGroup.
        """
        return deepcopy(self)
