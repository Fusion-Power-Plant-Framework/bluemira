# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Utility for sets of coordinates
"""

from typing import Iterable
import numpy as np
from pyquaternion import Quaternion

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.constants import EPS
from bluemira.geometry.error import CoordinatesError
from bluemira.geometry._deprecated_tools import (
    get_perimeter_3d,
    get_area_3d,
    get_centroid_3d,
    check_ccw_3d,
)


def principal_components(xyz_array):
    """
    Principal component analysis.
    """
    mean = np.mean(xyz_array, axis=1)
    xyz_shift = xyz_array - mean.reshape((3, 1))

    cov = np.cov(xyz_shift)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def _parse_to_xyz_array(xyz_array):
    """
    Make a 3, N xyz array out of just about anything.
    """
    if isinstance(xyz_array, np.ndarray):
        xyz_array = _parse_array(xyz_array)
    elif isinstance(xyz_array, dict):
        xyz_array = _parse_dict(xyz_array)
    elif isinstance(xyz_array, Iterable):
        xyz_array = _parse_iterable(xyz_array)
    else:
        raise CoordinatesError(f"Cannot instantiate Coordinates with: {type(xyz_array)}")
    return xyz_array


def _parse_array(xyz_array):
    try:
        xyz_array = np.array(np.atleast_2d(xyz_array), dtype=np.float64)
    except ValueError:
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        )

    shape = xyz_array.shape
    if len(shape) > 2:
        raise NotImplementedError

    n, m = shape
    if n == 3:
        if m == 3:
            bluemira_warn(
                "You are creating Coordinates with a (3, 3) array, defaulting to (3, N)."
            )

    elif m == 3:
        xyz_array = xyz_array.T

    else:
        raise CoordinatesError(
            "Cannot instantiate Coordinates where either n or m != 3."
        )

    if not np.allclose([len(xyz_array[i]) for i in range(3)], len(xyz_array[0])):
        raise CoordinatesError(
            "Cannot instantiate Coordinates with a ragged (3, N | M) array."
        )

    return xyz_array


def _parse_dict(xyz_dict):
    x = np.atleast_1d(xyz_dict.get("x", 0))
    y = np.atleast_1d(xyz_dict.get("y", 0))
    z = np.atleast_1d(xyz_dict.get("z", 0))

    shape_lengths = np.array([len(c.shape) for c in [x, y, z]])

    if np.any(shape_lengths > 1):
        raise CoordinatesError(
            "Cannot instantiate Coordinates from dict with coordinate vectors that are not 1-D."
        )

    lengths = [len(c) for c in [x, y, z]]

    usable_lengths = []
    for length in lengths:
        if length != 1:
            usable_lengths.append(length)

    if not np.allclose(usable_lengths, usable_lengths[0]):
        raise CoordinatesError(
            "Cannot instantiate Coordinate from dict with a ragged set of vectors."
        )

    # Backfill single-value coordinates
    actual_length = usable_lengths[0]
    if len(x) == 1:
        x = x[0] * np.ones(actual_length)
    if len(y) == 1:
        y = y[0] * np.ones(actual_length)
    if len(z) == 1:
        z = z[0] * np.ones(actual_length)

    return np.array([x, y, z])


def _parse_iterable(xyz_iterable):
    # We temporarily set the dtype to object to avoid a VisibleDeprecationWarning
    xyz_array = np.array(xyz_iterable, dtype=object)
    return _parse_array(xyz_array)


class Coordinates:
    """
    Coordinates object for storing ordered sets of coordinates.

    The following are enforced by default:
    * Shape of (3, N)
    * Counter-clockwise direction [TBD]

    Parameters
    ----------
    xyz_array: Union[np.ndarray, dict, Iterable[Iterable]]

    Notes
    -----
    This is a utility class for dealing with sets of coordinates in a number of different
    contexts. It should not be used for the creation of CAD geometries.
    """

    __slots__ = ("_array", "_is_planar", "_normal_vector")
    # =============================================================================
    # Instantiation
    # =============================================================================

    def __init__(self, xyz_array, enforce_ccw=False):
        self._array = _parse_to_xyz_array(xyz_array)
        self._set_plane_props()

        if enforce_ccw:
            self.set_ccw()

    # =============================================================================
    # Checks
    # =============================================================================

    def _set_plane_props(self):
        """
        Set the planar properties of the Coordinates.
        """
        eigenvalues, eigenvectors = principal_components(self._array)

        if np.isclose(eigenvalues[-1], 0.0):
            self._is_planar = True
        else:
            self._is_planar = False

        self._normal_vector = eigenvectors[:, -1]

    @property
    def is_planar(self):
        """
        Whether or not the Coordinates are planar.
        """
        return self._is_planar

    @property
    def normal_vector(self):
        """
        The normal vector of the best-fit plane of the Coordinates.
        """
        return self._normal_vector

    def check_ccw(self, axis=None) -> bool:
        """
        Whether or not the Coordinates are ordered in the counter-clockwise direction
        about a specified axis. If None is specified, the Coordinates normal vector will
        be used.
        """
        if axis is None:
            axis = self.normal_vector
        else:
            axis = np.array(axis, dtype=float)
            if not axis.size == 3:
                raise CoordinatesError("Base vector must be of size 3.")
            axis /= np.linalg.norm(axis)

        return check_ccw_3d(*self._array, axis)

    def set_ccw(self, axis=None):
        """
        Set the Coordinates to be counter-clockwise about a specified axis. If None is
        specified, the Coordinates normal vector will be used.
        """
        if not check_ccw_3d(axis=axis):
            self.reverse()

    # =============================================================================
    # Property access
    # =============================================================================

    @property
    def x(self):
        """
        The x coordinate vector
        """
        return self._array[0]

    @property
    def y(self):
        """
        The y coordinate vector
        """
        return self._array[1]

    @property
    def z(self):
        """
        The z coordinate vector
        """
        return self._array[2]

    @property
    def xy(self):
        """
        The x-y coordinate array
        """
        return self._array[[0, 1], :]

    @property
    def xz(self):
        """
        The x-z coordinate array
        """
        return self._array[[0, 2], :]

    @property
    def yz(self):
        """
        The y-z coordinate array
        """
        return self._array[[1, 2], :]

    @property
    def points(self):
        """
        A list of the individual points of the Coordinates.
        """
        return list(self.T)

    # =========================================================================
    # Conversions
    # =========================================================================

    def as_dict(self):
        """
        Cast the Coordinates as a dictionary.

        Returns
        -------
        d: dict
            Dictionary with {'x': [], 'y': [], 'z':[]}
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    # =============================================================================
    # Useful properties
    # =============================================================================

    @property
    def closed(self) -> bool:
        """
        Whether or not this is a closed set of Coordinates.
        """
        if len(self) > 2:
            if np.allclose(self[:, 0], self[:, -1], rtol=0, atol=EPS):
                return True
        return False

    @property
    def length(self) -> float:
        """
        Perimeter length of the coordinates.
        """
        return get_perimeter_3d(*self._array)

    @property
    def area(self) -> float:
        """
        Enclosed area of the Coordinates. 0 if the Coordinates are open.
        """
        if not self.closed:
            return 0.0

        if not self.is_planar:
            bluemira_warn(
                "Cannot get the area of a non-planar set of Coordinates. Returning 0.0"
            )
            return 0.0

        return get_area_3d(*self._array)

    @property
    def center_of_mass(self) -> tuple:
        """
        Geometrical centroid of the Coordinates.
        """
        # [sic] coordinates do not have a "mass", but named such for consistency with
        # other geometry objects.
        return tuple(get_centroid_3d(*self._array))

    # =============================================================================
    # Array-like behaviour
    # =============================================================================

    @property
    def T(self):  # noqa(N802)
        """
        Transpose of the Coordinates
        """
        return self._array.T

    @property
    def shape(self):
        """
        Shape of the Coordinates
        """
        return self._array.shape

    # =============================================================================
    # Modification
    # =============================================================================
    def reverse(self):
        """
        Reverse the direction of the Coordinates.
        """
        self._array = self._array[:, ::-1]

    def open(self):
        """
        Open the Coordinates (if they are closed)
        """
        if self.closed:
            self._array = self._array[:, :-1]

    def close(self):
        """
        Close the Coordinates (if they are open)
        """
        if not self.closed:
            self._array = np.vstack((self._array.T, self._array[:, 0])).T

    def rotate(
        self, base: tuple = (0, 0, 0), direction: tuple = (0, 0, 1), degree: float = 0.0
    ):
        """
        Rotate the Coordinates.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle [degrees]
        """
        if degree == 0.0:
            return

        base = np.array(base, dtype=float)
        if not base.size == 3:
            raise CoordinatesError("Base vector must be of size 3.")

        direction = np.array(direction, dtype=float)
        if not direction.size == 3:
            raise CoordinatesError("Direction vector must be of size 3.")
        direction /= np.linalg.norm(direction)  # normalise rotation axis

        points = self._array - base.reshape(3, 1)
        quart = Quaternion(axis=direction, angle=np.deg2rad(degree))
        r_matrix = quart.rotation_matrix
        new_array = points.T @ r_matrix.T
        self._array = new_array.T

        self._set_plane_props()

    def translate(self, vector: tuple = (0, 0, 0)):
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        vector = np.array(vector)
        if not vector.size == 3:
            raise CoordinatesError("Translation vector must be of size 3.")

        self._array += vector.T
        self._set_plane_props()

    # =============================================================================
    # Dunders (with different behaviour to array)
    # =============================================================================

    def __eq__(self, other):
        """
        Check the Coordinates for equality with other Coordinates.

        Parameters
        ----------
        other: Coordinates
            The other Coordinates to compare against

        Returns
        -------
        equal: bool
            Whether or not the Coordinates are identical

        Notes
        -----
        Coordinates with identical coordinates but different orderings will not be
        counted as identical.
        """
        if isinstance(other, self.__class__):
            return np.all(np.allclose(self._array, other._array, rtol=0, atol=EPS))
        return False

    def __len__(self):
        """
        The number of points in the Coordinates.
        """
        return self.shape[1]

    # =============================================================================
    # Array-like dunders
    # =============================================================================

    def __repr__(self):
        """
        Representation of the Coordinates.
        """
        r = self._array.__repr__()
        return f"{self.__class__.__name__}{r[5:]}"

    def __getitem__(self, *args, **kwargs):
        """
        Array-like indexing and slicing.
        """
        return self._array.__getitem__(*args, **kwargs)

    def __iter__(self):
        """
        Array-like unpacking.
        """
        return self._array.__iter__()
