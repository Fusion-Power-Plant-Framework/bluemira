# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of tools used for position interpolation.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from bluemira.geometry.constants import D_TOLERANCE, VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.error import PositionerError
from bluemira.utilities.tools import is_num

if TYPE_CHECKING:
    import numpy.typing as npt


class XZGeometryInterpolator(abc.ABC):
    """
    Abstract base class for 2-D x-z geometry interpolation to normalised [0, 1] space.

    By convention, normalised x-z space is oriented counter-clockwise w.r.t. [0, 1, 0].

    Parameters
    ----------
    geometry:
        Geometry to interpolate with
    """

    def __init__(self, geometry: BluemiraWire | BluemiraFace):
        self.geometry = geometry

    def _get_xz_coordinates(self, num_pts):
        """
        Get discretised x-z coordinates of the geometry.

        Returns
        -------
        :
            The xz coordinates
        """
        coordinates = self.geometry.discretise(
            byedges=True, dl=self.geometry.length / num_pts
        )
        coordinates.set_ccw([0, 1, 0])
        return coordinates.xz

    @abc.abstractmethod
    def to_xz(
        self, l_values: npt.ArrayLike
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Convert parametric-space 'L' values to physical x-z space.
        """
        ...

    @abc.abstractmethod
    def to_L(self, x: npt.ArrayLike, z: npt.ArrayLike) -> float | np.ndarray:
        """
        Convert physical x-z space values to parametric-space 'L' values.
        """
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """
        The dimension of the parametric space
        """
        ...


class PathInterpolator(XZGeometryInterpolator):
    """
    Sets up an x-z path for a point to move along.

    The path is treated as flat in the x-z plane.
    """

    def __init__(self, geometry: BluemiraWire):
        super().__init__(geometry)

    def to_xz(
        self, l_values: npt.ArrayLike
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Convert parametric-space 'L' values to physical x-z space.

        Returns
        -------
        :
            The xz coordinates
        """
        l_values = np.clip(l_values, 0.0, 1.0)
        if is_num(l_values):
            return self.geometry.value_at(alpha=l_values)[[0, 2]]

        x, z = np.zeros(len(l_values)), np.zeros(len(l_values))
        for i, lv in enumerate(l_values):
            x[i], z[i] = self.geometry.value_at(alpha=lv)[[0, 2]]

        return x, z

    def to_L(self, x: npt.ArrayLike, z: npt.ArrayLike) -> float | np.ndarray:
        """
        Convert physical x-z space values to parametric-space 'L' values.

        Returns
        -------
        :
            The normalised coordinates
        """
        if is_num(x):
            return self.geometry.parameter_at([x, 0, z], tolerance=VERY_BIG)

        l_values = np.zeros(len(x))
        for i, (xi, zi) in enumerate(zip(x, z, strict=False)):
            l_values[i] = self.geometry.parameter_at([xi, 0, zi], tolerance=VERY_BIG)
        return l_values

    @property
    def dimension(self) -> int:
        """
        Dimension of the parametric space of the PathInterpolator
        """
        return 1


class RegionInterpolator(XZGeometryInterpolator):
    """
    Map ``[0, 1]^2`` to a closed, possibly non-convex x-z region.

    For boundary wires, horizontal boundary intersections are sorted and paired
    according to the even-odd fill rule:

        [x_0, x_1], [x_2, x_3], ...

    For faces, ``slice_shape`` returns the portions of the slicing line lying
    inside the face. Those sliced edges are converted directly into intervals.

    The first normalised coordinate, ``l_0``, measures cumulative distance
    across the union of the filled intervals at a given z coordinate. Exterior
    gaps and holes are excluded from this distance.

    Parameters
    ----------
    geometry:
        A closed boundary wire or a face. A face may contain inner wires
        representing holes.

    Notes
    -----
    A continuous bijection cannot exist between a square and a region
    containing a hole. Consequently, a square-based parameterisation of a
    multiply-connected region necessarily contains seams.

    The seam convention used here is right-continuous. An exact seam value maps
    to the left endpoint of the interval to its right, except for ``l_0 == 1``.

    Points outside the region, including points inside holes, are projected
    horizontally onto the closest valid interval at the clamped z coordinate.
    """

    _MAX_SLICE_RETRIES = 8

    def __init__(self, geometry: BluemiraWire | BluemiraFace):
        super().__init__(geometry)

        if (isinstance(geometry, BluemiraWire) and not geometry.is_closed()) or (
            isinstance(geometry, BluemiraFace)
            and not all(g.is_closed() for g in geometry.boundary)
        ):
            raise PositionerError(
                "RegionInterpolator can only handle faces or closed wires."
            )

        self.z_min = geometry.bounding_box.z_min
        self.z_max = geometry.bounding_box.z_max
        self._z_span = self.z_max - self.z_min

        if self._z_span <= D_TOLERANCE:
            raise PositionerError(
                "RegionInterpolator requires a region with non-zero z extent."
            )

    @staticmethod
    def _coordinate_tolerance(values: np.ndarray) -> float:
        """
        Calculate a scale-dependent coordinate tolerance.

        Parameters
        ----------
        values:
            Coordinates from which the geometric scale is determined.

        Returns
        -------
        :
            Scale-dependent absolute tolerance.
        """
        values = np.asarray(values, dtype=float)

        finite_values = values[np.isfinite(values)]

        scale = (
            1.0
            if finite_values.size == 0
            else max(1.0, float(np.max(np.abs(finite_values))))
        )

        return max(D_TOLERANCE, 32.0 * np.finfo(float).eps * scale)

    @staticmethod
    def _intersection_x_values(intersections) -> np.ndarray:
        """
        Extract sorted, finite and tolerance-deduplicated x coordinates.

        This method is intended for boundary intersections, such as the points
        returned when slicing a wire.

        CAD kernels can report the same intersection more than once when a
        slicing plane passes through a shared vertex. Numerically identical
        intersections are therefore combined before applying the even-odd
        filling rule.

        Parameters
        ----------
        intersections:
            Boundary intersection points returned by ``slice_shape``.

        Returns
        -------
        :
            Sorted array of distinct x coordinates.
        """
        if intersections is None:
            return np.empty(0, dtype=float)

        values = [intersection[0] for intersection in intersections]
        if not values:
            return np.empty(0, dtype=float)

        values_array = np.sort(np.asarray(values, dtype=float))
        tolerance = RegionInterpolator._coordinate_tolerance(values_array)

        unique_values = [values_array[0]]

        for value in values_array[1:]:
            if value - unique_values[-1] > tolerance:
                unique_values.append(value)
            else:
                # Use the mean of numerically duplicated values instead of
                # arbitrarily retaining one of them.
                unique_values[-1] = 0.5 * (unique_values[-1] + value)

        return np.asarray(unique_values, dtype=float)

    @staticmethod
    def _merge_intervals(intervals: np.ndarray) -> np.ndarray:
        """
        Merge intervals that overlap or touch within geometric tolerance.

        This removes small CAD-kernel artefacts without joining intervals
        separated by a genuine hole.

        Parameters
        ----------
        intervals:
            Array with shape ``(n, 2)`` containing ``[left, right]`` pairs.

        Returns
        -------
        :
            Sorted, non-overlapping intervals.

        Raises
        ------
        PositionerError
            Interval problems
        """
        intervals = np.atleast_2d(np.asarray(intervals, dtype=float).squeeze())

        if intervals.size == 0:
            return np.empty((0, 2), dtype=float)

        if intervals.shape[1] != 2:  # noqa: PLR2004
            raise PositionerError("Horizontal intervals must have shape (n, 2).")

        # Do not depend on the orientation of the sliced edge.
        intervals = np.sort(intervals, axis=1)

        # Sort primarily by the left endpoint and secondarily by the right.
        sort_order = np.lexsort((intervals[:, 1], intervals[:, 0]))
        intervals = intervals[sort_order]

        tolerance = RegionInterpolator._coordinate_tolerance(intervals)

        merged = [intervals[0, (0, 1)]]

        for left, right in intervals[1:]:
            previous = merged[-1]

            if left <= previous[1] + tolerance:
                previous[1] = max(previous[1], right)
            else:
                merged.append([left, right])

        return np.asarray(merged, dtype=float)

    def _face_intervals_at(self, intersections: list[BluemiraWire] | None) -> np.ndarray:
        """
        Slice a face and return its filled horizontal intervals.

        Slicing a face returns edges representing portions of the slicing line
        that lie inside the face. Each edge is therefore converted directly
        into one filled interval.

        In particular, if a face contains a hole, a horizontal slice through
        the hole should produce two or more separate edges. These edges are
        retained as separate intervals.

        Parameters
        ----------
        intersections:
            intesections with a plane at z

        Returns
        -------
        :
            Sorted and merged filled intervals.
        """
        if intersections is None:
            return np.empty((0, 2), dtype=float)

        intervals: list[tuple[float, float]] = []

        for sliced_edge in intersections:
            try:
                start_x = sliced_edge.start_point().x
                end_x = sliced_edge.end_point().x
            except (AttributeError, TypeError):
                continue

            left, right = sorted((start_x, end_x))
            intervals.append((left, right))

        if not intervals:
            return np.empty((0, 2), dtype=float)

        return self._merge_intervals(np.asarray(intervals, dtype=float))

    def _wire_intersections_at(self, intersections: np.ndarray) -> np.ndarray:
        """
        Slice a boundary wire and return its distinct x intersections.

        Parameters
        ----------
        intersections:
            intesections with a plane at z

        Returns
        -------
        :
            Sorted array of distinct x coordinates.
        """
        x_values = self._intersection_x_values(intersections)

        if len(x_values) == 0:
            return np.empty((0, 2), dtype=float)

        if len(x_values) == 1:
            return np.array([[x_values[0], x_values[0]]], dtype=float)

        if len(x_values) % 2 == 0:
            return self._merge_intervals(x_values.reshape(-1, 2))

        # An odd number greater than one is usually a tangent or vertex
        # degeneracy. The caller will try a nearby interior slice.
        return np.empty((0, 2), dtype=float)

    def _try_intervals_at(self, z: float) -> np.ndarray:
        """
        Attempt to construct filled intervals without perturbing z.

        Parameters
        ----------
        z:
            Physical z coordinate.

        Returns
        -------
        :
            Valid intervals, or an empty array when the slice is degenerate or
            cannot be consistently interpreted.
        """
        intersections = slice_shape(
            self.geometry,
            BluemiraPlane.from_3_points([0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]),
        )

        if isinstance(self.geometry, BluemiraFace):
            return self._face_intervals_at(intersections)

        return self._wire_intersections_at(intersections)

    def _intervals_at(self, z: float) -> np.ndarray:
        """
        Filled horizontal intervals at a given z coordinate.

        If the exact slice is degenerate, nearby slices towards the interior of
        the bounding box are attempted. This handles horizontal planes passing
        exactly through vertices or tangent points.

        Parameters
        ----------
        z:
            Physical z coordinate.

        Returns
        -------
        :
            Sorted, non-overlapping filled intervals.

        Raises
        ------
        PositionerError
            If no valid interval decomposition can be constructed.
        """
        z = float(np.clip(z, self.z_min, self.z_max))

        intervals = self._try_intervals_at(z)

        if len(intervals) > 0:
            return intervals

        perturbation = max(self._z_span * 1e-10, float(D_TOLERANCE) * 4.0)

        midpoint = 0.5 * (self.z_min + self.z_max)

        # Move towards the interior of the bounding box.
        preferred_direction = 1.0 if z <= midpoint else -1.0

        for attempt in range(self._MAX_SLICE_RETRIES):
            offset = perturbation * (2**attempt)

            # Try the preferred inward direction first, then the opposite
            # direction. The second attempt is useful for internal tangent
            # points and non-convex regions.
            for direction in (preferred_direction, -preferred_direction):
                trial_z = float(np.clip(z + direction * offset, self.z_min, self.z_max))

                if trial_z == z:
                    continue

                intervals = self._try_intervals_at(trial_z)

                if len(intervals) > 0:
                    return intervals

        if isinstance(self.geometry, BluemiraFace):
            geometry_type = "face"
            intersection_description = "filled slice segments"
        else:
            geometry_type = "wire"
            intersection_description = "boundary intersections"

        raise PositionerError(
            "Could not form horizontal intervals at "
            f"z={z!r} for the supplied {geometry_type}. No valid "
            f"{intersection_description} were found at the requested "
            "coordinate or at nearby coordinates. The geometry may be "
            "self-intersecting, degenerate, outside the x-z plane, or "
            "unsupported by slice_shape."
        )

    @staticmethod
    def _from_normalised(l_0: float, intervals: np.ndarray) -> float:
        """
        Convert one normalised horizontal coordinate to physical x.

        Distance is measured over the union of all filled intervals. Gaps
        between intervals, including holes, are omitted.

        Parameters
        ----------
        l_0:
            Normalised horizontal coordinate.
        intervals:
            Filled horizontal intervals.

        Returns
        -------
        :
            Physical x coordinate.
        """
        intervals = np.asarray(intervals, dtype=float)
        widths = intervals[:, 1] - intervals[:, 0]
        total_width = float(np.sum(widths))

        if total_width <= D_TOLERANCE:
            return intervals[0, 0]

        l_0 = np.clip(l_0, 0.0, 1.0)
        distance = l_0 * total_width

        if distance >= total_width:
            return float(intervals[-1, 1])

        cumulative_widths = np.cumsum(widths)

        # Using side="right" implements the documented right-continuous seam
        # convention. At an exact seam, the interval to the right is selected.
        interval_index = min(
            int(np.searchsorted(cumulative_widths, distance, side="right")),
            len(intervals) - 1,
        )

        previous_width = (
            0.0 if interval_index == 0 else float(cumulative_widths[interval_index - 1])
        )

        local_distance = distance - previous_width

        return float(intervals[interval_index, 0] + local_distance)

    @staticmethod
    def _to_normalised(x: float, intervals: np.ndarray) -> float:
        """
        Convert one physical x coordinate to its normalised coordinate.

        If x does not belong to a filled interval, it is projected onto the
        nearest interval. This includes points outside the outer boundary and
        points inside holes.

        Parameters
        ----------
        x:
            Physical x coordinate.
        intervals:
            Filled horizontal intervals.

        Returns
        -------
        :
            Normalised horizontal coordinate.
        """
        intervals = np.asarray(intervals, dtype=float)
        widths = intervals[:, 1] - intervals[:, 0]
        total_width = float(np.sum(widths))

        if total_width <= D_TOLERANCE:
            return 0.0

        prefix_width = 0.0
        best_distance = np.inf
        best_l_0 = 0.0

        final_interval_index = len(intervals) - 1

        for interval_index, ((left, right), width) in enumerate(
            zip(intervals, widths, strict=True)
        ):
            projected_x = float(np.clip(x, left, right))
            projection_distance = abs(float(x) - projected_x)

            candidate = (prefix_width + projected_x - left) / total_width

            # A seam corresponds to two distinct physical boundary points:
            # the right endpoint of one interval and the left endpoint of the
            # next interval.
            #
            # The forward map is right-continuous, so an exact seam maps to the
            # next interval. Encode the preceding right endpoint as the
            # floating-point value immediately below the seam, allowing that
            # physical endpoint to round-trip without entering the gap.
            is_nonfinal_right_endpoint = (
                interval_index < final_interval_index
                and projected_x == right
                and candidate > 0.0
            )

            if is_nonfinal_right_endpoint:
                candidate = float(np.nextafter(candidate, -np.inf))

            # Retaining the first result for equal projection distances gives
            # a deterministic tie break towards the smaller normalised value.
            if projection_distance < best_distance:
                best_distance = projection_distance
                best_l_0 = candidate

            prefix_width += width

        return float(np.clip(best_l_0, 0.0, 1.0))

    def to_xz(
        self,
        l_values: npt.ArrayLike | tuple[float, float] | tuple[np.ndarray, np.ndarray],
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Convert parametric-space 'L' values to physical x-z space.

        Parameters
        ----------
        l_values:
            Coordinates in normalised space

        Returns
        -------
        x:
            x coordinate in real space
        z:
            z coordinate in real space

        """
        l_0, l_1 = np.broadcast_arrays(
            np.clip(np.asarray(l_values[0], dtype=float), 0.0, 1.0),
            np.clip(np.asarray(l_values[1], dtype=float), 0.0, 1.0),
        )

        z = self.z_min + self._z_span * l_1
        x = np.empty_like(z, dtype=float)

        for index in np.ndindex(z.shape):
            intervals = self._intervals_at(float(z[index]))
            x[index] = self._from_normalised(float(l_0[index]), intervals)

        if x.ndim == 0:
            return float(x), float(z)

        return x, z

    def to_L(
        self, x: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Convert physical x-z coordinates to normalised coordinates.

        z is clipped to the region's vertical range. At the resulting z
        coordinate, x is projected onto the nearest filled horizontal interval
        before being normalised.

        Parameters
        ----------
        x:
            Physical x coordinate or array.
        z:
            Physical z coordinate or array.

        Returns
        -------
        l_0:
            Normalised horizontal coordinate or array.
        l_1:
            Normalised vertical coordinate or array.

        """
        x_array, z_array = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        )

        clipped_z = np.clip(z_array, self.z_min, self.z_max)

        l_1 = (clipped_z - self.z_min) / self._z_span

        l_0 = np.empty_like(l_1, dtype=float)

        for index in np.ndindex(l_1.shape):
            intervals = self._intervals_at(float(clipped_z[index]))
            l_0[index] = self._to_normalised(float(x_array[index]), intervals)

        if l_0.ndim == 0:
            return float(l_0), float(l_1)

        return l_0, l_1

    @property
    def dimension(self) -> int:
        """
        Dimension of the parametric space of the RegionInterpolator
        """
        return 2


class PositionMapper:
    """
    Positioning tool for use in optimisation

    Parameters
    ----------
    interpolators:
        The ordered list of geometry interpolators
    """

    def __init__(self, interpolators: dict[str, XZGeometryInterpolator]):
        self.interpolators = interpolators

    def _check_length(self, thing):
        """
        Check that something is the same length as the number of available interpolators.

        Raises
        ------
        PositionerError
            the number of iterators is not equal to the number of objects
        """
        if len(thing) != len(self.interpolators):
            raise PositionerError(
                f"Object of length: {len(thing)} not of length {len(self.interpolators)}"
            )

    def _vector_to_list(self, l_values):
        """
        Convert a vector of l_values into a ragged list if necessary

        Returns
        -------
        :
            The list of normalised values
        """
        list_values = []
        for i, interpolator in enumerate(self.interpolators.values()):
            values = l_values[i : i + interpolator.dimension]
            list_values.append(values)
        return list_values

    def to_xz(self, l_values: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Convert a set of parametric-space values to physical x-z coordinates.

        Parameters
        ----------
        l_values:
            The set of parametric-space values to convert

        Returns
        -------
        x:
            Array of x coordinates
        z:
            Array of z coordinates
        """
        l_values = self._vector_to_list(l_values)
        self._check_length(l_values)
        return np.array([
            tool.to_xz(l_values[i]) for i, tool in enumerate(self.interpolators.values())
        ]).T

    def to_xz_dict(self, l_values: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert a set of parametric space values to physical coordinates in a dictionary
        form.

        Parameters
        ----------
        l_values:
            The set of parametric-space values to convert

        Returns
        -------
        Dictionary of x-z values corresponding to each interpolator
        """
        l_values = self._vector_to_list(l_values)
        self._check_length(l_values)
        xz_dict = {}
        for i, (key, tool) in enumerate(self.interpolators.items()):
            xz_dict[key] = np.asarray(tool.to_xz(l_values[i]))
        return xz_dict

    def to_L(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Convert a set of physical x-z coordinates to parametric-space values.

        Parameters
        ----------
        x:
            The x coordinates to convert
        z:
            The z coordinates to convert

        Returns
        -------
        l_values:
            The set of parametric-space values
        """
        self._check_length(x)
        self._check_length(z)
        l_values = np.zeros(self.dimension)
        for i, tool in enumerate(self.interpolators.values()):
            l_values[i : i + tool.dimension] = tool.to_L(x[i], z[i])
        return np.array(l_values)

    @property
    def dimension(self) -> int:
        """
        The total dimension of the parametric space
        """
        return sum(interp.dimension for interp in self.interpolators.values())

    @property
    def interpolator_names(self) -> list[str]:
        """
        The names of the interpolators
        """
        return list(self.interpolators.keys())
