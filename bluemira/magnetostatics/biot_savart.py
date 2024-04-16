# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Biot-Savart filament object
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bluemira.base.constants import EPS, MU_0, MU_0_4PI, ONE_4PI
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.coordinates import rotation_matrix
from bluemira.magnetostatics.baseclass import CurrentSource
from bluemira.magnetostatics.tools import process_coords_array, process_xyz_array
from bluemira.utilities import tools
from bluemira.utilities.plot_tools import Plot3D

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.pyplot import Axes

    from bluemira.geometry.coordinates import Coordinates

__all__ = ["BiotSavartFilament"]


class BiotSavartFilament(CurrentSource):
    """
    Class to calculate field and vector potential from an arbitrary filament.

    Parameters
    ----------
    arrays:
        The arbitrarily shaped closed current Coordinates. Alternatively provide the
        list of Coordinates objects.
    radius:
        The nominal radius of the coil [m].
    current:
        The current flowing through the filament [A]. Defaults to 1 A to enable
        current to be optimised separately from the field response.
    """

    def __init__(
        self,
        arrays: Coordinates
        | npt.NDArray[np.float64]
        | list[Coordinates]
        | list[npt.NDArray[np.float64]],
        radius: float,
        current: float = 1.0,
    ):
        if not isinstance(arrays, list):
            # Handle single Coordinates/array
            arrays = [arrays]
        arrays = [process_coords_array(array) for array in arrays]

        # Handle list of Coordinates/arrays (potentially of different sizes)
        d_ls, mids_points = [], []
        points = []
        for i, xyz in enumerate(arrays):
            d_l = np.diff(xyz, axis=0)
            self._check_discretisation(d_l)

            mid_points = xyz[:-1, :] + 0.5 * d_l
            d_ls.append(d_l)
            points.append(xyz[:-1, :])
            mids_points.append(mid_points)
            if i == 0:
                # Take the first Coordinates as a reference for inductance calculation
                self.ref_mid_points = mid_points
                self.ref_d_l = d_l

                lengths = np.sqrt(np.sum(d_l**2, axis=1))
                self.length = np.sum(lengths)
                self.length_scale = np.min(lengths)

        # Assemble arrays and vector
        self._d_l = np.vstack(d_ls)
        self._d_l_hat = np.linalg.norm(self._d_l, axis=1)
        self._mid_points = np.vstack(mids_points)
        self._points = np.vstack(points)
        self._arrays = arrays
        self._radius = radius
        self.current = current

    @staticmethod
    def _check_discretisation(d_l: npt.NDArray[np.float64]):
        """
        Check the discretisation of the array.
        """
        lengths = np.sqrt(np.sum(d_l**2, axis=1))
        total = np.sum(lengths)
        max_d_l = np.max(lengths)
        if max_d_l > 0.03 * total:
            bluemira_warn("Biot-Savart discretisation possibly insufficient.")
        # TODO: Improve check and modify discretisation

    @process_xyz_array
    def potential(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the vector potential of an arbitrarily shaped Coordinates.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the potential
        y:
            The y coordinate(s) of the points at which to calculate the potential
        z:
            The z coordinate(s) of the points at which to calculate the potential


        Returns
        -------
        The vector potential at the point due to the arbitrarily shaped Coordinates
        """
        point = np.array([x, y, z])
        r = point - self._points
        r_mag = tools.norm(r, axis=1)
        r_mag[r_mag < EPS] = EPS
        core = r_mag / self._radius
        core[r_mag > self._radius] = 1

        # The below einsum operation is equivalent to:
        # self.current * np.sum(core * self.d_l.T / r_mag, axis=0) / (4 * np.pi)
        return np.einsum(
            "i, ji, ... -> j", core, self._d_l / r_mag[None], ONE_4PI * self.current
        )

    @process_xyz_array
    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the field due to the arbitrarily shaped Coordinates.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        The field at the point(s) due to the arbitrarily shaped Coordinates

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{Idl \\times\\mathbf{r^{'}}}{|\\mathbf{r^{'}}|^{3}}`

        This is the original Biot-Savart equation, without centre-averaged
        smoothing. Do not use for values near the coil current centreline.
        """  # noqa: W505, E501
        point = np.array([x, y, z])
        r = point - self._mid_points
        r3 = np.linalg.norm(r, axis=1) ** 3

        ds = np.cross(self._d_l, r)

        # Coil core correction
        d_l_hat = self._d_l_hat[:, None]
        ds_mag = np.linalg.norm(ds / d_l_hat, axis=1)
        ds_mag = np.tile(ds_mag, (3, 1)).T
        ds_mag[ds_mag < EPS] = EPS
        core = ds_mag**2 / self._radius**2
        core[ds_mag > self._radius] = 1
        return MU_0_4PI * self.current * np.sum(core * ds / r3[:, np.newaxis], axis=0)

    def inductance(self) -> float:
        """
        Calculate the total inductance of the BiotSavartFilament.

        Returns
        -------
        The total inductance (including self-inductance of reference Coordinates) [H]

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{d\\mathbf{x_{1}} \\cdot d\\mathbf{r_{x}}}{|\\mathbf{x_{1}}-\\mathbf{x_{2}}|}`

        https://arxiv.org/pdf/1204.1486.pdf

        You probably shouldn't use this if you are actually interested in the
        inductance of an arbitrarily shaped Coordinates...
        """  # noqa: W505, E501
        # TODO: Validate inductance calculate properly and compare stored
        # energy of systems
        inductance = 0
        for _i, (x1, dx1) in enumerate(
            zip(self.ref_mid_points, self.ref_d_l, strict=False)
        ):
            # We create a mask to drop the point where x1 == x2
            r = x1 - self._mid_points
            mask = np.sum(r**2, axis=1) > 0.5 * self.length_scale
            inductance += np.sum(
                np.dot(dx1, self._d_l[mask].T) / np.linalg.norm(r[mask], axis=1)
            )

        # Self-inductance correction (Y = 0.5 for homogenous current distribution)
        # Equation 6 of https://arxiv.org/pdf/1204.1486.pdf
        error_tail = 0
        a, b = self._radius, 0.5 * self.length_scale
        if b > 10 * a:
            # Equation A.4 of https://arxiv.org/pdf/1204.1486.pdf
            error_tail = a**2 / b**2 - 3 / (8 * b**4) * (a**4 - 2 * a**2)
        l_hat_0 = self.length * (2 * np.log(2 * b / a) + 0.5) + error_tail

        return MU_0_4PI * (inductance + l_hat_0)

    def rotate(self, angle: float, axis: str | np.ndarray):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [degree]
        axis:
            The axis of rotation
        """
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self._points = self._points @ r
        self._d_l = self._d_l @ r
        self._mid_points = self._mid_points @ r
        self.ref_d_l = self.ref_d_l @ r
        self.ref_mid_points = self.ref_mid_points @ r
        self._arrays = [array @ r for array in self._arrays]

    def plot(self, ax: Axes | None = None, *, show_coord_sys: bool = False):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax:
            The matplotlib axes to plot on
        show_coord_sys:
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()
            # If no ax provided, we assume that we want to plot only this source,
            # and thus set aspect ratio equality on this term only
            # Invisible bounding box to set equal aspect ratio plot
            xbox, ybox, zbox = BoundingBox.from_xyz(*self._points.T).get_box_arrays()
            ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for array in self._arrays:
            ax.plot(*array.T, color="b", linewidth=1)

        # Plot local coordinate system
        if show_coord_sys:
            origin = [0, 0, 0]
            dcm = np.eye(3)
            ax.scatter([0, 0, 0], color="k")
            ax.quiver(*origin, *dcm[0], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[1], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[2], length=self.length_scale, color="r")


def Bz_coil_axis(
    r: float, z: float = 0, pz: npt.ArrayLike = 0, current: float = 1
) -> float | npt.NDArray[np.float64]:
    """
    Calculate the theoretical vertical magnetic field of a filament coil
    (of radius r and centred in (0, z)) on a point on the coil axis at
    a distance pz from the axis origin.

    Parameters
    ----------
    r:
        Coil radius [m]
    z:
        Vertical position of the coil centroid [m]
    pz:
        Vertical position of the point on the axis on which the magnetic field
        shall be calculated [m]
    current:
        Current of the coil [A]

    Returns
    -------
    Vertical magnetic field on the axis [T]

    Notes
    -----
    \t:math:`\\dfrac{1}{2}\\dfrac{\\mu_{0}Ir^2}{(r^{2}+(pz-z)^{2})^{3/2}}`
    """
    return 0.5 * MU_0 * current * r**2 / (r**2 + (np.asarray(pz) - z) ** 2) ** 1.5
