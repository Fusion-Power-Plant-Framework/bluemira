# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Biot-Savart filament object
"""
import numpy as np
from bluemira.base.constants import EPS, MU_0_4PI, ONE_4PI
from bluemira.utilities import tools
from bluemira.utilities.plot_tools import Plot3D
from bluemira.geometry.tools import rotation_matrix, bounding_box, close_coordinates
from bluemira.magnetostatics.tools import process_loop_array, process_xyz_array
from bluemira.magnetostatics.baseclass import CurrentSource

__all__ = ["BiotSavartFilament"]


class BiotSavartFilament(CurrentSource):
    """
    Class to calculate field and vector potential from an arbitrary filament.

    Parameters
    ----------
    arrays: Union[Loop, np.array(n, 3), List[Loop, np.array(n, 3)]]
        The arbitrarily shaped closed current Loop. Alternatively provide the
        list of Loop objects.
    radius: float
        The nominal radius of the coil
    current: float
        The current flowing through the filament [A]. Defaults to 1 A to enable
        current to be optimised separately from the field response.
    """

    def __init__(self, arrays, radius, current=1.0):

        # TODO: Check / modify discretisation

        if not isinstance(arrays, list):
            # Handle single Loop/array
            arrays = [arrays]

        # Handle list of Loops/arrays (potentially of different sizes)
        d_ls, mids_points = [], []
        points = []
        for i, array in enumerate(arrays):
            array = process_loop_array(array)
            # Ensure array is closed
            xyz = np.array(close_coordinates(*array.T)).T

            diff = np.diff(xyz, axis=0)
            d_l = np.r_[[diff[-1, :]], diff]  # prepend
            # central difference average segment length vectors
            d_l = 0.5 * (d_l[1:] + d_l[:-1])

            mid_points = xyz[:-1, :] + 0.5 * d_l
            d_ls.append(d_l)
            points.append(xyz[:-1, :])
            mids_points.append(mid_points)
            if i == 0:
                # Take the first loop as a reference for inductance calculation
                self.ref_mid_points = mid_points
                self.ref_d_l = d_l

                lengths = np.sqrt(np.sum(diff ** 2, axis=1))
                self.length = np.sum(lengths)
                self.length_scale = np.min(lengths)

        # Assemble arrays and vector
        self.d_l = np.vstack(d_ls)
        self.d_l_hat = np.linalg.norm(self.d_l, axis=1)
        self.mid_points = np.vstack(mids_points)
        self.points = np.vstack(points)
        self._array_lengths = [len(p) for p in points]
        self.radius = radius
        self.current = current

    @process_xyz_array
    def potential(self, x, y, z):
        """
        Calculate the vector potential of an arbitrarily shaped loop.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the potential
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the potential
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the potential


        Returns
        -------
        potential: np.array(3)
            The vector potential at the point due to the arbitrarily shaped loop
        """
        point = np.array([x, y, z])
        r = point - self.points
        r_mag = tools.norm(r, axis=1)
        r_mag[r_mag < EPS] = EPS
        core = r_mag / self.radius
        core[r_mag > self.radius] = 1

        # The below einsum operation is equivalent to:
        # np.sum(core * self.d_l.T / r_mag, axis=0) / (4 * np.pi)
        return np.einsum(
            "i, ji, ... -> j", core, self.d_l / r_mag[None], ONE_4PI * self.current
        )

    @process_xyz_array
    def field_old(self, x, y, z):
        """
        Calculate the field due to the arbitrarily shaped loop.

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
        B: np.array
            The field at the point(s) due to the arbitrarily shaped loop

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{Idl \\times\\mathbf{r^{'}}}{|\\mathbf{r^{'}}|^{3}}`

        This is the original Biot-Savart equation, without centre-averaged
        smoothing. Do not use for values near the coil current centreline.
        """  # noqa (W505)
        point = np.arary([x, y, z])
        r = point - self.mid_points
        r3 = np.linalg.norm(r, axis=1) ** 3

        ds = np.cross(self.d_l, r)

        # Coil core correction
        d_l_hat = self.d_l_hat[:, None]
        ds_mag = np.linalg.norm(ds / d_l_hat, axis=1)
        ds_mag = np.tile(ds_mag, (3, 1)).T
        ds_mag[ds_mag < EPS] = EPS
        core = ds_mag ** 2 / self.radius ** 2
        core[ds_mag > self.radius] = 1
        return MU_0_4PI * self.current * np.sum(core * ds / r3[:, np.newaxis], axis=0)

    @process_xyz_array
    def field(self, x, y, z):
        """
        Calculate the field due to the arbitrarily shaped loop.

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
        B: np.array
            The field(s) at the point(s) due to the arbitrarily shaped loop

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{Idl \\times\\mathbf{r^{'}}}{|\\mathbf{r^{'}}|^{3}}`

        Uses Simon McIntosh's centre-averaged difference approach to smooth
        field near filaments.

        Masking about coil core.
        """  # noqa (W505)
        # point array -> point-segment vectors
        point = np.array([x, y, z])
        r = np.atleast_2d(point) - self.points
        r1 = r - self.d_l / 2
        r2 = r + self.d_l / 2

        r1_hat = r1 / tools.norm(r1, axis=1)[:, None]
        r2_hat = r2 / tools.norm(r2, axis=1)[:, None]

        d_l_hat = self.d_l_hat[:, None]

        ds = np.cross(self.d_l, r) / d_l_hat
        ds_mag = tools.norm(ds, axis=1)
        ds = np.cross(self.d_l, ds) / d_l_hat
        ds_mag[ds_mag < EPS] = EPS
        core = ds_mag ** 2 / self.radius ** 2
        core[ds_mag > self.radius] = 1
        # The below einsum operation is equivalent to:
        # MU_0_4PI * sum(core * np.cross(ds, r2_hat - r1_hat) / ds_mag ** 2)
        return np.einsum(
            "..., i, ij -> j",
            MU_0_4PI * self.current,
            core / ds_mag ** 2,
            np.cross(ds, r2_hat - r1_hat),
        )

    def inductance(self):
        """
        Calculate the total inductance of the BiotSavartLoop.

        Returns
        -------
        inductance: float
            The total inductance (including self-inductance of reference loop)
            in Henries [H]

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{d\\mathbf{x_{1}} \\cdot d\\mathbf{r_{x}}}{|\\mathbf{x_{1}}-\\mathbf{x_{2}}|}`

        https://arxiv.org/pdf/1204.1486.pdf

        You probably shouldn't use this if you are actually interested in the
        inductance of an arbitrarily shaped loop...
        """  # noqa (W505)
        # TODO: Validate inductance calculate properly and compare stored
        # energy of systems
        inductance = 0
        for i, (x1, dx1) in enumerate(zip(self.ref_mid_points, self.ref_d_l)):
            # We create a mask to drop the point where x1 == x2
            r = x1 - self.mid_points
            mask = np.sum(r ** 2, axis=1) > self.radius
            inductance += np.sum(
                np.dot(dx1, self.d_l[mask].T) / np.linalg.norm(r[mask], axis=1)
            )

        # Self-inductance correction (Y = 0.5 for homogenous current distribution)
        inductance += (
            2 * self.length * (np.log(2 * self.length_scale / self.radius) + 0.25)
        )

        return MU_0_4PI * inductance

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
        r = rotation_matrix(angle, axis)
        self.points = self.points @ r
        self.d_l = self.d_l @ r
        self.mid_points = self.mid_points @ r
        self.ref_d_l = self.ref_d_l @ r
        self.ref_mid_points = self.ref_mid_points @ r

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
            # Invisible bounding box to set equal aspect ratio plot
            xbox, ybox, zbox = bounding_box(*self.points.T)
            ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        # Split sub-filaments up for plotting purposes
        i = 0
        for length in self._array_lengths:
            ax.plot(*self.points[i : i + length].T, color="b", linewidth=1)
            i += length

        # Plot local coordinate system
        if show_coord_sys:
            origin = [0, 0, 0]
            dcm = np.eye(3)
            ax.scatter([0, 0, 0], color="k")
            ax.quiver(*origin, *dcm[0], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[1], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[2], length=self.length_scale, color="r")
