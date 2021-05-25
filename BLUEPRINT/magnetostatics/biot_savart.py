# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Green field loop object
"""
import numpy as np
from scipy.linalg import norm


class BiotSavartLoop:
    """
    Class to calculate field and vector potential from an arbitrary loop shape.

    Parameters
    ----------
    loops: Loop or List[Loop]
        The arbitrarily shaped closed current Loop. Alternatively provide the
        list of Loop objects.
    radius: float
        The nominal radius of the coil
    """

    def __init__(self, loops, radius):

        if not isinstance(loops, list):
            # Handle single Loop
            loops = [loops]

        # Handle list of Loops (potentially of different sizes)
        d_ls, d_l_hats, mids_points = [], [], []
        points = []
        for i, loop in enumerate(loops):
            loop = loop.copy()
            # Ensure loop is closed
            loop.close()
            d_l = np.diff(loop.xyz).T
            d_l = np.append(np.reshape(d_l[-1, :], (1, 3)), d_l, axis=0)  # prepend
            # central difference average segment length vectors
            d_l = (d_l[1:] + d_l[:-1]).T / 2

            d_l_hat = np.linalg.norm(d_l.T, axis=1)
            mid_points = loop.xyz[:, :-1] + d_l / 2
            d_ls.append(d_l)
            d_l_hats.append(d_l_hat)
            points.append(loop.xyz[:, :-1])
            mids_points.append(mid_points)
            if i == 0:
                # Take the first loop as a reference for inductance calculation
                self.ref_loop = loop
                self.ref_mid_points = mid_points
                self.ref_d_l = d_l

        # Assemble arrays and vector
        self.d_l = np.hstack(d_ls)
        self.d_l_hat = np.hstack(d_l_hats)
        self.mid_points = np.hstack(mids_points)
        self.points = np.hstack(points)

        self.loop = loops[0] if len(loops) == 1 else loops
        self.radius = radius
        self.length_scale = self.ref_loop.get_min_length()

    def potential(self, point):
        """
        Calculate the vector potential of an arbitrarily shaped loop.

        Parameters
        ----------
        point: np.array(3)
            The point at which to calculate the field

        Returns
        -------
        potential: np.array(3)
            The vector potential at the point due to the arbitrarily shaped loop
        """
        # TODO: Remove once regression tests complete
        # (stored energy now calculated via inductance)
        r = point - self.points.T
        r_mag = np.tile(norm(r, axis=1), (3, 1)).T
        r_mag[r_mag < 1e-16] = 1e-16
        core = r_mag / self.radius
        core[r_mag > self.radius] = 1
        return np.sum(core * self.d_l.T / r_mag, axis=0) / (4 * np.pi)

    def field_old(self, point):
        """
        Calculate the field due to the arbitrarily shaped loop.

        Parameters
        ----------
        point: np.array(3)
            The point at which to calculate the field

        Returns
        -------
        B: np.array(3)
            The field at the point due to the arbitrarily shaped loop

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{Idl \\times\\mathbf{r^{'}}}{|\\mathbf{r^{'}}|^{3}}`

        This is the original Biot-Savart equation, without centre-averaged
        smoothing. Do not use for values near the coil current centreline.
        """  # noqa (W505)
        r = point - self.mid_points.T
        r3 = np.linalg.norm(r, axis=1) ** 3

        ds = np.cross(self.d_l.T, r)

        # Coil core correction
        d_l_hat = np.tile(norm(self.d_l.T, axis=1), (3, 1)).T
        ds_mag = np.linalg.norm(ds / d_l_hat, axis=1)
        ds_mag = np.tile(ds_mag, (3, 1)).T
        ds_mag[ds_mag < 1e-16] = 1e-16
        core = ds_mag ** 2 / self.radius ** 2
        core[ds_mag > self.radius] = 1

        return 1e-7 * np.sum(core * ds / r3[:, np.newaxis], axis=0)

    def field(self, point):
        """
        Calculate the field due to the arbitrarily shaped loop.

        Parameters
        ----------
        point: np.array(3)
            The point at which to calculate the field

        Returns
        -------
        B: np.array(3)
            The field at the point due to the arbitrarily shaped loop

        Notes
        -----
        \t:math:`\\dfrac{\\mu_{0}}{4\\pi}\\oint \\dfrac{Idl \\times\\mathbf{r^{'}}}{|\\mathbf{r^{'}}|^{3}}`

        Uses Simon McIntosh's centre-averaged difference approach to smooth
        field near filaments.

        Masking about coil core.
        """  # noqa (W505)
        r = point - self.points.T

        r1 = r - self.d_l.T / 2
        r1_hat = r1 / np.tile(norm(r1, axis=1), (3, 1)).T
        r2 = r + self.d_l.T / 2
        r2_hat = r2 / np.tile(norm(r2, axis=1), (3, 1)).T
        d_l_hat = np.tile(norm(self.d_l.T, axis=1), (3, 1)).T
        ds = np.cross(self.d_l.T, r) / d_l_hat
        ds_mag = np.tile(norm(ds, axis=1), (3, 1)).T
        ds = np.cross(self.d_l.T, ds) / d_l_hat
        ds_mag[ds_mag < 1e-16] = 1e-16
        core = ds_mag ** 2 / self.radius ** 2
        core[ds_mag > self.radius] = 1

        return 1e-7 * sum(core * np.cross(ds, r2_hat - r1_hat) / ds_mag ** 2)

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
        for i, (x1, dx1) in enumerate(zip(self.ref_mid_points.T, self.ref_d_l.T)):
            # We create a mask to drop the point where x1 == x2
            r = x1 - self.mid_points.T
            mask = np.sum(r ** 2, axis=1) > self.radius
            inductance += np.sum(
                np.dot(dx1, self.d_l.T[mask].T) / np.linalg.norm(r[mask], axis=1)
            )

        # Self-inductance correction (Y = 0.5 for homogenous current distribution)
        inductance += (
            2
            * self.ref_loop.length
            * (np.log(2 * self.length_scale / self.radius) + 0.25)
        )

        return 1e-7 * inductance


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
