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
Grid object and grid operations
"""
from BLUEPRINT.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.equilibria.constants import X_AXIS_MIN
from BLUEPRINT.equilibria.plotting import GridPlotter
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import trapz
import numpy as np


class Grid:
    """
    Grid object

    Parameters
    ----------
    x_min: float > 0
        Minimum X grid coordinate [m]
    x_max: float
        Maximum X grid coordinate [m]
    z_min: float
        Minimum Z grid coordinate [m]
    z_max: float
        Maximum Z grid coordinate [m]
    nx: int
        Number of X grid points
    nz: int
        Number of Z grid points
    """

    def __init__(self, x_min, x_max, z_min, z_max, nx, nz):
        if x_min <= 0:  # Cannot calculate flux on machine axis - (divide by 0)
            x_min = X_AXIS_MIN
        if x_min > x_max:
            print("")  # stdout flusher
            bluemira_warn("Equilibria::Grid Xmin sollte niemals mehr als Xmax sein!")
            x_min, x_max = x_max, x_min

        if z_min > z_max:
            print("")  # stdout flusher
            bluemira_warn("Equilibria::Grid Zmin sollte niemals mehr als Zmax sein!")
            z_min, z_max = z_max, z_min

        self.x_min = x_min
        self.x_max = x_max
        self.x_size = x_max - x_min
        self.x_mid = (x_max - x_min) / 2
        self.z_min = z_min
        self.z_max = z_max
        self.z_size = abs(z_max) + abs(z_min)
        self.z_mid = z_min + (z_max - z_min) / 2
        self.x_1d = np.linspace(x_min, x_max, nx)
        self.z_1d = np.linspace(z_min, z_max, nz)
        self.x, self.z = np.meshgrid(self.x_1d, self.z_1d, indexing="ij")
        self.dx = np.diff(self.x_1d[:2])[0]  # Grid sizes
        self.dz = np.diff(self.z_1d[:2])[0]
        self.nx, self.nz = nx, nz
        self.bounds = [
            [x_min, x_max, x_max, x_min, x_min],  # Grid corners
            [z_min, z_min, z_max, z_max, z_min],
        ]
        self.edges = np.concatenate(
            [
                [(x, 0) for x in range(nx)],  # Grid edges
                [(x, nz - 1) for x in range(nx)],
                [(0, z) for z in range(nz)],
                [(nx - 1, z) for z in range(nz)],
            ]
        )

    @classmethod
    def from_eqdict(cls, e):
        """
        Initialises a Grid object from an EQDSK dictionary

        Parameters
        ----------
        e: dict
            EQDSK dictionary
        """
        return cls(
            e["xgrid1"],
            e["xgrid1"] + e["xdim"],
            e["zmid"] - 0.5 * e["zdim"],
            e["zmid"] + 0.5 * e["zdim"],
            e["nx"],
            e["nz"],
        )

    def point_inside(self, x, z):
        """
        Determines if a point is inside the rectangular grid (includes edges)

        Parameters
        ----------
        x, z: float, float
            The X, Z coordinates of the point

        Returns
        -------
        inside: bool
            Whether or not the point is inside the grid
        """
        return (
            (x >= self.x_min)
            and (x <= self.x_max)
            and (z >= self.z_min)
            and (z <= self.z_max)
        )

    def plot(self, ax=None, **kwargs):
        """
        Plots the Grid object onto an ax
        """
        return GridPlotter(self, ax=ax, **kwargs)


def integrate_dx_dz(func, d_x, d_z):
    """
    Returns the double-integral of a function over the space

    \t:math:`\\int_Z\\int_X f(x, z) dXdZ`

    Parameters
    ----------
    func: np.array(N, M)
        A 2-D function map
    d_x: float
        The discretisation size in the X coordinate
    d_z: float
        The discretisation size in the Z coordinate

    Returns
    -------
    integral: float
        The integral value of the field in 2-D
    """
    return trapz(trapz(func)) * d_x * d_z


def volume_integral(func, x, d_x, d_z):
    """
    Calculates the volume integral of a field in quasi-cylindrical coordinates

    Parameters
    ----------
    func: 2-D np.array
        Field to volume intregrate
    x: 2-D np.array
        X coordinate grid
    d_x: float
        Grid X cell size
    d_z: float
        Grid Z cell size

    Returns
    -------
    integral: float
        The integral value of the field in 3-D space
    """
    return 2 * np.pi * integrate_dx_dz(func * x, d_x, d_z)


def regrid(eq, grid, **kwargs):
    """
    Casts an EqObject onto a different sized grid, and different resolution.

    Parameters
    ----------
    eq: EqObject
        The Equilibrium object to regrid
    grid: Grid
        The Grid upon which to re-grid the Equilibrium

    Returns
    -------
    neweq: EqObject
        The new Equilibrium object
    """
    from BLUEPRINT.equilibria.equilibrium import Equilibrium
    from BLUEPRINT.equilibria.find import find_OX
    from BLUEPRINT.equilibria.boundary import apply_boundary

    neweq = Equilibrium(
        eq.coilset,
        grid,
        limiter=eq.limiter,
        Ip=eq._Ip,
        vcontrol=None,
        li=eq._li,
        **kwargs
    )
    psi = eq.psi()
    o_points, x_points = find_OX(eq.x, eq.z, psi, limiter=eq.limiter)
    profiles = eq._profiles
    jtor = profiles.jtor(eq.x, eq.z, psi, o_points, x_points)
    jtor_f = RectBivariateSpline(eq.x[:, 0], eq.z[0, :], jtor)
    jtor_new = jtor_f(neweq.x, neweq.z, grid=False)

    neweq.boundary(jtor_new, neweq.plasma_psi)
    rhs = -MU_0 * neweq.x * jtor_new  # RHS of GS equation

    apply_boundary(rhs, neweq.plasma_psi)

    plasma_psi = neweq.solve_GS(rhs)
    neweq._update_plasma_psi(plasma_psi)
    neweq._remap_greens()
    return neweq


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
