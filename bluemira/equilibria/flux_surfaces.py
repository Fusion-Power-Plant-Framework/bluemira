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
Flux surface utility classes and calculations
"""

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property
from scipy.integrate import odeint

from bluemira.geometry._deprecated_tools import (
    get_angle_between_points,
    loop_plane_intersect,
    join_intersect,
    check_linesegment,
)
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.error import FluxSurfaceError


def connection_length(x, z, Bp, Bt):
    """
    Calculate the parallel connection length along a field line (i.e. flux surface).

    Parameters
    ----------
    x: np.ndarray
        Radial coordinates of a flux surface
    z: np.ndarray
        Vertical coordinates of a flux surface
    Bp: np.ndarray
        Poloidal field values at (x, z)
    Bt: np.ndarray
        Toroidal field values at (x, z)

    Returns
    -------
    l_par: float
        Connection length from the start of the flux surface to the end of the flux
        surface
    """
    dx = np.diff(x)
    dz = np.diff(z)
    Bp = 0.5 * (Bp[1:] + Bp[:-1])
    Bt = 0.5 * (Bt[1:] + Bt[:-1])
    B_ratio = Bt / Bp
    dl = np.sqrt(1 + B_ratio ** 2) * np.hypot(dx, dz)
    return np.sum(dl)


class FluxSurface:
    """
    Flux surface base class.
    """

    __slots__ = ["loop"]

    def __init__(self, geometry):
        self.loop = geometry

    @property
    def x_start(self):
        """
        Start radial coordinate of the FluxSurface.
        """
        return self.loop.x[0]

    @property
    def z_start(self):
        """
        Start vertical coordinate of the FluxSurface.
        """
        return self.loop.z[0]

    @property
    def x_end(self):
        """
        End radial coordinate of the FluxSurface.
        """
        return self.loop.x[-1]

    @property
    def z_end(self):
        """
        End vertical coordinate of the FluxSurface.
        """
        return self.loop.z[-1]

    def connection_length(self, eq):
        """
        Calculate the parallel connection length along a field line (i.e. flux surface).

        Parameters
        ----------
        eq: Equilibrium
            Equilibrium from which the FluxSurface was extracted

        Returns
        -------
        l_par: float
            Connection length from the start of the flux surface to the end of the flux
            surface
        """
        Bp = eq.Bp(self.loop.x, self.loop.z)
        Bt = eq.Bt(self.loop.x)
        return connection_length(self.loop.x, self.loop.z, Bp, Bt)

    def plot(self, ax=None, **kwargs):
        """
        Plot the FluxSurface.
        """
        if ax is None:
            ax = plt.gca()

        kwargs["linewidth"] = kwargs.get("linewidth", 0.01)
        kwargs["color"] = kwargs.get("color", "r")

        self.loop.plot(ax, **kwargs)

    def copy(self):
        """
        Make a deep copy of the FluxSurface.
        """
        return deepcopy(self)


class ClosedFluxSurface(FluxSurface):
    """
    Utility class for closed flux surfaces.
    """

    def __init__(self, geometry):
        if not geometry.closed:
            raise FluxSurfaceError(
                "Cannot make a ClosedFluxSurface from an open geometry."
            )
        super().__init__(geometry)

    @cached_property
    def major_radius(self):
        """
        Major radius of the ClosedFluxSurface.
        """
        return self.loop.centroid[0]

    @cached_property
    def minor_radius(self):
        """
        Minor radius of the ClosedFluxSurface.
        """
        return 0.5 * (np.max(self.loop.x) - np.min(self.loop.x))

    @cached_property
    def aspect_ratio(self):
        """
        Aspect ratio of the ClosedFluxSurface.
        """
        return self.major_radius / self.minor_radius

    @cached_property
    def kappa(self):
        """
        Average elongation of the ClosedFluxSurface.
        """
        return 0.5 * (self.kappa_upper + self.kappa_lower)

    @cached_property
    def kappa_upper(self):
        """
        Upper elongation of the ClosedFluxSurface.
        """
        return (np.max(self.loop.z) - self.loop.centroid[1]) / self.minor_radius

    @cached_property
    def kappa_lower(self):
        """
        Lower elongation of the ClosedFluxSurface.
        """
        return abs(np.max(self.loop.z) - self.loop.centroid[1]) / self.minor_radius

    @cached_property
    def delta(self):
        """
        Average triangularity of the ClosedFluxSurface.
        """
        return 0.5 * (self.delta_upper + self.delta_lower)

    @cached_property
    def delta_upper(self):
        """
        Upper triangularity of the ClosedFluxSurface.
        """
        arg_z_max = np.argmax(self.loop.z)
        x_z_max = self.loop.x[arg_z_max]
        return (self.major_radius - x_z_max) / self.minor_radius

    @cached_property
    def delta_lower(self):
        """
        Lower triangularity of the ClosedFluxSurface.
        """
        arg_z_min = np.argmin(self.loop.z)
        x_z_min = self.loop.x[arg_z_min]
        return (self.major_radius - x_z_min) / self.minor_radius

    @cached_property
    def area(self):
        """
        Enclosed area of the ClosedFluxSurface.
        """
        return self.loop.area

    @cached_property
    def volume(self):
        """
        Volume of the ClosedFluxSurface.
        """
        return 2 * np.pi * self.loop.area * self.loop.centroid[0]

    def safety_factor(self, eq):
        """
        Calculate the cylindrical safety factor of a closed flux surface. The ratio of
        toroidal turns to a single full poloidal turn.

        Parameters
        ----------
        eq: Equilibrium
            Equilibrium with which to calculate the safety factor

        Returns
        -------
        q: float
            Cylindrical safety factor of the closed flux surface
        """
        return self.connection_length(eq) / self.loop.length


class OpenFluxSurface(FluxSurface):
    """
    Utility class for handling open flux surface geometries.
    """

    __slots__ = ["alpha"]

    def __init__(self, loop):
        if loop.closed:
            raise FluxSurfaceError(
                "OpenFluxSurface cannot be made from a closed geometry."
            )
        super().__init__(loop)

        # Constructors
        self.alpha = None

    def flux_expansion(self):
        pass

    def clip(self, first_wall):
        """
        Clip the LFS and HFS geometries to a first wall.

        Parameters
        ----------
        first_wall: Loop
            The geometry of the first wall to clip the OpenFluxSurface to
        """
        first_wall = first_wall.copy()

        args = join_intersect(self.loop, first_wall, get_arg=True)

        # Because we oriented the loop the "right" way, the first intersection
        # is at the smallest argument
        loop = Loop.from_array(self.loop[: min(args) + 1])
        self.loop = self._reset_direction(loop)

        fw_arg = int(first_wall.argmin([self.x_end, self.z_end]))

        if fw_arg + 1 == len(first_wall):
            pass
        elif check_linesegment(
            first_wall.d2.T[fw_arg],
            first_wall.d2.T[fw_arg + 1],
            np.array([self.x_end, self.z_end]),
        ):
            fw_arg = fw_arg + 1

        # Relying on the fact that first wall is ccw, get the intersection angle
        self.alpha = get_angle_between_points(loop[-2], loop[-1], first_wall[fw_arg])


def connection_length_sm(eq, x, z):
    dx = x[:-1] - x[1:]
    dz = z[:-1] - z[1:]
    x_mp = x[:-1] + 0.5 * dx
    z_mp = z[:-1] + 0.5 * dz
    Bx = eq.Bx(x_mp, z_mp)
    Bz = eq.Bz(x_mp, z_mp)
    Bt = eq.Bt(x_mp)
    dtheta = np.hypot(dx, dz)
    dphi = Bt / np.hypot(Bx, Bz) * dtheta / x_mp
    dl = dphi * np.sqrt((dx / dphi) ** 2 + (dz / dphi) ** 2 + (x[:-1] + 0.5 * dx) ** 2)
    return np.sum(dl)


class FLT:
    def __init__(self, eq):
        self.eq = eq

    def dy_dt(self, xz, angles):
        Bx = self.eq.Bx(*xz[:2])
        Bz = self.eq.Bz(*xz[:2])
        Bt = self.eq.Bt(xz[0])
        B = np.sqrt(Bx ** 2 + Bz ** 2 + Bt ** 2)
        return xz[0] / Bt * np.array([Bx, Bz, B])

    def trace_fl(self, x, z, n_points=100):
        angles = np.linspace(0, 20 * 2 * np.pi, n_points)
        result = odeint(self.dy_dt, np.array([x, z, 0]), angles)
        return result.T


def flux_expansion(eq, x1, z1, x2, z2):
    return x1 * eq.Bp(x1, z1) / (x2 * eq.Bp(x2, z2))


def split_flux_surface(flux_surface, plane, o_point):
    """
    Split an OpenFluxSurface into two separate OpenFluxSurfaces.

    Parameters
    ----------
    flux_surface: OpenFluxSurface
        The open flux surface to split into two
    plane: Plane
        The x-y cutting plane
    o_point: O-point
        The magnetic centre of the plasma

    Returns
    -------
    down, up: Iterable[OpenFluxSurface]
        The downwards and upwards open flux surfaces from the splitting point
    """
    if not isinstance(flux_surface, OpenFluxSurface):
        raise FluxSurfaceError("Can only split an OpenFluxSurface.")

    def reset_direction(self, loop):
        if loop.argmin([self.x_mp, self.z_mp]) != 0:
            loop.reverse()
        return loop

    ref_loop = flux_surface.loop.copy()
    intersections = loop_plane_intersect(ref_loop, plane)
    x_inter = intersections.T[0]

    # Pick the first intersection, travelling from the o_point outwards
    deltas = x_inter - o_point.x
    arg_inter = np.argmax(deltas > 0)
    x_mp = x_inter[arg_inter]
    z_mp = o_point.z

    # Split the flux surface geometry into LFS and HFS geometries

    delta = 1e-1 if o_point.x < x_mp else -1e-1
    radial_line = Loop(x=[o_point.x, x_mp + delta], z=[z_mp, z_mp])
    # Add the intersection point to the loop
    arg_inter = join_intersect(ref_loop, radial_line, get_arg=True)[0]

    # Split the flux surface geometry
    loop1 = Loop.from_array(ref_loop[: arg_inter + 1])
    loop2 = Loop.from_array(ref_loop[arg_inter:])

    loop1 = reset_direction(loop1)
    loop2 = reset_direction(loop2)

    # Sort the segments into down / outboard and up / inboard geometries
    if loop1.z[1] > z_mp:
        lfs_loop = loop2
        hfs_loop = loop1
    else:
        lfs_loop = loop1
        hfs_loop = loop2
    return OpenFluxSurface(lfs_loop), OpenFluxSurface(hfs_loop)
