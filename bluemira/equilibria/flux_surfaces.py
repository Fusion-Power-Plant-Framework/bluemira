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

from dataclasses import dataclass
from typing import Iterable
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.integrate import odeint

from bluemira.utilities.tools import cartesian_to_polar
from bluemira.geometry._deprecated_tools import (
    get_angle_between_points,
    loop_plane_intersect,
    join_intersect,
    check_linesegment,
)
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_base import Plane
from bluemira.equilibria.error import FluxSurfaceError
from bluemira.equilibria.constants import PSI_NORM_TOL


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

    def _dl(self, eq):
        x, z = self.loop.x, self.loop.z
        Bp = eq.Bp(x, z)
        Bt = eq.Bt(x)
        dx = np.diff(x)
        dz = np.diff(z)
        Bp = 0.5 * (Bp[1:] + Bp[:-1])
        Bt = Bt[:-1] * x[:-1] / (x[:-1] + 0.5 * dx)
        B_ratio = Bt / Bp
        return np.sqrt(1 + B_ratio ** 2) * np.hypot(dx, dz)

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
        return np.sum(self._dl(eq))

    def plot(self, ax=None, **kwargs):
        """
        Plot the FluxSurface.
        """
        if ax is None:
            ax = plt.gca()

        kwargs["linewidth"] = kwargs.get("linewidth", 0.05)
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

    __slots__ = []

    def __init__(self, geometry):
        if not geometry.closed:
            raise FluxSurfaceError(
                "Cannot make a ClosedFluxSurface from an open geometry."
            )
        super().__init__(geometry)

    @property
    @lru_cache(1)
    def major_radius(self):
        """
        Major radius of the ClosedFluxSurface.
        """
        # debatable... could also be x_min + 0.5 * (x_max - x_min)
        return self.loop.centroid[0]

    @property
    @lru_cache(1)
    def minor_radius(self):
        """
        Minor radius of the ClosedFluxSurface.
        """
        return 0.5 * (np.max(self.loop.x) - np.min(self.loop.x))

    @property
    @lru_cache(1)
    def aspect_ratio(self):
        """
        Aspect ratio of the ClosedFluxSurface.
        """
        return self.major_radius / self.minor_radius

    @property
    @lru_cache(1)
    def kappa(self):
        """
        Average elongation of the ClosedFluxSurface.
        """
        return 0.5 * (self.kappa_upper + self.kappa_lower)

    @property
    @lru_cache(1)
    def kappa_upper(self):
        """
        Upper elongation of the ClosedFluxSurface.
        """
        return (np.max(self.loop.z) - self.loop.centroid[1]) / self.minor_radius

    @property
    @lru_cache(1)
    def kappa_lower(self):
        """
        Lower elongation of the ClosedFluxSurface.
        """
        return abs(np.min(self.loop.z) - self.loop.centroid[1]) / self.minor_radius

    @property
    @lru_cache(1)
    def delta(self):
        """
        Average triangularity of the ClosedFluxSurface.
        """
        return 0.5 * (self.delta_upper + self.delta_lower)

    @property
    @lru_cache(1)
    def delta_upper(self):
        """
        Upper triangularity of the ClosedFluxSurface.
        """
        arg_z_max = np.argmax(self.loop.z)
        x_z_max = self.loop.x[arg_z_max]
        return (self.major_radius - x_z_max) / self.minor_radius

    @property
    @lru_cache(1)
    def delta_lower(self):
        """
        Lower triangularity of the ClosedFluxSurface.
        """
        arg_z_min = np.argmin(self.loop.z)
        x_z_min = self.loop.x[arg_z_min]
        return (self.major_radius - x_z_min) / self.minor_radius

    @property
    @lru_cache(1)
    def area(self):
        """
        Enclosed area of the ClosedFluxSurface.
        """
        return self.loop.area

    @property
    @lru_cache(1)
    def volume(self):
        """
        Volume of the ClosedFluxSurface.
        """
        return 2 * np.pi * self.loop.area * self.loop.centroid[0]

    def shafranov_shift(self, eq):
        """
        Calculate the Shafranov shift of the ClosedFluxSurface.

        Parameters
        ----------
        eq: Equilibrium
            Equilibrium with which to calculate the safety factor

        Returns
        -------
        dx_shaf: float
            Radial Shafranov shift
        dz_shaf: float
            Vertical Shafranov shift
        """
        o_point = eq.get_OX_points()[0][0]  # magnetic axis
        return o_point.x - self.major_radius, o_point.z - self.loop.centroid[1]

    def safety_factor(self, eq):
        """
        Calculate the cylindrical safety factor of the ClosedFluxSurface. The ratio of
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
        x, z = self.loop.x, self.loop.z
        dx, dz = np.diff(x), np.diff(z)
        r, _ = cartesian_to_polar(
            x[:-1] + dx, z[:-1] + dz, self.major_radius, self.loop.centroid[1]
        )
        return np.sum(self._dl(eq) * r / (x[:-1] + dx)) / (2 * np.pi)


class OpenFluxSurface(FluxSurface):
    """
    Utility class for handling open flux surface geometries.
    """

    __slots__ = []

    def __init__(self, loop):
        if loop.closed:
            raise FluxSurfaceError(
                "OpenFluxSurface cannot be made from a closed geometry."
            )
        super().__init__(loop)

    def split(self, o_point, plane=None):
        """
        Split an OpenFluxSurface into two separate PartialOpenFluxSurfaces about a
        horizontal plane.

        Parameters
        ----------
        flux_surface: OpenFluxSurface
            The open flux surface to split into two
        o_point: O-point
            The magnetic centre of the plasma
        plane: Optional[Plane]
            The x-y cutting plane. Will default to the O-point x-y plane

        Returns
        -------
        down, up: Iterable[OpenFluxSurface]
            The downwards and upwards open flux surfaces from the splitting point
        """

        def reset_direction(loop):
            if loop.argmin([x_mp, z_mp]) != 0:
                loop.reverse()
            return loop

        if plane is None:
            plane = Plane(
                [o_point.x, 0, o_point.z],
                [o_point.x + 1, 0, o_point.z],
                [o_point.x, 1, o_point.z],
            )

        ref_loop = self.loop.copy()
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
        return PartialOpenFluxSurface(lfs_loop), PartialOpenFluxSurface(hfs_loop)


class PartialOpenFluxSurface(OpenFluxSurface):
    """
    Utility class for a partial open flux surface, i.e. an open flux surface that has
    been split at the midplane and only has one intersection with the wall.
    """

    __slots__ = ["alpha"]

    def __init__(self, loop):
        super().__init__(loop)

        self.alpha = None

    def clip(self, first_wall):
        """
        Clip the PartialOpenFluxSurface to a first wall.

        Parameters
        ----------
        first_wall: Loop
            The geometry of the first wall to clip the OpenFluxSurface to
        """
        first_wall = first_wall.copy()

        args = join_intersect(self.loop, first_wall, get_arg=True)

        # Because we oriented the loop the "right" way, the first intersection
        # is at the smallest argument
        self.loop = Loop.from_array(self.loop[: min(args) + 1], enforce_ccw=False)
        # self.loop = self._reset_direction(loop)

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
        self.alpha = get_angle_between_points(
            self.loop[-2], self.loop[-1], first_wall[fw_arg]
        )

    def flux_expansion(self, eq):
        """
        Flux expansion of the PartialOpenFluxSurface.

        Parameters
        ----------
        eq: Equilibrium
            Equilibrium with which to calculate the flux expansion

        Returns
        -------
        f_x: float
            Target flux expansion
        """
        return (
            self.x_start
            * eq.Bp(self.x_start, self.z_start)
            / (self.x_end * eq.Bp(self.x_end, self.z_end))
        )


def analyse_plasma_core(eq, n_points=50):
    """
    Analyse plasma core parameters across the normalised 1-D flux coordinate.

    Returns
    -------
    results: CoreResults
        Results dataclass
    """
    psi_n = np.linspace(PSI_NORM_TOL, 1 - PSI_NORM_TOL, n_points, endpoint=False)
    loops = [eq.get_flux_surface(pn) for pn in psi_n]
    loops.append(eq.get_LCFS())
    psi_n = np.append(psi_n, 1.0)
    flux_surfaces = [ClosedFluxSurface(loop) for loop in loops]
    return CoreResults(
        psi_n,
        [fs.major_radius for fs in flux_surfaces],
        [fs.minor_radius for fs in flux_surfaces],
        [fs.aspect_ratio for fs in flux_surfaces],
        [fs.area for fs in flux_surfaces],
        [fs.volume for fs in flux_surfaces],
        [fs.kappa for fs in flux_surfaces],
        [fs.kappa_upper for fs in flux_surfaces],
        [fs.kappa_lower for fs in flux_surfaces],
        [fs.delta for fs in flux_surfaces],
        [fs.delta_upper for fs in flux_surfaces],
        [fs.delta_lower for fs in flux_surfaces],
        [fs.safety_factor(eq) for fs in flux_surfaces],
        [fs.shafranov_shift(eq)[0] for fs in flux_surfaces],
    )


@dataclass
class CoreResults:
    """
    Dataclass for core results.
    """

    psi_n: Iterable
    R_0: Iterable
    a: Iterable
    A: Iterable
    area: Iterable
    V: Iterable
    kappa: Iterable
    kappa_upper: Iterable
    kappa_lower: Iterable
    delta: Iterable
    delta_lower: Iterable
    delta_upper: Iterable
    q: Iterable
    Delta_shaf: Iterable


class FieldLineTracer:
    """
    Field line tracing tool.
    """

    def __init__(self, eq, first_wall=None):
        """
        Parameters
        ----------
        eq: Equilibrium
            Equilibrium in which to trace a field line
        first_wall: Union[Grid, Loop]
            Boundary at which to stop tracing the field line
        """
        self.eq = eq
        if first_wall is None:
            first_wall = self.eq.grid
        self.first_wall = first_wall

    def trace_field_line(self, x, z, n_points=100, forward=True, n_turns_max=10):
        """
        Trace a single field line starting at a point.

        Parameters
        ----------
        x: float
            Radial coordinate of the starting point
        z: float
            Vertical coordinate of the starting point
        n_points: int
            Number of points along the field line
        forward: bool
            Whether or not to step forward or backward (+B or -B)
        n_turns_max: Union[int, float]
            Maximum number of toroidal turns to trace the field line

        Returns
        -------
        x_fl: np.ndarray
            Radial coordinates of the field line
        z_fl: np.ndarray
            Vertical coordiantes of the field line
        dl: np.ndarray
            Distance travelled along the field line
        """
        phi = np.linspace(0, n_turns_max * 2 * np.pi, n_points)
        result = odeint(self._d_phi_dt, np.array([x, z, 0]), phi, args=(forward,))
        return result.T

    def _d_phi_dt(self, xz, phi, forward):
        f = 1.0 if forward is True else -1.0
        if self.first_wall.point_inside(*xz[:2]):  # point_in_poly
            Bx = self.eq.Bx(*xz[:2])
            Bz = self.eq.Bz(*xz[:2])
            Bt = self.eq.Bt(xz[0])
            B = np.sqrt(Bx ** 2 + Bz ** 2 + Bt ** 2)
            dx, dz, dl = xz[0] / Bt * np.array([f * Bx, f * Bz, B])
        else:
            dx, dz, dl = np.zeros(3, dtype=np.int)
        return dx, dz, dl


def estimate_field(x1, z1, Bp1, Bt1, x2, z2, Bp2, x15, z15):
    dl = np.hypot(x2 - x1, z2 - z1)
    dl15 = np.hypot(x15 - x1, z15 - z1)
    Bt15 = Bt1 * x1 / x15
    Bp15 = Bp1 + dl15 / dl * (Bp2 - Bp1)
    return Bp15, Bt15


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


def safety_factor(x, z, Bp, Bt):
    """
    Safety factor draft
    """
    dx = np.diff(x)
    dz = np.diff(z)
    Bp = 0.5 * (Bp[1:] + Bp[:-1])
    Bt = 0.5 * (Bt[1:] + Bt[:-1])
    B_ratio = Bt / Bp
    r, _ = cartesian_to_polar(x[:-1] + dx, z[:-1] + dz, np.average(x), np.average(z))
    dl = np.sqrt(1 + B_ratio ** 2) * np.hypot(dx, dz)
    return np.sum(dl * r / (x[:-1] + dx)) / (2 * np.pi)
