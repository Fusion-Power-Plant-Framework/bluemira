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
import numba as nb
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.integrate import solve_ivp

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
from bluemira.equilibria.find import find_flux_surface_through_point


@nb.jit(nopython=True, cache=True)
def _flux_surface_dl(x, z, dx, dz, Bp, Bt):
    Bp = 0.5 * (Bp[1:] + Bp[:-1])
    Bt = Bt[:-1] * x[:-1] / (x[:-1] + 0.5 * dx)
    B_ratio = Bt / Bp
    return np.sqrt(1 + B_ratio ** 2) * np.hypot(dx, dz)


class FluxSurface:
    """
    Flux surface base class.

    Parameters
    ----------
    geometry: Loop
        Flux surface geometry object
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
        return _flux_surface_dl(x, z, np.diff(x), np.diff(z), Bp, Bt)

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
        x = x[:-1] + 0.5 * dx  # Segment centre-points
        z = z[:-1] + 0.5 * dz
        dl = np.hypot(dx, dz)  # Poloidal plane dl
        Bp = eq.Bp(x, z)
        Bt = eq.Bt(x)
        return np.sum(dl * Bt / (Bp * x)) / (2 * np.pi)


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
    vars = ["major_radius", "minor_radius", "aspect_ratio", "area", "volume"]
    vars += [f"{v}{end}" for end in ["", "_upper", "_lower"] for v in ["kappa", "delta"]]
    return CoreResults(
        psi_n,
        *[[getattr(fs, var) for fs in flux_surfaces] for var in vars],
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


class FieldLine:
    """
    Field line object.

    Parameters
    ----------
    loop: Loop
        Geometry of the FieldLine
    connection_length: float
        Connection length of the FieldLine
    """

    def __init__(self, loop, connection_length):
        self.loop = loop
        self.connection_length = connection_length

    def plot(self, ax=None, **kwargs):
        """
        Plot the FieldLine.

        Parameters
        ----------
        ax: Optional[Axes]
            Matplotlib axes onto which to plot
        """
        self.loop.plot(ax=ax, **kwargs)

    def pointcare_plot(self, ax=None):
        """
        Pointcaré plot of the field line intersections with the half-xz-plane.

        Parameters
        ----------
        ax: Optional[Axes]
            Matplotlib axes onto which to plot
        """
        if ax is None:
            ax = plt.gca()

        xz_plane = Plane([0, 0, 0], [1, 0, 0], [0, 0, 1])
        xi, _, zi = loop_plane_intersect(self.loop, xz_plane).T
        idx = np.where(xi >= 0)
        xi = xi[idx]
        zi = zi[idx]
        ax.plot(xi, zi, linestyle="", marker="o", color="r", ms=5)
        ax.set_aspect("equal")


class FieldLineTracer:
    """
    Field line tracing tool.

    Parameters
    ----------
    eq: Equilibrium
        Equilibrium in which to trace a field line
    first_wall: Union[Grid, Loop]
        Boundary at which to stop tracing the field line

    Notes
    -----
    Totally pinched some maths from Ben Dudson's FreeGS here... Perhaps one day I can
    return the favours.

    I needed it to compare the analytical connection length calculation with something,
    so I nicked this but changed the way the equation is solved.

    Note that this will properly trace field lines through Coils as it doesn't rely on
    the psi map (which is inaccurate inside Coils).
    """

    class CollisionTerminator:
        """
        Private Event handling object for solve_ivp

        Parameters
        ----------
        boundary: Union[Grid, Loop]
            Boundary at which to stop tracing the field line.
        """

        def __init__(self, boundary):
            self.boundary = boundary
            self.terminal = True

        def __call__(self, phi, xz, *args):
            """
            Function handle for the CollisionTerminator.
            """
            if self.boundary.point_inside(xz[:2]):
                return np.min(self.boundary.distance_to(xz[:2]))
            else:
                return -np.min(self.boundary.distance_to(xz[:2]))

    def __init__(self, eq, first_wall=None):
        self.eq = eq
        if first_wall is None:
            first_wall = self.eq.grid
        self.first_wall = first_wall

    def trace_field_line(self, x, z, n_points=200, forward=True, n_turns_max=20):
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
        field_line: FieldLine
            Resulting field line
        """
        phi = np.linspace(0, 2 * np.pi * n_turns_max, n_points)

        result = solve_ivp(
            self._dxzl_dphi,
            y0=np.array([x, z, 0]),
            t_span=(0, 2 * np.pi * n_turns_max),
            t_eval=phi,
            events=self.CollisionTerminator(self.first_wall),
            method="LSODA",
            args=(forward,),
        )
        r, z, phi, connection_length = self._process_result(result)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        loop = Loop(x=x, y=y, z=z, enforce_ccw=False)
        return FieldLine(loop, connection_length)

    def _dxzl_dphi(self, phi, xz, forward):
        """
        Credit: Dr. B. Dudson, FreeGS.
        """
        f = 1.0 if forward is True else -1.0
        Bx = self.eq.Bx(*xz[:2])
        Bz = self.eq.Bz(*xz[:2])
        Bt = self.eq.Bt(xz[0])
        B = np.sqrt(Bx ** 2 + Bz ** 2 + Bt ** 2)
        dx, dz, dl = xz[0] / Bt * np.array([f * Bx, f * Bz, B])
        return np.array([dx, dz, dl])

    @staticmethod
    def _process_result(result):
        if len(result["y_events"][0]) != 0:
            # Field line tracing was terminated by a collision
            end = len(result["y"][0])
            r, z = result["y"][0][:end], result["y"][1][:end]
            phi = result["t"][:end]
            termination = result["y_events"][0].flatten()
            r = np.append(r, termination[0])
            z = np.append(z, termination[1])
            connection_length = termination[2]
            phi = np.append(phi, result["t_events"][0][0])

        else:
            # Field line tracing was not terminated by a collision
            r, z, length = result["y"][0], result["y"][1], result["y"][2]
            phi = result["t"]
            connection_length = length[-1]
        return r, z, phi, connection_length


def calculate_connection_length_flt(
    eq, x, z, forward=True, first_wall=None, n_turns_max=50
):
    """
    Calculate the parallel connection length from a starting point to a flux-intercepting
    surface using a field line tracer.

    Parameters
    ----------
    eq: Equilibrium
        Equilibrium in which to calculate the connection length
    x: float
        Radial coordinate of the starting point
    z: float
        Vertical coordinate of the starting point
    forward: bool (default = True)
        Whether or not to follow the field line forwards or backwards (+B or -B)
    first_wall: Union[Loop, Grid]
        Flux-intercepting surface. Defaults to the grid of the equilibrium
    n_turns_max: Union[int, float]
        Maximum number of toroidal turns to trace the field line

    Returns
    -------
    connection_length: float
        Parallel connection length along the field line from the starting point to the
        intersection point [m]

    Notes
    -----
    More mathematically accurate, but needs additional configuration. Will not likely
    return a very accurate flux inteception point. Also works for closed flux surfaces,
    but can't tell the difference. Not sensitive to equilibrium grid discretisation.
    Will work correctly for flux surfaces passing through Coils, but really they should
    be intercepted beforehand!
    """
    flt = FieldLineTracer(eq, first_wall)
    field_line = flt.trace_field_line(
        x, z, forward=forward, n_points=2, n_turns_max=n_turns_max
    )
    return field_line.connection_length


def calculate_connection_length_fs(eq, x, z, forward=True, first_wall=None):
    """
    Calculate the parallel connection length from a starting point to a flux-intercepting
    surface using flux surface geometry.

    Parameters
    ----------
    eq: Equilibrium
        Equilibrium in which to calculate the connection length
    x: float
        Radial coordinate of the starting point
    z: float
        Vertical coordinate of the starting point
    forward: bool (default = True)
        Whether or not to follow the field line forwards or backwards
    first_wall: Union[Loop, Grid]
        Flux-intercepting surface. Defaults to the grid of the equilibrium

    Returns
    -------
    connection_length: float
        Parallel connection length along the field line from the starting point to the
        intersection point [m]

    Raises
    ------
    FluxSurfaceError
        If the flux surface at the point in the equilibrium is not an open flux surface

    Notes
    -----
    Less mathematically accurate. Will return exact intersection point. Sensitive to
    equilibrium grid discretisation. Presently does not correctly work for flux surfaces
    passing through Coils, but really they should be intercepted beforehand!
    """
    if first_wall is None:
        x1, x2 = eq.grid.x_min, eq.grid.x_max
        z1, z2 = eq.grid.z_min, eq.grid.z_max
        first_wall = Loop(x=[x1, x2, x2, x1, x1], z=[z1, z1, z2, z2, z1])

    xfs, zfs = find_flux_surface_through_point(eq.x, eq.z, eq.psi(), x, z, eq.psi(x, z))
    f_s = OpenFluxSurface(Loop(x=xfs, z=zfs))

    class Point:
        def __init__(self, x, z):
            self.x = x
            self.z = z

    lfs, hfs = f_s.split(Point(x=x, z=z))
    if forward:
        fs = lfs
    else:
        fs = hfs

    fs.clip(first_wall)
    return fs.connection_length(eq)
