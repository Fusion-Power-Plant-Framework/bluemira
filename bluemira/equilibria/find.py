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
Methods for finding O- and X-points and flux surfaces on 2-D arrays.
"""

import numpy as np
import numba as nb
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from matplotlib._contour import QuadContourGenerator
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.error import EquilibriaError
from bluemira.geometry._deprecated_tools import in_polygon, get_area_2d
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.constants import X_TOLERANCE, B_TOLERANCE

__all__ = [
    "Xpoint",
    "Opoint",
    "Lpoint",
    "get_contours",
    "find_OX_points",
    "find_LCFS_separatrix",
    "find_flux_surf",
    "in_zone",
    "in_plasma",
    "grid_2d_contour",
]


# =============================================================================
# O-, X- and L-point classes and finding + sorting functions
# =============================================================================


class PsiPoint:
    """
    Abstract object for psi-points with list indexing and point behaviour.
    """

    __slots__ = ["x", "z", "psi"]

    def __init__(self, x, z, psi):
        self.x, self.z = x, z
        self.psi = psi

    def __iter__(self):
        """
        Imbue PsiPoint with generator-like behaviour
        """
        yield self.x
        yield self.z
        yield self.psi

    def __getitem__(self, i):
        """
        Imbue PsiPoint with list-like behaviour
        """
        return [self.x, self.z, self.psi][i]

    def __str__(self):
        """
        A better string representation of the PsiPoint.
        """
        return (
            f"{self.__class__.__name__} x: {self.x:.2f}, z:{self.z:.2f}, "
            f"psi: {self.psi:.2f}"
        )


class Xpoint(PsiPoint):
    """
    X-point class.
    """

    __slots__ = []
    pass


class Opoint(PsiPoint):
    """
    O-point class.
    """

    __slots__ = []
    pass


class Lpoint(PsiPoint):
    """
    Limiter point class.
    """

    __slots__ = []
    pass


def find_local_minima(f):
    """
    Finds all local minima in a 2-D function map

    Parameters
    ----------
    f: np.array(N, M)
        The 2-D field on which to find local minima

    Returns
    -------
    i, j: np.array(int64, ..), np.array(int64, ..)
        The indices of local minima on the field map

    Notes
    -----
    Cannot find minima on the corners of an array.
    Cannot find "plateau" minima.
    For our use case, neither limitation is relevant.
    """
    return np.where(
        (
            (f < np.roll(f, 1, 0))
            & (f < np.roll(f, -1, 0))
            & (f <= np.roll(f, 0, 1))
            & (f <= np.roll(f, 0, -1))
            & (f < np.roll(f, 1, 1))
            & (f < np.roll(f, -1, 1))
        )
    )


def find_local_Bp_minima_scipy(f_Bp2, x0, z0, radius):  # noqa (N802)
    """
    Find local Bp^2 minima on a grid (precisely) using a scipy optimiser.

    Parameters
    ----------
    f_Bp2: callable
        The function handle for Bp^2 interpolation
    x0: float
        The local grid minimum x coordinate
    z0: float
        The local grid minimum z coordinate
    radius: float
        The search radius

    Returns
    -------
    x: Union[float, None]
        The x coordinate of the minimum. None if the minimum is not valid.
    z: float
        The z coordinate of the minimum
    """
    x_0 = np.array([x0, z0])
    # BFGS expensive with many grid points?
    # bounds = [
    #     [x0 - 2 * dx, x0 + 2 * dx],
    #     [z0 - 2 * dz, z0 + 2 * dz],
    # ]  # TODO: Implement and figure out why so slow
    res = minimize(f_Bp2, x_0, method="BFGS", options={"disp": False})  # bounds=bounds,
    if np.sqrt(res.fun) < B_TOLERANCE:
        xn, zn = res.x[0], res.x[1]
        if np.sqrt((x0 - xn) ** 2 + (z0 - zn) ** 2) < 5 * radius:
            return [res.x[0], res.x[1]]
    return None


@nb.jit(nopython=True, cache=True)
def inv_2x2_matrix(a, b, c, d):
    """
    Inverse of a 2 x 2 [[a, b], [c, d]] matrix.
    """
    return np.array([[d, -b], [-c, a]]) / (a * d - b * c)


def find_local_Bp_minima_cg(f_psi, x0, z0, radius):
    """
    Find local Bp minima on a grid (precisely) using a local Newton/Powell
    conjugate gradient search.

    Parameters
    ----------
    f_psi: callable
        The function handle for psi interpolation
    x0: float
        The local grid minimum x coordinate
    z0: float
        The local grid minimum z coordinate
    radius: float
        The search radius

    Returns
    -------
    x: Union[float, None]
        The x coordinate of the minimum. None if the minimum is not valid.
    z: float
        The z coordinate of the minimum
    """
    xi, zi = x0, z0
    count = 0
    while True:
        Bx = -f_psi(xi, zi, dy=1, grid=False) / xi
        Bz = f_psi(xi, zi, dx=1, grid=False) / xi
        if np.hypot(Bx, Bz) < B_TOLERANCE:
            return [xi, zi]
        else:
            a = -Bx / xi - f_psi(xi, zi, dy=1, dx=1)[0][0] / xi
            b = -f_psi(xi, zi, dy=2)[0][0] / xi
            c = -Bz / xi + f_psi(xi, zi, dx=2) / xi
            d = f_psi(xi, zi, dx=1, dy=1)[0][0] / xi
            inv_jac = inv_2x2_matrix(float(a), float(b), float(c), float(d))
            delta = np.dot(inv_jac, [Bx, Bz])
            xi -= delta[0]
            zi -= delta[1]
            count += 1
            if ((xi - x0) ** 2 + (zi - z0) ** 2 > radius) or (count > 50):
                return None


def drop_space_duplicates(points, tol=X_TOLERANCE):
    """
    Drop duplicates from a list of points if closer together than tol
    """
    stack = []
    for p1 in points:
        duplicate = False
        for p2 in stack:
            if (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 < tol:
                duplicate = True
                break
        if not duplicate:
            stack.append(p1)
    return stack


def triage_OX_points(f_psi, points):
    """
    Triage the local Bp minima into O- and X-points.

    Parameters
    ----------
    f_psi: callable
        The function handle for psi interpolation
    points: List[List]

    Returns
    -------
    o_points: List[List]
        The O-point locations
    x_points: List[List]
        The X-point locations

    Notes
    -----
    O-points have a positive 2nd derivative and X-points have a negative one:

    \t:math:`S=\\bigg(\\dfrac{{\\partial}{\\psi}^{2}}{{\\partial}X^{2}}\\bigg)`
    \t:math:`\\bigg(\\dfrac{{\\partial}{\\psi}^{2}}{{\\partial}Z^{2}}\\bigg)`
    \t:math:`-\\bigg(\\dfrac{{\\partial}{\\psi}^{2}}{{\\partial}X{\\partial}Z}`
    \t:math:`\\bigg)^{2}`
    """
    o_points, x_points = [], []
    for (xi, zi) in points:
        d2dx2 = f_psi(xi, zi, dx=2, grid=False)
        d2dz2 = f_psi(xi, zi, dy=2, grid=False)
        d2dxdz = f_psi(xi, zi, dx=1, dy=1, grid=False)
        s_value = d2dx2 * d2dz2 - d2dxdz ** 2

        if s_value < 0:
            x_points.append(Xpoint(xi, zi, f_psi(xi, zi)[0][0]))
        else:  # Note: Low positive values are usually dubious O-points, and
            # possibly should be labelled as X-points.
            o_points.append(Opoint(xi, zi, f_psi(xi, zi)[0][0]))

    return o_points, x_points


def find_OX_points(x, z, psi, limiter=None, x_min=None):  # noqa (N802)
    """
    Finds O-points and X-points by minimising the poloidal field.

    Parameters
    ----------
    x: np.array(N, M)
        The spatial x coordinates of the grid points [m]
    z: np.array(N, M)
        The spatial z coordinates of the grid points [m]
    psi: np.array(N, M)
        The poloidal magnetic flux map [V.s/rad]
    limiter: Union[None, Limiter]
        The limiter to use (if any)
    x_min: Union[None, float]
        The inner x cut-off point for searching O, X points (useful when using
        big grids and avoiding singularities in the CS due to Greens functions)

    Returns
    -------
    o_points: List[Opoint]
        The O-points in the psi map
    x_points: List[Union[Xpoint,LPoint]]
        The X-points and L-points in the psi map

    Notes
    -----
    \t:math:`\\lvert{\\nabla}{\\psi}{\\lvert}^{2} = 0`

    Local minima brute-forced, and subsequent accurate locations of the points
    found by local optimisation.

    For speed, does this on existing psi map (not exactly at optimum).

    Points are order w.r.t. central grid coordinates.
    """
    d_x, d_z = x[1, 0] - x[0, 0], z[0, 1] - z[0, 0]  # Grid resolution
    x_m, z_m = (x[0, 0] + x[-1, 0]) / 2, (z[0, 0] + z[0, -1]) / 2  # Grid centre

    if x_min is None:
        f = RectBivariateSpline(x[:, 0], z[0, :], psi)  # Spline for psi interpolation
    else:
        # Truncate grid to avoid many OX points on solenoid (CREATE relic)
        i_x = np.argmin(abs(x[:, 0] - x_min)) - 1
        f = RectBivariateSpline(x[i_x:, 0], z[0, :], psi[i_x:, :])
        x = x[i_x:0]
        psi = psi[i_x:, :]

    nx, nz = psi.shape  # Grid shape (including truncation)

    radius = min(0.5, 2 * (d_x ** 2 + d_z ** 2))  # Search radius

    def f_bp(x_opt):
        """
        Poloidal field optimiser function handle
        \t:math:`B_{p} = B_{x}^{2}+B_{z}^{2}`

        Notes
        -----
        No points square-rooting here, we don't care about actual values.
        """
        return (-f(x_opt[0], x_opt[1], dy=1, grid=False) / x_opt[0]) ** 2 + (
            f(x_opt[0], x_opt[1], dx=1, grid=False) / x_opt[0]
        ) ** 2

    Bp2 = f_bp(np.array([x, z]))

    points = []
    for i, j in zip(*find_local_minima(Bp2)):
        if i > nx - 3 or i < 3 or j > nz - 3 or j < 3:
            continue  # Edge points uninteresting and mess up S calculation.

        if nx * nz <= 4225:  # scipy method faster on small grids
            point = find_local_Bp_minima_scipy(f_bp, x[i, j], z[i, j], radius)

        else:  # Local Newton/Powell CG method faster on large grids
            point = find_local_Bp_minima_cg(f, x[i, j], z[i, j], radius)

        if point:
            points.append(point)

    points = drop_space_duplicates(points)

    o_points, x_points = triage_OX_points(f, points)

    if len(o_points) == 0:
        print("")  # stdout flusher
        bluemira_warn("EQUILIBRIA::find_OX: No O-points found during an iteration.")
        return o_points, x_points

    def _cntr_sort(p):
        return (p[0] - x_m) ** 2 + (p[1] - z_m) ** 2

    def _psi_sort(p):
        return (p[2] - psio) ** 2

    o_points.sort(key=_cntr_sort)
    x_op, z_op, psio = o_points[0]  # Primary O-point
    useful_x, useless_x = [], []
    if limiter is not None:
        limit_x = []
        for lim in limiter:
            limit_x.append(Lpoint(*lim, f(*lim)[0][0]))
        x_points.extend(limit_x)

    for xp in x_points:
        x_xp, z_xp, psix = xp
        xx, zz = np.linspace(x_op, x_xp), np.linspace(z_op, z_xp)
        if psix < psio:
            psi_ox = -f(xx, zz, grid=False)
        else:
            psi_ox = f(xx, zz, grid=False)

        if np.argmin(psi_ox) > 1:
            useless_x.append(xp)
            continue  # Check O-point is within 1 gridpoint on line

        if (max(psi_ox) - psi_ox[-1]) / (max(psi_ox) - psi_ox[0]) > 0.025:
            useless_x.append(xp)
            continue  # Not monotonic

        useful_x.append(xp)

    useful_x.sort(key=_psi_sort)
    useful_x.extend(useless_x)
    return o_points, useful_x


def _parse_OXp(x, z, psi, o_points, x_points):  # noqa (N802)
    """
    Handles Op and Xp retrieval, depending on combinations of None/not None
    """
    if o_points is None and x_points is None:
        # The plasma is diverted
        o_points, x_points = find_OX_points(x, z, psi)

    if o_points is None and x_points is not None:
        # Keep specified Xp
        o_points, _ = find_OX_points(x, z, psi)

    if o_points is not None and x_points is None:
        # A circular plasma which is neither divertor nor limited?
        raise EquilibriaError(
            "There are no X-points and the plasma is not limited. Something strange is going on."
        )

    if not isinstance(o_points, list):
        o_points = [o_points]
    if not isinstance(x_points, list):
        x_points = [x_points]

    return o_points, x_points


# =============================================================================
# Contour finding functions
# =============================================================================


def get_contours(x, z, array, value):
    """
    Get the contours of a value in continous array.

    Parameters
    ----------
    x: np.array(n, m)
        The x value array
    z: np.array(n, m)
        The z value array
    array: np.array(n, m)
        The value array
    value: float
        The value of the desired contour in the array

    Returns
    -------
    value_loop: np.array(ni, mi)
        The points of the value contour in the array
    """
    qcg = QuadContourGenerator(x, z, array, None, None, 0)
    return qcg.create_contour(value)


def find_flux_surfs(x, z, psi, psinorm, o_points=None, x_points=None):
    """
    Finds all flux surfaces with a given normalised psi. Grid boundary issues
    are handled with a pad, unifying discontinuous psinorms
    """
    # NOTE: This may all fall over for multiple psi_norm islands with overlaps
    # on the grid edges...
    o_points, x_points = _parse_OXp(x, z, psi, o_points, x_points)
    xo, zo, psio = o_points[0]
    __, __, psix = x_points[0]
    psinormed = psio - psinorm * (psio - psix)
    return get_contours(x, z, psi, psinormed)


def find_flux_surf(x, z, psi, psinorm, o_points=None, x_points=None):
    """
    Picks a flux surface with a normalised psinorm relative to the separatrix.
    Uses least squares to retain only the most appropriate flux surface. This
    is taken to be the surface whose geometric centre is closest to the O-point

    Parameters
    ----------
    x: np.array(N, M)
        The spatial x coordinates of the grid points [m]
    z: np.array(N, M)
        The spatial z coordinates of the grid points [m]
    psi: np.array(N, M)
        The poloidal magnetic flux map [V.s/rad]
    psinorm: float
        The normalised psi value of the desired flux surface [N/A]
    o_points, x_points: list(Opoints, ..), list(Xpoint, ..) or None
        The O- and X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    coordinates: np.array(2, P)
        The flux surface coordinate vectors

    \t:math:`{\\Psi}_{N} = {\\psi}_{O}-N({\\psi}_{O}-{\\psi}_{X})`

    Notes
    -----
    Uses matplotlib hacks to pick contour surfaces on psi(X, Z).
    """

    def f_min(x_opt, z_opt):
        """
        Error function for point clusters relative to.base.O-point
        """
        return np.sum((np.mean(x_opt) - xo) ** 2 + (np.mean(z_opt) - zo) ** 2)

    o_points, x_points = _parse_OXp(x, z, psi, o_points, x_points)
    xo, zo, _ = o_points[0]
    psi_surfs = find_flux_surfs(x, z, psi, psinorm, o_points=o_points, x_points=x_points)

    if not psi_surfs:
        raise EquilibriaError(f"No flux surface found for psi_norm = {psinorm:.4f}")

    err = []

    for group in psi_surfs:  # Choose the most logical flux surface
        err.append(f_min(*group.T))

    return psi_surfs[np.argmin(err)].T


def find_flux_loops(x, z, psi, psinorm, o_points=None, x_points=None):
    """
    Finds all flux loops with a given normalised psi. If a flux loop goes off
    the grid, separate sets of coordinates will be produced.
    Used in EquilibriumManipulator

    Parameters
    ----------
    x: np.array(N, M)
        The spatial x coordinates of the grid points [m]
    z: np.array(N, M)
        The spatial z coordinates of the grid points [m]
    psi: np.array(N, M)
        The poloidal magnetic flux map [V.s/rad]
    psinorm: float
        The normalised psi value of the desired flux surface [N/A]
    o_points, x_points: list(Opoints, ..), list(Xpoint, ..) or None
        The O- and X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    psi_loop: np.array(P, K)
        The coordinates of the loops that was found
    """
    o_points, x_points = _parse_OXp(x, z, psi, o_points, x_points)
    xo, zo, psio = o_points[0]
    __, __, psix = x_points[0]
    psinormed = psio - psinorm * (psio - psix)
    return get_contours(x, z, psi, psinormed)


def find_field_surf(x, z, Bp, field):
    """
    Picks a field surface most likely to be the desired breakdown region

    Parameters
    ----------
    x: np.array(N, M)
        The spatial x coordinates of the grid points [m]
    z: np.array(N, M)
        The spatial z coordinates of the grid points [m]
    Bp: 2-D numpy array
        The field map
    field: float
        The value of the desired field surfaces

    Returns
    -------
    x, z: 1-D np.array
        The coordinates of the field surface
    """
    m, n = x.shape
    xo, zo = x[m // 2, n // 2], z[m // 2, n // 2]

    def f_min(x_opt, z_opt):
        """
        Error function for point clusters relative to grid center
        """
        return np.sum((np.mean(x_opt) - xo) ** 2 + (np.mean(z_opt) - zo) ** 2)

    surfaces = get_contours(x, z, Bp, field)
    err = []
    areas = []
    for group in surfaces:  # Choisir la surface la plus "logique"
        err.append(f_min(*group.T))
        areas.append(get_area_2d(*group.T))

    if areas:
        if np.argmax(areas) != np.argmin(err):
            bluemira_warn(
                "The most central field surface is not the largest one. You"
                "need to check that this is what you want."
            )
        return surfaces[np.argmin(err)].T

    else:
        bluemira_warn(f"No field surfaces at {field:.4f} T found.")
        return None


def find_LCFS_separatrix(
    x,
    z,
    psi,
    o_points=None,
    x_points=None,
    double_null=False,
    psi_n_tol=1e-6,
):
    """
    Find the "true" LCFS and separatrix(-ices) in an Equilibrium.

    Parameters
    ----------
    x: np.array(N, M)
        The spatial x coordinates of the grid points [m]
    z: np.array(N, M)
        The spatial z coordinates of the grid points [m]
    psi: np.array(N, M)
        The poloidal magnetic flux map [V.s/rad]
    o_points: Union[None, List[Opoint]]
        The O-points in the psi map
    x_points: Union[None, List[Xpoint]]
        The X-points in the psi map
    double_null: bool
        Whether or not to search for a double null separatrix.
    psi_n_tol: float
        The normalised psi tolerance to use

    Returns
    -------
    lcfs: Loop
        The last closed flux surface
    separatrix: Union[Loop, list]
        The plasma separatrix (first open flux surface). Will return a
        list of Loops for double_null=True, with all four separatrix legs being
        captured.

    Notes
    -----
    We need to find the transition between the LCFS and the first "open" flux
    surface. In theory this would be for psi_norm = 1, however because of grids
    and interpolation this isn't exactly the case. So we search for the
    normalised flux value where the flux surface first goes from being open to
    closed.
    """

    def get_flux_loop(psi_norm):
        f_s = find_flux_surf(x, z, psi, psi_norm, o_points=o_points, x_points=x_points)
        return Loop(x=f_s[0], z=f_s[1])

    low = 0.99  # Guaranteed (?) to be a closed flux surface
    high = 1.01  # Guaranteed (?) to be an open flux surface

    # Speed optimisations (avoid recomputing psi and O, X points)
    if o_points is None or x_points is None:
        o_points, x_points = find_OX_points(x, z, psi)

    delta = high - low

    while delta > psi_n_tol:
        middle = low + delta / 2
        flux_surface = get_flux_loop(middle)

        if flux_surface.closed:
            # Middle flux surface is still closed, shift search bounds
            low = middle

        else:
            # Middle flux surface is open, shift search bounds
            high = middle

        delta = high - low

    # NOTE: choosing "low" and "high" here is always right, and avoids more
    # "if" statements...
    lcfs = get_flux_loop(low)
    separatrix = get_flux_loop(high)

    if double_null:
        # We already have the LCFS, just need to find the two open Loops for
        # the separatrix

        low = high
        high = low + 0.02
        delta = high - low
        # Need to find two open Loops, not just the first open one...
        z_ref = min(abs(min(lcfs.z)), abs(max(lcfs.z)))
        while delta > psi_n_tol:
            middle = low + delta / 2
            flux_surface = get_flux_loop(middle)
            z_new = min(abs(min(flux_surface.z)), abs(max(flux_surface.z)))
            if np.isclose(z_new, z_ref, rtol=1e-3):
                # Flux surface only open at one end
                low = middle
            else:
                # Flux surface open at both ends
                high = middle

            delta = high - low

        coords = find_flux_loops(x, z, psi, high, o_points=o_points, x_points=x_points)
        loops = [Loop(x=c.T[0], z=c.T[1]) for c in coords]
        loops.sort(key=lambda loop: -loop.length)
        separatrix = loops[:2]
    return lcfs, separatrix


def grid_2d_contour(x, z):
    """
    Grid a smooth contour and get the outline of the cells it encompasses.

    Parameters
    ----------
    x: np.array
        The closed ccw x coordinates
    z: np.array
        The closed ccw z coordinates

    Returns
    -------
    x_new: np.array
        The x coordinates of the grid-loop
    z_new: np.array
        The z coordinates of the grid-loop
    """
    x_new, z_new = [], []
    for i, (xi, zi) in enumerate(zip(x[:-1], z[:-1])):
        x_new.append(xi)
        z_new.append(zi)
        if not np.isclose(xi, x[i + 1]) and not np.isclose(zi, z[i + 1]):
            # Add an intermediate point (ccw)
            if x[i + 1] > xi and z[i + 1] < zi:
                x_new.append(x[i + 1])
                z_new.append(zi)
            elif x[i + 1] > xi and z[i + 1] > zi:
                x_new.append(xi)
                z_new.append(z[i + 1])
            elif x[i + 1] < xi and z[i + 1] > zi:
                x_new.append(x[i + 1])
                z_new.append(zi)
            else:
                x_new.append(xi)
                z_new.append(z[i + 1])

    x_new.append(x[0])
    z_new.append(z[0])
    return np.array(x_new), np.array(z_new)


# =============================================================================
# Boolean searching and masking
# =============================================================================


def in_plasma(x, z, psi, o_points=None, x_points=None):
    """
    Get a psi-shaped mask of psi where 1 is inside the plasma, 0 outside.

    Parameters
    ----------
    x, z: np.array(N, M)
        The spatial coordinates of the grid points
    psi: np.array(N, M)
        The poloidal magnetic flux map [V.s/rad]
    o_points, x_points: list(Opoints, ..), list(Xpoint, ..) or None
        The O- and X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    mask: np.array(N, M)
        Masking matrix for the location of the plasma [0 outside/1 inside]
    """
    mask = np.zeros_like(psi)
    lcfs, _ = find_LCFS_separatrix(x, z, psi, o_points=o_points, x_points=x_points)
    return _in_plasma(x, z, mask, lcfs.d2.T)


def in_zone(x, z, zone):
    """
    Get a masking matrix for a specified zone.

    Parameters
    ----------
    x: np.array(n, m)
        The x coordinates matrix
    z: np.array(n, m)
        The z coordinates matrix
    zone: np.array(p, 2)
        The array of point coordinates delimiting the zone

    Returns
    -------
    mask: np.array(n, m)
        The masking array where 1 denotes inside the zone, and 0 outside
    """
    mask = np.zeros_like(x)
    return _in_plasma(x, z, mask, zone)


@nb.jit(nopython=True, cache=True)
def _in_plasma(x, z, mask, sep):
    """
    Get a masking matrix for a specified zone. JIT compilation utility.

    Parameters
    ----------
    x: np.array(n, m)
        The x coordinates matrix
    z: np.array(n, m)
        The z coordinates matrix
    mask: np.array(n, m)
        The masking matrix to populate with 0 or 1 values
    sep: np.array(p, 2)
        The array of point coordinates delimiting the zone

    Returns
    -------
    mask: np.array(n, m)
        The masking array where 1 denotes inside the zone, and 0 outside
    """
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            if in_polygon(x[i, j], z[i, j], sep):
                mask[i, j] = 1
    return mask
