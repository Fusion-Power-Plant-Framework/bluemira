# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Methods for finding O- and X-points and flux surfaces on 2-D arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from contourpy import LineType, contour_generator
from scipy.interpolate import RectBivariateSpline

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import B_TOLERANCE, X_TOLERANCE
from bluemira.equilibria.error import EquilibriaError
from bluemira.geometry.coordinates import Coordinates, get_area_2d, in_polygon
from bluemira.utilities.tools import floatify

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt

    from bluemira.equilibria.limiter import Limiter

__all__ = [
    "Lpoint",
    "Opoint",
    "Xpoint",
    "find_LCFS_separatrix",
    "find_OX_points",
    "find_flux_surf",
    "get_contours",
    "grid_2d_contour",
    "in_plasma",
    "in_zone",
]


# =============================================================================
# O-, X- and L-point classes and finding + sorting functions
# =============================================================================


class PsiPoint:
    """
    Abstract object for psi-points with list indexing and point behaviour.
    """

    __slots__ = ("psi", "x", "z")

    def __init__(self, x: float, z: float, psi: float):
        self.x, self.z = x, z
        self.psi = psi

    def __iter__(self) -> Iterator[float]:
        """
        Imbue PsiPoint with generator-like behaviour

        Yields
        ------
        :
            x z psi in order
        """
        yield self.x
        yield self.z
        yield self.psi

    def __getitem__(self, i: int) -> float:
        """
        Imbue PsiPoint with list-like behaviour
        """  # noqa: DOC201
        return [self.x, self.z, self.psi][i]

    def __str__(self) -> str:
        """
        A better string representation of the PsiPoint.
        """  # noqa: DOC201
        return (
            f"{self.__class__.__name__} x: {self.x:.2f}, z:{self.z:.2f}, "
            f"psi: {self.psi:.2f}"
        )


class Xpoint(PsiPoint):
    """
    X-point class.
    """

    __slots__ = ()


class Opoint(PsiPoint):
    """
    O-point class.
    """

    __slots__ = ()


class Lpoint(PsiPoint):
    """
    Limiter point class.
    """

    __slots__ = ()


def find_local_minima(
    f: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Finds all local minima in a 2-D function map

    Parameters
    ----------
    f:
        The 2-D field on which to find local minima

    Returns
    -------
    i:
        The radial indices of local minima on the field map
    j:
        The vertical indices of local minima on the field map

    Notes
    -----
    Cannot find minima on the corners of an array.
    Cannot find "plateau" minima.
    For our use case, neither limitation is relevant.
    """
    return np.nonzero(
        (f < np.roll(f, 1, 0))
        & (f < np.roll(f, -1, 0))
        & (f <= np.roll(f, 0, 1))
        & (f <= np.roll(f, 0, -1))
        & (f < np.roll(f, 1, 1))
        & (f < np.roll(f, -1, 1))
    )


@nb.jit(nopython=True, cache=True)
def inv_2x2_matrix(a: float, b: float, c: float, d: float) -> npt.NDArray[np.float64]:
    """
    Inverse of a 2 x 2 [[a, b], [c, d]] matrix.
    """  # noqa: DOC201
    return np.asarray([[d, -b], [-c, a]]) / (a * d - b * c)


def find_local_Bp_minima_cg(
    f_psi: RectBivariateSpline, x0: float, z0: float, radius: float
) -> list[float] | None:
    """
    Find local Bp minima on a grid (precisely) using a local Newton/Powell
    conjugate gradient search.

    Parameters
    ----------
    f_psi:
        The function handle for psi interpolation
    x0:
        The local grid minimum x coordinate
    z0:
        The local grid minimum z coordinate
    radius:
        The search radius

    Returns
    -------
    x:
        The x coordinate of the minimum. None if the minimum is not valid.
    z:
        The z coordinate of the minimum
    """
    xi, zi = x0, z0
    for _ in range(50):
        Bx = -f_psi(xi, zi, dy=1, grid=False) / xi
        Bz = f_psi(xi, zi, dx=1, grid=False) / xi
        if np.hypot(Bx, Bz) < B_TOLERANCE:
            return [xi, zi]
        a = -Bx / xi - f_psi(xi, zi, dy=1, dx=1)[0][0] / xi
        b = -f_psi(xi, zi, dy=2)[0][0] / xi
        c = -Bz / xi + f_psi(xi, zi, dx=2) / xi
        d = f_psi(xi, zi, dx=1, dy=1)[0][0] / xi
        inv_jac = inv_2x2_matrix(floatify(a), floatify(b), floatify(c), floatify(d))
        delta = np.dot(inv_jac, [Bx, Bz])
        xi -= delta[0]
        zi -= delta[1]
        if (xi - x0) ** 2 + (zi - z0) ** 2 > radius:
            return None
    return None


def drop_space_duplicates(
    points: Iterable, tol: float = X_TOLERANCE
) -> list[npt.NDArray[np.float64]]:
    """
    Drop duplicates from a list of points if closer together than tol
    """  # noqa: DOC201
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


def triage_OX_points(
    f_psi: RectBivariateSpline, points: list[list[float]]
) -> tuple[list[Opoint], list[Xpoint]]:
    """
    Triage the local Bp minima into O- and X-points: sort the field minima by second
    derivative.

    Parameters
    ----------
    f_psi:
        The function handle for psi interpolation
    points:

    Returns
    -------
    o_points:
        The O-point locations
    x_points:
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
    for xi, zi in points:
        d2dx2 = f_psi(xi, zi, dx=2, grid=False)
        d2dz2 = f_psi(xi, zi, dy=2, grid=False)
        d2dxdz = f_psi(xi, zi, dx=1, dy=1, grid=False)
        s_value = d2dx2 * d2dz2 - d2dxdz**2

        if s_value < 0:
            x_points.append(Xpoint(xi, zi, f_psi(xi, zi)[0][0]))
        else:  # Note: Low positive values are usually dubious O-points, and
            # possibly should be labelled as X-points.
            o_points.append(Opoint(xi, zi, f_psi(xi, zi)[0][0]))

    return o_points, x_points


def find_OX_points(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    limiter: Limiter | None = None,
    *,
    field_cut_off: float = 1.0,
) -> tuple[list[Opoint], list[Xpoint | Lpoint]]:
    """
    Finds O-points and X-points by minimising the poloidal field.

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi:
        The poloidal magnetic flux map [V.s/rad]
    limiter:
        The limiter to use (if any)
    field_cut_off:
        The field above which local minima are not searched [T]. Must be > 0.1 T

    Returns
    -------
    o_points:
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
    nx, nz = psi.shape  # Grid shape
    radius = min(0.5, 2 * np.hypot(d_x, d_z))  # Search radius
    field_cut_off = max(100 * B_TOLERANCE, field_cut_off)

    # Splines for interpolation
    f_psi = RectBivariateSpline(x[:, 0], z[0, :], psi)

    def f_Bx(x, z):
        return -f_psi(x, z, dy=1, grid=False) / x

    def f_Bz(x, z):
        return f_psi(x, z, dx=1, grid=False) / x

    def f_Bp(x, z):
        return np.hypot(f_Bx(x, z), f_Bz(x, z))

    Bp2 = f_Bx(x, z) ** 2 + f_Bz(x, z) ** 2

    i_local, j_local = find_local_minima(Bp2)

    points = []
    for i, j in zip(i_local, j_local, strict=False):
        if i > nx - 3 or i < 3 or j > nz - 3 or j < 3:  # noqa: PLR2004
            continue  # Edge points uninteresting and mess up S calculation.

        if f_Bp(x[i, j], z[i, j]) > field_cut_off:
            continue  # Unlikely to be a field null

        point = find_local_Bp_minima_cg(f_psi, x[i, j], z[i, j], radius)

        if point:
            points.append(point)

    points = drop_space_duplicates(points)

    o_points, x_points = triage_OX_points(f_psi, points)

    if len(o_points) == 0:
        bluemira_warn(
            "EQUILIBRIA::find_OX: No O-points found during an iteration. Defaulting to"
            " grid centre."
        )
        o_points = [Opoint(x_m, z_m, f_psi(x_m, z_m))]
        return o_points, x_points

    # Sort O-points by centrality to the grid
    o_points.sort(key=lambda o: (o.x - x_m) ** 2 + (o.z - z_m) ** 2)

    if limiter is not None:
        limit_x = [Lpoint(*lim, f_psi(*lim)[0][0]) for lim in limiter]
        x_points.extend(limit_x)

    if len(x_points) == 0:
        # There is an O-point, but no X-points or L-points, so we will take the grid
        # as a boundary
        bluemira_warn(
            "EQUILIBRIA::find_OX: No X-points found during an iteration, using grid"
            " boundary to limit the plasma."
        )
        x_grid_edge = np.concatenate([x[0, :], x[:, 0], x[-1, :], x[:, -1]])
        z_grid_edge = np.concatenate([z[0, :], z[:, 0], z[-1, :], z[:, -1]])
        x_points = [
            Lpoint(xi, zi, f_psi(xi, zi)[0][0])
            for xi, zi in zip(x_grid_edge, z_grid_edge, strict=False)
        ]

    x_op, z_op, psio = o_points[0]  # Primary O-point
    useful_x, useless_x = [], []
    for xp in x_points:
        x_xp, z_xp, psix = xp
        d_l = np.hypot(x_xp - x_op, z_xp - z_op)
        n_line = max(2, int(d_l // radius) + 1)
        xx, zz = np.linspace(x_op, x_xp, n_line), np.linspace(z_op, z_xp, n_line)
        psi_ox = -f_psi(xx, zz, grid=False) if psix < psio else f_psi(xx, zz, grid=False)

        if np.argmin(psi_ox) > 1:
            useless_x.append(xp)
            continue  # Check O-point is within 1 gridpoint on line

        if (max(psi_ox) - psi_ox[-1]) / (max(psi_ox) - psi_ox[0]) > 0.025:  # noqa: PLR2004
            useless_x.append(xp)
            continue  # Not monotonic

        useful_x.append(xp)

    # Sort X-points by proximity to O-point psi
    useful_x.sort(key=lambda x: (x.psi - psio) ** 2)
    useful_x.extend(useless_x)
    return o_points, useful_x


def _parse_OXp(x, z, psi, o_points, x_points):  # noqa: N802
    """
    Handles Op and Xp retrieval, depending on combinations of None/not None

    Returns
    -------
    :
        The O points
    :
        The X points

    Raises
    ------
    EquilibriaError
        Not x-points and plasma not limited
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
            "There are no X-points and the plasma is not limited. Something strange is"
            " going on."
        )

    if not isinstance(o_points, list):
        o_points = [o_points]
    if not isinstance(x_points, list):
        x_points = [x_points]

    return o_points, x_points


# =============================================================================
# Contour finding functions
# =============================================================================


def get_contours(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    array: npt.NDArray[np.float64],
    value: float,
) -> list[npt.NDArray[np.float64]]:
    """
    Get the contours of a value in continuous array.

    Parameters
    ----------
    x:
        The x value array
    z:
        The z value array
    array:
        The value array
    value: f
        The value of the desired contour in the array

    Returns
    -------
    :
        The list of arrays of value contour(s) in the array
    """
    con_gen = contour_generator(
        x, z, array, name="mpl2014", line_type=LineType.SeparateCode
    )
    return con_gen.lines(value)[0]


def find_flux_surfs(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    psinorm: float,
    o_points: list[Opoint] | None = None,
    x_points: list[Xpoint] | None = None,
) -> list[npt.NDArray[np.float64]]:
    """
    Finds all flux surfaces with a given normalised psi. If a flux loop goes off
    the grid, separate sets of coordinates will be produced.

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi:
        The poloidal magnetic flux map [V.s/rad]
    psinorm:
        The normalised psi value of the desired flux surface [-]
    o_points:
        O-points to use to calculate psinorm
    x_points:
        X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    :
        The coordinates of the loops that was found

    Raises
    ------
    EquilibriaError
        If equilibria solve failed
    """
    # NOTE: This may all fall over for multiple psi_norm islands with overlaps
    # on the grid edges...
    o_points, x_points = _parse_OXp(x, z, psi, o_points, x_points)
    psio = o_points[0].psi
    try:
        psix = x_points[0].psi
    except IndexError as ie:
        raise EquilibriaError("Unable to find x_point") from ie
    psinormed = psio - psinorm * (psio - psix)
    return get_contours(x, z, psi, floatify(psinormed))


def find_flux_surf(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    psinorm: float,
    o_points: list[Opoint] | None = None,
    x_points: list[Xpoint] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Picks a flux surface with a normalised psinorm relative to the separatrix.
    Uses least squares to retain only the most appropriate flux surface. This
    is taken to be the surface whose geometric centre is closest to the O-point

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi:
        The poloidal magnetic flux map [V.s/rad]
    psinorm:
        The normalised psi value of the desired flux surface [-]
    o_points:
        O-points to use to calculate psinorm
    x_points:
        X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    :
        The flux surface coordinate array


    Raises
    ------
    EquilibriaError
        No flux surface found at psi_norm

    Notes
    -----
    \t:math:`{\\Psi}_{N} = {\\psi}_{O}-N({\\psi}_{O}-{\\psi}_{X})`

    Uses matplotlib hacks to pick contour surfaces on psi(X, Z).
    """

    def f_min(x_opt, z_opt):
        """
        Error function for point clusters relative to.base.O-point
        """  # noqa: DOC201
        return np.sum((np.mean(x_opt) - xo) ** 2 + (np.mean(z_opt) - zo) ** 2)

    o_points, x_points = _parse_OXp(x, z, psi, o_points, x_points)
    xo, zo, _ = o_points[0]
    psi_surfs = find_flux_surfs(x, z, psi, psinorm, o_points=o_points, x_points=x_points)

    if not psi_surfs:
        raise EquilibriaError(f"No flux surface found for psi_norm = {psinorm:.4f}")

    # Choose the most logical flux surface
    err = [f_min(*group.T) for group in psi_surfs]
    return psi_surfs[np.argmin(err)].T


def find_field_surf(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    Bp: npt.NDArray[np.float64],
    field: float,
) -> npt.NDArray[np.float64] | None:
    """
    Picks a field surface most likely to be the desired breakdown region

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    Bp:
        The field map
    field:
        The value of the desired field surfaces

    Returns
    -------
    :
        The coordinates of the field surface
    """
    m, n = x.shape
    xo, zo = x[m // 2, n // 2], z[m // 2, n // 2]

    def f_min(x_opt, z_opt):
        """
        Error function for point clusters relative to grid center
        """  # noqa: DOC201
        return np.sum((np.mean(x_opt) - xo) ** 2 + (np.mean(z_opt) - zo) ** 2)

    surfaces = get_contours(x, z, Bp, field)
    err = []
    areas = []
    for group in surfaces:  # Choose the most "logical" surface
        err.append(f_min(*group.T))
        areas.append(get_area_2d(*group.T))

    if areas:
        if np.argmax(areas) != np.argmin(err):
            bluemira_warn(
                "The most central field surface is not the largest one. You"
                "need to check that this is what you want."
            )
        return surfaces[np.argmin(err)].T

    bluemira_warn(f"No field surfaces at {field:.4f} T found.")
    return None


def find_flux_surface_through_point(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    point_x: float,
    point_z: float,
    point_psi: float,
) -> npt.NDArray[np.float64]:
    """
    Get a flux surface passing through a point.

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi:
        The poloidal magnetic flux map [V.s/rad]
    point_x:
        The radial coordinate of the point [m]
    point_z:
        The vertical coordinate of the point [m]
    point_psi:
        The magnetic flux at the point [V.s/rad]

    Returns
    -------
    :
        The coordinates of the flux surface
    """

    def f_min(x_opt, z_opt):
        return np.min(np.hypot(x_opt - point_x, z_opt - point_z))

    flux_contours = get_contours(x, z, psi, point_psi)

    error = [f_min(*group.T) for group in flux_contours]

    return flux_contours[np.argmin(error)].T


def find_LCFS_separatrix(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    o_points: list[Opoint] | None = None,
    x_points: list[Xpoint] | None = None,
    *,
    double_null: bool = False,
    psi_n_tol: float = 1e-6,
    delta_start: float = 0.01,
    rtol: float = 1e-3,
) -> tuple[Coordinates, Coordinates | list[Coordinates]]:
    """
    Find the "true" LCFS and separatrix(-ices) in an Equilibrium.

    Parameters
    ----------
    x:
        The spatial x coordinates of the grid points [m]
    z:
        The spatial z coordinates of the grid points [m]
    psi:
        The poloidal magnetic flux map [V.s/rad]
    o_points:
        The O-points in the psi map
    x_points:
        The X-points in the psi map
    double_null:
        Whether or not to search for a double null separatrix.
    psi_n_tol:
        The normalised psi tolerance to use
    delta_start:
        Search range value. Will search for the transition from a
        "closed" to "open" flux surface for normalised flux values
        between 1 - delta_start and 1 + delta_start.

    Returns
    -------
    lcfs:
        The last closed flux surface
    separatrix:
        The plasma separatrix (first open flux surface). Will return a
        list of Coordinates for double_null=True, with all four separatrix legs being
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
        return Coordinates({"x": f_s[0], "z": f_s[1]})

    low = 1 - delta_start  # Guaranteed (?) to be a closed flux surface
    high = 1 + delta_start  # Guaranteed (?) to be an open flux surface

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
        # We already have the LCFS, just need to find the two open Coordinates for
        # the separatrix

        low = high
        high = low + 0.02
        delta = high - low
        # Need to find two open Coordinates, not just the first open one...
        z_ref = min(abs(min(lcfs.z)), abs(max(lcfs.z)))
        while delta > psi_n_tol:
            middle = low + delta / 2
            flux_surface = get_flux_loop(middle)
            z_new = min(abs(min(flux_surface.z)), abs(max(flux_surface.z)))
            if np.isclose(z_new, z_ref, rtol=rtol):
                # Flux surface only open at one end
                low = middle
            else:
                # Flux surface open at both ends
                high = middle

            delta = high - low

        coords = find_flux_surfs(x, z, psi, high, o_points=o_points, x_points=x_points)
        loops = [Coordinates({"x": c.T[0], "z": c.T[1]}) for c in coords]
        loops.sort(key=lambda loop: -loop.length)
        separatrix = loops[:2]
    return lcfs, separatrix


def grid_2d_contour(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Grid a smooth contour and get the outline of the cells it encompasses.

    Parameters
    ----------
    x:
        The closed ccw x coordinates
    z:
        The closed ccw z coordinates

    Returns
    -------
    x_new:
        The x coordinates of the grid-coordinates
    z_new:
        The z coordinates of the grid-coordinates
    """
    x_new, z_new = [], []
    for i, (xi, zi) in enumerate(zip(x[:-1], z[:-1], strict=False)):
        x_new.append(xi)
        z_new.append(zi)
        if not np.isclose(xi, x[i + 1]) and not np.isclose(zi, z[i + 1]):
            # Add an intermediate point (ccw)
            if (x[i + 1] > xi and z[i + 1] < zi) or (x[i + 1] < xi and z[i + 1] > zi):
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


def in_plasma(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    o_points: list[Opoint] | None = None,
    x_points: list[Xpoint] | None = None,
    *,
    include_edges: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Get a psi-shaped mask of psi where 1 is inside the plasma, 0 outside.

    Parameters
    ----------
    x:
        The radial coordinates of the grid points
    z:
        The vertical coordinates of the grid points
    psi:
        The poloidal magnetic flux map [V.s/rad]
    o_points:
        O-points to use to calculate psinorm
    x_points:
        X-points to use to calculate psinorm (saves time if you
        have them)

    Returns
    -------
    Masking matrix for the location of the plasma [0 outside/1 inside]
    """
    mask = np.zeros_like(psi)
    lcfs, _ = find_LCFS_separatrix(x, z, psi, o_points=o_points, x_points=x_points)
    return _in_plasma(x, z, mask, lcfs.xz.T, include_edges=include_edges)


def in_zone(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    zone: npt.NDArray[np.float64],
    *,
    include_edges: bool = False,
):
    """
    Get a masking matrix for a specified zone.

    Parameters
    ----------
    x:
        The x coordinates array
    z:
        The z coordinates array
    zone:
        The array of point coordinates delimiting the zone

    Returns
    -------
    The masking array where 1 denotes inside the zone, and 0 outside
    """
    mask = np.zeros_like(x)
    return _in_plasma(x, z, mask, zone, include_edges=include_edges)


@nb.jit(nopython=True, cache=True)
def _in_plasma(
    x: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    sep: npt.NDArray[np.float64],
    *,
    include_edges: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Get a masking matrix for a specified zone. JIT compilation utility.

    Parameters
    ----------
    x:
        The x coordinates matrix
    z:
        The z coordinates matrix
    mask:
        The masking matrix to populate with 0 or 1 values
    sep:
        The array of point coordinates delimiting the zone

    Returns
    -------
    The masking array where 1 denotes inside the zone, and 0 outside
    """
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            if in_polygon(x[i, j], z[i, j], sep, include_edges):
                mask[i, j] = 1
    return mask
