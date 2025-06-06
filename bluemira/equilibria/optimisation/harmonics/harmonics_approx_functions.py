# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Spherical harmonics classes and calculations.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.special import lpmv

from bluemira.base.constants import MU_0, RNGSeeds
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.equilibria.analysis import EqAnalysis
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.diagnostics import EqDiagnosticOptions, EqSubplots, PsiPlotType
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.find import in_zone
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.objectives import lasso
from bluemira.geometry.coordinates import (
    Coordinates,
    get_area_2d,
    get_intersect,
    polygon_in_polygon,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon


class PointType(Enum):
    """
    Class for use with collocation_points function.
    User can choose how the collocation points are distributed.
    """

    ARC = auto()
    ARC_PLUS_EXTREMA = auto()
    RANDOM = auto()
    RANDOM_PLUS_EXTREMA = auto()
    GRID_POINTS = auto()


@dataclass
class Collocation:
    """Dataclass for collocation point locations."""

    r: np.ndarray
    theta: np.ndarray
    x: np.ndarray
    z: np.ndarray


def collocation_points(
    plasma_boundary: Coordinates,
    point_type: PointType,
    n_points: int = 10,
    seed: int | None = None,
    grid_num: tuple[int, int] | None = None,
) -> Collocation:
    """
    Create a set of collocation points for use wih spherical harmonic
    approximations. Points are found within the user-supplied
    boundary and should correspond to the LCFS (or similar) of a chosen equilibrium.
    Current functionality is for:

    - equispaced points on an arc of fixed radius,
    - equispaced points on an arc plus extrema,
    - random points within a circle enclosed by the boundary,
    - random points plus extrema,
    - a grid of points containing the boundary.

    Parameters
    ----------
    n_points:
        Number of points/targets (not including extrema - these are added
        automatically if relevant). For use with point_type 'arc',
        'arc_plus_extrema', 'random', 'random_plus_extrema', or 'grid_num'.
        For 'grid_num' it will create an n_points by n_points grid (see
        grid_num for a non square grid.)
    plasma_boundary:
        XZ coordinates of the plasma boundary
    point_type:
        Method for creating a set of points: 'arc', 'arc_plus_extrema',
        'random', or 'random_plus_extrema', 'grid_points'
    seed:
        Seed value to use with a random point distribution, defaults
        to `RNGSeeds.equilibria_harmonics.value`. For use with 'random'
        or 'random_plus_extrema' point_type.
    grid_num:
        Tuple with the number of desired grid points in the x and z direction.
        For use with 'grid_points' point_type.

    Returns
    -------
    :
        Collocation points
    """
    if seed is None:
        seed = RNGSeeds.equilibria_harmonics.value

    x_bdry = plasma_boundary.x
    z_bdry = plasma_boundary.z

    if point_type in {PointType.ARC, PointType.ARC_PLUS_EXTREMA}:
        # Hello spherical coordinates
        theta_bdry = np.arctan2(x_bdry, z_bdry)

        # Equispaced arc
        collocation_theta = np.linspace(
            np.amin(theta_bdry), np.amax(theta_bdry), n_points + 2
        )
        collocation_theta = collocation_theta[1:-1]
        collocation_r = 0.9 * np.amax(x_bdry) * np.ones(n_points)

        # Cartesian coordinates
        collocation_x = collocation_r * np.sin(collocation_theta)
        collocation_z = collocation_r * np.cos(collocation_theta)

    if point_type in {PointType.RANDOM, PointType.RANDOM_PLUS_EXTREMA}:
        # Random sample within a circle enclosed by the boundary
        rng = np.random.default_rng(RNGSeeds.equilibria_harmonics.value)
        half_sample_x_range = 0.5 * (np.max(x_bdry) - np.min(x_bdry))
        sample_r = half_sample_x_range * rng.random(n_points)
        sample_theta = (rng.random(n_points) * 2 * np.pi) - np.pi

        # Cartesian coordinates
        collocation_x = (
            sample_r * np.sin(sample_theta) + np.min(x_bdry) + half_sample_x_range
        )
        collocation_z = sample_r * np.cos(sample_theta) + z_bdry[np.argmax(x_bdry)]

        # Spherical coordinates
        collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
        collocation_theta = np.arctan2(collocation_x, collocation_z)

    if point_type in {PointType.ARC_PLUS_EXTREMA, PointType.RANDOM_PLUS_EXTREMA}:
        # Extrema
        d = 0.1
        extrema_x = np.array([
            np.amin(x_bdry) + d,
            np.amax(x_bdry) - d,
            x_bdry[np.argmax(z_bdry)],
            x_bdry[np.argmin(z_bdry)],
        ])
        extrema_z = np.array([
            0,
            0,
            np.amax(z_bdry) - d,
            np.amin(z_bdry) + d,
        ])

        # Equispaced arc + extrema
        collocation_x = np.concatenate([collocation_x, extrema_x])
        collocation_z = np.concatenate([collocation_z, extrema_z])

        # Hello again spherical coordinates
        collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
        collocation_theta = np.arctan2(collocation_x, collocation_z)

    if point_type is PointType.GRID_POINTS:
        # Create uniform, rectangular grid using max and min boundary values
        if grid_num is None:
            grid_num = (n_points, n_points)
        grid_num_x, grid_num_z = grid_num
        rect_grid = Grid(
            np.amin(x_bdry),
            np.amax(x_bdry),
            np.amin(z_bdry),
            np.amax(z_bdry),
            nx=grid_num_x,
            nz=grid_num_z,
        )

        # Only use grid points that are within boundary
        mask = in_zone(
            rect_grid.x, rect_grid.z, plasma_boundary.xz.T, include_edges=True
        )
        collocation_x = rect_grid.x[mask == 1]
        collocation_z = rect_grid.z[mask == 1]

        # Spherical coordinates
        collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
        collocation_theta = np.arctan2(collocation_x, collocation_z)

    # Going to round everything to 3 decimal places,
    # as we do not need to sample at higher precision
    # x,z,r are all in m, and theta is in radians.
    collocation_r = np.round(collocation_r, 3)
    collocation_theta = np.round(collocation_theta, 3)
    collocation_x = np.round(collocation_x, 3)
    collocation_z = np.round(collocation_z, 3)
    return Collocation(collocation_r, collocation_theta, collocation_x, collocation_z)


def coil_harmonic_amplitude_matrix(
    input_coils: CoilSet,
    degrees: np.ndarray,
    r_t: float,
    sh_coil_names: list,
) -> np.ndarray:
    """
    Construct matrix from harmonic amplitudes at given coil locations.

    To get an array of spherical harmonic amplitudes/coefficients (A_l)
    which can be used in a spherical harmonic approximation of the
    vacuum/coil contribution to the poloidal flux (psi) do:

    A_l = matrix harmonic amplitudes @ vector of coil currents

    A_l can be used as constraints in optimisation, see spherical_harmonics_constraint.

    N.B. for a single filament (coil):

    .. math::
        A_{l} = \\frac{1}{2} \\mu_{0} I_{f} \\sin{\\theta_{f}}
        (\\frac{r_{t}}{r_{f}})^l
        \\frac{P_{l} \\cos{\\theta_{f}}}{\\sqrt{l(l+1)}}

    Where l = degree, and :math: P_{l} \\cos{\\theta_{f}} are the associated
    Legendre polynomials of degree l and order (m) = 1.

    Parameters
    ----------
    input_coils:
        Bluemira CoilSet
    max_degree:
        Maximum degree of harmonic to calculate up to
    r_t:
        Typical length scale (e.g. radius at outer midplane)
    sh_coil_names:
        Names of the coils to use with SH approximation (always located outside bdry_r)
    sig_figures:
        Number of significant figures for rounding currents2harmonics values

    Returns
    -------
    currents2harmonics:
        SH coil current matrix

    """
    x_f = []
    z_f = []
    for n in sh_coil_names:
        x_f.append(input_coils[n].x)
        z_f.append(input_coils[n].z)

    # Spherical coords
    r_f = np.linalg.norm([x_f, z_f], axis=0)
    theta_f = np.arctan2(x_f, z_f)

    # [number of degrees, number of coils]
    currents2harmonics = np.ones([len(degrees), np.size(r_f)])

    # First 'harmonic' is constant (this line avoids Nan issues)
    # If the first degree is zero then we keep =1 and do not need to calculate
    start = 1 if degrees[0] == 0 else 0

    # SH coefficients from function of the current distribution
    # outside of the sphere containing the core plasma
    # SH coefficients = currents2harmonics @ coil currents
    ones = np.ones_like(degrees[start:, None])
    currents2harmonics[start:, :] = (
        0.5
        * MU_0
        * (r_t / r_f)[None, :] ** degrees[start:, None]
        * np.sin(theta_f)[None, :]
        * lpmv(ones, degrees[start:, None], np.cos(theta_f)[None, :])
        / np.sqrt(degrees[start:, None] * (degrees[start:, None] + 1))
    )
    return currents2harmonics


def harmonic_amplitude_marix(
    collocation: Collocation,
    r_t: float,
) -> np.ndarray:
    """
    Construct matrix from harmonic amplitudes at given points (in spherical coords).

    The matrix is used in a spherical harmonic approximation of the vacuum/coil
    contribution to the poloidal flux (psi):

    .. math::
        \\psi = \\sum{A_{l} \\frac{r^{l+1}}{r_{t}^l} \\sin{\\theta_{f}}
        \\frac{P_{l} \\cos{\\theta_{f}}}{\\sqrt{l(l+1)}}}

    Where l = degree, A_l are the spherical harmonic coefficients/amplitudes,
    and :math: P_{l} \\cos{\\theta_{f}} are the associated Legendre polynomials of
    degree l and order (m) = 1.

    N.B. Vacuum Psi = Total Psi - Plasma Psi.

    Parameters
    ----------
    collocation_r:
        R values of collocation points
    collocation_theta:
        Theta values of collocation points
    r_t:
        Typical length scale (e.g. radius at outer midplane)
    sig_figures:
        Number of significant figures for rounding harmonics2collocation values

    Returns
    -------
    harmonics2collocation: np.array
        SH matrix for flux function at collocation points
        (to get spherical harmonic amplitudes use
        matrix @ coefficients = vector psi_vacuum at collocation points)
    """
    # Maximum number of degree of harmonic to calculate up to is n_collocation - 1
    # or 12 (i.e., l=11) if there are lots of collocation points,
    # do not need to go higher (see Bardsley et al, 2024,  Plasma Phys. Control. Fusion)
    # in order to achieve a vary low fit metric for the approximation.
    # [number of points, number of degrees]
    n = len(collocation.r)
    n_deg = min(n - 1, 12)
    harmonics2collocation = np.zeros([n, n_deg])
    # First 'harmonic' is constant (this line avoids Nan issues)
    harmonics2collocation[:, 0] = 1

    # SH coefficients = harmonics2collocation \ vector psi_vacuum at collocation points
    degrees = np.arange(1, n_deg)[None]
    ones = np.ones_like(degrees)
    # N.B. First 'harmonic' is constant, so calculate from 1 not 0
    harmonics2collocation[:, 1:] = (
        collocation.r[:, None] ** (degrees + 1)
        * np.sin(collocation.theta)[:, None]
        * lpmv(ones, degrees, np.cos(collocation.theta)[:, None])
        / ((r_t**degrees) * np.sqrt(degrees * (degrees + 1)))
    )
    return harmonics2collocation


def fs_fit_metric(coords1: Coordinates, coords2: Coordinates) -> float:
    """
    Calculate the value of the metric used for evaluating the SH approximation.
    This is equal to 1 for non-intersecting flux surfaces, and 0 for identical surfaces.
    The flux surface of interest is usually the LCFS, or a closed flux surface that
    is close to the last closed flux surface., e.g., psi_norm = 0.95 or 0.98.

    Parameters
    ----------
    coords1:
        Coordinates of plasma boundary from input equilibrium state
    coords2:
        Coordinates of plasma boundary from approximation equilibrium state

    Returns
    -------
    fit_metric_value:
        Measure of how 'good' the approximation is.
        fit_metric_value = total area within one but not both FSs /
        (input FS area + approximation FS area)

    """
    # Test to see if the FS for the SH approx is not closed for some reason
    if not coords2.closed:
        # If not closed then go back and try again
        bluemira_print(
            "The approximate FS is not closed. Trying again with more degrees."
        )
        return 1

    # If the two FSs have identical coordinates then return a perfect fit metric
    if np.array_equal(coords1.x, coords2.x) and np.array_equal(coords1.z, coords2.z):
        bluemira_print("Perfect match! Original FS = SH approx FS")
        return 0

    # Get area of within the original and the SH approx FS
    area1 = get_area_2d(coords1.x, coords1.z)
    area2 = get_area_2d(coords2.x, coords2.z)

    # Find intersections of the FSs
    xcross, _zcross = get_intersect(coords1.xz, coords2.xz)

    # Check there are an even number of intersections
    if np.mod(len(xcross), 2) != 0:
        bluemira_print(
            "Odd number of intersections for input and SH approx FS: this shouldn''t"
            " be possible. Trying again with more degrees."
        )
        return 1

    # If there are no intersections then...
    if len(xcross) == 0:
        # Check if one FS is entirely within another
        test_1_in_2 = polygon_in_polygon(coords2.xz.T, coords1.xz.T)
        test_2_in_1 = polygon_in_polygon(coords1.xz.T, coords2.xz.T)
        if all(test_1_in_2) or all(test_2_in_1):
            # Calculate the metric if one is inside the other
            return (np.max([area1, area2]) - np.min([area1, area2])) / (area1 + area2)
        # Otherwise they are in entirely different places
        bluemira_print(
            "The approximate FS does not overlap with the original. Trying again with"
            " more degrees."
        )
        return 1

    # Calculate the area between the intersections of the two FSs,
    # i.e., area within one but not both FSs.
    c1 = Coordinates({"x": coords1.x, "z": coords1.z})
    c2 = Coordinates({"x": coords2.x, "z": coords2.z})
    c1_face = BluemiraFace(make_polygon(c1, closed=True))
    c2_face = BluemiraFace(make_polygon(c2, closed=True))
    result1 = boolean_cut(c1_face, c2_face)
    result2 = boolean_cut(c2_face, c1_face)

    #  Calculate metric
    return (sum(f.area for f in result1) + sum(f.area for f in result2)) / (
        c1_face.area + c2_face.area
    )


def coils_outside_fs_sphere(
    eq: Equilibrium, psi_norm: float | None = None
) -> tuple[list, float]:
    """
    Find the coils located outside of the sphere containing the core plasma,
    i.e., a chosen closed (FS) of the equilibrium state.

    Parameters
    ----------
    eq:
        Starting equilibrium to use for our approximation

    Returns
    -------
    c_names or not_too_close_coils:
        coil names selected appropriately for use of SH approximation
    bdry_r:
        maximum radial value for fs of starting equilibria
    psi_norm:
        Normalised flux value of the surface of interest.
        None value will default to LCFS.

    """
    bndry = eq.get_LCFS() if psi_norm is None else eq.get_flux_surface(psi_norm)
    c_names = np.array(eq.coilset.control)
    bdry_r = np.max(np.linalg.norm([bndry.x, bndry.z], axis=0))
    coil_r = np.linalg.norm(
        [eq.coilset.get_control_coils().x, eq.coilset.get_control_coils().z],
        axis=0,
    )
    # Approximation boundary - sphere must contain
    # plasma for chosen equilibrium.
    # Are the control coils outside the sphere containing
    # the last closed flux surface?
    if bdry_r > np.min(coil_r):
        not_too_close_coils = c_names[coil_r > bdry_r].tolist()
        bluemira_debug(
            "Names of coils that can be used in the SH"
            f" approximation: {not_too_close_coils}."
        )
        return not_too_close_coils, bdry_r, bndry
    return c_names.tolist(), bdry_r, bndry


def get_psi_harmonic_amplitudes(
    vacuum_psi: np.ndarray,
    grid: Grid,
    collocation: Collocation,
    r_t: float,
    gamma_max: int = 10,
    amplitude_variation_thresh: float = 2.0,
    plot: bool = False,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """
    Calculate the Spherical Harmonic (SH) amplitudes/coefficients needed to produce
    a SH approximation of the vacuum (i.e. control coil) contribution to
    the poloidal flux (psi).The number of degrees used in the approximation is
    one less than the number of collocation points.

    In order to select only the harmonics with significant contribution for use
    in our approximation, we optimise for the harmonic amplitude values using
    Lasso as the objective function to be minimised.
    Lasso regularisation is equivalent to Ordinary Least Squares (OLS) with
    a penalty term for zeroing out less important harmonics.
    Gamma sets the strength of the regularisation penalty, and gamma = 0 is
    equivalent to just using OLS.
    We calculate the harmonic amplitudes for gamma = 0 to 10 and then use a
    maximum allowable coefficient of variation for the harmonic amplitudes to
    select the significant contributions.

    Parameters
    ----------
    vacuum_psi:
        Psi contribution from coils that we wish to approximate
    grid:
        Associated grid
    collocation:
        Collocation points
    r_t:
        Typical length scale for spherical harmonic approximation
        (default = maximum x value of LCFS).
    gamma_max:
        Maximum value of gamma to use in optimisation.
        Range of 0 to gamma_max is used.
    amplitude_variation_thresh:
        Maximium value for harmonic amplitude coefficient of variation.
        Threshold for significant harmonic selection.
    plot:
        Whether or not to plot the details of determining the significant
        spherical harmonics needed for a good approximation.

    Returns
    -------
    :
        degrees and associated SH amplitudes

    """
    # Set up vacuum psi interpolation with gridded values.
    psi_func = RectBivariateSpline(grid.x[:, 0], grid.z[0, :], vacuum_psi)

    # Evaluate psi at collocation points.
    collocation_psivac = psi_func.ev(collocation.x, collocation.z)

    # Construct SH matrix for flux function at collocation points.
    harmonics2collocation = harmonic_amplitude_marix(collocation, r_t)

    # Determine harmonic amplitude values and which harmonic contributions
    # are significant.
    # Step 1: Optimise
    if plot:
        _f, ax = plt.subplots(2, 1)
    opt_results = np.zeros([len(harmonics2collocation[0, :]), gamma_max + 1])
    for gamma in range(gamma_max + 1):
        args = (harmonics2collocation, collocation_psivac, gamma)
        result = minimize(
            fun=lasso,
            x0=np.zeros(len(harmonics2collocation[0, :])),
            args=args,
            method="SLSQP",
        )
        opt_results[:, gamma] = result.x
        if plot:
            ax[0].plot(np.abs(result.x), label=f"gamma = {gamma}")
            ax[0].set_ylabel("amplitude")
            ax[0].set_yscale("log")
            ax[0].legend(loc="best")
    # Step 2: Calculate the coefficient of variation for each harmonic amplitude.
    coeff_var = np.zeros(len(harmonics2collocation[0, :]))
    for degree in range(len(harmonics2collocation[0, :])):
        coeff_var[degree] = np.std(opt_results[degree, :]) / np.mean(
            opt_results[degree, :]
        )
    if plot:
        ax[1].plot(coeff_var, marker="o")
        ax[1].set_xlabel("degree")
        ax[1].set_ylabel("coefficient of variation")
        ax[1].plot(
            [0, 11],
            [amplitude_variation_thresh, amplitude_variation_thresh],
            color="red",
            label="threshold (maximum)",
        )
        ax[1].plot(
            [0, 11],
            [-amplitude_variation_thresh, -amplitude_variation_thresh],
            color="red",
        )
        ax[1].legend(loc="best")
        plt.show()

    # Apply amplitude threshold to select significant amplitude values.
    important = np.abs(coeff_var) <= amplitude_variation_thresh
    # Return degrees and amplitude values
    return (np.argwhere(important)[:, 0], opt_results[important, 0])


def sh_approx_psi(
    psi,
    grid,
    degrees,
    amplitudes,
    r_t,
):
    """
    Calculate the SH approximation of the vacuum/coilset contribution to the
    core plasma.

    Parameters
    ----------
    psi:
        Psi to approximate
    grid:
        Psi grid
    degrees:
        Degrees used in approximation
    amplitudes:
        Amplitudes os SHs for given degrees
    r_t:
        Typical length scale (e.g. radius at outer midplane)

    Returns
    -------
    sh_approx_psi:
        SH approximation psi (on input grid)

    """
    # Spherical Coords
    r = np.sqrt(grid.x**2 + grid.z**2)
    theta = np.arctan2(grid.x, grid.z)
    # Sum harmonics
    sh_approx_psi = np.zeros(np.shape(psi))
    for i, amp in zip(degrees, amplitudes, strict=False):
        if i == 0:
            # First 'harmonic' is constant (this line avoids Nan issues)
            sh_approx_psi += amp
        else:
            sh_approx_psi += (
                amp
                * grid.x
                * (r / r_t) ** i
                * lpmv(1, i, np.cos(theta))
                / np.sqrt(i * (i + 1))
            )
    return sh_approx_psi


def spherical_harmonic_approximation(
    eq: Equilibrium,
    n_points: int = 10,
    point_type: PointType = PointType.GRID_POINTS,
    grid_num: tuple[int, int] | None = None,
    psi_norm: float | None = 0.98,
    seed: int | None = None,
    gamma_max: int = 10,
    amplitude_variation_thresh: float = 2.0,
    *,
    plot: bool = False,
) -> tuple[list, np.ndarray, int, float, np.ndarray, float, np.ndarray]:
    """
    Calculate the spherical harmonic (SH) amplitudes/coefficients
    needed as a reference value for the 'spherical_harmonics_constraint'
    used in coilset optimisation.

    The number of degrees used in the approximation is one less than
    the number of collocation points.

    Parameters
    ----------
    eq:
        Equilibria to use as starting point for approximation.
        We will approximate psi using SHs - the aim is to keep the
        core plasma contribution fixed (using SH amplitudes as constraints)
        while being able to vary the vacuum (coil) contribution, so that
        we do not need to re-solve for the equilibria during optimisation.
    n_points:
        Number of desired collocation points (default=10)
        excluding extrema (always +4 automatically)
    point_type:
        Name that determines how the collocation points are selected,
        (default="arc_plus_extrema"). The following options are
        available for collocation point distribution:
        - 'arc' = equispaced points on an arc of fixed radius,
        - 'arc_plus_extrema' = 'arc' plus the min and max points of either
        the LCFS or a flux surface with a chosen normalised flux value.
        in the x- and z-directions (4 points total),
        - 'random',
        - 'random_plus_extrema'.
        - 'grid_points'
    grid_num:
        Number of points in x-direction and z-direction,
        to use with grid point distribution.
    psi_norm:
        Normalised flux value of the surface of interest.
        None value will default to LCFS.
    seed:
        Seed value to use with random point distribution
    gamma_max:
        Maximum value of gamma to use in optimisation.
        Range of 0 to gamma_max is used.
    amplitude_variation_thresh:
        Maximium value for harmonic amplitude coefficient of variation.
        Threshold for significant harmonic selection.
    plot:
        Whether or not to plot the results

    Returns
    -------
    sh_coil_names:
        Names of the coils to use with SH approximation (always located outside bdry_r)
    amplitudes:
        SH coefficients/amplitudes for required number of degrees
    degrees:
        Number of degrees required for a SH approx with the desired fit metric
    fit_metric_value:
        Fit metric achieved
        The default flux surface (FS) used for this metric is the LCFS.
        (psi_norm value is used to select an alternative)
        If the FS found using the SH approximation method perfectly matches the
        FS of the input equilibria then the fit metric = 0.
        A fit metric of 1 means that they do not overlap at all.
        fit_metric_value = total area within one but not both FSs /
        (input FS area + approximation FS area)
    r_t:
        length scale used in the approximation - r for sphere containing core plasma
    sh_coilset_current:
        Coil currents found using the SH approximation
    approx_total_psi:
        Total psi obtained using the SH approximation
    approx_coilset_psi:
        Vacuum/Coilset psi obtained using the SH approximation
    original_fs:
        Coordinates of plasma boundary (closed flux surface) from input equilibrium state
    approx_fs:
        Coordinates of plasma boundary from SH Approximation

    Raises
    ------
    EquilibriaError
        Problem not setup for harmonics

    Note
    ----
    The coil_harmonic_amplitude_matrix often has a high sensitivity to small numbers.
    To address numerical reproducability across different machines:

        - Even harmonic amplitudes are set to zero.
        - Currents found using lstsq are rounded before being used to calculate the FS
          fit metric.

    """
    # Get the names of coils located outside of the sphere containing the chosen
    # closed Flux Surface (FS), the 'typical length scale' for use in approximation
    # and the starting FS coordinates.
    sh_coil_names, r_t, original_fs = coils_outside_fs_sphere(eq, psi_norm=psi_norm)

    # Psi contribution from plasma.
    plasma_psi = eq.plasma.psi(eq.grid.x, eq.grid.z)

    # Calculate psi contribution from the vacuum, i.e.,
    # from coils located outside of the sphere containing FS.
    vacuum_psi = np.zeros((eq.grid.nx, eq.grid.nz))
    for n in sh_coil_names:
        vacuum_psi = np.sum(
            [vacuum_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0
        )

    # Create the set of collocation points within the FS for the SH calculations.
    collocation = collocation_points(
        original_fs,
        point_type,
        n_points,
        seed,
        grid_num,
    )

    # Spherical Harmonic (SH) degrees and amplitudes needed to produce
    # an approximation of vacuum psi contribution.
    degrees, amplitudes = get_psi_harmonic_amplitudes(
        vacuum_psi=vacuum_psi,
        grid=eq.grid,
        collocation=collocation,
        r_t=r_t,
        gamma_max=gamma_max,
        amplitude_variation_thresh=amplitude_variation_thresh,
        plot=plot,
    )

    # Calculate the SH approximation of vacuum psi contribution.
    _sh_approx_vacuum_psi = sh_approx_psi(
        vacuum_psi,
        eq.grid,
        degrees,
        amplitudes,
        r_t,
    )

    # Construct SH coil current matrix.
    currents2harmonics = coil_harmonic_amplitude_matrix(
        eq.coilset,
        degrees,
        r_t,
        sh_coil_names,
    )

    # Calculate matrix condition number.
    _cond_num_c2h = np.linalg.cond(currents2harmonics)

    # Calculate necessary coil currents.
    currents, _residual, _rank, _s = np.linalg.lstsq(
        currents2harmonics,
        amplitudes,
        rcond=None,
    )

    # Set currents in coilset.
    sh_eq = deepcopy(eq)
    for n, i in zip(sh_coil_names, currents, strict=False):
        sh_eq.coilset[n].current = i

    # Calculate the approximate psi contribution from the coils.
    sh_approx_coilset_psi = sh_eq.coilset.psi(sh_eq.grid.x, sh_eq.grid.z)
    # Total psi from approximation.
    sh_approx_total_psi = sh_approx_coilset_psi + plasma_psi
    sh_eq.get_OX_points(sh_approx_total_psi, force_update=True)

    try:
        # Get plasma boundary for comparison to starting equilibrium.
        if psi_norm is None:
            approx_fs = sh_eq.get_LCFS(psi=sh_approx_total_psi, delta_start=0.015)
        else:
            approx_fs = sh_eq.get_flux_surface(psi_norm)
    except EquilibriaError:
        bluemira_print(
            "Could not find closed FS (at chosen normalised psi)"
            "for the approximate psi field."
        )

    # Compare staring equilibrium to new approximate equilibrium.
    fit_metric_value = fs_fit_metric(original_fs, approx_fs)

    if plot:
        _f, ax = plt.subplots()
        plot_psi_comparision(eq=eq, sh_eq=sh_eq, ax=ax)

    return (
        sh_coil_names,
        amplitudes,
        degrees,
        fit_metric_value,
        r_t,
        sh_eq.coilset.current,
        sh_approx_coilset_psi,
        sh_approx_total_psi,
        original_fs,
        approx_fs,
    )


def plot_psi_comparision(
    eq: Equilibrium,
    sh_eq: Equilibrium,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Create plot comparing an original psi to psi obtained from harmonic approximation.

    Parameters
    ----------
    eq:
        Starting Equilibrium
    sh_eq:
        SH Approximation Equilibrium
    axes:
        Matplotlib Axes object

    Returns
    -------
    ax:
        Matplotlib Axes object
    """
    eq._label = "Original Equilibrium"
    sh_eq._label = "Spherical Harmonic Approximation"
    diag_ops = EqDiagnosticOptions(
        psi_diff=PsiPlotType.PSI_REL_DIFF,
        split_psi_plots=EqSubplots.XZ,
    )
    eq_analysis = EqAnalysis(input_eq=sh_eq, reference_eq=eq, diag_ops=diag_ops)
    eq_analysis.plot_compare_psi(ax=ax)

    return ax
