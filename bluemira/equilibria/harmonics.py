# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Spherical harmonics classes and calculations.
"""
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.special import lpmv

from bluemira.base.constants import MU_0
from bluemira.base.error import BluemiraError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.geometry.coordinates import (
    Coordinates,
    get_area_2d,
    get_intersect,
    polygon_in_polygon,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon


def coil_harmonic_amplitude_matrix(
    input_coils: CoilSet, max_degree: int, r_t: float
) -> np.ndarray:
    """
    Construct matrix from harmonic amplitudes at given coil locations.

    To get an array of spherical harmonic amplitudes/coeffcients (A_l)
    which can be used in a spherical harmonic approximation of the
    vacuum/coil contribution to the polodial flux (psi) do:
        A_l = matrix harmonic amplitudes @ vector of coil currents

    A_l can be used as contraints in optimisation, see spherical_harmonics_constraint.

    N.B. for a single filament (coil):

        A_l =  1/2 * mu_0 * I_f * sin(theta_f) * (r_t/r_f)**l *
                    ( P_l * cos(theta_f) / sqrt(l*(l+1)) )

    Where l = degree, and P_l * cos(theta_f) are the associated
    Legendre polynomials of degree l and order (m) = 1.

    Parmeters
    ----------
    input_coils:
        Bluemira CoilSet
    max_degree: integer
        Maximum degree of harmonic to calculate up to
    r_t: float
        Typical length scale (e.g. radius at outer midplane)

    Returns
    -------
    currents2harmonics: np.array
        Matrix of harmonic amplitudes

    """
    x_f = input_coils.get_control_coils().x
    z_f = input_coils.get_control_coils().z

    # Spherical coords
    r_f = np.sqrt(x_f**2 + z_f**2)
    theta_f = np.arctan2(x_f, z_f)

    # [number of degrees, number of coils]
    currents2harmonics = np.zeros([max_degree, np.size(r_f)])
    # First 'harmonic' is constant (this line avoids Nan isuues)
    currents2harmonics[0, :] = 1  #

    # SH coefficients from function of the current distribution
    # outside of the sphere coitaining the LCFS
    # SH coeffs = currents2harmonics @ coil currents
    degrees = np.arange(1, max_degree)[:, None]
    ones = np.ones_like(degrees)
    currents2harmonics[1:, :] = (
        0.5
        * MU_0
        * (r_t / r_f)[None, :] ** degrees
        * np.sin(theta_f)[None, :]
        * lpmv(ones, degrees, np.cos(theta_f)[None, :])
        / np.sqrt(degrees * (degrees + 1))
    )

    return currents2harmonics


def harmonic_amplitude_marix(
    collocation_r: np.ndarray, collocation_theta: np.ndarray, r_t: float
) -> np.ndarray:
    """
    Construct matrix from harmonic amplitudes at given points (in spherical coords).

    The matrix is used in a spherical harmonic approximation of the vacuum/coil
    contribution to the poloidal flux (psi):

        psi = SUM(
            A_l * ( r**(l+1) / r_t**l ) * sin (theta) *
            ( P_l * cos(theta_f) / sqrt(l*(l+1)) )
        )

    Where l = degree, A_l are the spherical harmonic coeffcients/ampletudes,
    and is P_l * cos(theta_f) are the associated Legendre polynomials of
    degree l and order (m) = 1.

    N.B. Vacuum Psi = Total Psi - Plasma Psi.

    Parameters
    ----------
    collocation_r: np.array
        R values of collocation points
    collocation_theta: np.array
        Theta values of collocation points
    r_t: float
        Typical length scale (e.g. radius at outer midplane)

    Returns
    -------
    harmonics2collocation: np.array
        Matrix of harmonic amplitudes (to get spherical harmonic coefficents
        use matrix @ coefficents = vector psi_vacuum at colocation points)
    """
    # Maximum degree of harmonic to calculate up to = n_collocation - 1
    # [number of points, number of degrees]
    n = len(collocation_r)
    harmonics2collocation = np.zeros([n, n - 1])
    # First 'harmonic' is constant (this line avoids Nan isuues)
    harmonics2collocation[:, 0] = 1

    # SH coeffcient matrix
    # SH coeffs = harmonics2collocation \ vector psi_vacuum at colocation points
    degrees = np.arange(1, n - 1)[None]
    ones = np.ones_like(degrees)
    harmonics2collocation[:, 1:] = (
        collocation_r[:, None] ** (degrees + 1)
        * np.sin(collocation_theta)[:, None]
        * lpmv(ones, degrees, np.cos(collocation_theta)[:, None])
        / ((r_t**degrees) * np.sqrt(degrees * (degrees + 1)))
    )

    return harmonics2collocation


class PointType(Enum):
    """
    Class for use with collocation_points function.
    User can choose how the collocation points are distributed.
    """

    ARC = auto()
    ARC_PLUS_EXTREMA = auto()
    RANDOM = auto()
    RANDOM_PLUS_EXTREMA = auto()


def collocation_points(
    n_points: int, plamsa_bounday: np.ndarray, point_type: str
) -> Dict:
    """
    Create a set of collocation points for use wih spherical harmonic
    approximations. Points are found within the user-supplied
    boundary and should correspond to the LCFS of a chosen equilibrium.
    Curent functionality is for:
        - equispaced points on an arc of fixed radius,
        - equispaced points on an arc plus extrema,
        - random points within a circle enclosed by the LCFS,
        - random points plus extrema.

    Parameters
    ----------
    n_points: integer
        Number of points/targets (not including extrema - these are added
        automatically if relevent).
    plamsa_bounday:
        XZ coordinates of the plasma boundary
    point_type: string
        Method for creating a set of points: 'arc', 'arc_plus_extrema',
        'random', or 'random_plus_extrema'

    Returns
    -------
    collocation: dict
        Dictionary containing collocation points:
        - "x" and "z" values of collocation points.
        - "r" and "theta" values of collocation points.

    """
    point_type = PointType[point_type.upper()]
    x_bdry = plamsa_bounday.x
    z_bdry = plamsa_bounday.z

    if point_type in (PointType.ARC, PointType.ARC_PLUS_EXTREMA):
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

    if point_type in (PointType.RANDOM, PointType.RANDOM_PLUS_EXTREMA):
        # Random sample within a circle enclosed by the LCFS
        half_sample_x_range = 0.5 * (np.max(x_bdry) - np.min(x_bdry))
        sample_r = half_sample_x_range * np.random.rand(n_points)
        sample_theta = (np.random.rand(n_points) * 2 * np.pi) - np.pi

        # Cartesian coordinates
        collocation_x = (
            sample_r * np.sin(sample_theta) + np.min(x_bdry) + half_sample_x_range
        )
        collocation_z = sample_r * np.cos(sample_theta) + z_bdry[np.argmax(x_bdry)]

        # Spherical coordinates
        collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
        collocation_theta = np.arctan2(collocation_x, collocation_z)

    if point_type in (PointType.ARC_PLUS_EXTREMA, PointType.RANDOM_PLUS_EXTREMA):
        # Extrema
        d = 0.1
        extrema_x = np.array(
            [
                np.amin(x_bdry) + d,
                np.amax(x_bdry) - d,
                x_bdry[np.argmax(z_bdry)],
                x_bdry[np.argmin(z_bdry)],
            ]
        )
        extrema_z = np.array(
            [
                0,
                0,
                np.amax(z_bdry) - d,
                np.amin(z_bdry) + d,
            ]
        )

        # Equispaced arc + extrema
        collocation_x = np.concatenate([collocation_x, extrema_x])
        collocation_z = np.concatenate([collocation_z, extrema_z])

        # Hello again spherical coordinates
        collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
        collocation_theta = np.arctan2(collocation_x, collocation_z)

    return {
        "r": collocation_r,
        "theta": collocation_theta,
        "x": collocation_x,
        "z": collocation_z,
    }


def lcfs_fit_metric(coords1: np.ndarray, coords2: np.ndarray) -> Dict:
    """
    Calculate the value of the metric used for evaluating the SH aprroximation.
    This is equal to 1 for non-intersecting LCFSs, and 0 for identical LCFSs.

    Parameters
    ----------
    coords1: np.array
        Coordinates of plamsa bounday from input equlibrum state
    coords2: np.array
        Coordinates of plamsa bounday from approximation equlibrum state

    Returns
    -------
    fit_metric_value: float
        Measure of how 'good' the approximation is.
        fit_metric_value = total area within one but not both LCFSs /
                            (input LCFS area + approximation LCFS area)

    """
    # Test to see if the LCFS for the SH approx is not closed for some reason
    if coords2.x[0] != coords2.x[-1] or coords2.z[0] != coords2.z[-1]:
        # If not closed then go back and try again
        bluemira_print(
            "The approximate LCFS is not closed. Trying again with more degrees."
        )
        return 1

    # If the two LCFSs have identical coordinates then return a perfect fit metric
    if np.array_equal(coords1.x, coords2.x) and np.array_equal(coords1.z, coords2.z):
        bluemira_print("Perfect match! Original LCFS = SH approx LCFS")
        return 0

    # Get area of within the original and the SH approx LCFS
    area1 = get_area_2d(coords1.x, coords1.z)
    area2 = get_area_2d(coords2.x, coords2.z)

    # Find intersections of the LCFSs
    xcross, zcross = get_intersect(coords1.xz, coords2.xz)

    # Check there are an even number of intersections
    if np.mod(len(xcross), 2) != 0:
        bluemira_print(
            "Odd number of intersections for input and SH approx LCFS: this shouldn''t be possible. Trying again with more degrees."
        )
        return 1

    # If there are no intersections then...
    if len(xcross) == 0:
        # Check if one LCFS is entirely within another
        test_1_in_2 = polygon_in_polygon(coords2.xz.T, coords1.xz.T)
        test_2_in_1 = polygon_in_polygon(coords1.xz.T, coords2.xz.T)
        if all(test_1_in_2) or all(test_2_in_1):
            # Calculate the metric if one is inside the other
            return (np.max([area1, area2]) - np.min([area1, area2])) / (area1 + area2)
        else:
            # Otherwise they are in entirely different places
            bluemira_print(
                "The approximate LCFS does not overlap with the original. Trying again with more degrees."
            )
            return 1

    # Calculate the area between the intersections of the two LCFSs,
    # i.e., area within one but not both LCFSs.
    c1 = Coordinates({"x": coords1.x, "z": coords1.z})
    c2 = Coordinates({"x": coords2.x, "z": coords2.z})
    c1_face = BluemiraFace(make_polygon(c1, closed=True))
    c2_face = BluemiraFace(make_polygon(c2, closed=True))
    result1 = boolean_cut(c1_face, c2_face)
    result2 = boolean_cut(c2_face, c1_face)

    #  Calculate metric
    return (sum([f.area for f in result1]) + sum([f.area for f in result2])) / (
        c1_face.area + c2_face.area
    )


def coils_outside_sphere_vacuum_psi(
    eq: Equilibrium,
) -> Tuple[np.ndarray, np.ndarray, CoilSet]:
    """
    Calculate the poloidal flux (psi) contribution from the vacuumn/coils
    located outside of the sphere containing the plamsa, i.e., LCFS of
    equlibrium state. N.B., currents from coilset are not considered here.

    Parameters
    ----------
    eq: Bluemira Equilibrium
        Starting equilibrium to use for our approximation

    Returns
    -------
    vacuum_psi: ndarray
        Psi contributuion from control coils
        (only control coils! - be careful how you use it)
    plasma_psi: ndarray
        Psi contribution from plasma
    new_coilset: Bluemira CoilSet
        Coilset with control coils selected appropriately for use of SH approximation

    """
    # Psi contribution from the coils = total - plasma contribution
    plasma_psi = eq.plasma.psi(eq.grid.x, eq.grid.z)
    vacuum_psi = eq.psi() - plasma_psi

    # Approximation boundary - sphere must contain
    # plasma/LCFS for chosen equilibrium.
    # Are the control coils outside the sphere containing
    # the last closed flux surface?

    c_names = np.array(eq.coilset.name)

    max_bdry_r = np.max(np.sqrt(eq.get_LCFS().x ** 2 + eq.get_LCFS().z ** 2))
    coil_r = np.sqrt(np.array(eq.coilset.x) ** 2 + np.array(eq.coilset.z) ** 2)

    if max_bdry_r > np.min(coil_r):
        too_close_coils = c_names[coil_r <= max_bdry_r]
        not_too_close_coils = c_names[coil_r > max_bdry_r].tolist()
        bluemira_print(
            f"One or more of your coils is too close to the LCFS to be used in the SH approximation. Coil names: {too_close_coils}."
        )

        # Need a coilset with control coils outside sphere
        new_coils = []
        for n in eq.coilset.name:
            new_coils.append(eq.coilset[n])
        new_coilset = CoilSet(*new_coils, control_names=not_too_close_coils)
    else:
        new_coilset = deepcopy(eq.coilset)

    # If not using all coils in approx (have set control coils)
    if len(new_coilset.get_control_coils().name) != len(new_coilset.name):
        # Calculate psi contributuion from non-control coils
        # This shouldn't matter if none of the coil currents have been set
        non_ccoil_cont = new_coilset.psi(
            eq.grid.x, eq.grid.z
        ) - new_coilset.get_control_coils().psi(eq.grid.x, eq.grid.z)
        # Remove contributuion from non-control coils
        vacuum_psi -= non_ccoil_cont

    return vacuum_psi, plasma_psi, new_coilset


def get_psi_harmonic_ampltidues(
    vacuum_psi: np.ndarray, grid: Grid, collocation: Dict, r_t: float
) -> np.ndarray:
    """
    Calculate the Spherical Harmoic (SH) amplitudes/coefficients needed to produce
    a SH approximation of the vaccum (i.e. control coil) contribution to
    the poloidal flux (psi).The number of degrees used in the approximation is
    one less than the number of collocation points.

    Parameters
    ----------
    vacuum_psi: ndarray
        Psi contributuion from coils that we wish to approximate
    grid: Bluemira Grid
        Associated grid
    collocation: dict
        Dictionary containing collocation points information
    r_t: float (default = maximum x value of LCFS)
        Typical lengthscale for spherical harmonic approximation.

    Returns
    -------
    psi_harmonic_ampltidues: np.array
        SH coefficients for given number of degrees

    """
    # Set up interpolation with gridded values
    psi_func = RectBivariateSpline(grid.x[:, 0], grid.z[0, :], vacuum_psi)

    # Evaluate at collocation points
    collocation_psivac = psi_func.ev(collocation["x"], collocation["z"])

    # Construct matrix from SH amplitudes for flux function at collocation points
    harmonics2collocation = harmonic_amplitude_marix(
        collocation["r"], collocation["theta"], r_t
    )

    # Fit harmonics to match values at collocation points
    psi_harmonic_ampltidues, residual, rank, s = np.linalg.lstsq(
        harmonics2collocation, collocation_psivac, rcond=None
    )

    return psi_harmonic_ampltidues


def spherical_harmonic_approximation(
    eq: Equilibrium,
    n_points: int = None,
    point_type: str = None,
    acceptable_fit_metric: float = None,
    r_t: float = None,
    plot: bool = False,
) -> Tuple[CoilSet, float, np.ndarray, int, float, np.ndarray]:
    """
    Calculate the spherical harmonic (SH) amplitudes/coefficients
    needed as a reference value for the 'spherical_harmonics_constraint'
    used in coilset optimisation.

    Use a LCFS fit metric to determine the rquired number of degrees.

    The number of degrees used in the approximation is one less than
    the number of collocation points.

    Parameters
    ----------
    eq: Bluemira Equilibrium
        Equilibria to use as starting point for approximation.
        We will approximate psi using SHs - the aim is to keep the
        core plasma contribution fixed (using SH amplitudes as constraints)
        while being able to vary the vacuum (coil) contribution, so that
        we do not need to re-solve for the equilibria during oiptimisation.
    n_points: integer (default=8)
        Number of desired collocation points
        excluding extrema (always +4 automatically)
    point_type: string (default="arc_plus_extrema")
        Name that determines how the collocation points are selected.
        The following options are available for colocation point distibution:
            - 'arc' = equispaced points on an arc of fixed radius,
            - 'arc_plus_extrema' = 'arc' plus the min and max points of the LSFS
                in the x- and z-directions (4 points total),
            - 'random',
            - 'random_plus_extrema'.
    acceptable_fit_metric: float(default=0.01)
        Value between 0 and 1 chosen by user.
        If the LCFS found using the SH approximation method perfectly matches the
        LCFS of the input equilibria then the fit metric = 0.
        A fit metric of 1 means that they do not overlap at all.
        fit_metric_value = total area within one but not both LCFSs /
                            (input LCFS area + approximation LCFS area)
    r_t: float (default = maximum x value of LCFS)
        Typical lengthscale for spherical harmonic approximation.
    extra_info: bool (default = False)
        If False, return only the information needed for use in optimisation.
        If True, return additional information and a plot comparing orginal psi
        to the SH approximation.

    Returns
    -------
    sh_coilset: Bluemira Coilset
        Coilset to use with SH approximation
    r_t: float
        typical lengthscale for spherical harmonic approximation
    coil_current_harmonic_ampltidues: ndarray
        SH coefficients/amplitudes for required number of degrees
    degree: int
        number of degrees required for a SH approx with the desired fit metric
    fit_metric_value: float
        fit metric acheived
    approx_total_psi: ndarray
        the total psi obtained using the SH approximation
    """
    # Default values if not input
    if acceptable_fit_metric is None:
        acceptable_fit_metric = 0.01
    if n_points is None:
        n_points = 8
    if point_type is None:
        point_type = "arc_plus_extrema"

    # Get the nessisary boundary locations and lengthscale
    # for use in spherical harmonic approximations.
    # Starting LCFS
    eq_copy = deepcopy(eq)
    original_LCFS = eq_copy.get_LCFS()

    # Typical lengthscale default if not chosen by user
    if r_t is None:
        r_t = np.amax(original_LCFS.x)

    # Grid keep the same as input equilibrium
    grid = eq_copy.grid

    # Contribution to psi that we would like to approximate
    # (and plasma contribution for later), also make sure
    # control coils are in acceptable locations for approximation
    vacuum_psi, plasma_psi, sh_coilset = coils_outside_sphere_vacuum_psi(eq_copy)

    # Create the set of collocation points within the LCFS for the SH calculations
    collocation = collocation_points(
        n_points,
        original_LCFS,
        point_type,
    )

    # SH amplitudes needed to produce an approximation of vaccum psi contribution
    psi_harmonic_ampltidues = get_psi_harmonic_ampltidues(
        vacuum_psi, grid, collocation, r_t
    )

    # Set min to save some time
    min_degree = 2
    max_degree = len(collocation["x"]) - 1

    for degree in np.arange(min_degree, max_degree + 1):
        # Construct matrix from harmonic amplitudes for coils
        currents2harmonics = coil_harmonic_amplitude_matrix(
            sh_coilset.get_control_coils(), degree, r_t
        )

        # Calculate necessary coil currents
        currents, residual, rank, s = np.linalg.lstsq(
            currents2harmonics, psi_harmonic_ampltidues[:degree], rcond=None
        )

        # Calculate the coilset SH amplitudes for use in optimisisation
        coil_current_harmonic_ampltidues = currents2harmonics @ currents

        # Set currents in coilset
        sh_coilset.get_control_coils().current = currents

        # Calculate the approximate Psi contribution from the coils
        coilset_approx_psi = sh_coilset.psi(grid.x, grid.z)

        # Total
        approx_total_psi = coilset_approx_psi + plasma_psi

        # Get plasma boundary for comparison to starting equilibrium using fit metric
        eq_copy.coilset = sh_coilset
        approx_LCFS = eq_copy.get_LCFS(psi=approx_total_psi)

        # Compare staring equlibrium to new approximate equilibrium
        fit_metric_value = lcfs_fit_metric(original_LCFS, approx_LCFS)

        if fit_metric_value <= acceptable_fit_metric:
            bluemira_print(
                f"The fit metric value acheived is {fit_metric_value} using {degree} degrees."
            )
            break
        elif degree == max_degree:
            raise BluemiraError(
                f"Uh oh, you may need to use more degrees for a fit metric of {acceptable_fit_metric}! Use a greater number of collocation points please."
            )

    # plot comparing orginal psi to the SH approximation
    if plot:
        plot_psi_comparision(
            grid,
            eq.psi(grid.x, grid.z),
            approx_total_psi,
            eq.coilset.psi(grid.x, grid.z),
            coilset_approx_psi,
        )

    return (
        sh_coilset,
        r_t,
        coil_current_harmonic_ampltidues,
        degree,
        fit_metric_value,
        approx_total_psi,
    )


def plot_psi_comparision(
    grid: Grid,
    tot_psi_org: np.ndarray,
    tot_psi_app: np.ndarray,
    vac_psi_org: np.ndarray,
    vac_psi_app: np.ndarray,
):
    """
    Create plot comparing an orginal psi to psi obtained from approximation.

    Parameters
    ----------
    grid: Bluemira Grid
        Nedd x and z values to plot psi.
    tot_psi_org: ndarray
        Original Total Psi
    tot_psi_app: ndarray
        Approximation Total Psi
    vac_psi_org: ndarray
        Original Vacuum Psi (contribution from entire coilset)
    vac_psi_app: ndarray
        Approximation Vacuum Psi (contribution from entire coilset)

    """
    nlevels = 50  # PLOT_DEFAULTS["psi"]["nlevels"]
    cmap = PLOT_DEFAULTS["psi"]["cmap"]
    clevels = np.linspace(np.amin(tot_psi_org), np.amax(tot_psi_org), nlevels)

    plot1 = plt.subplot2grid((5, 4), (0, 0), rowspan=2, colspan=1)
    plot1.set_title("Original, Total Psi")
    plot1.contour(grid.x, grid.z, tot_psi_org, levels=clevels, cmap=cmap, zorder=8)
    plot2 = plt.subplot2grid((5, 4), (0, 2), rowspan=2, colspan=1)
    plot2.set_title("SH Approximation, Total Psi")
    plot2.contour(grid.x, grid.z, tot_psi_app, levels=clevels, cmap=cmap, zorder=8)
    plot3 = plt.subplot2grid((5, 4), (3, 0), rowspan=2, colspan=1)
    plot3.set_title("Original, Vacuum Psi")
    plot3.contour(grid.x, grid.z, vac_psi_org, levels=clevels, cmap=cmap, zorder=8)
    plot4 = plt.subplot2grid((5, 4), (3, 2), rowspan=2, colspan=1)
    plot4.set_title("SH Approximation, Vacuum Psi")
    plot4.contour(grid.x, grid.z, vac_psi_app, levels=clevels, cmap=cmap, zorder=8)
