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

import numpy as np
from scipy.special import lpmv

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_print
from bluemira.geometry.coordinates import get_area_2d, get_intersect, polygon_in_polygon


def coil_harmonic_amplitudes(input_coils, i_f, max_degree, r_t):
    """
    Returns spherical harmonics coefficients/amplitudes (A_l) to be used
    in a spherical harmonic approximation of the vacuum/coil contribution
    to the polodial flux (psi). Vacuum Psi = Total Psi - Plasma Psi.
    These coefficients can be used as contraints in optimisation.

    For a single filement (coil):

        A_l =  1/2 * mu_0 * I_f * sin(theta_f) * (r_t/r_f)**l *
                    ( P_l * cos(theta_f) / sqrt(l*(l+1)) )

    Where l = degree, and P_l * cos(theta_f) are the associated
    Legendre polynomials of degree l and order (m) = 1.

    Parmeters
    ----------
    input_coils:
        Bluemira CoilSet
    i_f: np.array
        Currents of filaments (coils)
    max_degree: integer
        Maximum degree of harmonic to calculate up to
    r_t: float
        Typical length scale (e.g. radius at outer midplane)

    Returns
    -------
    amplitudes: np.array
        Array of spherical harmonic amplitudes from given coil potitions and currents
     max_valid_r: float
        Maximum spherical radius for which the spherical harmonics apply
    """
    # SH coefficients from fuction of the current distribution outside of the sphere
    # containing the plamsa, i.e., LCFS (r_lcfs)
    # SH coeffs = currents2harmonics @ coil currents
    # N.B., max_valid_r >= r_lcfs,
    # i.e., cannot use coil located within r_lcfs as part of this method.
    currents2harmonics, max_valid_r = coil_harmonic_amplitude_matrix(
        input_coils, max_degree, r_t
    )

    return currents2harmonics @ i_f, max_valid_r


def coil_harmonic_amplitude_matrix(input_coils, max_degree, r_t):
    """
    Construct matrix from harmonic amplitudes at given coil locations.

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
        Matrix of harmonic amplitudes (to get spherical harmonic coefficents
        -> matrix @ vector of coil currents, see coil_harmonic_amplitudes)
     max_valid_r: float
        Maximum spherical radius for which the spherical harmonic
        approximation is valid
    """
    x_f = input_coils.get_control_coils().x
    z_f = input_coils.get_control_coils().z

    # Spherical coords
    r_f = np.sqrt(x_f**2 + z_f**2)
    theta_f = np.arctan2(x_f, z_f)
    # Maxmimum r value for the sphere whithin which harmonics apply
    max_valid_r = np.amin(r_f)

    # [number of degrees, number of coils]
    currents2harmonics = np.zeros([max_degree, np.size(r_f)])
    # First 'harmonic' is constant (this line avoids Nan isuues)
    currents2harmonics[0, :] = 1  #

    # SH coeffcients from fuction of the current distribution
    # outside of the sphere coitaining the LCFS
    # SH coeffs = currents2harmonics @ coil currents
    for degree in np.arange(1, max_degree):
        currents2harmonics[degree, :] = (
            0.5
            * MU_0
            * (r_t / r_f) ** degree
            * np.sin(theta_f)
            * lpmv(1, degree, np.cos(theta_f))
            / np.sqrt(degree * (degree + 1))
        )

    return currents2harmonics, max_valid_r


def harmonic_amplitude_marix(
    collocation_r, collocation_theta, n_collocation, max_degree, r_t
):
    """
    Construct matrix from harmonic amplitudes at given points (in spherical coords).

    The matrix is used in a spherical harmonic approximation of the vacuum/coil
    contribution to the poilodal flux (psi):

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
    n_collocation: integer
        Number of collocation points
    max_degree: integer
        Maximum degree of harmonic to calculate up to
    r_t: float
        Typical length scale (e.g. radius at outer midplane)

    Returns
    -------
    harmonics2collocation: np.array
        Matrix of harmonic amplitudes (to get spherical harmonic coefficents
        use matrix @ coefficents = vector psi_vacuum at colocation points)
    """
    # [number of points, number of degrees]
    harmonics2collocation = np.zeros([n_collocation, max_degree])
    # First 'harmonic' is constant (this line avoids Nan isuues)
    harmonics2collocation[:, 0] = 1

    # SH coeffcient matrix
    # SH coeffs = harmonics2collocation \ vector psi_vacuum at colocation points
    for degree in np.arange(1, max_degree):
        harmonics2collocation[:, degree] = (
            collocation_r ** (degree + 1)
            * np.sin(collocation_theta)
            * lpmv(1, degree, np.cos(collocation_theta))
            / ((r_t**degree) * np.sqrt(degree * (degree + 1)))
        )

    return harmonics2collocation


def collocation_points(n_points, plamsa_bounday, point_type):
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
    collocation_r: np.array
        R values of collocation points
    collocation_theta: np.array
        Theta values of collocation points
    n_collocation: integer
        Number of collocation points
    """
    x_bdry = plamsa_bounday.x
    z_bdry = plamsa_bounday.z

    if point_type == "arc" or point_type == "arc_plus_extrema":
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

    if point_type == "random" or point_type == "random_plus_extrema":
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

    if point_type == "arc_plus_extrema" or point_type == "random_plus_extrema":
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

    # Number of collocation points
    n_collocation = np.size(collocation_x)

    return collocation_r, collocation_theta, collocation_x, collocation_z, n_collocation


def lcfs_fit_metric(coords1, coords2):
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
        # raise BluemiraError('hmmm')
        bluemira_print(
            f"The approximate LCFS is not closed. Trying again with more degrees."
        )
        fit_metric_value = 1
        return fit_metric_value

    # If the two LCFSs have identical coordinates then return a perfect fit metric
    if np.array_equal(coords1.x, coords2.x) and np.array_equal(coords1.z, coords2.z):
        bluemira_print(f"Perfect match! Original LCFS = SH approx LCFS")
        fit_metric_value = 0
        return fit_metric_value

    # Get area of within the original and the SH approx LCFS
    area1 = get_area_2d(coords1.x, coords1.z)
    area2 = get_area_2d(coords2.x, coords2.z)

    # Find intersections of the LCFSs
    xcross, zcross = get_intersect(coords1.xz, coords2.xz)

    # Check there are an even number of intersections
    if np.mod(len(xcross), 2) != 0:
        bluemira_print(
            f"Odd number of intersections for input and SH approx LCFS: this shouldn''t be possible. Trying again with more degrees."
        )
        fit_metric_value = 1
        return fit_metric_value

    # If there are no intersections then...
    if len(xcross) == 0:
        # Check if one LCFS is entirely within another
        test_1_in_2 = polygon_in_polygon(coords2.xz.T, coords1.xz.T)
        test_2_in_1 = polygon_in_polygon(coords1.xz.T, coords2.xz.T)
        if all(test_1_in_2) or all(test_2_in_1):
            # Calculate the metric if one is inside the other
            fit_metric_value = (np.max([area1, area2]) - np.min([area1, area2])) / (
                area1 + area2
            )
            return fit_metric_value
        else:
            # Otherwise they are in entirely different places
            bluemira_print(
                f"The approximate LCFS does not overlap with the original. Trying again with more degrees."
            )
            fit_metric_value = 1
            return fit_metric_value

    # Calculate the area between the intersections of the two LCFSs,
    # i.e., area within one but not both LCFSs.

    # Initial value
    area_between = 0

    # Add first intersection to the end
    xcross = np.append(xcross, xcross[0])
    zcross = np.append(zcross, zcross[0])

    # Scan over intersections
    for i in np.arange(len(xcross) - 1):
        # Find indeces of start and end of the segment of LCFSs between
        # intersections
        start1 = np.argmin(abs(coords1.x - xcross[i]) + abs(coords1.z - zcross[i]))
        start2 = np.argmin(abs(coords2.x - xcross[i]) + abs(coords2.z - zcross[i]))
        end1 = np.argmin(abs(coords1.x - xcross[i + 1]) + abs(coords1.z - zcross[i + 1]))
        end2 = np.argmin(abs(coords2.x - xcross[i + 1]) + abs(coords2.z - zcross[i + 1]))

        if end1 < start1:
            # If segment overlaps start of line defining LCFS
            seg1 = np.append(coords1.xz[:, start1:], coords1.xz[:, :end1], axis=1)
        else:
            seg1 = coords1.xz[:, start1:end1]

        if end2 < start2:
            # If segment overlaps start of line defining LCFS
            seg2 = np.append(coords2.xz[:, start2:], coords2.xz[:, :end2], axis=1)
        else:
            seg2 = coords2.xz[:, start2:end2]

        # Generate co-ordinates defining a polygon between these two
        # intersections.
        x = np.array([xcross[i], xcross[i + 1], xcross[i]])
        z = np.array([zcross[i], zcross[i + 1], zcross[i]])
        x = np.insert(x, 2, np.flip(seg2[0, :]), axis=0)
        z = np.insert(z, 2, np.flip(seg2[1, :]), axis=0)
        x = np.insert(x, 1, seg1[0, :], axis=0)
        z = np.insert(z, 1, seg1[1, :], axis=0)

        # Calculate the area of the polygon
        area_between = area_between + get_area_2d(x, z)

    #  Calculate metric
    fit_metric_value = area_between / (area1 + area2)

    return fit_metric_value
