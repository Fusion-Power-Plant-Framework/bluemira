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
1-D radiation model inspired by the PROCESS function "plot_radprofile" in plot_proc.py.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, fields
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shp

from typing import Dict, List, Type, Union
from bluemira.base import constants
from bluemira.base.constants import ureg
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.flux_surfaces import calculate_connection_length_flt
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.physics import calc_psi_norm
from bluemira.display.plotter import plot_coordinates
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from cherab.core.math import sample3d
from matplotlib.collections import LineCollection
from raysect.core import Point3D, Vector3D, rotate_basis, translate
from raysect.optical.observer import PowerPipeline0D
from raysect.optical.observer.nonimaging.pixel import Pixel
from scipy.interpolate import LinearNDInterpolator, interp1d, interp2d

if TYPE_CHECKING:
    from step_reactor.temp_flux_surface_maker import TempFsSolver


SEP_CORRECTOR = 5e-2  # [m]
E_CHARGE = ureg.Quantity("e").to_base_units().magnitude


def upstream_temperature(
    b_pol: float,
    b_tot: float,
    lambda_q_near: float,
    p_sol: float,
    eq: Equilibrium,
    r_sep_mp: float,
    z_mp: float,
    k_0: float,
    firstwall_geom: Grid,
    connection_length=None,
):
    """
    Calculate the upstream temperature, as suggested from "Pitcher, 1997".
    Knowing the power entering the SOL, and assuming large temperature gradient
    between upstream and target.

    Parameters
    ----------
    b_pol: float
        Poloidal magnetic field at the midplane [T]
    b_tot: float
        Total magnetic field at the midplane [T]
    lambda_q_near: float
        Power decay length in the near SOL [m]
    p_sol: float
        Total power entering the SOL [W]
    eq: Equilibrium
        Equilibrium in which to calculate the upstream temperature
    r_sep_mp: float
        Upstream location radial coordinate [m]
    z_mp: float
        Upstream location z coordinate [m]
    k_0: float
        Material's conductivity
    firstwall_geom: grid
        First wall geometry
    lfs: Boolean
        True for the outboard. False for the inboard

    Returns
    -------
    t_upstream_kev: float
        upstream temperature. Unit [keV]
    """
    # SoL cross-section at the midplane
    a_par = 4 * np.pi * r_sep_mp * lambda_q_near * (b_pol / b_tot)

    # upstream power density
    q_u = p_sol / a_par

    # connection length from the midplane to the target
    if connection_length is None:
        l_tot = calculate_connection_length_flt(
            eq,
            r_sep_mp,
            z_mp,
            first_wall=firstwall_geom,
        )
    else:
        l_tot = connection_length

    # upstream temperature [keV]
    t_upstream_ev = (3.5 * (q_u / k_0) * l_tot) ** (2 / 7)
    t_upstream_kev = constants.raw_uc(t_upstream_ev, "eV", "keV")

    return t_upstream_kev


def target_temperature(
    p_sol: float,
    t_u: float,
    n_u: float,
    gamma: float,
    eps_cool: float,
    f_ion_t: float,
    b_pol_tar: float,
    b_pol_u: float,
    alpha_pol_deg: float,
    r_u: float,
    r_tar: float,
    lambda_q_near: float,
    b_tot_tar: float,
):
    """
    Calculate the target as suggested from the 2PM.
    It includes hydrogen recycle loss energy.
    Ref. Stangeby, "The Plasma Boundary of Magnetic Fusion
    Devices", 2000.

    Parameters
    ----------
    p_sol: float
        Total power entering the SOL [W]
    t_u: float
        Upstream temperature. Unit [eV]
    n_u: float
        Electron density at the upstream [1/m^3]
    gamma: float
        Sheath heat transmission coefficient
    eps_cool: float
        Electron energy loss [eV]
    f_ion_t: float
        Hydrogen first ionization [eV]
    b_pol_tar: float
        Poloidal magnetic field at the target [T]
    b_pol_u: float
        Poloidal magnetic field at the midplane [T]
    alpha: float
        Incident angle between separatrix and target plate as
        poloidal projection [deg]
    r_u: float
        Upstream location radial coordinate [m]
    r_tar: float
        stike point radial coordinate [m]
    lambda_q_near: float
        Power decay length in the near SOL at the midplane [m]
    b_tot_tar: float
        Total magnetic field at the target [T]
    lfs: boolean
        low field side. Default value True.
        If False it stands for high field side (hfs).

    Returns
    -------
    t_tar: float
        target temperature. Unit [keV]
    """
    # flux expansion at the target location
    f_exp = (r_u * b_pol_u) / (r_tar * b_pol_tar)

    # lambda target
    # lambda_q_tar = lambda_q_near * f_exp * (1 / np.sin(np.deg2rad(alpha_pol_deg)))
    lambda_q_tar = lambda_q_near * f_exp * (1 / np.sin(np.deg2rad(alpha_pol_deg)))
    # wet area as poloidal section
    a_wet = 4 * np.pi * r_tar * lambda_q_tar

    # parallel cross section
    a_par = a_wet * (b_pol_tar / b_tot_tar)

    # parallel power flux density
    q_u = p_sol / a_par

    # ion mass in kg (it should be DT = 2.5*amu)
    m_i_amu = constants.D_MOLAR_MASS
    m_i_kg = constants.raw_uc(m_i_amu, "amu", "kg")

    # converting upstream temperature
    # upstream electron density - no fifference hfs/lfs?
    # Numerator and denominator of the upstream forcing function
    #print(q_u)
    num_f = m_i_kg * 4 * (q_u**2)
    den_f = 2 * E_CHARGE * (gamma**2) * (E_CHARGE**2) * (n_u**2) * (t_u**2)
    #print(num_f, den_f)
    # forcing function
    f_ev = num_f / den_f

    # Critical target temperature
    t_crit = eps_cool / gamma

    # Finding roots of the target temperature quadratic equation
    #print(eps_cool, gamma, f_ev)
    coeff_2 = 2 * (eps_cool / gamma) - f_ev
    coeff_3 = (eps_cool**2) / (gamma**2)
    coeff = [1, coeff_2, coeff_3]
    roots = np.roots(coeff)

    if roots.dtype == complex:
        t_tar = f_ion_t
    else:
        # Excluding unstable solution
        #print(roots)
        sol_i = np.where(roots > t_crit)[0][0]

        # Target temperature
        t_tar = roots[sol_i]

    # target temperature in keV
    t_tar = constants.raw_uc(t_tar, "eV", "keV")

    #test
    boltz = 1.380649e-23/(8.61e-5) #[J/K]

    c_st = (2 * boltz * 5/m_i_kg)**(1/2)

    q_t = 7*boltz*5*7.5e20*c_st

    return t_tar


def random_point_temperature(
    x_p: float,
    z_p: float,
    t_u: float,
    p_sol: float,
    lambda_q_near: float,
    eq: Equilibrium,
    r_sep_mp: float,
    z_mp: float,
    k_0: float,
    firstwall_geom: Grid,
    lfs=True,
    connection_length=None,
):
    """
    Calculate the temperature at a random point above the x-point.

    Parameters
    ----------
    x_p: float
        x coordinate of the point [m]
    z_p: float
        z coordinate of the point [m]
    t_u: float
        upstream temperature [eV]
    p_sol: float
        Total power entering the SOL [W]
    lambda_q_near: float
        Power decay length in the near SOL at the midplane [m]
    eq: Equilibrium
        Equilibrium in which to calculate the point temperature
    r_sep_omp: float
        Upstream location radial coordinate [m]
    z_mp: float
        Upstream location z coordinate [m]
    k_0: float
        Material's conductivity
    firstwall_geom: grid
        first wall geometry
    lfs: boolean
        low (toroidal) field side (outer wall side). Default value True.
        If False it stands for high field side (hfs).

    Returns
    -------
    t_p: float
        point temperature. Unit [keV]
    """
    # Flux expansion at the point
    b_pol_sep_mp = eq.Bp(r_sep_mp, z_mp)
    b_pol_p = eq.Bp(x_p, z_p)
    b_tor_p = eq.Bt(x_p)
    b_tot = np.hypot(b_pol_p, b_tor_p)
    f_exp = (r_sep_mp * b_pol_sep_mp) / (x_p * b_pol_p)

    # lambda target
    lambda_q_local = lambda_q_near * f_exp

    # parallel cross section
    a_par = (4 * np.pi * x_p * lambda_q_local) * (b_pol_p / b_tot)

    # parallel power flux density
    q_par = p_sol / a_par

    # Distinction between lfs and hfs
    if lfs:
        d = SEP_CORRECTOR
    else:
        d = -SEP_CORRECTOR

    # Distance between the chosen point and the the target
    if lfs:
        forward = True
    else:
        forward = False
    l_p = calculate_connection_length_flt(
        eq,
        x_p + (d * f_exp),
        z_p,
        forward=forward,
        first_wall=firstwall_geom,
    )
    # connection length from the midplane to the target
    if connection_length is None:
        l_tot = calculate_connection_length_flt(
            eq,
            r_sep_mp,
            z_mp,
            forward=forward,
            first_wall=firstwall_geom,
        )
    else:
        l_tot = connection_length
    # connection length from mp to p point
    s_p = l_tot - l_p
    if round(abs(z_p)) == 0:
        s_p = 0
    # Local temperature
    t_p = ((t_u**3.5) - 3.5 * (q_par / k_0) * s_p) ** (2 / 7)

    # From eV to keV
    t_p = constants.raw_uc(t_p, "eV", "keV")

    return t_p


def electron_density_and_temperature_sol_decay(
    t_sep: float,
    n_sep: float,
    lambda_q_near: float,
    lambda_q_far: float,
    dx_mp: float,
    f_exp=1,
    near_sol_gradient: float = 0.99,
):
    """
    Generic radial esponential decay to be applied from a generic starting point
    at the separatrix (not only at the mid-plane).
    The vertical location is dictated by the choice of the flux expansion f_exp.
    By default f_exp = 1, meaning mid-plane.
    The boolean "omp" set by default as "True", gives the option of choosing
    either outer mid-plane (True) or inner mid-plane (False)
    From the power decay length it calculates the temperature decay length and the
    density decay length.

    Parameters
    ----------
    t_sep: float
        initial temperature value at the separatrix [keV]
    n_sep: float
        initial density value at the separatrix [1/m^3]
    lambda_q_near: float
        Power decay length in the near SOL [m]
    lambda_q_far: float
        Power decay length in the far SOL [m]
    dx_mp: [list]
        Gaps between flux tubes at the mp [m]
    f_exp: float
        flux expansion. Default value=1 referred to the mid-plane
    near_sol_gradient: float
        temperature and density drop within the near scrape-off layer
        from the separatrix value
    lfs: boolean
        low (toroidal) field side (outer wall side). Default value True.
        If False it stands for high field side (hfs).

    Returns
    -------
    te_sol: np.array
        radial decayed temperatures through the SoL. Unit [keV]
    ne_sol: np.array
        radial decayed densities through the SoL. unit [1/m^3]
    """
    # temperature and density decay factors
    t_factor = 7 / 2
    n_factor = 1

    # radial distance of flux tubes from the separatrix
    dr = dx_mp * f_exp

    # temperature and density percentage decay within the far SOL
    far_sol_gradient = 1 - near_sol_gradient

    # power decay length modified according to the flux expansion
    lambda_q_near = lambda_q_near * f_exp
    lambda_q_far = lambda_q_far * f_exp

    # Assuming conduction-limited regime.
    lambda_t_near = t_factor * lambda_q_near
    lambda_t_far = t_factor * lambda_q_far
    lambda_n_near = n_factor * lambda_t_near
    lambda_n_far = n_factor * lambda_t_far

    te_sol = (near_sol_gradient * t_sep) * np.exp(-dr / lambda_t_near) + (
        far_sol_gradient * t_sep
    ) * np.exp(-dr / lambda_t_far)
    ne_sol = (near_sol_gradient * n_sep) * np.exp(-dr / lambda_n_near) + (
        far_sol_gradient * n_sep
    ) * np.exp(-dr / lambda_n_far)

    return te_sol, ne_sol


def gaussian_decay(max_value: float, min_value: float, no_points: float, decay=True):
    """
    Generic gaussian decay to be applied between two extreme values and for a
    given number of points.

    Parameters
    ----------
    max_value: float
        maximum value of the parameters
    min_value: float
        minimum value of the parameters
    no_points: float
        number of points through which make the parameter decay

    Returns
    -------
    dec_param: np.array
        decayed parameter
    """
    no_points = max(no_points, 1)

    # setting values on the horizontal axis
    x = np.linspace(no_points, 0, no_points)

    # centering the gaussian on its highest value
    mu = max(x)

    # setting sigma to be consistent with min_value
    frac = max_value / min_value
    lg = np.log(frac)
    denominator = 2 * lg * (1 / (mu**2))
    sigma = np.sqrt(1 / denominator)
    h = 1 / np.sqrt(2 * sigma**2)

    # decaying param
    dec_param = max_value * (np.exp((-(h**2)) * ((x - mu) ** 2)))
    if decay is False:
        dec_param = dec_param[::-1]
    i_near_minimum = np.where(dec_param < min_value)
    dec_param[i_near_minimum[0]] = min_value

    return dec_param


def exponential_decay(max_value: float, min_value: float, no_points: float, decay=False):
    """
    Generic exponential decay to be applied between two extreme values and for a
    given number of points.

    Parameters
    ----------
    max_value: float
        maximum value of the parameters
    min_value: float
        minimum value of the parameters
    no_points: float
        number of points through which make the parameter decay
    decay: boolean
        to define either a decay or increment

    Returns
    -------
    dec_param: np.array
        decayed parameter
    """
    no_points = max(no_points, 1)
    if no_points <= 1:
        return np.linspace(max_value, min_value, no_points)
    else:
        x = np.linspace(1, no_points, no_points)
        a = np.array([x[0], min_value])
        b = np.array([x[-1], max_value])
        if decay:
            arg = x / b[0]
            base = a[0] / b[0]
        else:
            arg = x / a[0]
            base = b[0] / a[0]

        my_log = np.log(arg) / np.log(base)

        f = a[1] + (b[1] - a[1]) * my_log

    return f


def ion_front_distance(
    x_strike: float,
    z_strike: float,
    eq: Equilibrium,
    x_pt_z: float,
    t_tar: float = None,
    avg_ion_rate: float = None,
    avg_momentum_rate: float = None,
    n_r: float = None,
    rec_ext: float = None,
):
    """
    Manual definition of ion penetration depth.
    TODO: Find sv_i and sv_m

    Parameters
    ----------
    x_strike: float [m]
        x coordinate of the strike point
    z_strike: float [m]
        z coordinate of the strike point
    eq: Equilibrium
        Equilibrium in which to calculate the x-point temperature
    x_pt_z: float
        x-point location z coordinate [m]
    t_tar: float [keV]
        target temperature
    avg_ion_rate: float
        average ionization rate
    avg_momentum_rate: float
        average momentum loss rate
    n_r: float
        density at the recycling region entrance [1/m^3]
    rec_ext: float [m]
        recycling region extention (along the field line)
        from the target

    Returns
    -------
    z_front: float [m]
        z coordinate of the ionization front
    """
    # Speed of light to convert kg to eV/c^2
    light_speed = constants.C_LIGHT

    # deuterium ion mass
    m_i_amu = constants.D_MOLAR_MASS
    m_i_kg = constants.raw_uc(m_i_amu, "amu", "kg")
    m_i = m_i_kg / (light_speed**2)

    # Magnetic field at the strike point
    b_pol = eq.Bp(x_strike, z_strike)
    b_tor = eq.Bt(x_strike)
    b_tot = np.hypot(b_pol, b_tor)

    # From total length to poloidal length
    pitch_angle = b_tot / b_pol
    if rec_ext is None:
        den_lambda = 3 * np.pi * m_i * avg_ion_rate * avg_momentum_rate
        z_ext = np.sqrt((8 * t_tar) / den_lambda) ** (1 / n_r)
    else:
        z_ext = rec_ext * np.sin(pitch_angle)

    # z coordinate (from the midplane)
    z_front = abs(z_strike - x_pt_z) - abs(z_ext)

    return z_front


def calculate_z_species(t_ref, z_ref, species_frac, te):
    """
    Calculation of species ion charge, in condition of quasi-neutrality.

    Parameters
    ----------
    t_ref: np.array
        temperature reference [keV]
    z_ref: np.array
        effective charge reference [m]
    species_frac: float
        fraction of relevant impurity
    te: array
        electron temperature [keV]

    Returns
    -------
    species_frac*z_val**2: np.array
        species ion charge
    """
    z_interp = interp1d(t_ref, z_ref)
    z_val = z_interp(te)

    return species_frac * z_val**2


def radiative_loss_function_values(te, t_ref, l_ref):
    """
    By interpolation, from reference values, it returns the
    radiative power loss values for a given set of electron temperature.

    Parameters
    ----------
    te: np.array
        electron temperature [keV]
    t_ref: np.array
        temperature reference [keV]
    l_ref: np.array
        radiative power loss reference [Wm^3]

    Returns
    -------
    interp_func(te): np.array [W m^3]
        local values of the radiative power loss function
    """
    te_i = np.where(te < min(t_ref))
    te[te_i] = min(t_ref) + (np.finfo(float).eps)
    interp_func = interp1d(t_ref, l_ref)

    return interp_func(te)


def radiative_loss_function_plot(t_ref, lz_val, species):
    """
    Radiative loss function plot for a set of given impurities.

    Parameters
    ----------
    t_ref: np.array
        temperature reference [keV]
    l_z: [np.array]
        radiative power loss reference [Wm^3]
    species: [string]
        name species
    """
    fig, ax = plt.subplots()
    plt.title("Radiative loss functions vs Electron Temperature")
    plt.xlabel(r"$T_e~[keV]$")
    plt.ylabel(r"$L_z~[W.m^{3}]$")

    [ax.plot(t_ref, lz_specie, label=name) for lz_specie, name in zip(lz_val, species)]
    plt.xscale("log")
    plt.xlim(None, 10)
    plt.yscale("log")
    ax.legend(loc="best", borderaxespad=0, fontsize=12)

    return ax


def calculate_line_radiation_loss(ne, p_loss_f, species_frac):
    """
    Calculation of Line radiation losses.
    For a given impurity this is the total power lost, per unit volume,
    by all line-radiation processes INCLUDING Bremsstrahlung.

    Parameters
    ----------
    ne: np.array
        electron density [1/m^3]
    p_loss_f: np.array
        local values of the radiative power loss function
    species_frac: float
        fraction of relevant impurity

    Returns
    -------
    (species_frac[0] * (ne**2) * p_loss_f) * 1e-6: np.array
        Line radiation losses [MW m^-3]
    """

    return (species_frac * (ne**2) * p_loss_f) * 1e-6


def linear_interpolator(x, z, field):
    """
    Interpolating function calculated over 1D coordinate
    arrays and 1D field value array.

    Parameters
    ----------
    x: np.array
        x coordinates of given points [m]
    z: np.array
        z coordinates of given points [m]
    field: np.array
        set of punctual field values associated to the given points

    Returns
    -------
    interpolated_function: LinearNDInterpolator object
    """
    return LinearNDInterpolator(list(zip(x, z)), field, fill_value=0)


def interpolated_field_values(x, z, linear_interpolator):
    """
    Interpolated field values for a given set of points.

    Parameters
    ----------
    x: float, np.array
        x coordinates of point in which interpolate [m]
    z: float, np.array
        z coordinates of point in which interpolate [m]
    linear_interpolator: Interpolating function
        LinearNDInterpolator object

    Returns
    -------
    field_grid: matrix len(x) x len(z)
        2D grid of interpolated field values
    """
    xx, zz = np.meshgrid(x, z)
    return linear_interpolator(xx, zz)


def grid_interpolator(x, z, field_grid):
    """
    Interpolated field function obtainded for a given grid.
    Needed: length(xx) = m, length(zz) = n, field_grid.shape = n x m.

    Parameters
    ----------
    x: np.array
        x coordinates. length(xx)=m [m]
    z: np.array
        z coordinates. length(xx)=n [m]
    field_grid: matrix n x m
        corresponding field values arranged according to
        the grid of given points

    Returns
    -------
    interpolated_function: scipy.interpolate.interp2d object
        interpolated field function, to be used to
        calculate the field values for a new set of points
        or to be provided to a tracing code such as CHEARAB
    """
    return interp2d(x, z, field_grid)


def pfr_filter(separatrix: Grid, x_point_z: float):
    """
    To filter out from the radiation interpolation domain the private flux regions

    Parameters
    ----------
    separatrix: [Grid]
        Object containing x and z coordinates of each separatrix half
    x_point_z: float
        z coordinates of the x-point [m]

    Returns
    -------
    domains_x: np.array
        x coordinates of pfr domain
    domains_z: np.array
        z coordinates of pfr domain
    """
    # Identifying whether lower or upper x-point
    if x_point_z < 0:
        fact = 1
    else:
        fact = -1
    
    if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]
    # Selecting points between null and targets (avoiding the x-point singularity)
    z_ind = [
        np.where((halves.z * fact) < (x_point_z * fact - 0.01)) for halves in separatrix
    ]
    domains_x = [halves.x[list_ind] for list_ind, halves in zip(z_ind, separatrix)]
    domains_z = [halves.z[list_ind] for list_ind, halves in zip(z_ind, separatrix)]
    
    if len(domains_x) !=1:
        # Closing the domain
        domains_x[1] = np.append(domains_x[1], domains_x[0][0])
        domains_z[1] = np.append(domains_z[1], domains_z[0][0])

    return np.concatenate(domains_x), np.concatenate(domains_z)


def filtering_in_or_out(domain_x: list, domain_z: list, include_points=True):
    """
    To exclude from the calculation a specific region which is
    either contained or not contained within a given domain

    Parameters
    ----------
    domain_x: [float]
        list of x coordinates defining the domain
    domain_z: [float]
        list of x coordinates defining the domain
    include_points: boolean
        wheter the points inside or outside the domain must be excluded

    Returns
    -------
    include: shapely method
        method which includes or excludes from the domain a given point
    """
    region_data = np.zeros((len(domain_x), 2))
    region_data[:, 0] = domain_x
    region_data[:, 1] = domain_z
    shapely_region = shp.Polygon(region_data)

    def include(point):
        if include_points:
            return shapely_region.contains(point)
        else:
            return not shapely_region.contains(point)

    return include


# Adapted functions from Stuart
def detect_radiation(wall_detectors, n_samples, world):
    """
    To sample the wall and detect radiation
    """
    # Storage lists for results
    power_density = []
    detector_numbers = []
    distance = []
    detected_power = []
    detected_power_stdev = []
    detector_area = []
    power_density_stdev = []

    running_distance = 0
    cherab_total_power = 0

    # Loop over each tile detector
    for i, detector in enumerate(wall_detectors):

        print("detector {} / {}".format(i, len(wall_detectors) - 1))

        # extract the dimensions and orientation of the tile
        x_width = detector[1]
        y_width = detector[2]
        centre_point = detector[3]
        normal_vector = detector[4]
        y_vector = detector[5]
        pixel_area = x_width * y_width

        # Use the power pipeline to record total power arriving at the surface
        power_data = PowerPipeline0D()


        pixel_transform = translate(
            centre_point.x, centre_point.y, centre_point.z
        ) * rotate_basis(normal_vector, y_vector)
        # Use pixel_samples argument to increase amount of sampling and reduce noise
        pixel = Pixel(
            [power_data],
            x_width=x_width,
            y_width=y_width,
            name="pixel-{}".format(i),
            spectral_bins=1,
            transform=pixel_transform,
            parent=world,
            pixel_samples=n_samples,
        )
        # make detector sensitivity 1nm so that radiation function
        # is effectively W/m^3/str

        # Start collecting samples
        pixel.observe()

        # Append the collected data to the storage lists
        detector_radius = np.sqrt(centre_point.x**2 + centre_point.y**2)

        detector_area.append(pixel_area)
        power_density.append(
            power_data.value.mean / pixel_area
        )  # convert to W/m^2 !!!!!!!!!!!!!!!!!!!
        power_density_stdev.append(np.sqrt(power_data.value.variance) / pixel_area)
        detected_power.append(
            power_data.value.mean / pixel_area * (y_width * 2 * np.pi * detector_radius)
        )
        detected_power_stdev.append(np.sqrt(power_data.value.variance))
        detector_numbers.append(i)

        running_distance += 0.5 * y_width  # with Y_WIDTH instead of y_width
        distance.append(running_distance)
        running_distance += 0.5 * y_width  # with Y_WIDTH instead of y_width

        # For checking energy conservation.
        # Revolve this tile around the CYLINDRICAL z-axis
        # to get total power collected by these tiles.
        # Add up all the tile contributions to get total power collected.
        cherab_total_power += (power_data.value.mean / pixel_area) * (
            y_width * 2 * np.pi * detector_radius
        )

    output = {
        "power_density": power_density,
        "power_density_stdev": power_density_stdev,
        "detected_power": detected_power,
        "detected_power_stdev": detected_power_stdev,
        "detector_area": detector_area,
        "detector_numbers": detector_numbers,
        "distance": distance,
        "total_power": cherab_total_power,
    }

    return output


def build_wall_detectors(wall_r, wall_z, max_wall_len, x_width, debug=False):
    """
    To build the detectors on the wall
    """
    # number of detectors
    num = np.shape(wall_r)[0] - 2

    print("\n\n...building detectors...")

    # further initializations
    wall_detectors = []

    ctr = 0

    if debug:
        plt.figure()

    for index in range(0, num + 1):
        p1x = wall_r[index]
        p1y = wall_z[index]
        p1 = Point3D(p1x, 0, p1y)

        p2x = wall_r[index + 1]
        p2y = wall_z[index + 1]
        p2 = Point3D(p2x, 0, p2y)

        if p1 != p2:  # Ignore duplicate points

            # evaluate y_vector
            y_vector_full = p1.vector_to(p2)
            y_vector = y_vector_full.normalise()
            y_width = y_vector_full.length

            # Check if the wall element is small enough
            if y_width > max_wall_len:
                n_split = np.int(np.ceil(y_width / max_wall_len))

                # evaluate normal_vector
                normal_vector = Vector3D(p1y - p2y, 0.0, p2x - p1x).normalise()

                # Split up the wall component if necessary
                splitwall_x = np.linspace(p1x, p2x, num=n_split + 1)
                splitwall_y = np.linspace(p1y, p2y, num=n_split + 1)

                y_width = y_width / n_split

                for k in np.arange(n_split):
                    # evaluate the central point of the detector
                    detector_center = Point3D(
                        0.5 * (splitwall_x[k] + splitwall_x[k + 1]),
                        0,
                        0.5 * (splitwall_y[k] + splitwall_y[k + 1]),
                    )

                    # to populate it step by step
                    wall_detectors = wall_detectors + [
                        (
                            (ctr),
                            x_width,
                            y_width,
                            detector_center,
                            normal_vector,
                            y_vector,
                        )
                    ]

                    ctr = ctr + 1

            else:

                # evaluate normal_vector
                normal_vector = Vector3D(p1y - p2y, 0.0, p2x - p1x).normalise()
                # normal_vector = (detector_center.vector_to(plasma_axis_3D)).normalise()
                # inward pointing

                # evaluate the central point of the detector
                detector_center = Point3D(0.5 * (p1x + p2x), 0, 0.5 * (p1y + p2y))

                # to populate it step by step
                wall_detectors = wall_detectors + [
                    ((ctr), x_width, y_width, detector_center, normal_vector, y_vector)
                ]

                if debug:
                    plt.plot([p1x, p2x], [p1y, p2y], "k")
                    plt.plot([p1x, p2x], [p1y, p2y], ".k")
                    pc = detector_center
                    pcn = pc + normal_vector * 0.05
                    plt.plot([pc.x, pcn.x], [pc.z, pcn.z], "r")

                ctr = ctr + 1

    if debug:
        plt.show()

    return wall_detectors


def plot_radiation_loads(radiation_function, wall_detectors, wall_loads, plot_title, fw_shape):
    """
    To plot the radiation on the wall as MW/m^2
    """
    #min_r = 5.5
    #max_r = 12.5
    #min_z = -7
    #max_z = 5

    min_r = min(fw_shape.x)
    max_r = max(fw_shape.x)
    min_z = min(fw_shape.z)
    max_z = max(fw_shape.z)

    t_r, _, t_z, t_samples = sample3d(
        radiation_function, (min_r, max_r, 500), (0, 0, 1), (min_z, max_z, 1000)
    )

    # Plot the wall and radiation distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    gs = plt.GridSpec(2, 2, top=0.93, wspace=0.23)

    ax1 = plt.subplot(gs[0:2, 0])
    ax1.imshow(
        np.squeeze(t_samples).transpose() * 1.0e-6,
        extent=[min_r, max_r, min_z, max_z],
        clim=(0.0, np.amax(t_samples) * 1.0e-6),
    )

    segs = []

    for i in np.arange(len(wall_detectors)):
        wall_cen = wall_detectors[i][3]
        y_vector = wall_detectors[i][5]
        y_width = wall_detectors[i][2]

        end1 = wall_cen - 0.5 * y_width * y_vector
        end2 = wall_cen + 0.5 * y_width * y_vector

        segs.append([[end1.x, end1.z], [end2.x, end2.z]])

    wall_powerload = np.array(wall_loads["power_density"])

    line_segments = LineCollection(segs, cmap="hot")
    line_segments.set_array(wall_powerload)

    ax1.add_collection(line_segments)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=np.max(t_samples) * 1.0e-6)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.hot)
    cmap.set_array([])

    heat_cbar = plt.colorbar(cmap)
    heat_cbar.set_label(r"Wall Load ($MW.m^{-2}$)")

    ax2 = plt.subplot(gs[0, 1])

    ax2.plot(
        np.array(wall_loads["distance"]), np.array(wall_loads["power_density"]) * 1.0e-6
    )
    ax2.grid(True)
    ax2.set_ylim([0.0, 1.1 * np.max(np.array(wall_loads["power_density"]) * 1.0e-6)])

    plt.ylabel(r"Radiation Load ($MW.m^{-2}$)")

    ax3 = plt.subplot(gs[1, 1])

    plt.plot(
        np.array(wall_loads["distance"]),
        np.cumsum(np.array(wall_loads["detected_power"]) * 1.0e-6),
    )

    plt.ylabel(r"Total Power $[MW]$")
    plt.xlabel(r"Poloidal Distance $[m]$")
    ax3.grid(True)

    plt.suptitle(plot_title)

    plt.show()


class Radiation:
    """
    Initial and generic class (no distinction between core and SOL)
    to calculate radiation source within the flux tubes.
    """

    # fmt: off
    default_params = [
        ["n_e_sep", "Electron density at separatrix", 3e+19, "1/m^3", None, "Input"],
        ["T_e_sep", "Electron temperature at separatrix", 2e-01, "keV", None, "Input"],
    ]
    # fmt: on

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: TempFsSolver,
        params: Union[Dict, ParameterFrame],
    ):
        self.params = params
        
        self.flux_surf_solver = flux_surf_solver
        self.eq = eq

        # Constructors
        self.x_sep_omp = None
        self.x_sep_imp = None

        self.collect_x_and_o_point_coordinates()

        # Separatrix parameters
        self.collect_separatrix_parameters()

    def collect_x_and_o_point_coordinates(self):
        """
        Magnetic axis coordinates and x-point(s) coordinates.
        """
        o_point, x_point = self.eq.get_OX_points()
        self.points = {
            "x_point": {
                "x": x_point[0][0],
                "z_low": x_point[0][1],
                "z_up": x_point[1][1],
            },
            "o_point": {"x": o_point[0][0], "z": round(o_point[0][1], 5)},
        }
        if self.points["x_point"]["z_low"] > self.points["x_point"]["z_up"]:
            self.points["x_point"]["z_low"] = x_point[1][1]
            self.points["x_point"]["z_up"] = x_point[0][1]

    def collect_separatrix_parameters(self):
        """
        Radiation source relevant parameters at the separatrix
        """
        self.separatrix = self.eq.get_separatrix()
        z_mp = self.points["o_point"]["z"]
        if self.eq.is_double_null:
            # The two halves
            self.sep_lfs = self.separatrix[0]
            self.sep_hfs = self.separatrix[1]
        else:
            ob_ind = np.where(self.separatrix.x > self.points["x_point"]["x"])
            ib_ind = np.where(self.separatrix.x < self.points["x_point"]["x"])
            self.sep_ob = Coordinates({"x": self.separatrix.x[ob_ind], "z": self.separatrix.z[ob_ind]})
            self.sep_ib = Coordinates({"x": self.separatrix.x[ib_ind], "z": self.separatrix.z[ib_ind]})

        # The mid-plane radii
        self.x_sep_omp = self.flux_surf_solver.x_sep_omp
        # To move away from the mathematical separatrix which would
        # give infinite connection length
        self.r_sep_omp = self.x_sep_omp + SEP_CORRECTOR
        # magnetic field components at the midplane
        self.b_pol_sep_omp = self.eq.Bp(self.x_sep_omp, z_mp)
        b_tor_sep_omp = self.eq.Bt(self.x_sep_omp)
        self.b_tot_sep_omp = np.hypot(self.b_pol_sep_omp, b_tor_sep_omp)

        if self.eq.is_double_null:
            self.x_sep_imp = self.flux_surf_solver.x_sep_imp
            self.r_sep_imp = self.x_sep_imp - SEP_CORRECTOR
            self.b_pol_sep_imp = self.eq.Bp(self.x_sep_imp, z_mp)
            b_tor_sep_imp = self.eq.Bt(self.x_sep_imp)
            self.b_tot_sep_imp = np.hypot(self.b_pol_sep_imp, b_tor_sep_imp)

    def collect_flux_tubes(self, psi_n):
        """
        Collect flux tubes according to the normalised psi.
        For now only used for the core as for the SoL the
        flux surfaces to calculate the heat flux from charged
        particles are used.

        Parameters
        ----------
        psi_n: np.array
            normalised psi

        Returns
        -------
        flux tubes: list
            list of flux tubes
        """
        return [self.eq.get_flux_surface(psi) for psi in psi_n]

    def flux_tube_pol_t(
        self,
        flux_tube,
        te_mp,
        core=False,
        t_rad_in=None,
        t_rad_out=None,
        rad_i=None,
        rec_i=None,
        t_tar=None,
        x_point_rad=False,
        main_chamber_rad=False,
    ):
        """
        Along a single flux tube, it assigns different temperature values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        te_mp: float
            electron temperature at the midplane [keV]
        core: boolean
            if True, t is constant along the flux tube. If false,it varies
        t_rad_in: float
            temperature value at the entrance of the radiation region [keV]
        t_rad_out: float
            temperature value at the exit of the radiation region [keV]
        rad_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        t_tar: float
            electron temperature at the target [keV]
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        te: np.array
            poloidal distribution of electron temperature
        """
        te = np.array([te_mp] * len(flux_tube))

        if core is True:
            return te

        if rad_i is not None and len(rad_i) == 1:
            te[rad_i] = t_rad_in
        elif rad_i is not None and len(rad_i) > 1:
            te[rad_i] = gaussian_decay(t_rad_in, t_rad_out, len(rad_i), decay=True)

        if rec_i is not None and x_point_rad:
            te[rec_i] = exponential_decay(
                t_rad_out * 0.95, t_tar, len(rec_i), decay=True
            )
        elif rec_i is not None and x_point_rad is False:
            te[rec_i] = t_tar

        if main_chamber_rad:
            mask = np.ones_like(te, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            te[mask] = np.linspace(te_mp, t_rad_in, len(te[mask]))

        return te

    def flux_tube_pol_n(
        self,
        flux_tube,
        ne_mp,
        core=False,
        n_rad_in=None,
        n_rad_out=None,
        rad_i=None,
        rec_i=None,
        n_tar=None,
        main_chamber_rad=False,
    ):
        """
        Along a single flux tube, it assigns different density values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        ne_mp: float
            electron density at the midplane [1/m^3]
        core: boolean
            if True, n is constant along the flux tube. If false,it varies
        n_rad_in: float
            density value at the entrance of the radiation region [1/m^3]
        n_rad_out: float
            density value at the exit of the radiation region [1/m^3]
        rad_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        n_tar: float
            electron density at the target [1/m^3]
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        ne: np.array
            poloidal distribution of electron density
        """
        # initializing ne with same values all along the flux tube
        ne = np.array([ne_mp] * len(flux_tube))
        if core is True:
            return ne

        if rad_i is not None and len(rad_i) == 1:
            ne[rad_i] = n_rad_in
        elif rad_i is not None and len(rad_i) > 1:
            ne[rad_i] = gaussian_decay(n_rad_out, n_rad_in, len(rad_i), decay=False)

        if rec_i is not None and len(rad_i) == 1:
            ne[rec_i] = n_tar
        elif rec_i is not None and len(rec_i) > 0:
            ne[rec_i] = gaussian_decay(n_rad_out, n_tar, len(rec_i))
        if rec_i is not None and len(rec_i) > 0 and len(rad_i) > 1:
            gap = ne[rad_i[-1]] - ne[rad_i[-2]]
            ne[rec_i] = gaussian_decay(n_rad_out - gap, n_tar, len(rec_i))

        if main_chamber_rad:
            mask = np.ones_like(ne, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            ne[mask] = np.linspace(ne_mp, n_rad_in, len(ne[mask]))

        return ne

    def mp_profile_plot(self, rho, rad_power, imp_name, ax=None):
        """
        1D plot of the radation power distribution along the midplane.
        if rad_power.ndim > 1 the plot shows as many line as the number
        of impurities, plus the total radiated power

        Parameters
        ----------
        rho: np.array
            dimensionless radius. Values between 0 and 1 for the plasma core.
            Values > 1 for the scrape-off layer
        rad_power: np.array
            radiated power at each mid-plane location corresponding to rho [Mw/m^3]
        imp_name: [strings]
            impurity neames
        """
        if ax is None:
            fig, ax = plt.subplots()
            plt.xlabel(r"$\rho$")
            plt.ylabel(r"$[MW.m^{-3}]$")

        if len(rad_power) == 1:
            ax.plot(rho, rad_power)
        else:
            [
                ax.plot(rho, rad_part, label=name)
                for rad_part, name in zip(rad_power, imp_name)
            ]
            rad_tot = np.sum(np.array(rad_power, dtype=object), axis=0).tolist()
            plt.title("Core radiated power density")
            ax.plot(rho, rad_tot, label="total radiation")
            ax.legend(loc="best", borderaxespad=0, fontsize=12)

        return ax


class CoreRadiation(Radiation):
    """
    Specific class to calculate the core radiation source.
    Temperature and density are assumed to be constant along a
    single flux tube.
    """

    # fmt: off
    default_params = Radiation.default_params + [
        ["n_e_0", "Electron density on axis", 1.81e+20, "1/m^3", None, "Input"],
        ["T_e_0", "Electron temperature on axis", 2.196e+01, "keV", None, "Input"],
        ["rho_ped_n", "Density pedestal r/a location", 9.4e-01, "dimensionless", None, "Input"],
        ["rho_ped_t", "Temperature pedestal r/a location", 9.76e-01 , "dimensionless", None, "Input"],
        ["n_e_ped", "Electron density pedestal height", 1.086e+20, "1/m^3", None, "Input"],
        ["T_e_ped", "Electron temperature pedestal height", 3.74, "keV", None, "Input"],
        ["alpha_n", "Density profile factor", 1.15, "dimensionless", None, "Input"],
        ["alpha_t", "Temperature profile index", 1.905, "dimensionless", None, "Input"],
        ["t_beta", "Temperature profile index beta", 2, "dimensionless", None, "Input"],
    ]
    # fmt: on

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: TempFsSolver,
        params: ParameterFrame,
        impurity_content,
        impurity_data,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.H_content = impurity_content["H"]
        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "Ar"
        ]
        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "Ar"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "Ar"
        ]
        self.impurity_symbols = impurity_content.keys()

        # Adimensional radius at the mid-plane.
        # From the core to the last closed flux surface
        self.rho_core = self.collect_rho_core_values()

        # For each flux tube, density and temperature at the mid-plane
        self.ne_mp, self.te_mp = self.core_electron_density_temperature_profile(
            self.rho_core
        )

    def collect_rho_core_values(self):
        """
        Calculation of core dimensionless radial coordinate rho.

        Returns
        -------
        rho_core: np.array
            dimensionless core radius. Values between 0 and 1
        """
        # The plasma bulk is divided into plasma core and plasma mantle according to rho
        # rho is a nondimensional radial coordinate: rho = r/a (r varies from 0 to a)
        self.rho_ped = (self.params.rho_ped_n + self.params.rho_ped_t) / 2.0

        # Plasma core for rho < rho_core
        rho_core1 = np.linspace(0.01, 0.95 * self.rho_ped, 30)
        rho_core2 = np.linspace(0.95 * self.rho_ped, self.rho_ped, 15)
        rho_core = np.append(rho_core1, rho_core2)

        # Plasma mantle for rho_core < rho < 1
        rho_sep = np.linspace(self.rho_ped, 0.99, 10)

        rho_core = np.append(rho_core, rho_sep)

        return rho_core

    def core_electron_density_temperature_profile(self, rho_core):
        """
        Calculation of electron density and electron temperature,
        as function of rho, from the magnetic axis to the separatrix,
        along the midplane.
        The region that extends through the plasma core until its
        outer layer, named pedestal, is referred as "interior".
        The region that extends from the pedestal to the separatrix
        is referred as "exterior".

        Parameters
        ----------
        rho_core: np.array
            dimensionless core radius. Values between 0 and 1

        Returns
        -------
        ne: np.array
            electron densities at the mid-plane. Unit [1/m^3]
        te: np.array
            electron temperature at the mid-plane. Unit [keV]
        """
        i_interior = np.where((rho_core >= 0) & (rho_core <= self.rho_ped))[0]

        n_grad_ped0 = self.params.n_e_0 - self.params.n_e_ped
        t_grad_ped0 = self.params.T_e_0 - self.params.T_e_ped

        rho_ratio_n = (
            1 - ((rho_core[i_interior] ** 2) / (self.rho_ped**2))
        ) ** self.params.alpha_n

        rho_ratio_t = (
            1
            - (
                (rho_core[i_interior] ** self.params.t_beta)
                / (self.rho_ped**self.params.t_beta)
            )
        ) ** self.params.alpha_t

        ne_i = self.params.n_e_ped + (n_grad_ped0 * rho_ratio_n)
        te_i = self.params.T_e_ped + (t_grad_ped0 * rho_ratio_t)

        i_exterior = np.where((rho_core > self.rho_ped) & (rho_core <= 1))[0]

        n_grad_sepped = self.params.n_e_ped - self.params.n_e_sep
        t_grad_sepped = self.params.T_e_ped - self.params.T_e_sep

        rho_ratio = (1 - rho_core[i_exterior]) / (1 - self.rho_ped)

        ne_e = self.params.n_e_sep + (n_grad_sepped * rho_ratio)
        te_e = self.params.T_e_sep + (t_grad_sepped * rho_ratio)

        ne_core = np.append(ne_i, ne_e)
        te_core = np.append(te_i, te_e)

        return ne_core, te_core

    def build_mp_radiation_profile(self):
        """
        1D profile of the line radiation loss at the mid-plane
        through the scrape-off layer
        """
        # Radiative loss function values for each impurity species
        loss_f = [
            radiative_loss_function_values(self.te_mp, t_ref, l_ref)
            for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
        ]

        # Line radiation loss. Mid-plane distribution through the SoL
        self.rad = [
            calculate_line_radiation_loss(self.ne_mp, loss, fi)
            for loss, fi in zip(loss_f, self.impurities_content)
        ]

    def plot_mp_radiation_profile(self):
        """
        Plot one dimensional behaviour of line radiation
        against the adimensional radius
        """
        self.mp_profile_plot(self.rho_core, self.rad, self.impurity_symbols)

    def build_core_distribution(self):
        """
        Build poloidal distribution (distribution along the field lines) of
        line radiation loss in the plasma core.

        Returns
        -------
        rad: [np.array]
            Line core radiation.
            For specie and each closed flux line in the core
        """
        # Closed flux tubes within the separatrix
        psi_n = self.rho_core**2
        self.flux_tubes = self.collect_flux_tubes(psi_n)

        # For each flux tube, poloidal density profile.
        self.ne_pol = [
            self.flux_tube_pol_n(ft, n, core=True)
            for ft, n in zip(self.flux_tubes, self.ne_mp)
        ]

        # For each flux tube, poloidal temperature profile.
        self.te_pol = [
            self.flux_tube_pol_t(ft, t, core=True)
            for ft, t in zip(self.flux_tubes, self.te_mp)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        self.loss_f = [
            [radiative_loss_function_values(t, t_ref, l_ref) for t in self.te_pol]
            for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        self.rad = [
            [
                calculate_line_radiation_loss(n, l_f, fi)
                for n, l_f in zip(self.ne_pol, ft)
            ]
            for ft, fi in zip(self.loss_f, self.impurities_content)
        ]

        return self.rad

    def build_core_radiation_map(self):
        """
        Build core radiation map.
        """
        rad = self.build_core_distribution()
        self.total_rad = np.sum(np.array(rad, dtype=object), axis=0).tolist()

        self.x_tot = np.concatenate([flux_tube.x for flux_tube in self.flux_tubes])
        self.z_tot = np.concatenate([flux_tube.z for flux_tube in self.flux_tubes])
        self.rad_tot = np.concatenate(self.total_rad)

    def radiation_distribution_plot(self, flux_tubes, power_density, ax=None):
        """
        2D plot of the core radation power distribution.

        Parameters
        ----------
        flux_tubes: np.array
            array of the closed flux tubes within the separatrix.
        power_density: np.array([np.array])
            arrays containing the power radiation density of the
            points lying on each flux tube [MW/m^3]
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min([np.amin(p) for p in power_density])
        p_max = max([np.amax(p) for p in power_density])

        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(flux_tubes, power_density):
            cm = ax.scatter(
                flux_tube.x,
                flux_tube.z,
                c=p,
                s=10,
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=40,
            )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax

    def plot_radiation_distribution(self):
        """
        Plot poloiadal radiation distribution
        within the plasma core
        """
        self.radiation_distribution_plot(self.flux_tubes, self.total_rad)

    def plot_lz_vs_tref(self):
        """
        Plot radiative loss function for a set of given impurities
        against the reference temperature.
        """
        radiative_loss_function_plot(
            self.imp_data_t_ref[0], self.imp_data_l_ref, self.impurity_symbols
        )


class ScrapeOffLayerRadiation(Radiation):
    """
    Specific class to calculate the SOL radiation source.
    In the SOL is assumed a conduction dominated regime until the
    x-point, with no heat sinks, and a convection dominated regime
    between x-point and target.
    """

    # fmt: off
    default_params = Radiation.default_params + [
        ['P_sep', 'Radiation power', 150, 'MW', None, 'Input'],
        ['fw_lambda_q_near_omp', 'Lambda_q near SOL omp', 0.002, 'm', None, 'Input'],
        ['fw_lambda_q_far_omp', 'Lambda_q far SOL omp', 0.10, 'm', None, 'Input'],
        ['fw_lambda_q_near_imp', 'Lambda_q near SOL imp', 0.002, 'm', None, 'Input'],
        ['fw_lambda_q_far_imp', 'Lambda_q far SOL imp', 0.10, 'm', None, 'Input'],
        ["k_0", "material's conductivity", 2000, "dimensionless", None, "Input"],
        ["gamma_sheath", "sheath heat transmission coefficient", 7, "dimensionless", None, "Input"],
        ["eps_cool", "electron energy loss", 25, "eV", None, "Input"],
        ["f_ion_t", "Hydrogen first ionization", 0.01, "keV", None, "Input"],
        ["det_t", "Detachment target temperature", 0.0015, "keV", None, "Input"],
        ["lfs_p_fraction", "lfs fraction of SoL power", 0.9, "dimensionless", None, "Input"],
        ["theta_outer_target", "Outer divertor poloidal angle with the separatrix flux line", 5, "deg", None, "Input"],
        ["theta_inner_target", "Inner divertor poloidal angle with the separatrix flux line", 5, "deg", None, "Input"],
    ]
    # fmt: on

    def collect_rho_sol_values(self):
        """
        Calculation of SoL dimensionless radial coordinate rho.

        Returns
        -------
        rho_sol: np.array
            dimensionless sol radius. Values higher than 1
        """
        # The dimensionless radius has plasma-core upper limit equal 1.
        # Such limit is the bottom one for the SoL
        r_o_point = self.points["o_point"]["x"]
        a = self.flux_surf_solver.x_sep_omp - r_o_point
        rho_sol = (a + self.flux_surf_solver.dx_omp) / a

        return rho_sol

    def x_point_radiation_z_ext(self, main_ext=None, pfr_ext=0.3, low_div=True):
        """
        Simple definition of a radiation region around the x-point.
        The region is supposed to extend from an arbitrary z coordinate on the
        main plasma side, to an arbitrary z coordinate on the private flux region side.

        Parameters
        ----------
        main_ext: float [m]
            region extension on the main plasma side
        pfr_ext: float [m]
            region extension on the private flux region side
        low_div: boolean
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the main plasma
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr
        """
        if main_ext is None:
            main_ext = abs(self.points["x_point"]["z_low"])

        if low_div is True:
            z_main = self.points["x_point"]["z_low"] + main_ext
            z_pfr = self.points["x_point"]["z_low"] - pfr_ext
        else:
            z_main = self.points["x_point"]["z_up"] - main_ext
            z_pfr = self.points["x_point"]["z_up"] + pfr_ext

        return z_main, z_pfr

    def radiation_region_ends(self, z_main, z_pfr, lfs=True):
        """
        Entering and exiting points (x, z) of the radiation region
        detected on the separatrix.
        The separatrix is supposed to be given by relevant half.

        Parameters
        ----------
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the main plasma
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr
        lfs: boolean
            default=True for the low field side (right half).
            If False, high field side (left half).

        Returns
        -------
        entrance_x, entrance_z: float, float
            x, z coordinates of the radiation region starting point
        exit_x, exit_z: float, float
            x, z coordinates of the radiation region ending point
        """
        if self.eq.is_double_null:
            sep_loop = self.sep_lfs if lfs else self.sep_hfs
        else:
            sep_loop = self.sep_ob if lfs else self.sep_ib
        if z_main > z_pfr:
            reg_i = np.where((sep_loop.z < z_main) & (sep_loop.z > z_pfr))[0]
            i_in = np.where(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]
            i_out = np.where(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
        else:
            reg_i = np.where((sep_loop.z > z_main) & (sep_loop.z < z_pfr))[0]
            i_in = np.where(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
            i_out = np.where(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]

        entrance_x, entrance_z = sep_loop.x[i_in], sep_loop.z[i_in]
        exit_x, exit_z = sep_loop.x[i_out], sep_loop.z[i_out]

        return entrance_x[0], entrance_z[0], exit_x[0], exit_z[0]

    def radiation_region_points(self, flux_tube, z_main, z_pfr, lower=True):
        """
        For a given flux tube, indexes of points which fall respectively
        into the radiation and recycling region

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region toward
            the main plasma.
            Taken on the separatrix
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr.
            Taken on the separatrix
        lower: boolean
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        rad_i: np.array
            indexes of the points within the radiation region
        rec_i: np.array
            indexes pf the points within the recycling region
        """
        if lower:
            rad_i = np.where((flux_tube.z < z_main) & (flux_tube.z > z_pfr))[0]
            rec_i = np.where(flux_tube.z < z_pfr)[0]
        else:
            rad_i = np.where((flux_tube.z > z_main) & (flux_tube.z < z_pfr))[0]
            rec_i = np.where(flux_tube.z > z_pfr)[0]

        return rad_i, rec_i

    def mp_electron_density_temperature_profiles(self, te_sep=None, omp=True):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at midplane.
        It uses the customised version for the mid-plane of the exponential
        decay law described in "electron_density_and_temperature_sol_decay".

        Parameters
        ----------
        te_sep: float
            electron temperature at the separatrix [keV]
        omp: boolean
            outer mid-plane. Default value True. If False it stands for inner mid-plane

        Returns
        -------
        te_sol: np.array
            radial decayed temperatures through the SoL at the mid-plane. Unit [keV]
        ne_sol: np.array
            radial decayed densities through the SoL at the mid-plane. Unit [1/m^3]
        """
        if omp or not self.eq.is_double_null:
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp
            dx = self.flux_surf_solver.dx_omp
        else:
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp
            dx = self.flux_surf_solver.dx_imp

        if te_sep is None:
            te_sep = self.params.T_e_sep
        ne_sep = self.params.n_e_sep

        te_sol, ne_sol = electron_density_and_temperature_sol_decay(
            te_sep,
            ne_sep,
            fw_lambda_q_near,
            fw_lambda_q_far,
            dx,
        )

        return te_sol, ne_sol

    def any_point_density_temperature_profiles(
        self,
        x_p: float,
        z_p: float,
        t_p: float,
        t_u: float,
        lfs=True,
    ):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL, starting from any point on the separatrix.
        (The z coordinate is the same. While the x coordinate changes)
        Using the equation to calculate T(s||).

        Parameters
        ----------
        x_p: float
            x coordinate of the point at the separatrix [m]
        z_p: float
            z coordinate of the point at the separatrix [m]
        t_p: float
            point temperature [keV]
        t_u: float
            upstream temperature [keV]
        lfs: boolean
            low (toroidal) field side (outer wall side). Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        te_prof: np.array
            radial decayed temperatures through the SoL. Unit [keV]
        ne_prof: np.array
            radial decayed densities through the SoL. Unit [1/m^3]
        """
        # Distinction between lfs and hfs
        if lfs is True or not self.eq.is_double_null:
            r_sep_mp = self.r_sep_omp
            b_pol_sep_mp = self.b_pol_sep_omp
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp
            dx = self.flux_surf_solver.dx_omp
        else:
            r_sep_mp = self.r_sep_imp
            b_pol_sep_mp = self.b_pol_sep_imp
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp
            dx = self.flux_surf_solver.dx_imp

        # magnetic field components at the local point
        b_pol_p = self.eq.Bp(x_p, z_p)

        # flux expansion
        f_p = (r_sep_mp * b_pol_sep_mp) / (x_p * b_pol_p)

        # Ratio between upstream and local temperature
        f_t = t_u / t_p

        # Local electron density
        n_p = self.params.n_e_sep * f_t

        # Temperature and density profiles across the SoL
        te_prof, ne_prof = electron_density_and_temperature_sol_decay(
            t_p,
            n_p,
            fw_lambda_q_near,
            fw_lambda_q_far,
            dx,
            f_exp=f_p,
        )

        return te_prof, ne_prof

    def tar_electron_densitiy_temperature_profiles(
        self, ne_div, te_div, f_m=1, detachment=False
    ):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at the target.
        From the pressure balance, considering friction losses.
        f_m is the fractional loss of pressure due to friction.
        It can vary between 0 and 1.

        Parameters
        ----------
        ne_div: np.array
            density of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [1/m^3]
        te_div: np.array
            temperature of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [keV]
        f_m: float
            fractional loss factor

        Returns
        -------
        te_t: np.array
            target temperature [keV]
        ne_t: np.array
            target density [1/m^3]
        """
        if not detachment:
            te_t = te_div
            f_m = f_m
        else:
            te_t = [self.params.det_t] * len(te_div)
            f_m = 0.1
        ne_t = (f_m * ne_div) / 2

        return te_t, ne_t

    def build_sector_distributions(
        self,
        flux_tubes,
        x_strike,
        z_strike,
        main_ext,
        firstwall_geom,
        pfr_ext=None,
        rec_ext=None,
        x_point_rad=False,
        detachment=False,
        lfs=True,
        low_div=True,
        main_chamber_rad=False,
    ):
        """
        Temperature and density profiles builder.
        Within the scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        flux_tubes: array
            set of flux tubes
        x_strike: float
            x coordinate of the first open flux surface strike point [m]
        z_strike: float
            z coordinate of the first open flux surface strike point [m]
        main_ext: float
            extention of the radiation region from the x-point
            towards the main plasma [m]
        firstwall_geom: grid
            first wall geometry
        pfr_ext: float
            extention of the radiation region from the x-point
            towards private flux region [m]
        rec_ext: float
            extention of the recycling region,
            along the separatrix, from the target [m]
        x_point_rad: boolean
            if True, it assumes there is no radiation at all
            in the recycling region, and pfr_ext MUST be provided.
        detachment: boolean
            if True, it makes the temperature decay through the
            recycling region from the H ionization temperature to
            the assign temperature for detachment at the target.
            Else temperature is constant through the recycling region.
        lfs: boolean
            low field side. Default value True.
            If False it stands for high field side (hfs)
        low_div: boolean
            default=True for the lower divertor.
            If False, upper divertor
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        t_pol: array
            temperature poloidal profile along each
            flux tube within the specified set [keV]
        n_pol: array
            density poloidal profile along each
            flux tube within the specified set [1/m^3]
        """
        # Validity condition for not x-point radiative
        if not x_point_rad and rec_ext is None:
            raise BuilderError("Required recycling region extention: rec_ext")
        elif not x_point_rad and rec_ext is not None and lfs:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.flux_surf_solver.eq,
                self.points["x_point"]["z_low"],
                rec_ext=2,
            )
            pfr_ext = abs(ion_front_z)

        elif not x_point_rad and rec_ext is not None and lfs is False:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.flux_surf_solver.eq,
                self.points["x_point"]["z_low"],
                rec_ext=0.2,
            )
            pfr_ext = abs(ion_front_z)

        # Validity condition for x-point radiative
        elif x_point_rad and pfr_ext is None:
            raise BuilderError("Required extention towards pfr: pfr_ext")

        # setting radiation and recycling regions
        z_main, z_pfr = self.x_point_radiation_z_ext(main_ext, pfr_ext, low_div)

        in_x, in_z, out_x, out_z = self.radiation_region_ends(z_main, z_pfr, lfs)

        reg_i = [
            self.radiation_region_points(f.coords, z_main, z_pfr, low_div)
            for f in flux_tubes
        ]

        # mid-plane parameters
        if lfs or not self.eq.is_double_null:
            t_u_kev = self.t_omp
            b_pol_tar = self.b_pol_out_tar
            b_pol_u = self.b_pol_sep_omp
            r_sep_mp = self.r_sep_omp
            # alpha = self.params.theta_outer_target
            alpha = self.alpha_lfs
            b_tot_tar = self.b_tot_out_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
        else:
            t_u_kev = self.t_imp
            b_pol_tar = self.b_pol_inn_tar
            b_pol_u = self.b_pol_sep_imp
            r_sep_mp = self.r_sep_imp
            # alpha = self.params.theta_inner_target
            alpha = self.alpha_hfs
            b_tot_tar = self.b_tot_inn_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp

        # Coverting needed parameter units
        t_u_ev = constants.raw_uc(t_u_kev, "keV", "eV")
        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")
        f_ion_t = constants.raw_uc(self.params.f_ion_t, "keV", "eV")

        if lfs and self.eq.is_double_null:
            p_sol = p_sol*self.params.lfs_p_fraction
        elif not lfs and self.eq.is_double_null:
            p_sol = p_sol*(1-self.params.lfs_p_fraction)

        t_mp_prof, n_mp_prof = self.mp_electron_density_temperature_profiles(
            t_u_kev, lfs
        )

        # entrance of radiation region
        t_rad_in = random_point_temperature(
            in_x,
            in_z,
            t_u_ev,
            p_sol,
            fw_lambda_q_near,
            self.eq,
            r_sep_mp,
            self.points["o_point"]["z"],
            self.params.k_0,
            firstwall_geom,
            lfs,
        )

        # exit of radiation region
        if x_point_rad and pfr_ext is not None:
            t_rad_out = self.params.f_ion_t
        elif detachment:
            t_rad_out = self.params.f_ion_t
        else:
            t_rad_out = target_temperature(
                p_sol,
                t_u_ev,
                self.params.n_e_sep,
                self.params.gamma_sheath,
                self.params.eps_cool,
                f_ion_t,
                b_pol_tar,
                b_pol_u,
                alpha,
                r_sep_mp,
                x_strike,
                fw_lambda_q_near,
                b_tot_tar,
            )

        # condition for occurred detachment
        if t_rad_out <= self.params.f_ion_t:
            x_point_rad = True
            detachment = True

        # profiles through the SoL
        t_in_prof, n_in_prof = self.any_point_density_temperature_profiles(
            in_x,
            in_z,
            t_rad_in,
            t_u_kev,
            lfs,
        )

        t_out_prof, n_out_prof = self.any_point_density_temperature_profiles(
            out_x,
            out_z,
            t_rad_out,
            t_u_kev,
            lfs,
        )

        t_tar_prof, n_tar_prof = self.tar_electron_densitiy_temperature_profiles(
            n_out_prof,
            t_out_prof,
            detachment=detachment,
        )

        # temperature poloidal distribution
        t_pol = [
            self.flux_tube_pol_t(
                f.coords,
                t,
                t_rad_in=t_in,
                t_rad_out=t_out,
                rad_i=reg[0],
                rec_i=reg[1],
                t_tar=t_t,
                x_point_rad=x_point_rad,
                main_chamber_rad=main_chamber_rad,
            )
            for f, t, t_in, t_out, reg, t_t, in zip(
                flux_tubes,
                t_mp_prof,
                t_in_prof,
                t_out_prof,
                reg_i,
                t_tar_prof,
            )
        ]
        #print(n_mp_prof[0], n_in_prof[0], n_out_prof[0], n_tar_prof[0])
        #print(t_mp_prof[0], t_in_prof[0], t_out_prof[0], t_tar_prof[0])
        # density poloidal distribution
        n_pol = [
            self.flux_tube_pol_n(
                f.coords,
                n,
                n_rad_in=n_in,
                n_rad_out=n_out,
                rad_i=reg[0],
                rec_i=reg[1],
                n_tar=n_t,
                main_chamber_rad=main_chamber_rad,
            )
            for f, n, n_in, n_out, reg, n_t, in zip(
                flux_tubes,
                n_mp_prof,
                n_in_prof,
                n_out_prof,
                reg_i,
                n_tar_prof,
            )
        ]

        return t_pol, n_pol

    def radiation_distribution_plot(self, flux_tubes, power_density, firstwall, ax=None):
        """
        2D plot of the radation power distribution.

        Parameters
        ----------
        flux_tubes: [np.array]
            list of arrays of the flux tubes of different sectors.
            Example: if len(flux_tubes) = 2, it could contain the set
            of flux tubes of the lower that go from the omp to the
            outer lower divertor, and the set of flux tubes that go from
            the imp to the inner lower divertor.
        power_density: [np.array]
            list of arrays containing the power radiation density of the
            points lying on the specified flux tubes.
            expected len(flux_tubes) = len(power_density)
        firstwall: Grid
            first wall geometry
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        tubes = sum(flux_tubes, [])
        power = sum(power_density, [])

        p_min = min([min(p) for p in power])
        p_max = max([max(p) for p in power])
        
        plot_coordinates(firstwall, ax=ax, linewidth=0.5, fill=False)
        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(tubes, power):
            cm = ax.scatter(
                flux_tube.coords.x,
                flux_tube.coords.z,
                c=p,
                s=10,
                marker=".",
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=40,
            )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax

    def poloidal_distribution_plot(
        self, flux_tubes, property, temperature=True, ax=None
    ):
        """
        2D plot of a generic property (density, temperature or radiation)
        as poloidal section the flux tube points.

        Parameters
        ----------
        flux_tubes: np.array
            array of the open flux tubes within the SoL.
        property: np.array([np.array])
            arrays containing the property values associated
            to the points of each flux tube.
        """
        if ax is None:
            _, ax = plt.subplots()
        else:
            _ = ax.figure
        plt.xlabel("Mid-Plane to Target")
        if temperature is True:
            plt.title("Temperature along flux surfaces")
            plt.ylabel(r"$T_e~[keV]$")
        else:
            plt.title("Density along flux surfaces")
            plt.ylabel(r"$n_e~[m^{-3}]$")

        [
            ax.plot(np.linspace(0, len(flux_tube.coords.x), len(flux_tube.coords.x)), val)
            for flux_tube, val in zip(flux_tubes, property)
        ]

        return ax

    def plot_t_vs_n(
        self, flux_tube, t_distribution, n_distribution, rad_distribution=None, ax1=None
    ):
        """
        2D plot of temperature and density of a single flux tube within the SoL

        Parameters
        ----------
        flux_tube: flux surface object as described in advective_transport.py
            open flux tube within the SoL.
        t_distribution: np.array([np.array])
            arrays containing the temperature values associated
            to the points of the flux tube.
        n_distribution: np.array([np.array])
            arrays containing the density values associated
            to the points of the flux tube.
        """
        if ax1 is None:
            _, ax1 = plt.subplots()
        else:
            _ = ax1.figure

        ax2 = ax1.twinx()
        x = np.linspace(0, len(flux_tube.coords.x), len(flux_tube.coords.x))
        y1 = t_distribution
        y2 = n_distribution
        ax1.plot(x, y1, "g-")
        ax2.plot(x, y2, "b-")

        plt.title("Electron Temperature vs Electron Density")
        ax1.set_xlabel("Mid-Plane to Target")
        ax1.set_ylabel(r"$T_e~[keV]$", color="g")
        ax2.set_ylabel(r"$n_e~[m^{-3}]$", color="b")

        return ax1, ax2


class DNScrapeOffLayerRadiation(ScrapeOffLayerRadiation):
    """
    Specific class to build the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: TempFsSolver,
        params: ParameterFrame,
        impurity_content,
        impurity_data,
        firstwall_geom,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]

        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.eq=eq
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs_low = self.flux_surf_solver.flux_surfaces_ob_lfs
        self.flux_tubes_hfs_low = self.flux_surf_solver.flux_surfaces_ib_lfs

        # partial flux tube from the mp to the target at the
        # outboard and inboard - upper divertor
        self.flux_tubes_lfs_up = self.flux_surf_solver.flux_surfaces_ob_hfs
        self.flux_tubes_hfs_up = self.flux_surf_solver.flux_surfaces_ib_hfs

        # strike points from the first open flux tube
        self.x_strike_lfs = self.flux_tubes_lfs_low[0].coords.x[-1]
        self.z_strike_lfs = self.flux_tubes_lfs_low[0].coords.z[-1]
        self.alpha_lfs = self.flux_tubes_lfs_low[0].alpha

        self.b_pol_out_tar = self.eq.Bp(self.x_strike_lfs, self.z_strike_lfs)
        self.b_tor_out_tar = self.eq.Bt(self.x_strike_lfs)
        self.b_tot_out_tar = np.hypot(self.b_pol_out_tar, self.b_tor_out_tar)

        self.x_strike_hfs = self.flux_tubes_hfs_low[0].coords.x[-1]
        self.z_strike_hfs = self.flux_tubes_hfs_low[0].coords.z[-1]
        self.alpha_hfs = self.flux_tubes_hfs_low[0].alpha

        self.b_pol_inn_tar = self.eq.Bp(self.x_strike_hfs, self.z_strike_hfs)
        self.b_tor_inn_tar = self.eq.Bt(self.x_strike_hfs)
        self.b_tot_inn_tar = np.hypot(self.b_pol_inn_tar, self.b_tor_inn_tar)

        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")
        p_sol_lfs = p_sol*self.params.lfs_p_fraction
        p_sol_hfs = p_sol*(1-self.params.lfs_p_fraction)

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp,
            p_sol=p_sol_lfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.z_mp,
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
            lfs=True,
        )

        self.t_imp = upstream_temperature(
            b_pol=self.b_pol_sep_imp,
            b_tot=self.b_tot_sep_imp,
            lambda_q_near=self.params.fw_lambda_q_near_imp,
            p_sol=p_sol_hfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_imp,
            z_mp=self.z_mp,
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
            lfs=False,
        )

    def build_sol_distribution(self, firstwall_geom: Grid):
        """
        Temperature and density profiles builder.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol["lfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol["hfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol["hfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        t_and_n_pol_inputs = {
            f"{side}_{low_up}": {
                "flux_tubes": getattr(self, f"flux_tubes_{side}_{low_up}"),
                "x_strike": getattr(self, f"x_strike_{side}"),
                "z_strike": getattr(self, f"z_strike_{side}"),
                "main_ext": None,
                "firstwall_geom": firstwall_geom,
                "pfr_ext": None,
                "rec_ext": 2,
                "x_point_rad": False,
                "detachment": False,
                "lfs": side == "lfs",
                "low_div": low_up == "low",
                "main_chamber_rad": True,
            }
            for side in ["lfs", "hfs"]
            for low_up in ["low", "up"]
        }

        self.t_and_n_pol = {}
        for side, var in t_and_n_pol_inputs.items():
            self.t_and_n_pol[side] = self.build_sector_distributions(**var)

        return self.t_and_n_pol

    def build_sol_radiation_distribution(
        self,
        t_and_n_pol_lfs_low,
        t_and_n_pol_lfs_up,
        t_and_n_pol_hfs_low,
        t_and_n_pol_hfs_up,
    ):
        """
        Radiation profiles builder.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        t_and_n_pol_lfs_low: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol_lfs_up: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol_hfs_low: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol_hfs_up: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss = {
            "lfs_low": t_and_n_pol_lfs_low[0],
            "lfs_up": t_and_n_pol_lfs_up[0],
            "hfs_low": t_and_n_pol_hfs_low[0],
            "hfs_up": t_and_n_pol_hfs_up[0],
        }

        for side, t_pol in loss.items():
            loss[side] = [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        self.rad = {
            "lfs_low": {"density": t_and_n_pol_lfs_low[1], "loss": loss["lfs_low"]},
            "lfs_up": {"density": t_and_n_pol_lfs_up[1], "loss": loss["lfs_up"]},
            "hfs_low": {"density": t_and_n_pol_hfs_low[1], "loss": loss["hfs_low"]},
            "hfs_up": {"density": t_and_n_pol_hfs_up[1], "loss": loss["hfs_up"]},
        }
        for side, ft in self.rad.items():
            self.rad[side] = [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content)
            ]
        return self.rad

    def build_sol_radiation_map(self, rad_lfs_low, rad_lfs_up, rad_hfs_low, rad_hfs_up):
        """
        Scrape off layer radiation map builder.

        Parameters
        ----------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        firstwall_geom: grid
            first wall geometry
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs_low = np.sum(
            np.array(rad_lfs_low, dtype=object), axis=0
        ).tolist()
        self.total_rad_lfs_up = np.sum(
            np.array(rad_lfs_up, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs_low = np.sum(
            np.array(rad_hfs_low, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs_up = np.sum(
            np.array(rad_hfs_up, dtype=object), axis=0
        ).tolist()

        rads = [
            self.total_rad_lfs_low,
            self.total_rad_hfs_low,
            self.total_rad_lfs_up,
            self.total_rad_hfs_up,
        ]
        power = sum(rads, [])

        flux_tubes = [
            self.flux_surf_solver.flux_surfaces_ob_lfs,
            self.flux_surf_solver.flux_surfaces_ib_lfs,
            self.flux_surf_solver.flux_surfaces_ob_hfs,
            self.flux_surf_solver.flux_surfaces_ib_hfs,
        ]
        flux_tubes = sum(flux_tubes, [])
    
        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(power)

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Grid):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall: Grid
            first wall geometry
        """
        self.radiation_distribution_plot(
            [
                self.flux_tubes_lfs_low,
                self.flux_tubes_hfs_low,
                self.flux_tubes_lfs_up,
                self.flux_tubes_hfs_up,
            ],
            [
                self.total_rad_lfs_low,
                self.total_rad_hfs_low,
                self.total_rad_lfs_up,
                self.total_rad_hfs_up,
            ],
            firstwall_geom,
        )

class SNScrapeOffLayerRadiation(ScrapeOffLayerRadiation):
    """
    Specific class to build the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: TempFsSolver,
        params: ParameterFrame,
        impurity_content,
        impurity_data,
        firstwall_geom,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]

        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.eq=eq
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs = self.flux_surf_solver.flux_surfaces_ob_lfs
        self.flux_tubes_hfs = self.flux_surf_solver.flux_surfaces_ob_hfs

        # strike points from the first open flux tube
        self.x_strike_lfs = self.flux_tubes_lfs[0].coords.x[-1]
        self.z_strike_lfs = self.flux_tubes_lfs[0].coords.z[-1]
        self.alpha_lfs = self.flux_tubes_lfs[0].alpha

        self.b_pol_out_tar = self.eq.Bp(self.x_strike_lfs, self.z_strike_lfs)
        self.b_tor_out_tar = self.eq.Bt(self.x_strike_lfs)
        self.b_tot_out_tar = np.hypot(self.b_pol_out_tar, self.b_tor_out_tar)

        self.x_strike_hfs = self.flux_tubes_hfs[0].coords.x[-1]
        self.z_strike_hfs = self.flux_tubes_hfs[0].coords.z[-1]
        self.alpha_hfs = self.flux_tubes_hfs[0].alpha

        self.b_pol_inn_tar = self.eq.Bp(self.x_strike_hfs, self.z_strike_hfs)
        self.b_tor_inn_tar = self.eq.Bt(self.x_strike_hfs)
        self.b_tot_inn_tar = np.hypot(self.b_pol_inn_tar, self.b_tor_inn_tar)

        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp,
            p_sol=p_sol,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.points["o_point"]["z"],
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
        )

    def build_sol_distribution(self, firstwall_geom: Grid):
        """
        Temperature and density profiles builder.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol["lfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol["hfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol["hfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        t_and_n_pol_inputs = {
            f"{side}": {
                "flux_tubes": getattr(self, f"flux_tubes_{side}"),
                "x_strike": getattr(self, f"x_strike_{side}"),
                "z_strike": getattr(self, f"z_strike_{side}"),
                "main_ext": 1,
                "firstwall_geom": firstwall_geom,
                "pfr_ext": None,
                "rec_ext": 2,
                "x_point_rad": False,
                "detachment": False,
                "lfs": side == "lfs",
                "low_div": True,
                "main_chamber_rad": True,
            }
            for side in ["lfs", "hfs"]
        }

        self.t_and_n_pol = {}
        for side, var in t_and_n_pol_inputs.items():
            self.t_and_n_pol[side] = self.build_sector_distributions(**var)

        return self.t_and_n_pol

    def build_sol_radiation_distribution(
        self,
        t_and_n_pol_lfs,
        t_and_n_pol_hfs,
    ):
        """
        Radiation profiles builder.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        t_and_n_pol_lfs_low: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol_lfs_up: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol_hfs_low: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol_hfs_up: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss = {
            "lfs": t_and_n_pol_lfs[0],
            "hfs": t_and_n_pol_hfs[0],
        }

        for side, t_pol in loss.items():
            loss[side] = [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        self.rad = {
            "lfs": {"density": t_and_n_pol_lfs[1], "loss": loss["lfs"]},
            "hfs": {"density": t_and_n_pol_hfs[1], "loss": loss["hfs"]},
        }
        for side, ft in self.rad.items():
            self.rad[side] = [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content)
            ]
        return self.rad

    def build_sol_radiation_map(self, rad_lfs, rad_hfs):
        """
        Scrape off layer radiation map builder.

        Parameters
        ----------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        firstwall_geom: grid
            first wall geometry
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs = np.sum(
            np.array(rad_lfs, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs = np.sum(
            np.array(rad_hfs, dtype=object), axis=0
        ).tolist()

        rads = [
            self.total_rad_lfs,
            self.total_rad_hfs,
        ]
        power = sum(rads, [])

        flux_tubes = [
            self.flux_tubes_lfs,
            self.flux_tubes_hfs,
        ]
        flux_tubes = sum(flux_tubes, [])

        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(power)

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Grid):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall: Grid
            first wall geometry
        """
        self.radiation_distribution_plot(
            [
                self.flux_tubes_lfs,
                self.flux_tubes_hfs,
            ],
            [
                self.total_rad_lfs,
                self.total_rad_hfs,
            ],
            firstwall_geom,
        )


class RadiationSolver:
    """
    Simplified solver to easily access the radiation model location inputs.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: TempFsSolver,
        params: ParameterFrame,
        impurity_content_core,
        impurity_data_core,
        impurity_content_sol,
        impurity_data_sol,
    ):

        self.eq = eq
        self.flux_surf_solver = flux_surf_solver
        self.params = self._make_params(params)
        self.imp_content_core = impurity_content_core
        self.imp_data_core = impurity_data_core
        self.imp_content_sol = impurity_content_sol
        self.imp_data_sol = impurity_data_sol
        self.lcfs = self.eq.get_LCFS()

        # To be calculated calling analyse
        self.core_rad = None
        self.sol_rad = None

        # To be calculated calling rad_map
        self.x_tot = None
        self.z_tot = None
        self.rad_tot = None

    def analyse(self, firstwall_geom: Grid):
        """
        Using core radiation model and sol radiation model
        to calculate the radiation source at all points

        Parameters
        ----------
        first_wall: Loop
            The closed first wall geometry

        Returns
        -------
        x_all: np.array
            The x coordinates of all the points included within the flux surfaces
        z_all: np.array
            The z coordinates of all the points included within the flux surfaces
        rad_all: np.array
            The local radiation source at all points included within
            the flux surfaces [MW/m^3]
        """
        self.core_rad = CoreRadiation(
            self.eq,
            self.flux_surf_solver,
            self.params,
            self.imp_content_core,
            self.imp_data_core,
        )

        if self.eq.is_double_null:
            self.sol_rad = DNScrapeOffLayerRadiation(
                self.eq,
                self.flux_surf_solver,
                self.params,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
            )
        else:
            self.sol_rad = SNScrapeOffLayerRadiation(
                self.eq,
                self.flux_surf_solver,
                self.params,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
            )

        return self.core_rad, self.sol_rad

    def rad_core_by_psi_n(self, psi_n):
        """
        Calculation of core radiation source for a given (set of) psi norm value(s)

        Parameters
        ----------
        psi_n: float (list)
            The normalised magnetic flux value(s)

        Returns
        -------
        rad_new: float (list)
            Local radiation source value(s) associated to the given psi_n
        """
        core_rad = CoreRadiation(
            self.eq,
            self.flux_surf_solver,
            self.params,
            self.imp_content_core,
            self.imp_data_core,
        )
        core_rad.build_mp_rad_profile()
        rad_tot = np.sum(np.array(core_rad.rad, dtype=object), axis=0).tolist()
        f_rad = interp1d(core_rad.rho_core, rad_tot)
        rho_new = np.sqrt(psi_n)
        rad_new = f_rad(rho_new)
        return rad_new

    def rad_core_by_points(self, x, z):
        """
        Calculation of core radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        self.rad_core_by_psi_n(psi_n): float (list)
            Local radiation source value(s) associated to the point(s)
        """
        psi = self.eq.psi(x, z)
        psi_n = calc_psi_norm(psi, *self.eq.get_OX_psis(psi))
        return self.rad_core_by_psi_n(psi_n)

    def rad_sol_by_psi_n(self, psi_n):
        """
        Calculation of sol radiation source for a given psi norm value

        Parameters
        ----------
        psi_n: float
            The normalised magnetic flux value

        Returns
        -------
        list
            Local radiation source values associated to the given psi_n
        """
        f_sol = linear_interpolator(self.x_sol, self.z_sol, self.rad_sol)
        fs = self.eq.get_flux_surface(psi_n)
        return np.concatenate(
            [interpolated_field_values(x, z, f_sol) for x, z in zip(fs.x, fs.z)]
        )

    def rad_sol_by_points(self, x_lst, z_lst):
        """
        Calculation of sol radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        list
            Local radiation source value(s) associated to the point(s)
        """
        f_sol = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return np.concatenate(
            [interpolated_field_values(x, z, f_sol) for x, z in zip(x_lst, z_lst)]
        )

    def rad_by_psi_n(self, psi_n):
        """
        Calculation of any radiation source for a given (set of) psi norm value(s)

        Parameters
        ----------
        psi_n: float (list)
            The normalised magnetic flux value(s)

        Returns
        -------
        rad_any: float (list)
            Local radiation source value(s) associated to the given psi_n
        """
        if psi_n < 1:
            return self.rad_core_by_psi_n(psi_n)
        else:
            return self.rad_sol_by_psi_n(psi_n)

    def rad_by_points(self, x, z):
        """
        Calculation of any radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        rad_any: float (list)
            Local radiation source value(s) associated to the point(s)
        """
        f = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return interpolated_field_values(x, z, f)

    def rad_map(self, firstwall_geom: Grid):
        """
        Mapping all the radiation values associated to all the points
        as three arrays containing x coordinates, z coordinates and
        local radiated power density [MW/m^3]
        """
        self.core_rad.build_core_radiation_map()

        t_and_n_sol_profiles = self.sol_rad.build_sol_distribution(firstwall_geom)
        rad_sector_profiles = self.sol_rad.build_sol_radiation_distribution(
            *t_and_n_sol_profiles.values()
        )
        self.sol_rad.build_sol_radiation_map(*rad_sector_profiles.values())

        self.x_tot = np.concatenate([self.core_rad.x_tot, self.sol_rad.x_tot])
        self.z_tot = np.concatenate([self.core_rad.z_tot, self.sol_rad.z_tot])
        self.rad_tot = np.concatenate([self.core_rad.rad_tot, self.sol_rad.rad_tot])

        return self.x_tot, self.z_tot, self.rad_tot

    def plot(self, ax=None):
        """
        Plot the RadiationSolver results.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min(self.rad_tot)
        p_max = max(self.rad_tot)

        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        cm = ax.scatter(
            self.x_tot,
            self.z_tot,
            c=self.rad_tot,
            s=10,
            cmap="plasma",
            vmin=p_min,
            vmax=p_max,
            zorder=40,
        )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax
    
    def _make_params(self, config):
        """Convert the given params to ``RadiationSolverParams``"""
        if isinstance(config, dict):
            try:
                return RadiationSolverParams(**config)
            except TypeError:
                unknown = [
                    k for k in config if k not in fields(RadiationSolverParams)
                ]
                raise TypeError(f"Unknown config parameter(s) {str(unknown)[1:-1]}")
        elif isinstance(config, RadiationSolverParams):
            return config
        else:
            raise TypeError(
                "Unsupported type: 'config' must be a 'dict', or "
                "'ChargedParticleSolverParams' instance; found "
                f"'{type(config).__name__}'."
            )
        
@dataclass
class RadiationSolverParams:
    rho_ped_n: float = 0.94
    """???"""

    n_e_0: float = 21.93e19
    """???"""

    n_e_ped: float = 8.117e19
    """???"""

    n_e_sep: float = 1.623e19
    """???"""

    alpha_n: float = 1.15
    """???"""

    rho_ped_t: float = 0.976
    """???"""

    T_e_0: float = 21.442
    """???"""

    T_e_ped: float = 5.059
    """???"""

    T_e_sep: float = 0.16
    """???"""

    alpha_t: float = 1.905
    """???"""

    t_beta: float = 2.0
    """???"""

    P_sep: float = 150
    """???"""

    k_0: float = 2000.0
    """???"""

    gamma_sheath: float = 7.0
    """???"""

    eps_cool: float = 25.0
    """???"""

    f_ion_t: float = 0.01
    """???"""

    det_t: float = 0.0015
    """???"""

    lfs_p_fraction: float = 0.9
    """???"""

    div_p_sharing: float = 0.5
    """???"""

    theta_outer_target: float = 5.0
    """???"""

    theta_inner_target: float = 5.0
    """???"""

    f_p_sol_near: float = 0.65
    """???"""

    fw_lambda_q_near_omp: float = 0.003
    """???"""

    fw_lambda_q_far_omp: float = 0.1
    """???"""

    fw_lambda_q_near_imp: float = 0.003
    """???"""

    fw_lambda_q_far_imp: float = 0.1
    """???"""

