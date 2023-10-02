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
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List
import bluemira.codes.process as process
from bluemira.base import constants
from bluemira.base.constants import ureg
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.flux_surfaces import calculate_connection_length_flt
from bluemira.equilibria.grid import Grid
from bluemira.geometry.coordinates import Coordinates, in_polygon
from bluemira.geometry.wire import BluemiraWire
from matplotlib.collections import LineCollection

# CHERAB imports
from cherab.core.math import AxisymmetricMapper, sample3d
from cherab.tools.emitters import RadiationFunction
from raysect.core import Point3D, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.observer import PowerPipeline0D
from raysect.optical.observer.nonimaging.pixel import Pixel
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder

# Scipy import
from scipy.interpolate import LinearNDInterpolator, interp1d, interp2d

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
    num_f = m_i_kg * 4 * (q_u**2)
    den_f = 2 * E_CHARGE * (gamma**2) * (E_CHARGE**2) * (n_u**2) * (t_u**2)
    # forcing function
    f_ev = num_f / den_f

    # Critical target temperature
    t_crit = eps_cool / gamma

    # Finding roots of the target temperature quadratic equation
    coeff_2 = 2 * (eps_cool / gamma) - f_ev
    coeff_3 = (eps_cool**2) / (gamma**2)
    coeff = [1, coeff_2, coeff_3]
    roots = np.roots(coeff)

    if roots.dtype == complex:
        t_tar = f_ion_t
    else:
        # Excluding unstable solution
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
    sep_corrector: float,
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
    d =sep_corrector if lfs else -sep_corrector

    # Distance between the chosen point and the the target
    l_p = calculate_connection_length_flt(
        eq,
        x_p + (d * f_exp),
        z_p,
        forward=lfs,
        first_wall=firstwall_geom,
    )
    # connection length from the midplane to the target
    if connection_length is None:
        l_tot = calculate_connection_length_flt(
            eq,
            r_sep_mp,
            z_mp,
            forward=lfs,
            first_wall=firstwall_geom,
        )
    else:
        l_tot = connection_length
    # connection length from mp to p point
    s_p = l_tot - l_p
    if round(abs(z_p)) == 0:
        s_p = 0
    # Return local temperature
    return  constants.raw_uc(((t_u**3.5) - 3.5 * (q_par / k_0) * s_p) ** (2 / 7), "eV", "keV")


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

    return (species_frac * (ne**2) * p_loss_f)/(4*np.pi) * 1e-6


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
    region = np.zeros((len(domain_x), 2))
    region[:, 0] = domain_x
    region[:, 1] = domain_z

    def include(point):
        if include_points:
            return in_polygon(point[0], point[1], region, include_edges=True)
        else:
            return not in_polygon(point[0], point[1], region, include_edges=True)

    return include

def get_impurity_data(impurities_list: list = ["H", "He"]):
    """
    Function getting the PROCESS impurity data
    """
    # This is a function
    imp_data_getter = process.Solver.get_species_data

    impurity_data = {}
    for imp in impurities_list:
        impurity_data[imp] = {
            "T_ref": imp_data_getter(imp)[0],
            "L_ref": imp_data_getter(imp)[1],
        }

    return impurity_data


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
                n_split = int(np.ceil(y_width / max_wall_len))

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
        origin='lower'  # Set the origin to 'lower' to flip the image
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

    norm = mpl.colors.Normalize(vmin=0.0, vmax=np.max(wall_powerload) * 1.0e-6)
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

class FirstWallRadiationSolver:
    """
    ...
    """

    def __init__(
        self,
        source_func: callable,
        firstwall_shape: BluemiraWire
    ):
        self.rad_source = source_func
        self.fw_shape = firstwall_shape

    def solve(self, plot=True):

        rad_3d = AxisymmetricMapper(self.rad_source)
        ray_stepsize = 1.0  # 2.0e-4
        emitter = VolumeTransform(
            RadiationFunction(rad_3d, step=ray_stepsize * 0.1),
            translate(0, 0, np.max(self.fw_shape.z)),
        )
        world = World()
        Cylinder(
            np.max(self.fw_shape.x),
            2.0 * np.max(self.fw_shape.z),
            transform=translate(0, 0, np.min(self.fw_shape.z)),
            parent=world,
            material=emitter,
        )
        max_wall_len = 10.0e-2
        X_WIDTH = 0.01
        wall_detectors = build_wall_detectors(self.fw_shape.x, self.fw_shape.z, max_wall_len, X_WIDTH)
        wall_loads = detect_radiation(wall_detectors, 500, world)

        if plot:
            plot_radiation_loads(
                rad_3d, wall_detectors, wall_loads, "SOL & divertor radiation loads", self.fw_shape
            )

        return wall_loads