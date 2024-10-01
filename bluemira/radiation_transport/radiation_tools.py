# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
1-D radiation model inspired by the PROCESS function "plot_radprofile" in plot_proc.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from rich.progress import track
from scipy.interpolate import (
    LinearNDInterpolator,
    interp1d,
    interp2d,
)
from scipy.spatial import Delaunay

from bluemira.base.constants import C_LIGHT, D_MOLAR_MASS, E_CHARGE, raw_uc
from bluemira.base.look_and_feel import bluemira_error
from bluemira.codes.utilities import get_code_interface
from bluemira.equilibria.flux_surfaces import calculate_connection_length_flt
from bluemira.geometry.coordinates import Coordinates, in_polygon

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy.typing as npt

    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.grid import Grid
    from bluemira.geometry.wire import BluemiraWire

try:
    from cherab.core.math import AxisymmetricMapper, sample3d
    from cherab.tools.emitters import RadiationFunction
    from raysect.core import Point3D, Vector3D, rotate_basis, translate
    from raysect.optical import World
    from raysect.optical.material import VolumeTransform
    from raysect.optical.observer import PowerPipeline0D
    from raysect.optical.observer.nonimaging.pixel import Pixel
    from raysect.primitive import Cylinder
except ImportError:
    bluemira_error("Cherab not installed")


@dataclass
class WallDetector:
    """Dataclass for wall detectors"""

    detector_id: int
    x_width: float
    y_width: float
    detector_center: Point3D
    normal_vector: Vector3D
    y_vector: Vector3D


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
    connection_length: float | None = None,
) -> float:
    """
    Calculate the upstream temperature, as suggested from
    "Pitcher et al., Plasma Physics and Controlled Fusion 39.6 (1997): 779".
    Knowing the power entering the SOL, and assuming large temperature gradient
    between upstream and target.

    Parameters
    ----------
    b_pol:
        Poloidal magnetic field at the midplane [T]
    b_tot:
        Total magnetic field at the midplane [T]
    lambda_q_near:
        Power decay length in the near SOL [m]
    p_sol:
        Total power entering the SOL [W]
    eq:
        Equilibrium in which to calculate the upstream temperature
    r_sep_mp:
        Upstream location radial coordinate [m]
    z_mp:
        Upstream location z coordinate [m]
    k_0:
        Material's conductivity
    firstwall_geom:
        First wall geometry
    connection_length:
        connection length from the midplane to the target

    Returns
    -------
    t_upstream_kev:
        upstream temperature. Unit [keV]
    """
    # SoL cross-section at the midplane
    a_par = 4 * np.pi * r_sep_mp * lambda_q_near * (b_pol / b_tot)

    # upstream power density
    q_u = p_sol / a_par

    # connection length from the midplane to the target
    l_tot = (
        calculate_connection_length_flt(
            eq,
            r_sep_mp,
            z_mp,
            first_wall=firstwall_geom,
        )
        if connection_length is None
        else connection_length
    )

    # upstream temperature [keV]
    return raw_uc((3.5 * (q_u / k_0) * l_tot) ** (2 / 7), "eV", "keV")


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
) -> float:
    """
    Calculate the target as suggested from the 2PM.
    It includes hydrogen recycle loss energy.
    Ref. Stangeby, "The Plasma Boundary of Magnetic Fusion
    Devices", 2000.

    Parameters
    ----------
    p_sol:
        Total power entering the SOL [W]
    t_u:
        Upstream temperature. Unit [eV]
    n_u:
        Electron density at the upstream [1/m^3]
    gamma:
        Sheath heat transmission coefficient
    eps_cool:
        Electron energy loss [eV]
    f_ion_t:
        Hydrogen first ionization [eV]
    b_pol_tar:
        Poloidal magnetic field at the target [T]
    b_pol_u:
        Poloidal magnetic field at the midplane [T]
    alpha:
        Incident angle between separatrix and target plate as
        poloidal projection [deg]
    r_u:
        Upstream location radial coordinate [m]
    r_tar:
        stike point radial coordinate [m]
    lambda_q_near:
        Power decay length in the near SOL at the midplane [m]
    b_tot_tar:
        Total magnetic field at the target [T]

    Returns
    -------
    t_tar:
        target temperature. Unit [eV]
    """
    # flux expansion at the target location
    f_exp = (r_u * b_pol_u) / (r_tar * b_pol_tar)

    # lambda target
    lambda_q_tar = lambda_q_near * f_exp * (1 / np.sin(np.deg2rad(alpha_pol_deg)))
    # wet area as poloidal section
    a_wet = 4 * np.pi * r_tar * lambda_q_tar

    # parallel cross section
    a_par = a_wet * (b_pol_tar / b_tot_tar)

    # parallel power flux density
    q_u = p_sol / a_par

    # ion mass in kg (it should be DT = 2.5*amu)
    m_i_kg = raw_uc(D_MOLAR_MASS, "amu", "kg")

    # converting upstream temperature
    # upstream electron density - no fifference hfs/lfs?
    # Numerator and denominator of the upstream forcing function
    # forcing function
    f_ev = (m_i_kg * 4 * (q_u**2)) / (
        2 * E_CHARGE * (gamma**2) * (E_CHARGE**2) * (n_u**2) * (t_u**2)
    )

    # Critical target temperature
    t_crit = eps_cool / gamma

    # Finding roots of the target temperature quadratic equation
    roots = np.roots([1, 2 * (eps_cool / gamma) - f_ev, (eps_cool**2) / (gamma**2)])

    # Target temperature excluding unstable solution
    return f_ion_t if roots.dtype == complex else roots[np.nonzero(roots > t_crit)[0][0]]


def specific_point_temperature(
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
    connection_length: float | None = None,
    *,
    lfs=True,
) -> float:
    """
    Calculate the temperature at a specific point above the x-point.

    Parameters
    ----------
    x_p:
        x coordinate of the point [m]
    z_p:
        z coordinate of the point [m]
    t_u:
        upstream temperature [eV]
    p_sol:
        Total power entering the SOL [W]
    lambda_q_near:
        Power decay length in the near SOL at the midplane [m]
    eq:
        Equilibrium in which to calculate the point temperature
    r_sep_omp:
        Upstream location radial coordinate [m]
    z_mp:
        Upstream location z coordinate [m]
    k_0:
        Material's conductivity
    firstwall_geom:
        first wall geometry
    connection_length:
        connection length from the midplane to the target
    lfs:
        low (toroidal) field side (outer wall side). Default value True.
        If False it stands for high field side (hfs).

    Returns
    -------
    t_p:
        point temperature. Unit [eV]
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
    d = sep_corrector if lfs else -sep_corrector

    # need to simplify this
    if lfs and z_p < z_mp:
        forward = True
    elif (lfs and z_p > z_mp) or (z_p < z_mp and not lfs):
        forward = False
    elif z_p > z_mp and not lfs:
        forward = True

    # Distance between the chosen point and the the target
    l_p = calculate_connection_length_flt(
        eq,
        x_p + (d * f_exp),
        z_p,
        forward=forward,
        first_wall=firstwall_geom,
    )
    # connection length from the midplane to the target
    l_tot = (
        calculate_connection_length_flt(
            eq,
            r_sep_mp,
            z_mp,
            forward=forward,
            first_wall=firstwall_geom,
        )
        if connection_length is None
        else connection_length
    )
    # connection length from mp to p point
    s_p = l_tot - l_p
    if round(abs(z_p)) == 0:
        s_p = 0
    # Return local temperature
    return ((t_u**3.5) - 3.5 * (q_par / k_0) * s_p) ** (2 / 7)


def electron_density_and_temperature_sol_decay(
    t_sep: float,
    n_sep: float,
    lambda_q_near: float,
    lambda_q_far: float,
    dx_mp: float,
    f_exp: float = 1,
) -> tuple[np.ndarray, ...]:
    """
    Generic radial esponential decay to be applied from a generic starting point
    at the separatrix (not only at the mid-plane).
    The vertical location is dictated by the choice of the flux expansion f_exp.
    By default f_exp = 1, meaning mid-plane.
    From the power decay length it calculates the temperature decay length and the
    density decay length.

    Parameters
    ----------
    t_sep:
        initial temperature value at the separatrix [keV]
    n_sep:
        initial density value at the separatrix [1/m^3]
    lambda_q_near:
        Power decay length in the near SOL [m]
    lambda_q_far:
        Power decay length in the far SOL [m]
    dx_mp:
        Gaps between flux tubes at the mp [m]
    f_exp:
        flux expansion. Default value=1 referred to the mid-plane
    near_sol_gradient:
        temperature and density drop within the near scrape-off layer
        from the separatrix value

    Returns
    -------
    te_sol:
        radial decayed temperatures through the SoL. Unit [eV]
    ne_sol:
        radial decayed densities through the SoL. unit [1/m^3]
    """
    # temperature and density decay factors
    if f_exp == 1:
        t_factor = 7 / 2
        n_factor = 1 / 3
    else:
        t_factor = 7
        n_factor = 1 / 7

    # radial distance of flux tubes from the separatrix
    dr = dx_mp * f_exp

    # power decay length modified according to the flux expansion
    lambda_q_near *= f_exp
    lambda_q_far *= f_exp

    # Assuming conduction-limited regime.
    lambda_t_near = t_factor * lambda_q_near
    lambda_n_near = n_factor * lambda_t_near

    te_sol = (t_sep) * np.exp(-dr / lambda_t_near)
    ne_sol = (n_sep) * np.exp(-dr / lambda_n_near)

    return te_sol, ne_sol


def gaussian_decay(
    max_value: float, min_value: float, no_points: float, *, decay: bool = True
) -> np.ndarray:
    """
    Generic gaussian decay to be applied between two extreme values and for a
    given number of points.

    Parameters
    ----------
    max_value:
        maximum value of the parameters
    min_value:
        minimum value of the parameters
    no_points:
        number of points through which make the parameter decay

    Returns
    -------
    dec_param:
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
    i_near_minimum = np.nonzero(dec_param < min_value)
    dec_param[i_near_minimum[0]] = min_value

    return dec_param


def exponential_decay(
    max_value: float, min_value: float, no_points: float, *, decay: bool = False
) -> np.ndarray:
    """
    Generic exponential decay to be applied between two extreme values and for a
    given number of points.

    Parameters
    ----------
    max_value:
        maximum value of the parameters
    min_value:
        minimum value of the parameters
    no_points:
        number of points through which make the parameter decay
    decay:
        to define either a decay or increment

    Returns
    -------
    dec_param:
        decayed parameter
    """
    no_points = max(no_points, 1)
    if no_points <= 1:
        return np.linspace(max_value, min_value, no_points)
    x = np.linspace(1, no_points, no_points)
    a = np.array([x[0], min_value])
    b = np.array([x[-1], max_value])
    if decay:
        arg = x / b[0]
        base = a[0] / b[0]
    else:
        arg = x / a[0]
        base = b[0] / a[0]

    return a[1] + (b[1] - a[1]) * (np.log(arg) / np.log(base))


def ion_front_distance(
    x_strike: float,
    z_strike: float,
    eq: Equilibrium,
    x_pt_z: float,
    t_tar: float | None = None,
    avg_ion_rate: float | None = None,
    avg_momentum_rate: float | None = None,
    n_r: float | None = None,
    rec_ext: float | None = None,
) -> float:
    """
    Manual definition of ion penetration depth.
    TODO: Find sv_i and sv_m

    Parameters
    ----------
    x_strike:
        x coordinate of the strike point [m]
    z_strike:
        z coordinate of the strike point [m]
    eq:
        Equilibrium in which to calculate the x-point temperature
    x_pt_z:
        x-point location z coordinate [m]
    t_tar:
        target temperature [keV]
    avg_ion_rate:
        average ionization rate
    avg_momentum_rate:
        average momentum loss rate
    n_r:
        density at the recycling region entrance [1/m^3]
    rec_ext:
        recycling region extention (along the field line)
        from the target [m]

    Returns
    -------
    z_front:
        z coordinate of the ionization front
    """
    # Speed of light to convert kg to eV/c^2
    # deuterium ion mass
    m_i = raw_uc(D_MOLAR_MASS, "amu", "kg") / (C_LIGHT**2)

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
    return abs(z_strike - x_pt_z) - abs(z_ext)


def calculate_zeff(
    impurities_content: np.ndarray,
    imp_data_z_ref: np.ndarray,
    imp_data_t_ref: np.ndarray,
    impurity_symbols: np.ndarray,
    te: np.ndarray,
):
    """
    Calculate the effective charge (Z_eff) for the plasma core.

    This function computes Z_eff based on the species information
    and the temperature profile.

    Parameters
    ----------
    impurities_content: np.array
        Content of each impurity species in the plasma.
    imp_data_z_ref: np.array
        Reference effective charge values corresponding to the reference temperatures.
    imp_data_t_ref: np.array
        Reference temperatures (in keV) for interpolation.
    impurity_symbols: np.array
        All the impurity species symbols in the plasma.
    te : np.ndarray
        Electron temperature profile (in keV) at various positions in the plasma.

    Returns
    -------
    zeff: np.ndarray
        Effective charge profile for each plasma position.
    avg_zeff: float
        Average Z_eff across the plasma.
    total_fraction: float
        Total fraction of impurities.
    intermediate_values: dict
        A dictionary containing species fractions, average charge states, and symbols.
    """
    # Get the electron temperature profile (flattened) [keV]
    te = np.concatenate(te)

    # Initialize lists to hold species fractions and charge states
    species_fractions = []
    species_zi = []
    symbols = []

    # Include other species
    for frac, z_r, t_r, symbol in zip(
        impurities_content,
        imp_data_z_ref,
        imp_data_t_ref,
        impurity_symbols,
        strict=False,
    ):
        species_fractions.append(frac)
        symbols.append(symbol)

        # Ensure data is in correct numerical format
        t_ref = np.array(t_r, dtype=float)
        z_ref = np.array(z_r, dtype=float)
        te = np.array(te, dtype=float)
        z_interp = interp1d(t_ref, z_ref, fill_value="extrapolate", bounds_error=False)
        z_i = z_interp(te)
        species_zi.append(z_i)

    # Convert lists to numpy arrays for vectorized operations
    z_i_all = np.array(species_zi)
    f_i_all = np.array(species_fractions)[:, np.newaxis]

    # Compute numerator and denominator for Zeff at each position
    numerator = np.sum(f_i_all * z_i_all**2, axis=0)
    denominator = np.sum(f_i_all * z_i_all, axis=0)

    # Calculate Zeff at each position
    zeff = numerator / denominator

    # Compute average Zeff over all positions
    avg_zeff = np.mean(zeff)

    # Calculate intermediate values to return as a dictionary
    intermediate_values = {
        "species_fractions": species_fractions,
        "species_zi": [np.mean(z_i) for z_i in species_zi],
        "symbols": symbols,
    }

    # Calculate total fraction of impurities
    total_fraction = np.sum(species_fractions)

    return zeff, avg_zeff, total_fraction, intermediate_values


def calculate_total_radiated_power(
    x: np.ndarray, z: np.ndarray, p_rad: np.ndarray
) -> float:
    """
    Calculate the total radiated power from the radiation map.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates (in meters) of the radiation map.
    z : np.ndarray
        Array of z-coordinates (in meters) of the radiation map.
    Prad : np.ndarray
        Array of radiation power density values (in MW/m³)
        at the corresponding x and z coordinates.

    Returns
    -------
    P_total : float
        Total radiated power in megawatts (MW).
    """
    # Stack x and z coordinates
    points = np.column_stack((x, z))

    # Delaunay triangulation
    tri = Delaunay(points)

    # Initialize total power
    p_total = 0.0

    # Loop over each triangle in the Delaunay triangulation
    for simplex in tri.simplices:
        indices = simplex
        x_vertices = x[indices]
        z_vertices = z[indices]
        p_rad_vertices = p_rad[indices]

        # Compute area of the triangle in x-z plane using determinant formula
        area = 0.5 * abs(
            (x_vertices[1] - x_vertices[0]) * (z_vertices[2] - z_vertices[0])
            - (x_vertices[2] - x_vertices[0]) * (z_vertices[1] - z_vertices[0])
        )
        if area <= 0:  # Check if area is non-zero and positive
            continue

        # Compute centroid (average position of vertices)
        x_centroid = np.mean(x_vertices)
        p_rad_centroid = np.mean(p_rad_vertices)

        # Compute volume element using cylindrical symmetry
        dv = 2 * np.pi * x_centroid * area  # Volume element in m^3

        # Compute differential power
        dp = p_rad_centroid * dv  # Power in MW (MW/m^3 * m^3)
        # Accumulate total power
        p_total += dp

    return p_total


def radiative_loss_function_values(
    te: np.ndarray, t_ref: np.ndarray, l_ref: np.ndarray
) -> np.ndarray:
    """
    By interpolation, from reference values, it returns the
    radiative power loss values for a given set of electron temperature.

    Parameters
    ----------
    te:
        electron temperature [keV]
    t_ref:
        temperature reference [eV]
    l_ref:
        radiative power loss reference [Wm^3]

    Returns
    -------
        interpolated local values of the radiative power loss function [W m^3]
    """
    te_i = np.nonzero(te < min(t_ref))
    te[te_i] = min(t_ref) + (np.finfo(float).eps)

    return interp1d(t_ref, l_ref)(te)


def radiative_loss_function_plot(
    t_ref: np.ndarray, lz_val: Iterable[np.ndarray], species: Iterable[str]
):
    """
    Radiative loss function plot for a set of given impurities.

    Parameters
    ----------
    t_ref:
        temperature reference [keV]
    l_z:
        radiative power loss reference [Wm^3]
    species:
        name species
    """
    _fig, ax = plt.subplots()
    plt.title("Radiative loss functions vs Electron Temperature")
    plt.xlabel(r"$T_e~[keV]$")
    plt.ylabel(r"$L_z~[W.m^{3}]$")

    [
        ax.plot(t_ref, lz_specie, label=name)
        for lz_specie, name in zip(lz_val, species, strict=False)
    ]
    plt.xscale("log")
    plt.xlim(None, 10)
    plt.yscale("log")
    ax.legend(loc="best", borderaxespad=0, fontsize=12)

    return ax


def calculate_line_radiation_loss(
    ne: np.ndarray, p_loss_f: np.ndarray, species_frac: float
) -> np.ndarray:
    """
    Calculation of Line radiation losses.
    For a given impurity this is the total power lost, per unit volume,
    by all line-radiation processes INCLUDING Bremsstrahlung.

    Parameters
    ----------
    ne:
        electron density [1/m^3]
    p_loss_f:
        local values of the radiative power loss function
    species_frac:
        fraction of relevant impurity

    Returns
    -------
        Line radiation losses [MW m^-3]
    """
    return raw_uc((species_frac * (ne**2) * p_loss_f), "W", "MW")


def linear_interpolator(
    x: np.ndarray, z: np.ndarray, field: np.ndarray
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Interpolating function calculated over 1D coordinate
    arrays and 1D field value array.

    Parameters
    ----------
    x:
        x coordinates of given points [m]
    z:
        z coordinates of given points [m]
    field:
        set of punctual field values associated to the given points

    Returns
    -------
    interpolated_function:
        LinearNDInterpolator object
    """
    return LinearNDInterpolator(list(zip(x, z, strict=False)), field, fill_value=0)


def interpolated_field_values(
    x: np.ndarray,
    z: np.ndarray,
    linear_interpolator: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Interpolated field values for a given set of points.

    Parameters
    ----------
    x:
        x coordinates of point in which interpolate [m]
    z:
        z coordinates of point in which interpolate [m]
    linear_interpolator:
        LinearNDInterpolator object

    Returns
    -------
    field_grid:
         matrix len(x) x len(z), 2D grid of interpolated field values
    """
    xx, zz = np.meshgrid(x, z)
    return linear_interpolator(xx, zz)


def grid_interpolator(
    x: np.ndarray, z: np.ndarray, field_grid: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Interpolated field function obtainded for a given grid.
    Needed: length(xx) = m, length(zz) = n, field_grid.shape = n x m.

    Parameters
    ----------
    x: np.array
        x coordinates. length(xx)=m [m]
    z: np.array
        z coordinates. length(xx)=n [m]
    field_grid:
        matrix n x m corresponding field values arranged according to
        the grid of given points

    Returns
    -------
    interpolated_function:
        interpolated field function, to be used to
        calculate the field values for a new set of points
        or to be provided to a tracing code such as CHEARAB
    """
    # scipy deprecation of interp2d ~3x slower:
    # grid = RegularGridInterpolator(
    #     (x, z), field_grid.T, bounds_error=False, fill_value=None, method="cubic"
    # )
    # return lambda xx, zz: grid((xx, zz))
    return interp2d(x, z, field_grid, kind="cubic")


def pfr_filter(
    separatrix: Iterable[Coordinates] | Coordinates, x_point_z: float
) -> tuple[np.ndarray, ...]:
    """
    To filter out from the radiation interpolation domain the private flux regions

    Parameters
    ----------
    separatrix:
        Object containing x and z coordinates of each separatrix half
    x_point_z:
        z coordinates of the x-point [m]

    Returns
    -------
    domains_x:
        x coordinates of pfr domain
    domains_z:
        z coordinates of pfr domain
    """
    # Identifying whether lower or upper x-point
    fact = 1 if x_point_z < 0 else -1

    if isinstance(separatrix, Coordinates):
        separatrix = [separatrix]
    # Selecting points between null and targets (avoiding the x-point singularity)
    z_ind = [
        np.nonzero((halves.z * fact) < (x_point_z * fact - 0.01))
        for halves in separatrix
    ]
    domains_x = [
        halves.x[list_ind] for list_ind, halves in zip(z_ind, separatrix, strict=False)
    ]
    domains_z = [
        halves.z[list_ind] for list_ind, halves in zip(z_ind, separatrix, strict=False)
    ]

    if len(domains_x) != 1:
        # Closing the domain
        domains_x[1] = np.append(domains_x[1], domains_x[0][0])
        domains_z[1] = np.append(domains_z[1], domains_z[0][0])

    return np.concatenate(domains_x), np.concatenate(domains_z)


def filtering_in_or_out(
    domain_x: list[float], domain_z: list[float], *, include_points: bool = True
) -> Callable[[Iterable[float]], bool]:
    """
    To exclude from the calculation a specific region which is
    either contained or not contained within a given domain

    Parameters
    ----------
    domain_x:
        list of x coordinates defining the domain
    domain_z:
        list of x coordinates defining the domain
    include_points:
        wheter the points inside or outside the domain must be excluded

    Returns
    -------
    include:
        method which includes or excludes from the domain a given point
    """
    region = np.zeros((len(domain_x), 2))
    region[:, 0] = domain_x
    region[:, 1] = domain_z

    if include_points:
        return lambda point: in_polygon(point[0], point[1], region, include_edges=True)
    return lambda point: not in_polygon(point[0], point[1], region, include_edges=True)


def get_impurity_data(
    impurities_list: Iterable[str] = ("H", "He"), confinement_time_ms: float = 0.1
):
    """
    Function getting the PROCESS impurity data
    """
    # This is a function
    imp_data_getter = get_code_interface("PROCESS").Solver.get_species_data

    impurity_data = {}
    for imp in impurities_list:
        impurity_data[imp] = {
            "T_ref": imp_data_getter(imp, confinement_time_ms)[0],
            "L_ref": imp_data_getter(imp, confinement_time_ms)[1],
            "z_ref": imp_data_getter(imp, confinement_time_ms)[2],
        }

    return impurity_data


@dataclass(repr=False)
class DetectedRadiation:
    """Detected radiation data"""

    power_density: npt.NDArray[np.float64]
    power_density_stdev: npt.NDArray[np.float64]
    detected_power: npt.NDArray[np.float64]
    detected_power_stdev: npt.NDArray[np.float64]
    detector_area: npt.NDArray[np.float64]
    detector_numbers: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    total_power: float


# Adapted functions from Stuart
def detect_radiation(
    wall_detectors: list[WallDetector],
    n_samples: int,
    world: World,
    *,
    verbose: bool = False,
):
    """
    To sample the wall and detect radiation
    """
    # Storage lists for results
    power_density = []
    distance = []
    detected_power = []
    detected_power_stdev = []
    detector_area = []
    power_density_stdev = []

    running_distance = 0
    cherab_total_power = 0

    quiet = not verbose

    # Loop over each tile detector
    for detector in track(
        wall_detectors,
        total=len(wall_detectors),
        description="Radiation detectors...",
    ):
        # extract the dimensions and orientation of the tile
        pixel_area = detector.x_width * detector.y_width

        # Use the power pipeline to record total power arriving at the surface
        power_data = PowerPipeline0D()

        # Use pixel_samples argument to increase amount of sampling and reduce noise
        pixel = Pixel(
            [power_data],
            x_width=detector.x_width,
            y_width=detector.y_width,
            name=f"pixel-{detector.detector_id}",
            spectral_bins=1,
            transform=translate(
                detector.detector_center.x,
                detector.detector_center.y,
                detector.detector_center.z,
            )
            * rotate_basis(detector.normal_vector, detector.y_vector),
            parent=world,
            pixel_samples=n_samples,
            quiet=quiet,
        )
        # make detector sensitivity 1nm so that radiation function
        # is effectively W/m^3/str

        # Start collecting samples
        pixel.observe()

        # Append the collected data to the storage lists
        detector_radius = np.sqrt(
            detector.detector_center.x**2 + detector.detector_center.y**2
        )

        detector_area.append(pixel_area)
        power_density.append(
            power_data.value.mean / pixel_area
        )  # convert to W/m^2 !!!!!!!!!!!!!!!!!!!

        power_density_stdev.append(np.sqrt(power_data.value.variance) / pixel_area)
        detected_power.append(
            power_data.value.mean
            / pixel_area
            * (detector.y_width * 2 * np.pi * detector_radius)
        )
        detected_power_stdev.append(np.sqrt(power_data.value.variance))

        running_distance += 0.5 * detector.y_width  # with Y_WIDTH instead of y_width
        distance.append(running_distance)
        running_distance += 0.5 * detector.y_width  # with Y_WIDTH instead of y_width

        # For checking energy conservation.
        # Revolve this tile around the CYLINDRICAL z-axis
        # to get total power collected by these tiles.
        # Add up all the tile contributions to get total power collected.
        cherab_total_power += (power_data.value.mean / pixel_area) * (
            detector.y_width * 2 * np.pi * detector_radius
        )

    return DetectedRadiation(
        np.asarray(power_density),
        np.asarray(power_density_stdev),
        np.asarray(detected_power),
        np.asarray(detected_power_stdev),
        np.asarray(detector_area),
        np.arange(len(wall_detectors), dtype=int),
        np.asarray(distance),
        cherab_total_power,
    )


def make_wall_detectors(
    wall_r, wall_z, max_wall_len, x_width, *, debug=False
) -> list[WallDetector]:
    """
    To make the detectors on the wall
    """
    # number of detectors
    num = np.shape(wall_r)[0] - 2

    # further initializations
    wall_detectors = []

    ctr = 0

    if debug:
        _fig, ax = plt.subplots()

    for index in range(num + 1):
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

                y_width /= n_split

                for k in np.arange(n_split):
                    # evaluate the central point of the detector
                    detector_center = Point3D(
                        0.5 * (splitwall_x[k] + splitwall_x[k + 1]),
                        0,
                        0.5 * (splitwall_y[k] + splitwall_y[k + 1]),
                    )

                    # to populate it step by step
                    wall_detectors.append(
                        WallDetector(
                            detector_id=ctr,
                            x_width=x_width,
                            y_width=y_width,
                            detector_center=detector_center,
                            normal_vector=normal_vector,
                            y_vector=y_vector,
                        )
                    )

            else:
                # evaluate normal_vector
                normal_vector = Vector3D(p1y - p2y, 0.0, p2x - p1x).normalise()
                # normal_vector = (detector_center.vector_to(plasma_axis_3D)).normalise()
                # inward pointing

                # evaluate the central point of the detector
                detector_center = Point3D(0.5 * (p1x + p2x), 0, 0.5 * (p1y + p2y))

                # to populate it step by step
                wall_detectors.append(
                    WallDetector(
                        detector_id=ctr,
                        x_width=x_width,
                        y_width=y_width,
                        detector_center=detector_center,
                        normal_vector=normal_vector,
                        y_vector=y_vector,
                    )
                )

                if debug:
                    ax.plot([p1x, p2x], [p1y, p2y], "k")
                    ax.plot([p1x, p2x], [p1y, p2y], ".k")
                    pcn = detector_center + normal_vector * 0.05
                    ax.plot([detector_center.x, pcn.x], [detector_center.z, pcn.z], "r")

            ctr += 1

    if debug:
        plt.show()

    return wall_detectors


def plot_radiation_loads(
    radiation_function, wall_detectors, wall_loads, plot_title, fw_shape
):
    """
    To plot the radiation on the wall as MW/m^2
    """
    min_r = min(fw_shape.x)
    max_r = max(fw_shape.x)
    min_z = min(fw_shape.z)
    max_z = max(fw_shape.z)

    _t_r, _, _t_z, t_samples = sample3d(
        radiation_function, (min_r, max_r, 500), (0, 0, 1), (min_z, max_z, 1000)
    )

    # Plot the wall and radiation distribution
    fig = plt.figure()

    gs = plt.GridSpec(2, 2, top=0.93, wspace=0.23)

    ax1 = plt.subplot(gs[0:2, 0])
    ax1.imshow(
        np.squeeze(t_samples).transpose() * 1.0e-6,
        extent=[min_r, max_r, min_z, max_z],
        clim=(0.0, np.amax(t_samples) * 1.0e-6),
        origin="lower",  # Set the origin to 'lower' to flip the image
    )

    segs = []

    for detector in wall_detectors:
        end1 = detector.detector_center - 0.5 * detector.y_width * detector.y_vector
        end2 = detector.detector_center + 0.5 * detector.y_width * detector.y_vector

        segs.append([[end1.x, end1.z], [end2.x, end2.z]])

    wall_powerload = np.array(wall_loads.power_density)

    line_segments = LineCollection(segs, cmap="hot")
    line_segments.set_array(wall_powerload)

    ax1.add_collection(line_segments)

    cmap = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(
            vmin=0.0, vmax=raw_uc(np.max(wall_powerload), "W", "MW")
        ),
        cmap=mpl.cm.hot,
    )
    cmap.set_array([])

    heat_cbar = fig.colorbar(cmap, ax=ax1)
    heat_cbar.set_label(r"Wall Load ($MW.m^{-2}$)")

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(
        np.array(wall_loads.distance),
        raw_uc(np.array(wall_loads.power_density), "W", "MW"),
    )
    ax2.set_ylim([
        0.0,
        raw_uc(1.1 * np.max(np.array(wall_loads.power_density)), "W", "MW"),
    ])
    ax2.grid(visible=True)
    ax2.set_ylabel(r"Radiation Load ($MW.m^{-2}$)")

    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(
        np.array(wall_loads.distance),
        np.cumsum(np.array(wall_loads.detected_power) * 1.0e-6),
    )

    ax3.set_ylabel(r"Total Power $[MW]$")
    ax3.set_xlabel(r"Poloidal Distance $[m]$")
    ax3.grid(visible=True)

    plt.suptitle(plot_title)
    plt.show()


class FirstWallRadiationSolver:
    """
    ...
    """

    def __init__(self, source_func: Callable, firstwall_shape: BluemiraWire):
        self.rad_source = source_func
        self.fw_shape = firstwall_shape

    def solve(
        self,
        max_wall_len: float = 0.1,
        x_width: float = 0.01,
        n_samples: int = 500,
        *,
        plot: bool = True,
        verbose: bool = False,
    ):
        """Solve first wall radiation problem"""
        shift = translate(0, 0, np.min(self.fw_shape.z))
        height = np.max(self.fw_shape.z) - np.min(self.fw_shape.z)
        rad_3d = AxisymmetricMapper(self.rad_source)
        ray_stepsize = 1.0  # 2.0e-4
        emitter = VolumeTransform(
            RadiationFunction(rad_3d, step=ray_stepsize * 0.1),
            translate(0, 0, np.max(self.fw_shape.z)),
        )
        world = World()
        Cylinder(
            np.max(self.fw_shape.x),
            height,
            transform=shift,
            parent=world,
            material=emitter,
        )
        wall_detectors = make_wall_detectors(
            self.fw_shape.x, self.fw_shape.z, max_wall_len, x_width, debug=verbose
        )
        wall_loads = detect_radiation(wall_detectors, n_samples, world, verbose=verbose)

        if plot:
            plot_radiation_loads(
                rad_3d,
                wall_detectors,
                wall_loads,
                "SOL & divertor radiation loads",
                self.fw_shape,
            )

        return wall_loads
