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
from typing import TYPE_CHECKING, cast

import numpy as np

from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.base.parameter_frame.typed import ParameterFrameLike


@dataclass
class MidplaneProfilesParams(ParameterFrame):
    """Midplane profiles parameters"""

    alpha_n: Parameter[float]
    """Density profile factor"""
    alpha_t: Parameter[float]
    """Temperature profile index"""
    n_e_0: Parameter[float]
    """Electron density on axis"""
    n_e_ped: Parameter[float]
    """Electron density pedestal height"""
    n_e_sep: Parameter[float]
    """Electron density at separatrix"""
    rho_ped_n: Parameter[float]
    """Density pedestal r/a location"""
    rho_ped_t: Parameter[float]
    """Temperature pedestal r/a location"""
    n_points_core_95: Parameter[int]
    """rho discretisation to 95% of core"""
    n_points_core_99: Parameter[int]
    """rho discretisation to 99% of core"""
    n_points_mantle: Parameter[int]
    """rho discretisation to separatrix"""
    t_beta: Parameter[float]
    """Temperature profile index beta"""
    T_e_0: Parameter[float]
    """Electron temperature on axis"""
    T_e_ped: Parameter[float]
    """Electron temperature pedestal height"""
    T_e_sep: Parameter[float]
    """Electron temperature at separatrix"""


@dataclass
class MidplaneProfiles:
    """midplane profiles"""

    psi_n: npt.NDArray[np.float64]
    """squared dimensionless core radius"""
    ne: npt.NDArray[np.float64]
    """electron densities at the mid-plane. Unit [1/m^3]"""
    te: npt.NDArray[np.float64]
    """electron temperature at the mid-plane. Unit [keV]"""


def midplane_profiles(params: ParameterFrameLike) -> MidplaneProfiles:
    """
    Calculate the core radiation source profiles.

    Temperature and density are assumed to be constant along a
    single flux tube.

    Parameters
    ----------
    params:
        Parameters defining the pedestal positions, profile exponents,
        densities [1/m³], temperatures, and radial sampling.

    Returns
    -------
    :
        Electron density [1/m³], temperature [keV], and the profile coordinate
        ``psi_n = rho²`` (an assumed radius-to-flux mapping, not an equilibrium
        flux calculation). Profiles use the common pedestal radius
        :math:`\\rho_p=(\\rho_{p,n}+\\rho_{p,T})/2`; see
        :func:`core_electron_density_temperature_profile`.
    """
    params = make_parameter_frame(params, MidplaneProfilesParams, allow_unknown=True)
    rho_ped = (params.rho_ped_n.value + params.rho_ped_t.value) / 2

    # A dimensionless radius at the mid-plane.
    # From the core to the last closed flux surface
    rho_core = collect_rho_core_values(
        rho_ped,
        params.n_points_core_95.value,
        params.n_points_core_99.value,
        params.n_points_mantle.value,
    )

    # For each flux tube, density and temperature at the mid-plane
    return core_electron_density_temperature_profile(params, rho_core, rho_ped)


def collect_rho_core_values(
    rho_ped: np.ndarray,
    n_points_core_95: int,
    n_points_core_99: int,
    n_points_mantle: int,
) -> np.ndarray:
    """
    Calculation of core dimensionless radial coordinate rho (between 0 and 1).

    Parameters
    ----------
    rho_ped:
        dimensionless pedestal radius. Values between 0 and 1
    n_points_core_95:
        no of discretisation points to 95% of core
    n_points_core_99:
        no of discretisation points to 99% of core
    n_points_mantle:
        no of discretisation points to separatrix

    Returns
    -------
    :
        radial coordinate rho [dimensionless].

    Notes
    -----
    The plasma bulk is divided into plasma core and plasma mantle according to rho
    rho is a nondimensional radial coordinate: rho = r/a (r varies from 0 to a)
    """
    # Plasma core for rho < rho_core
    rho_core1 = np.linspace(0.01, 0.95 * rho_ped, n_points_core_95)
    rho_core2 = np.linspace(0.95 * rho_ped, rho_ped, n_points_core_99)
    rho_core = np.append(rho_core1, rho_core2)

    # Plasma mantle for rho_core < rho < 1
    rho_sep = np.linspace(rho_ped, 0.99, n_points_mantle)

    return np.append(rho_core, rho_sep)


def core_electron_density_temperature_profile(
    params: MidplaneProfilesParams, rho_core: np.ndarray, rho_ped: np.ndarray
) -> MidplaneProfiles:
    """
    Calculation of electron density and electron temperature,
    as function of rho, from the magnetic axis to the separatrix,
    along the midplane.

    Parameters
    ----------
    params:
        midplane parameters
    rho_core:
        dimensionless core radius. Values between 0 and 1
    rho_ped:
        dimensionless pedestal radius. Values between 0 and 1

    Returns
    -------
    :
        electron density and electron temperature, stored as a MidplaneProfiles object.

    Notes
    -----
    With :math:`\\rho=r/a`, the core profiles, for
    :math:`0<\\rho\\leq\\rho_p`, are

    .. math::

        n_e(\\rho)=n_{e,p}+(n_{e,0}-n_{e,p})
        [1-(\\rho/\\rho_p)^2]^{\\alpha_n},

    .. math::

        T_e(\\rho)=T_{e,p}+(T_{e,0}-T_{e,p})
        [1-(\\rho/\\rho_p)^{\\beta_T}]^{\\alpha_T}.

    In the mantle, :math:`\\rho_p<\\rho<1`, either quantity
    :math:`f\\in\\{n_e,T_e\\}` is linearly interpolated:

    .. math::

        f(\\rho)=f_{sep}+(f_p-f_{sep})\\frac{1-\\rho}{1-\\rho_p}.

    Subscripts 0, p, and sep denote axis, pedestal, and separatrix values.
    ``t_beta`` supplies :math:`\\beta_T`. Densities are in [1/m³] and
    temperatures are converted to [keV]. The returned flux coordinate is
    the model assumption :math:`\\psi_N=\\rho^2`. Input radii must lie
    strictly between 0 and 1, with :math:`0<\\rho_p<1`.
    """
    i_core = np.nonzero((rho_core > 0) & (rho_core <= rho_ped))[0]
    te0_keV = cast("float", params.T_e_0.value_as("keV"))
    teped_keV = cast("float", params.T_e_ped.value_as("keV"))
    tesep_keV = cast("float", params.T_e_sep.value_as("keV"))

    n_grad_ped0 = params.n_e_0.value - params.n_e_ped.value
    t_grad_ped0 = te0_keV - teped_keV

    rho_ratio_n = (1 - ((rho_core[i_core] ** 2) / (rho_ped**2))) ** params.alpha_n.value

    rho_ratio_t = (
        1 - ((rho_core[i_core] ** params.t_beta.value) / (rho_ped**params.t_beta.value))
    ) ** params.alpha_t.value

    ne_i = params.n_e_ped.value + (n_grad_ped0 * rho_ratio_n)
    te_i = teped_keV + (t_grad_ped0 * rho_ratio_t)

    i_pedestal = np.nonzero((rho_core > rho_ped) & (rho_core < 1))[0]

    n_grad_sepped = params.n_e_ped.value - params.n_e_sep.value
    t_grad_sepped = teped_keV - tesep_keV

    rho_ratio = (1 - rho_core[i_pedestal]) / (1 - rho_ped)

    ne_e = params.n_e_sep.value + (n_grad_sepped * rho_ratio)
    te_e = tesep_keV + (t_grad_sepped * rho_ratio)

    return MidplaneProfiles(rho_core**2, np.append(ne_i, ne_e), np.append(te_i, te_e))
