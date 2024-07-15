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

import numpy as np

from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.base.parameter_frame.typing import ParameterFrameLike


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


def midplane_profiles(params: ParameterFrameLike):
    """
    Calculate the core radiation source profiles.

    Temperature and density are assumed to be constant along a
    single flux tube.
    """
    params = make_parameter_frame(params, MidplaneProfilesParams)
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


    Notes
    -----
    The region that extends through the plasma core until its
    outer layer is referred as core.
    The region that extends from the pedestal to the separatrix
    is referred as pedestal.
    """
    i_core = np.nonzero((rho_core > 0) & (rho_core <= rho_ped))[0]
    te0_keV = params.T_e_0.value_as("keV")
    teped_keV = params.T_e_ped.value_as("keV")
    tesep_keV = params.T_e_sep.value_as("keV")

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
