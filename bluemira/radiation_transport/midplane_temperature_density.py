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

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame


@dataclass
class MidplaneProfilesParams(ParameterFrame):
    n_e_sep: Parameter[float]
    T_e_sep: Parameter[float]
    n_e_0: Parameter[float]
    T_e_0: Parameter[float]
    rho_ped_n: Parameter[float]
    rho_ped_t: Parameter[float]
    n_e_ped: Parameter[float]
    T_e_ped: Parameter[float]
    alpha_n: Parameter[float]
    alpha_t: Parameter[float]
    t_beta: Parameter[float]
    P_sep: Parameter[float]
    fw_lambda_q_near_omp: Parameter[float]
    fw_lambda_q_far_omp: Parameter[float]
    fw_lambda_q_near_imp: Parameter[float]
    fw_lambda_q_far_imp: Parameter[float]
    k_0: Parameter[float]
    gamma_sheath: Parameter[float]
    eps_cool: Parameter[float]
    f_ion_t: Parameter[float]
    det_t: Parameter[float]
    lfs_p_fraction: Parameter[float]
    theta_outer_target: Parameter[float]
    theta_inner_target: Parameter[float]


class MidplaneProfiles:
    """
    Specific class to calculate the core radiation source.
    Temperature and density are assumed to be constant along a
    single flux tube.
    """

    param_cls = MidplaneProfilesParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
    ):
        self.params = make_parameter_frame(params, self.param_cls)

        # Adimensional radius at the mid-plane.
        # From the core to the last closed flux surface
        rho_core = self.collect_rho_core_values()

        # For each flux tube, density and temperature at the mid-plane
        (
            self.ne_mp,
            self.te_mp,
            self.psi_n,
        ) = self.core_electron_density_temperature_profile(rho_core)

    def collect_rho_core_values(self) -> np.ndarray:
        """
        Calculation of core dimensionless radial coordinate rho.

        Returns
        -------
        rho_core:
            dimensionless core radius. Values between 0 and 1
        """
        # The plasma bulk is divided into plasma core and plasma mantle according to rho
        # rho is a nondimensional radial coordinate: rho = r/a (r varies from 0 to a)
        self.rho_ped = (self.params.rho_ped_n.value + self.params.rho_ped_t.value) / 2.0

        # Plasma core for rho < rho_core
        rho_core1 = np.linspace(0.01, 0.95 * self.rho_ped, 30)
        rho_core2 = np.linspace(0.95 * self.rho_ped, self.rho_ped, 15)
        rho_core = np.append(rho_core1, rho_core2)

        # Plasma mantle for rho_core < rho < 1
        rho_sep = np.linspace(self.rho_ped, 0.99, 10)

        return np.append(rho_core, rho_sep)

    def core_electron_density_temperature_profile(
        self, rho_core: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Calculation of electron density and electron temperature,
        as function of rho, from the magnetic axis to the separatrix,
        along the midplane.

        Parameters
        ----------
        rho_core:
            dimensionless core radius. Values between 0 and 1

        Returns
        -------
        ne:
            electron densities at the mid-plane. Unit [1/m^3]
        te:
            electron temperature at the mid-plane. Unit [keV]
        psi_n:
            rho_core**2

        Notes
        -----
        The region that extends through the plasma core until its
        outer layer, named pedestal, is referred as "interior".
        The region that extends from the pedestal to the separatrix
        is referred as "exterior".
        """
        i_interior = np.where((rho_core >= 0) & (rho_core <= self.rho_ped))[0]
        te0_keV = self.params.T_e_0.value_as("keV")
        teped_keV = self.params.T_e_ped.value_as("keV")
        tesep_keV = self.params.T_e_sep.value_as("keV")

        n_grad_ped0 = self.params.n_e_0.value - self.params.n_e_ped.value
        t_grad_ped0 = te0_keV - teped_keV

        rho_ratio_n = (
            1 - ((rho_core[i_interior] ** 2) / (self.rho_ped**2))
        ) ** self.params.alpha_n.value

        rho_ratio_t = (
            1
            - (
                (rho_core[i_interior] ** self.params.t_beta.value)
                / (self.rho_ped**self.params.t_beta.value)
            )
        ) ** self.params.alpha_t.value

        ne_i = self.params.n_e_ped.value + (n_grad_ped0 * rho_ratio_n)
        te_i = teped_keV + (t_grad_ped0 * rho_ratio_t)

        i_exterior = np.where((rho_core > self.rho_ped) & (rho_core <= 1))[0]

        n_grad_sepped = self.params.n_e_ped.value - self.params.n_e_sep.value
        t_grad_sepped = teped_keV - tesep_keV

        rho_ratio = (1 - rho_core[i_exterior]) / (1 - self.rho_ped)

        ne_e = self.params.n_e_sep.value + (n_grad_sepped * rho_ratio)
        te_e = tesep_keV + (t_grad_sepped * rho_ratio)

        ne_core = np.append(ne_i, ne_e)
        te_core = np.append(te_i, te_e)
        psi_n = rho_core**2

        return ne_core, te_core, psi_n
