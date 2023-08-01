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
import numpy as np

from typing import Dict, Union
from bluemira.base.parameter_frame import ParameterFrame


class MidplaneProfiles:
    """
    Specific class to calculate the core radiation source.
    Temperature and density are assumed to be constant along a
    single flux tube.
    """

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
    ):
        self.params = self._make_params(params)
        

        # Adimensional radius at the mid-plane.
        # From the core to the last closed flux surface
        rho_core = self.collect_rho_core_values()

        # For each flux tube, density and temperature at the mid-plane
        self.ne_mp, self.te_mp, self.psi_n = self.core_electron_density_temperature_profile(rho_core)

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
        psi_n = rho_core**2

        return ne_core, te_core, psi_n
    
    def _make_params(self, config):
        """Convert the given params to ``MidplaneProfilesParams``"""
        if isinstance(config, dict):
            try:
                return MidplaneProfilesParams(**config)
            except TypeError:
                unknown = [
                    k for k in config if k not in fields(MidplaneProfilesParams)
                ]
                raise TypeError(f"Unknown config parameter(s) {str(unknown)[1:-1]}")
        elif isinstance(config, MidplaneProfilesParams):
            return config
        else:
            raise TypeError(
                "Unsupported type: 'config' must be a 'dict', or "
                "'ChargedParticleSolverParams' instance; found "
                f"'{type(config).__name__}'."
            )
        
@dataclass
class MidplaneProfilesParams:
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

    