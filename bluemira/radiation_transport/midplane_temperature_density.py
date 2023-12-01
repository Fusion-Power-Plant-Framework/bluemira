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
from bluemira.base.parameter_frame import ParameterFrame, Parameter, make_parameter_frame


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
        param_cls = MidplaneProfilesParams
        self.params = make_parameter_frame(params, param_cls)        

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
        self.rho_ped = (self.params.rho_ped_n.value + self.params.rho_ped_t.value) / 2.0

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

        n_grad_ped0 = self.params.n_e_0.value - self.params.n_e_ped.value
        t_grad_ped0 = self.params.T_e_0.value_as('keV') - self.params.T_e_ped.value_as('keV')

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
        te_i = self.params.T_e_ped.value_as('keV') + (t_grad_ped0 * rho_ratio_t)

        i_exterior = np.where((rho_core > self.rho_ped) & (rho_core <= 1))[0]

        n_grad_sepped = self.params.n_e_ped.value - self.params.n_e_sep.value
        t_grad_sepped = self.params.T_e_ped.value_as('keV') - self.params.T_e_sep.value_as('keV')

        rho_ratio = (1 - rho_core[i_exterior]) / (1 - self.rho_ped)

        ne_e = self.params.n_e_sep.value + (n_grad_sepped * rho_ratio)
        te_e = self.params.T_e_sep.value_as('keV') + (t_grad_sepped * rho_ratio)

        ne_core = np.append(ne_i, ne_e)
        te_core = np.append(te_i, te_e)
        psi_n = rho_core**2

        return ne_core, te_core, psi_n
        
@dataclass
class MidplaneProfilesParams(ParameterFrame):
    n_e_sep: Parameter[float] = Parameter(name="n_e_sep", long_name="Electron density at separatrix", value=3e19, unit='1/m^3', source="default")
    T_e_sep: Parameter[float] = Parameter(name="T_e_sep", long_name="Electron temperature at separatrix", value=2e-01, unit="keV", source="default")
    n_e_0: Parameter[float] =  Parameter(name="n_e_0", long_name="Electron density on axis", value=1.81e+20, unit="1/m^3", source="default")
    T_e_0: Parameter[float] =  Parameter(name="T_e_0", long_name="Electron temperature on axis", value=2.196e+01, unit="keV", source="default")
    rho_ped_n: Parameter[float] =  Parameter(name="rho_ped_n", long_name="Density pedestal r/a location", value=9.4e-01, unit="dimensionless", source="default")
    rho_ped_t: Parameter[float] =  Parameter(name="rho_ped_t", long_name="Temperature pedestal r/a location", value=9.76e-01 , unit="dimensionless", source="default")
    n_e_ped: Parameter[float] =  Parameter(name="n_e_ped", long_name="Electron density pedestal height", value=1.086e+20, unit="1/m^3", source="default")
    T_e_ped: Parameter[float] =  Parameter(name="T_e_ped", long_name="Electron temperature pedestal height", value=3.74, unit="keV", source="default")
    alpha_n: Parameter[float] =  Parameter(name="alpha_n", long_name="Density profile factor", value=1.15, unit="dimensionless", source="default")
    alpha_t: Parameter[float] =  Parameter(name="alpha_t", long_name="Temperature profile index", value=1.905, unit="dimensionless", source="default")
    t_beta: Parameter[float] =  Parameter(name="t_beta", long_name="Temperature profile index beta", value=2, unit="dimensionless", source="default")
    P_sep: Parameter[float] =  Parameter(name='P_sep', long_name='Radiation power', value=150, unit='MW', source="default")
    fw_lambda_q_near_omp: Parameter[float] =  Parameter(name='fw_lambda_q_near_omp', long_name='Lambda_q near SOL omp', value=0.002, unit='m', source="default")
    fw_lambda_q_far_omp: Parameter[float] =  Parameter(name='fw_lambda_q_far_omp', long_name='Lambda_q far SOL omp', value=0.10, unit='m', source="default")
    fw_lambda_q_near_imp: Parameter[float] =  Parameter(name='fw_lambda_q_near_imp', long_name='Lambda_q near SOL imp', value=0.002, unit='m', source="default")
    fw_lambda_q_far_imp: Parameter[float] =  Parameter(name='fw_lambda_q_far_imp', long_name='Lambda_q far SOL imp', value=0.10, unit='m', source="default")
    k_0: Parameter[float] =  Parameter(name="k_0", long_name="material's conductivity", value=2000, unit="dimensionless", source="default")
    gamma_sheath: Parameter[float] =  Parameter(name="gamma_sheath", long_name="sheath heat transmission coefficient", value=7, unit="dimensionless", source="default")
    eps_cool: Parameter[float] =  Parameter(name="eps_cool", long_name="electron energy loss", value=25, unit="eV", source="default")
    f_ion_t: Parameter[float] =  Parameter(name="f_ion_t", long_name="Hydrogen first ionization", value=0.01, unit="keV", source="default")
    det_t: Parameter[float] =  Parameter(name="det_t", long_name="Detachment target temperature", value=0.0015, unit="keV", source="default")
    lfs_p_fraction: Parameter[float] =  Parameter(name="lfs_p_fraction", long_name="lfs fraction of SoL power", value=0.9, unit="dimensionless", source="default")
    theta_outer_target: Parameter[float] =  Parameter(name="theta_outer_target", long_name="Outer divertor poloidal angle with the separatrix flux line", value=5, unit="deg", source="default")
    theta_inner_target: Parameter[float] =  Parameter(name="theta_inner_target", long_name="Inner divertor poloidal angle with the separatrix flux line", value=5, unit="deg", source="default")

    